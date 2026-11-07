"""Generate pilot tasks with the qualified image's tokenizer under supervision."""
import json
from pathlib import Path
import time

import campaign
import fixed_workload
import node_pilot
import node_smoke
import runtime
import supervisor
import verify_quality
import verify_fixed_task
import toolkit_integrity
import hashlib
import re


def reference_source(config,family,cell,allowed_root='/data/hf-cache'):
    reference=config.get('calibration_reference')
    if reference is None:return None
    if cell.get('cache')=='warm_prefix':raise ValueError('versioned warm report input cannot use a legacy calibration reference')
    if (family!='pilot' or cell['concurrency']!=1 or not isinstance(reference,dict) or
        set(reference)!={'manifest','sha256'} or not re.fullmatch(r'[a-f0-9]{64}',str(reference['sha256']))):
        raise ValueError('calibration reference requires a single pilot input and exact source pin')
    manifest=Path(reference['manifest'])
    if (manifest.is_symlink() or manifest.parent.is_symlink() or manifest.name!='MANIFEST.json' or
        manifest.parent.resolve().parent!=Path(allowed_root).resolve() or
        not re.fullmatch(r'dsv41-toolkit-[0-9]{8}T[0-9]{4}Z',manifest.parent.name) or
        manifest.parent.resolve()==Path(__file__).resolve().parents[1]):
        raise ValueError('calibration reference must be another immutable toolkit')
    if toolkit_integrity.verify(manifest)!=reference['sha256']:raise ValueError('calibration reference source changed')
    return manifest.parent/'reproduce'


def fixed_batch(session,cid,cell,repetition,block_size,timeout,root,prompt_version=fixed_workload.VERSION):
    if time.monotonic_ns()+int((timeout+120)*1e9)>session.deadline_ns:
        raise runtime.RuntimeErrorClosed('fixed batch and cleanup do not fit deadline')
    args=['sudo','-n','docker','exec',cid,'python3','-B','/campaign/fixed_workload.py','/model',
        str(cell['input_tokens']),str(cell['output_tokens']),cell['suite'],str(cell['concurrency']),str(repetition),'0',
        '--cache-state',cell['cache'],'--batch','--prompt-version',prompt_version]
    if block_size is not None:args+=['--block-size',str(block_size)]
    session.ensure_healthy();began=time.monotonic_ns()
    runtime.atomic_json(root/'generation-batch-intent.json',{'argv':args,'timeout_seconds':timeout,'started_ns':began})
    try:output=runtime.command(args,timeout=timeout)
    except BaseException as exc:
        for name in ('stdout','stderr'):
            value=getattr(exc,name,None)
            if isinstance(value,(str,bytes)):runtime.atomic_bytes(root/('generation-batch-error-'+name+'.txt'),value.encode() if isinstance(value,str) else value)
        runtime.atomic_json(root/'generation-batch-failure.json',{'type':type(exc).__name__,'started_ns':began,'completed_ns':time.monotonic_ns()})
        raise
    runtime.atomic_bytes(root/'generation-batch-stdout.txt',output.encode())
    tasks=json.loads(output.strip().splitlines()[-1])
    if not isinstance(tasks,list) or len(tasks)!=cell['concurrency'] or [t.get('session') for t in tasks]!=list(range(cell['concurrency'])):
        raise ValueError('fixed batch task population or ordering differs')
    for task in tasks:verify_fixed_task.verify(task)
    session.ensure_healthy()
    runtime.atomic_json(root/'generation-batch-result.json',{'status':'tokenizer_verified_tasks_captured',
        'tasks':len(tasks),'started_ns':began,'completed_ns':time.monotonic_ns(),
        'stdout_sha256':hashlib.sha256(output.encode()).hexdigest(),
        'scope':'one installed tokenizer process; inference and live usage remain unverified'})
    return tasks


def run(config):
    image_q, _, seconds = node_smoke.validate(config)
    matrix_path = Path(config['matrix'])
    # validate() already binds the manifest to the installed matrix. Select the
    # source comparison here without adding an early dependency before cell
    # and cache-contract validation.
    matrix_name='comparison-matrix.json' if node_smoke.read(matrix_path).get('schema')=='dsv41-b200-campaign/v3' else 'matrix.json'
    if matrix_path.read_bytes() != (campaign.HERE/matrix_name).read_bytes():
        raise ValueError('matrix differs from campaign source')
    matches = [x for x in campaign.ledger(matrix_path)['cells'] if x['id'] == config['cell_id']]
    if len(matches) != 1:
        raise ValueError('unknown cell')
    cell = matches[0]
    family=config.get('task_family','pilot')
    if family not in ('pilot','quality','fixed'):raise ValueError('unknown prompt task family')
    if cell['layout'] != config['layout']:raise ValueError('prompt layout mismatch')
    if family=='pilot' and cell['suite'] not in ('pilot_smoke','pilot_fit','pilot_concurrency','pilot_warm'):
        raise ValueError('only the selected pilot cell is supported')
    repetition=config.get('repetition',0)
    if type(repetition) is not int or not 0<=repetition<cell['repetitions']:raise ValueError('invalid prompt repetition')
    if family=='pilot' and repetition!=0:raise ValueError('pilot has one repetition')
    if family=='quality' and cell['suite']!='quality':
        raise ValueError('quality generator requires a quality cell')
    if family=='fixed' and cell['suite'] not in fixed_workload.SUITES:
        raise ValueError('fixed generator requires a declared fixed-output grid cell')
    prompt_version=config.get('fixed_prompt_version',fixed_workload.VERSION)
    if family=='fixed' and prompt_version not in fixed_workload.EXECUTION_VERSIONS:raise ValueError('unknown fixed prompt version')
    quality_cap=config.get('quality_output_cap',512)
    if family=='quality' and (type(quality_cap) is not int or quality_cap!=512):
        raise ValueError('unfrozen quality generation uses the declared initial512 cap only')
    block_size = None
    if cell['cache']=='warm_prefix':
        if not config.get('cache_config_evidence'): raise ValueError('live cache block size evidence required')
        evidence = Path(config['cache_config_evidence'])
        prior_root = evidence.parent
        runtime.verify_evidence_manifest(prior_root)
        prior = node_smoke.read(prior_root/'run-config.json')
        result = node_smoke.read(prior_root/'smoke-result.json')
        cache = node_smoke.read(evidence)
        old_manifest = node_smoke.read(prior_root/'manifest.json')
        new_manifest = node_smoke.read(config['manifest'])
        if (old_manifest.get('plan_sha256') != new_manifest.get('plan_sha256') or
            old_manifest.get('matrix_sha256') != new_manifest.get('matrix_sha256') or
            old_manifest['runtime']['starting_profiles'].get(config['layout']) != new_manifest['runtime']['starting_profiles'].get(config['layout'])):
            raise ValueError('prior cache evidence does not match current profile')
        if (evidence.name != 'cache-config.json' or prior.get('layout') != config['layout'] or
            prior.get('image') != config['image'] or result.get('status') != 'passed' or
            not isinstance(cache.get('block_size'),int) or cache['block_size'] <= 0):
            raise ValueError('prior live cache block size evidence invalid')
        block_size = cache['block_size']
    per_task = config.get('prompt_timeout_seconds')
    if not isinstance(per_task, int) or per_task < 1 or per_task > 3600:
        raise ValueError('invalid prompt generation timeout')
    reference=reference_source(config,family,cell)
    batch=config.get('batch_fixed_generation',False);batch_timeout=config.get('prompt_batch_timeout_seconds')
    if (type(batch) is not bool or (batch and (family!='fixed' or reference is not None or
            type(batch_timeout) is not int or not 1<=batch_timeout<=3600 or cell['input_tokens']*cell['concurrency']>8_000_000)) or
            not batch and batch_timeout is not None):
        raise ValueError('fixed batch generation requires declared fixed inputs and a bounded aggregate timeout')
    root = Path(config['run_dir'])
    run_id = config['run_id']
    deadline_ns = time.monotonic_ns() + int(seconds * 1e9)
    source = Path(__file__).resolve().parent
    model = Path(config['model_dir']).resolve(strict=True)
    if ',' in str(source) or ',' in str(model):
        raise ValueError('mount path contains comma')
    with supervisor.Supervisor(node_smoke.NODE_LOCK, root, run_id, deadline_ns, phase='prompt-generation') as session:
        runtime.atomic_json(root/'run-config.json', config)
        for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
            runtime.atomic_bytes(root/name,Path(config[key]).read_bytes())
        if reference is not None:
            (root/'reference-source').mkdir()
            for source_name,destination in (('vllm_prompt.py','vllm_prompt.py'),('workload.py','workload.py')):
                runtime.atomic_bytes(root/'reference-source'/destination,(reference/source_name).read_bytes())
            runtime.atomic_bytes(root/'reference-source/MANIFEST.json',(reference.parent/'MANIFEST.json').read_bytes())
        limit = time.monotonic() + 15
        while not (root/'watchdog-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic() >= limit: raise TimeoutError('watchdog startup timeout')
            time.sleep(.1)
        session.preflight([])
        runtime.atomic_json(root/'retirement-preflight.json', runtime.verify_retirement(config))
        argv = ['docker','run','-d','--network','none','--gpus','all','--restart','no',
                '--label','campaign=dsv41-1017','--label','run_id='+run_id,
                '--mount',f'type=bind,src={source},dst=/campaign,readonly',
                '--mount',f'type=bind,src={model},dst=/model,readonly',
                *(['--mount',f'type=bind,src={reference},dst=/reference,readonly'] if reference is not None else []),
                '--entrypoint','sleep',config['image'],'infinity']
        cid = session.create_start(argv)
        row = runtime.inspect_owned(cid, run_id)
        if row.get('Image') != image_q['image_id'] or row.get('State',{}).get('Running') is not True:
            raise runtime.RuntimeErrorClosed('prompt container image mismatch')
        tasks = []
        batch_tasks=fixed_batch(session,cid,cell,repetition,block_size,batch_timeout,root,prompt_version) if batch else None
        for index in range(cell['concurrency']):
            session.ensure_healthy()
            if not batch and time.monotonic_ns() + (per_task*(2 if reference is not None else 1) + 120)*1_000_000_000 > deadline_ns:
                raise runtime.RuntimeErrorClosed('prompt deadline and cleanup do not fit')
            if family=='quality':
                args=['sudo','-n','docker','exec',cid,'python3','-B','/campaign/quality_tasks.py','/model',
                      str(cell['input_tokens']),str(quality_cap),str(cell['concurrency']),str(repetition),str(index)]
            elif family=='fixed':
                args=['sudo','-n','docker','exec',cid,'python3','-B','/campaign/fixed_workload.py','/model',
                      str(cell['input_tokens']),str(cell['output_tokens']),cell['suite'],
                      str(cell['concurrency']),str(repetition),str(index),'--cache-state',cell['cache']]
                if prompt_version!=fixed_workload.VERSION:args+=['--prompt-version',prompt_version]
            else:
                script='/campaign/warm_pilot_prompt.py' if cell['suite']=='pilot_warm' else '/campaign/vllm_prompt.py'
                args = ['sudo','-n','docker','exec',cid,'python3','-B',script,'/model',
                        str(cell['input_tokens']),str(cell['output_tokens']),cell['suite'],
                        str(cell['concurrency']),'0',str(index)]
            if family=='quality' and cell['cache']=='warm_prefix':args+=['--cache-state','warm_prefix']
            if block_size is not None: args += ['--block-size',str(block_size)]
            if batch:task=batch_tasks[index]
            else:
                began=time.monotonic();output = runtime.command(args, timeout=per_task);candidate_seconds=time.monotonic()-began
                session.ensure_healthy()
                task = json.loads(output.strip().splitlines()[-1])
            if reference is not None:
                runtime.atomic_json(root/('generation-candidate-%03d.json'%index),task)
                if args.count('/campaign/vllm_prompt.py')!=1:raise ValueError('reference generator path differs')
                old_args=list(args);old_args[old_args.index('/campaign/vllm_prompt.py')]='/reference/vllm_prompt.py'
                began=time.monotonic();old_output=runtime.command(old_args,timeout=per_task);reference_seconds=time.monotonic()-began
                session.ensure_healthy();old_task=json.loads(old_output.strip().splitlines()[-1])
                runtime.atomic_json(root/('generation-reference-%03d.json'%index),old_task)
                parity={'status':'matched' if task==old_task else 'mismatch','reference_source_sha256':config['calibration_reference']['sha256'],
                    'candidate_seconds':candidate_seconds,'reference_seconds':reference_seconds,
                    'scope':'ordered installed-generator byte-equivalent task comparison; elapsed times are observations, not isolated speedup evidence'}
                runtime.atomic_json(root/('generation-parity-%03d.json'%index),parity)
                if task!=old_task:raise ValueError('installed reference generator task differs')
            runtime.atomic_json(root/('task-%03d.json'%index), task)
            tasks.append(task)
        if family=='pilot':node_pilot.prepare(matrix_path,config['cell_id'],tasks)
        elif family=='quality':
            for task in tasks:
                if task.get('cache_designation')!=cell['cache'] or task.get('prompt_version')!='quality-cache-v2':
                    raise ValueError('quality prompt cache state or version mismatch')
                verify_quality.evaluate(task,{'request_id':task['id'],'complete':False},repetition,cell['concurrency'],cell['input_tokens'])
        else:
            import verify_fixed_task
            for index,task in enumerate(tasks):
                if (task.get('cache_designation')!=cell['cache'] or task.get('prompt_version')!=prompt_version or
                    task.get('suite')!=cell['suite'] or task.get('target_input_tokens')!=cell['input_tokens'] or
                    task.get('output_tokens')!=cell['output_tokens'] or task.get('concurrency')!=cell['concurrency'] or
                    task.get('repetition')!=repetition or task.get('session')!=index):
                    raise ValueError('fixed task differs from selected cell')
                verify_fixed_task.verify(task)
        runtime.atomic_json(root/{'pilot':'pilot-tasks.json','quality':'quality-tasks.json','fixed':'fixed-tasks.json'}[family], tasks)
    runtime.atomic_json(root/'prompt-result.json', {'status':'passed','cleanup':'passed',
                        'cell_id':cell['id'],'tasks':len(tasks),'task_family':family,'repetition':repetition,
                        'scope':'tokenizer-generated; response usage unverified'})
    runtime.seal_evidence(root)
