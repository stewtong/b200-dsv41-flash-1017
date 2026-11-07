"""Run explicitly prepared pilot cells in one bounded supervised engine session."""
import hashlib
import json
import os
import os
from pathlib import Path
import time

import attempts
import campaign
import cancellation_control
import node_pilot
import node_smoke
import pilot_registry
import runtime
import verify_pilot
import verify_pilot_block

ALLOWED={'cell_id','tasks_file','request_timeout_seconds'}


def retain_shutdown_atomic_metadata(root,state):
    """Retain only a retired owned collector's noncanonical atomic snapshot.

    The canonical head and every raw sample remain unchanged. Unknown temporary
    files and measured failure records still prevent normal acceptance.
    """
    root=Path(root);identity=state.get('nvml_identity')
    if not isinstance(identity,dict):return None
    original=root/('nvml-live.json.tmp.'+str(identity['pid']))
    if not original.exists() and not original.is_symlink():return None
    if (original.is_symlink() or not original.is_file() or original.stat().st_nlink!=1 or
            original.stat().st_size>1_000_000):
        raise runtime.RuntimeErrorClosed('unsafe owned atomic metadata')
    if (not state.get('stopping') or not state.get('watchdog_stop') or
            identity not in state.get('owned_processes',[]) or
            any(runtime.process_alive(p) for p in [state['watchdog_identity'],*state['owned_processes']])):
        raise runtime.RuntimeErrorClosed('atomic metadata writer or shutdown remains unresolved')
    owner=state['driver_identity']
    if runtime.process_alive(owner) and owner!=runtime.process_identity(os.getpid()):
        raise runtime.RuntimeErrorClosed('another live driver owns atomic metadata')
    if (node_smoke.read(root/'cleanup-result.json').get('success') is not True or
            node_smoke.read(root/'gpu-idle-after-cleanup.json').get('idle') is not True):
        raise runtime.RuntimeErrorClosed('atomic metadata retention requires passed cleanup and idle proof')
    for name in ('driver-failure.json','watchdog-failure.json','sampler-failure.json',
                 'nvml-measurement-failure.json','nvml-cadence-failure.json'):
        if (root/name).exists():raise runtime.RuntimeErrorClosed('measured failure cannot be repaired by metadata retention')
    head=root/'nvml-live.json'
    if head.is_symlink() or node_smoke.read(head).get('pid')!=identity['pid']:
        raise runtime.RuntimeErrorClosed('canonical collector head differs')
    before=original.stat();data=original.read_bytes();after=original.stat()
    fingerprint=lambda s:(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if fingerprint(before)!=fingerprint(after):raise runtime.RuntimeErrorClosed('owned atomic metadata changed')
    parsed=None
    try:
        value=json.loads(data)
        if isinstance(value,dict):parsed={k:value.get(k) for k in ('pid','monotonic_ns','measurement_mode')}
    except (ValueError,UnicodeError):pass
    if parsed is not None and parsed['pid']!=identity['pid']:
        raise runtime.RuntimeErrorClosed('interrupted atomic metadata writer differs')
    parent=root/'postprocessing-forensics'
    if parent.is_symlink() or parent.exists() and not parent.is_dir():
        raise runtime.RuntimeErrorClosed('unsafe atomic metadata retention parent')
    parent.mkdir(exist_ok=True);target_dir=parent/'owned-atomic-metadata';target_dir.mkdir()
    target=target_dir/'nvml-live-interrupted-atomic.json'
    original.rename(target)
    digest=hashlib.sha256(data).hexdigest()
    if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
        raise runtime.RuntimeErrorClosed('retained atomic metadata bytes differ')
    receipt={'schema':'retained-owned-atomic-metadata/v1','run_id':state['run_id'],
        'writer_identity':identity,'original_path':original.name,'retained_path':target.relative_to(root).as_posix(),
        'sha256':digest,'bytes':len(data),'original_mtime_ns':before.st_mtime_ns,'parsed_snapshot':parsed,
        'canonical_head_unchanged':True,'scope':'Noncanonical interrupted snapshot retained after writer death; raw measured evidence and all acceptance checks remain unchanged.'}
    runtime.atomic_json(root/'interrupted-owned-metadata.json',receipt)
    return receipt


def project_shared(root,dest,owner,*,schema='shared-pilot-session/v1'):
    """Retain portable shared evidence, including the raw KV baseline excerpt."""
    if schema not in ('shared-pilot-session/v1','shared-fixed-qualification-session/v1'):
        raise ValueError('unknown shared session projection schema')
    root=Path(root);dest=Path(dest);config=node_smoke.read(root/'run-config.json')
    if (root.is_symlink() or dest.is_symlink() or not dest.is_dir() or dest.parent.resolve()!= (root/'cells').resolve() or
        not __import__('re').fullmatch(r'[A-Za-z0-9_.-]+',dest.name)):
        raise ValueError('shared projection destination is not an exact cell directory')
    state=node_smoke.read(root/'execution-state.json')
    if state.get('driver_identity')!=owner or state.get('run_id')!=config.get('run_id'):
        raise ValueError('shared projection driver or run identity differs')
    retain_shutdown_atomic_metadata(root,state)
    files=[p for p in root.iterdir() if p.is_file() and p.name!='SHA256SUMS']
    baseline=root/'kv-baseline-source'
    if config.get('kv_baseline_evidence') is not None and not (baseline/'SOURCE_SHA256SUMS').is_file():
        raise ValueError('shared KV baseline source is incomplete')
    if baseline.exists():
        if baseline.is_symlink() or not baseline.is_dir() or any(p.is_symlink() or not p.is_file() for p in baseline.iterdir()):
            raise ValueError('unsafe shared KV baseline excerpt')
        files.extend(baseline.iterdir())
    for name in ('worker-observation','worker-observer-source','worker-source-proof','worker-allocator-source-proof','kernel-warmup'):
        source=root/name
        if source.exists():
            if source.is_symlink() or not source.is_dir() or any(p.is_symlink() for p in source.rglob('*')):
                raise ValueError('unsafe shared worker observation source')
            files.extend(p for p in source.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in files):raise ValueError('symlink in shared projection')
    shared_raw=[p for p in files if p.suffix=='.jsonl' and
                (p.name.startswith(('telemetry-','sampler-timing-','nvml-samples-','safety-')) or
                 p.is_relative_to(root/'worker-observation') and p.name.startswith('samples-')) and p.stat().st_size>=1_000_000]
    if shared_raw:
        finished=(state.get('stopping') and state.get('watchdog_stop') and
                  (root/'smoke-result.json').exists() and node_smoke.read(root/'smoke-result.json').get('status')=='passed')
        if (not isinstance(state.get('watchdog_identity'),dict) or
                any(runtime.process_alive(p) for p in [state['watchdog_identity'],*state.get('owned_processes',[])]) or
                not finished and runtime.process_alive(owner)):
            raise runtime.RuntimeErrorClosed('shared raw stream still has a possible writer')
    hashes={}
    for file in sorted(files):
        name='shared-run-config.json' if file==root/'run-config.json' else file.relative_to(root).as_posix()
        path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
        if file in shared_raw:
            # POSIX rename can be a successful no-op when both names already
            # link the same inode. Avoid creating an alias in that case.
            if not path.exists() or not os.path.samefile(file,path):
                temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
                os.link(file,temporary);os.replace(temporary,path)
                fd=os.open(path.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
        else:runtime.atomic_bytes(path,file.read_bytes())
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    runtime.atomic_json(dest/'block-provenance.json',{'schema':schema,
        'run_id':config['run_id'],'driver_identity':owner,'shared_file_sha256':hashes,
        'scope':'one startup and cleanup shared across these explicitly registered cells'})
    return hashes


def prepare(config):
    entries=config.get('cells')
    if not isinstance(entries,list) or not entries or len(entries)>9:
        raise ValueError('one to nine explicit pilot cells required')
    result=[];seen=set();budget=0
    matrix=Path(config['matrix']).read_bytes()
    if matrix!=(campaign.HERE/'matrix.json').read_bytes():raise ValueError('matrix changed')
    for item in entries:
        if (not isinstance(item,dict) or set(item) not in (ALLOWED,ALLOWED|{'prompt_reuse'}) or
                'prompt_reuse' in item and not isinstance(item['prompt_reuse'],dict)):
            raise ValueError('invalid pilot block cell fields')
        child={k:v for k,v in config.items() if k!='cells'};child.update(item)
        if type(child.get('retry',False)) is not bool:raise ValueError('retry must be boolean')
        if child['cell_id'] in seen:raise ValueError('duplicate pilot block cell')
        seen.add(child['cell_id'])
        data=node_pilot.verified_tasks(child);tasks=json.loads(data)
        cell=node_pilot.prepare(child['matrix'],child['cell_id'],tasks)
        if cell['layout']!=config['layout']:raise ValueError('mixed block layouts')
        node_pilot.validate_rank_contract(child,cell)
        timeout=child['request_timeout_seconds']
        if type(timeout) not in (int,float) or not 0<timeout<=7200:raise ValueError('invalid pilot deadline')
        if not node_pilot.PILOT_METRICS.issubset(set(child.get('required_metric_names',[]))):
            raise ValueError('required pilot metric contract incomplete')
        budget+=timeout*(2 if cell['cache']=='warm_prefix' else 1)
        result.append({'config':child,'cell':cell,'tasks':tasks,'task_bytes':data,'attribution':{}})
    return result,matrix,budget


def validate_order(prepared,prior):
    """Require recorded lower-rung success before attempting a higher rung."""
    seen=set(prior)
    for row in prepared:
        cell=row['cell'];suite=cell['suite'];n=cell['input_tokens'];c=cell['concurrency']
        if suite!='pilot_smoke' and ('pilot_smoke',8000,1) not in seen:
            raise ValueError('pilot startup prerequisite missing')
        previous={512000:128000,1000000:512000}.get(n)
        if suite=='pilot_fit' and previous and ('pilot_fit',previous,1) not in seen:
            raise ValueError('lower context fit prerequisite missing')
        if suite in ('pilot_warm','pilot_concurrency') and ('pilot_fit',n,1) not in seen:
            raise ValueError('single-request fit prerequisite missing')
        if suite=='pilot_concurrency' and c==32 and ('pilot_concurrency',n,8) not in seen:
            raise ValueError('lower concurrency prerequisite missing')
        seen.add((suite,n,c))


def prior_cells(config):
    observed=[]
    for source in config.get('prior_pilot_evidence',[]):
        root=Path(source);runtime.verify_evidence_manifest(root)
        check=verify_pilot.verify(root)
        old=node_smoke.read(root/'run-config.json');manifest=node_smoke.read(root/'manifest.json')
        current=node_smoke.read(config['manifest'])
        if (check['status']!='passed' or
            any(manifest['runtime'].get(k)!=current['runtime'].get(k) for k in ('model_revision','tokenizer_revision','image_digest')) or
            old.get('layout')!=config['layout'] or
            old.get('image')!=config['image'] or manifest.get('plan_sha256')!=current.get('plan_sha256') or
            manifest['runtime']['starting_profiles'].get(config['layout'])!=current['runtime']['starting_profiles'].get(config['layout'])):
            raise ValueError('prior pilot evidence differs from block profile')
        cell=node_smoke.read(root/'pilot-cell.json')['cell']
        observed.append((cell['suite'],cell['input_tokens'],cell['concurrency']))
    return observed


def run(config):
    prepared,matrix,budget=prepare(config)
    controls=config.get('qualify_cancellation',False)
    if type(controls) is not bool:raise ValueError('cancellation qualification must be boolean')
    if controls:
        shapes=[(r['cell']['suite'],r['cell']['input_tokens'],r['cell']['concurrency'],r['cell']['cache']) for r in prepared]
        if shapes!=[('pilot_fit',128000,1,'cold'),('pilot_fit',512000,1,'cold')]:
            raise ValueError('cancellation qualification requires the128K then512K single-request fit block')
        budget+=360  # Two bounded request/absence/semantic controls, outside measured cells.
    validate_order(prepared,prior_cells(config))
    _,_,seconds=node_smoke.validate(config,session_budget='pilot-block')
    # Reserve startup, sampler qualification, bounded metadata and cleanup. Each
    # capture also checks its deadline against the actual remaining window.
    if budget+3600+600+120>seconds:raise ValueError('pilot block worst-case budget does not fit session')
    root=Path(config['run_dir']);owner=runtime.process_identity(os.getpid())
    specs=[{'cell_id':r['cell']['id'],'method_sha256':pilot_registry.method_fingerprint(
        r['config'],hashlib.sha256(r['task_bytes']).hexdigest())} for r in prepared]
    paths=attempts.begin_block(pilot_registry.DEFAULT_ROOT,specs,config['run_id'],owner,
                               retry=config.get('retry',False),retry_reason=config.get('retry_reason'))
    for row,path in zip(prepared,paths):
        row['attempt']=path
        runtime.atomic_json(path/'context.json',{'run_dir':str(root),'driver_identity':owner,
            'group_id':config['run_id'],'cell_artifact_dir':str(root/'cells'/row['cell']['id']),
            'task_sha256':hashlib.sha256(row['task_bytes']).hexdigest()})
    def capture(session,shared,endpoint):
        for row in prepared:
            session.ensure_healthy()
            runtime.atomic_json(row['attempt']/'dispatch-intent.json',{'cell_id':row['cell']['id'],
                'group_id':config['run_id'],'driver_identity':owner,'monotonic_ns':time.monotonic_ns()})
            cellroot=shared/'cells'/row['cell']['id'];cellroot.mkdir(parents=True)
            row['root']=cellroot
            runtime.atomic_json(cellroot/'run-config.json',row['config'])
            node_pilot.capture_cell(session,cellroot,endpoint,row['config'],row['cell'],row['tasks'],
                row['task_bytes'],matrix,row['attribution'],router_log_path=shared/'router.log',
                cache_config_path=shared/'cache-config.json')
            runtime.atomic_json(row['attempt']/'capture.json',{'cell_artifact_dir':str(cellroot),
                'attribution':row['attribution'],'status':'requests_captured_cleanup_pending'})
            attempts.capture_pending(row['attempt'],['context.json','capture.json'],config['run_id'],owner)
        # Fit is established before this larger unscored recovery request.
        # Recovery controls run after measured cells and use distinct prefixes.
        if controls:
            for mode in ('prefill','decode'):
                cancellation_control.run(session,endpoint,prepared[1]['tasks'][0],config['expected_ranks'],
                    shared/('cancellation-'+mode),mode,node_pilot.clock_domain(session.run_id))
    try:
        node_smoke.run(config,capture,session_budget='pilot-block')
        # Explicitly duplicate retained shared records for portable verification.
        # They describe one shared session, not a separate startup for each cell.
        for row in prepared:
            dest=row['root'];project_shared(root,dest,owner)
            result={'cell_id':row['cell']['id'],'status':'passed','requests_status':'passed',
                'tasks_sha256':hashlib.sha256(row['task_bytes']).hexdigest(),'attribution':row['attribution'],
                'scope':'feasibility pilot observation; not scored performance or a frozen-profile qualification','reason':None}
            runtime.atomic_json(dest/'pilot-result.json',result);runtime.seal_evidence(dest)
        # Verify the complete captured block before marking any attempt passed.
        checked=[verify_pilot.verify(row['root']) for row in prepared]
        runtime.atomic_json(root/'pilot-block-result.json',{'status':'passed','cells':checked,
            'scope':'unscored pilot cells sharing engine startup and cleanup'})
        runtime.seal_evidence(root)
        verify_pilot_block.verify(root)
        for row,verification in zip(prepared,checked):
            pilot_registry.finish(row['attempt'],row['root']/'pilot-result.json',verification)
    except Exception as exc:
        for path in paths:
            if node_smoke.read(path/'attempt.json').get('status')=='running':pilot_registry.fail(path,exc)
        raise
