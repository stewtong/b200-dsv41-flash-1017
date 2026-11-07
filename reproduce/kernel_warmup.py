"""Run one explicitly excluded, disjoint V1 warmup before scoped measurement."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import fixed_attribution, fixed_burst, node_pilot, routing_map, runtime, verify_fixed_task

PINS=('image_digest','model_revision','tokenizer_revision','sampling_source','temperature','top_p')


def prepare(config,manifest,measured_tasks,allowed_root='/data'):
    declaration=config.get('kernel_warmup')
    if (config.get('execution_phase')!='fixed-performance' or
            manifest.get('scope_id')!='context-concurrency-336-v1' or
            not isinstance(declaration,dict) or
            set(declaration)!={'tasks_file','source_manifest_sha256','request_timeout_seconds'}):
        raise ValueError('scoped performance requires an explicit excluded kernel warmup')
    path=Path(declaration['tasks_file']);source=path.parent;timeout=declaration['request_timeout_seconds']
    if (not path.is_absolute() or path.is_symlink() or source.is_symlink() or
            path.name!='fixed-tasks.json' or not source.resolve(strict=True).is_relative_to(Path(allowed_root).resolve()) or
            type(timeout) not in (int,float) or not 0<timeout<=1800):
        raise ValueError('invalid excluded warmup source or deadline')
    runtime.verify_evidence_manifest(source)
    seal=hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest()
    if seal!=declaration['source_manifest_sha256']:raise ValueError('warmup source seal changed')
    read=lambda name:json.loads((source/name).read_text())
    origin=read('run-config.json');result=read('prompt-result.json');old=read('manifest.json')
    tasks=read('fixed-tasks.json')
    if (origin.get('task_family')!='fixed' or origin.get('fixed_prompt_version')!='fixed-performance-v1' or
            origin.get('image')!=config['image'] or read('cleanup-result.json').get('success') is not True or
            read('reconciliation.json').get('status')!='complete' or
            result.get('status')!='passed' or result.get('cleanup')!='passed' or result.get('tasks')!=8 or
            len(tasks)!=8 or {t.get('id') for t in tasks}!={'s'+str(i) for i in range(8)} or
            origin.get('cell_id')!=result.get('cell_id') or
            any(old['runtime'].get(k)!=manifest['runtime'].get(k) for k in PINS)):
        raise ValueError('warmup generator or component pins differ')
    prefixes=set()
    for task in tasks:
        verify_fixed_task.verify(task)
        if (task.get('suite')!='anchor' or task.get('prompt_version')!='fixed-performance-v1' or
                task.get('target_input_tokens')!=8000 or task.get('concurrency')!=8 or
                task.get('output_tokens')!=2048 or task.get('cache_designation')!='cold' or task.get('repetition')!=0):
            raise ValueError('warmup must be the declared excluded8K/c8 V1 anchor population')
        prefixes.add(task['session_prefix'])
    if len(prefixes)!=8 or prefixes&{t['session_prefix'] for t in measured_tasks}:
        raise ValueError('kernel warmup overlaps a scored prompt namespace')
    return dict(source=source,source_seal=seal,tasks=tasks,timeout=timeout,
                task_bytes=path.read_bytes(),budget_seconds=timeout+2*fixed_attribution.BOUNDARY_SECONDS)


def run(session,shared,endpoint,config,manifest,prepared,mapping):
    source=prepared['source'];runtime.verify_evidence_manifest(source)
    if (hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest()!=prepared['source_seal'] or
            (source/'fixed-tasks.json').read_bytes()!=prepared['task_bytes']):
        raise ValueError('warmup source changed after preparation')
    root=Path(shared)/'kernel-warmup';root.mkdir()
    shutil.copytree(source,root/'input-source');runtime.verify_evidence_manifest(root/'input-source')
    placement=routing_map.select(mapping,8,0)
    ids={'s'+str(row['task_session']):row['session_header'] for row in placement}
    profile=manifest['runtime']['profile_hashes'][config['layout']]
    namespace=fixed_burst.namespace(config['run_id'],'excluded-kernel-warmup',0,profile)
    block=json.loads((Path(shared)/'cache-config.json').read_text())['block_size']
    started=time.monotonic_ns()
    observed=fixed_attribution.capture(session,endpoint,prepared['tasks'],root/'capture',prepared['timeout'],
        node_pilot.clock_domain(session.run_id),namespace,ids,config['expected_ranks'],Path(shared)/'router.log',block)
    if any(observed['session_routes'].get(row['session_header'])!=[row['endpoint'],row['rank']] for row in placement):
        raise ValueError('warmup route differs from frozen placement; scored dispatch prohibited')
    runtime.atomic_json(root/'warmup-result.json',dict(schema='excluded-kernel-warmup/v1',
        status='captured_cleanup_pending',source_seal=prepared['source_seal'],
        mapping_sha256=mapping['mapping_sha256'],placement=placement,started_ns=started,
        completed_ns=time.monotonic_ns(),scored_requests=0,
        scope='one excluded8K/c8 kernel warmup; no claim that every later context/graph shape is warmed',
        cold_cache_label='cache-busted cold; unique early system namespaces, no verified cache flush or reset'))
    runtime.seal_evidence(root)
    import verify_kernel_warmup
    verify_kernel_warmup.verify(root,[],manifest,config,mapping,time.monotonic_ns())
