"""Audit excluded warmup inputs, raw request work and ordering independently."""
import hashlib,json
from pathlib import Path
import routing_map,runtime,verify_fixed_attribution,verify_fixed_task


def verify(directory,measured_tasks,manifest,config,mapping,before_ns):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda path:json.loads(Path(path).read_text())
    result=read(root/'warmup-result.json');source=root/'input-source'
    runtime.verify_evidence_manifest(source)
    source_seal=hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest()
    declaration=config.get('kernel_warmup',{})
    origin=read(source/'run-config.json');generation=read(source/'prompt-result.json');old=read(source/'manifest.json')
    tasks=read(source/'fixed-tasks.json')
    if (set(declaration)!={'tasks_file','source_manifest_sha256','request_timeout_seconds'} or
            source_seal!=declaration['source_manifest_sha256'] or result.get('source_seal')!=source_seal or
            origin.get('task_family')!='fixed' or origin.get('fixed_prompt_version')!='fixed-performance-v1' or
            origin.get('image')!=config['image'] or read(source/'cleanup-result.json').get('success') is not True or
            read(source/'reconciliation.json').get('status')!='complete' or
            origin.get('cell_id')!=generation.get('cell_id') or generation.get('status')!='passed' or
            generation.get('cleanup')!='passed' or generation.get('tasks')!=8 or len(tasks)!=8 or
            any(old['runtime'].get(k)!=manifest['runtime'].get(k) for k in
                ('image_digest','model_revision','tokenizer_revision','sampling_source','temperature','top_p'))):
        raise ValueError('excluded warmup source or frozen component pins differ')
    prefixes=[]
    for task in tasks:
        verify_fixed_task.verify(task)
        if tuple(task.get(k) for k in ('suite','target_input_tokens','concurrency','output_tokens','cache_designation','repetition','prompt_version'))!=('anchor',8000,8,2048,'cold',0,'fixed-performance-v1'):
            raise ValueError('excluded warmup task population differs')
        prefixes.append(task['session_prefix'])
    if len(set(prefixes))!=8 or set(prefixes)&{t['session_prefix'] for t in measured_tasks}:
        raise ValueError('warmup and measurement namespaces overlap')
    captured=read(root/'capture/requests/measured/fixed-inputs.json')
    if captured['tasks']!=tasks:raise ValueError('transmitted warmup differs from retained generator')
    profile=manifest['runtime']['profile_hashes'][config['layout']]
    namespace='fx'+hashlib.sha256(json.dumps([config['run_id'],'excluded-kernel-warmup',0,profile],separators=(',',':')).encode()).hexdigest()[:24]
    contract=read(root/'capture/attribution-contract.json')
    state=read(root.parent/'execution-state.json')
    if (captured.get('request_namespace')!=namespace or
            captured.get('clock_id')!=state['driver_identity']['boot_id']+':'+config['run_id'] or
            contract.get('expected_ranks')!=config['expected_ranks']):
        raise ValueError('warmup belongs to another run, boot or engine population')
    placement=routing_map.select(mapping,8,0)
    if result.get('mapping_sha256')!=mapping['mapping_sha256'] or result.get('placement')!=placement:
        raise ValueError('warmup placement differs from frozen route map')
    checked=verify_fixed_attribution.verify(root/'capture')
    for row in placement:
        if checked['session_routes'].get(row['session_header'])!=[row['endpoint'],row['rank']]:
            raise ValueError('warmup actual route differs')
    requests=[read(root/'capture/requests/measured'/('s'+str(i)+'.json')) for i in range(8)]
    if (result.get('schema')!='excluded-kernel-warmup/v1' or result.get('status')!='captured_cleanup_pending' or
            result.get('scored_requests')!=0 or type(result.get('started_ns')) is not int or
            type(result.get('completed_ns')) is not int or
            not result['started_ns']<=min(r['start_ns'] for r in requests)<max(r['end_ns'] for r in requests)<=result['completed_ns']<=before_ns or
            result.get('cold_cache_label')!='cache-busted cold; unique early system namespaces, no verified cache flush or reset' or
            result.get('scope')!='one excluded8K/c8 kernel warmup; no claim that every later context/graph shape is warmed'):
        raise ValueError('warmup timing, exclusion or cache claim differs')
    return dict(status='excluded_disjoint_kernel_warmup_verified',requests=8,scored_requests=0,
                completed_ns=result['completed_ns'],cold_cache_label=result['cold_cache_label'])
