"""Audit one unscored tuning repetition without its dispatcher's preparation."""
import hashlib
import json
from pathlib import Path
import launcher,fixed_input_reuse
import campaign,engine_counters,routing_map,runtime,speculation_evidence,verify_engine_profile,verify_fixed_attribution,verify_fixed_task,verify_telemetry
import verify_worker_observations


def speculative_population(rows,max_tokens):
    observed=[speculation_evidence.from_stream(row['raw'],max_tokens) for row in rows]
    # The zero-variance rule applies to the cell population. Constant requests
    # within a variable cell are not separate violations.
    pooled=[sum(r['acceptance_histogram'][i] for r in observed) for i in range(max_tokens+1)]
    if sum(1 for count in pooled if count)<=1:
        raise ValueError('fixed speculative cell lacks active variable acceptance; investigate retained steps')
    return observed


def verify(directory):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda name:json.loads((root/name).read_text())
    startup=verify_engine_profile.verify(root);config=read('run-config.json');manifest=read('manifest.json');plan=read('fixed-qualification-plan.json')
    matrix=(root/'fixed-matrix.json').read_bytes()
    if hashlib.sha256(matrix).hexdigest()!=manifest.get('matrix_sha256'):raise ValueError('fixed qualification matrix differs')
    canonical={c['id']:c for c in campaign.expand(json.loads(matrix))};cell=canonical.get(config.get('cell_id'),{})
    scored=config.get('execution_phase')=='fixed-performance'
    repetition=config.get('repetition');profiles=manifest['runtime'].get('qualification_profiles',{});anchor=cell.get('suite') in ('anchor','anchor_greedy');trial=profiles.get(config.get('profile_id') if anchor else config.get('trial_profile_id'),{})
    if scored:
        import fixed_profiles
        campaign.validate_phase(manifest,'fixed-performance',cell_ids=[config['cell_id']],evidence=True)
        trial=fixed_profiles.validate(manifest['runtime'].get('fixed_profiles',{}).get(config['layout']),manifest)
    payload={k:v for k,v in trial.items() if k!='registration_sha256'}
    if (cell.get('suite') not in (('core','warm_prefix_panel') if scored else ('tuning_trial','tuning_extra','anchor','anchor_greedy')) or cell.get('layout')!=config['layout'] or
            type(repetition) is not int or not 0<=repetition<3 or
            trial.get('layout')!=config['layout'] or cell['id'] not in trial.get('fixed_cell_ids' if scored else 'deciding_cell_ids',[]) or
            hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=trial.get('registration_sha256') or
            trial.get('plan_sha256')!=manifest['plan_sha256'] or trial.get('matrix_sha256')!=manifest['matrix_sha256'] or
            trial.get('image_digest')!=config['image']):
        raise ValueError('fixed qualification trial registration differs')
    if scored:expected=trial['profile_sha256']
    elif anchor:
        if trial.get('profile_delta')!={}:raise ValueError('common anchor changed its starting profile')
        expected=trial['profile_sha256']
        if expected!=launcher.profile_sha256(config['layout']):raise ValueError('common anchor canonical starting profile differs')
    else:
        arm=cell['parameters']['arm'];expected=trial['profile_sha256'] if arm=='challenger' else trial['control_profile_sha256'] if arm in ('control_before','control_after') else None
    if expected!=startup['profile_sha256']:raise ValueError('fixed qualification used another control or candidate')
    ident='fixedqual-'+hashlib.sha256(json.dumps([cell['id'],expected,repetition],separators=(',',':')).encode()).hexdigest()
    if (plan.get('cell_id')!=cell['id'] or plan.get('attempt_cell_id')!=ident or plan.get('repetition')!=repetition or
            plan.get('profile_sha256')!=expected):raise ValueError('fixed qualification plan differs')
    source=root/'fixed-input-source';runtime.verify_evidence_manifest(source)
    if hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest()!=plan.get('source_manifest_sha256'):
        raise ValueError('fixed qualification input seal differs')
    generation=read('fixed-input-source/prompt-result.json');origin=read('fixed-input-source/run-config.json')
    if origin.get('generation_method') is not None or generation.get('generation_method') is not None:
        import streamed_input_source
        streamed_input_source.verify(source)
    flat=anchor or scored
    n=cell['input_tokens'] if flat else cell['parameters']['input']
    concurrency=cell['concurrency'] if flat else cell['parameters']['concurrency']
    suite=cell['suite'] if flat else 'core';cache=cell['cache'] if flat else 'cold';cap=cell['output_tokens'] if flat else 2048
    reuse=config.get('fixed_input_reuse')
    source_layout=reuse.get('source_layout') if isinstance(reuse,dict) else config['layout']
    source_cell=f"{suite}-{source_layout}-i{n}-c{concurrency}-{cache}-o{cap}"
    if (generation.get('status')!='passed' or generation.get('cleanup')!='passed' or generation.get('task_family')!='fixed' or
            generation.get('cell_id')!=source_cell or generation.get('repetition')!=repetition or origin.get('cell_id')!=source_cell or
            origin.get('image')!=config['image'] or origin.get('layout')!=source_layout or
            origin.get('task_family')!='fixed' or origin.get('repetition')!=repetition):
        raise ValueError('fixed qualification installed input origin differs')
    if reuse is not None:fixed_input_reuse.verify(config,source,manifest,root/'fixed-input-reuse-proof')
    elif (root/'fixed-input-reuse-proof').exists():raise ValueError('undeclared fixed input lineage')
    for name in ('manifest.json','checkpoint-receipt.json','image-qualification.json'):
        if name=='manifest.json' and reuse is not None:continue
        if (source/name).read_bytes()!=(root/name).read_bytes():raise ValueError('fixed qualification input dependency differs')
    inputs=read('fixed-attribution/requests/measured/fixed-inputs.json');tasks=read('fixed-input-source/fixed-tasks.json')
    process_contract=read('fixed-attribution/requests/fixed-session-contract.json').get('client_processes')
    workers=config.get('fixed_client_workers')
    if workers is not None:
        if (not scored or type(workers) is not int or workers!=1 or
                process_contract!={'workers':workers,'cpus':read('cpu-partition.json')['client_cpus']}):
            raise ValueError('scored measured client process or CPU budget differs from launch')
    elif process_contract is not None:
        raise ValueError('undeclared fixed measured client process')
    if (tasks!=inputs['tasks'] or len(tasks)!=concurrency or generation.get('tasks')!=len(tasks) or
            {t.get('id') for t in tasks}!={'s'+str(i) for i in range(concurrency)}):
        raise ValueError('fixed qualification input population differs from transmitted tasks')
    for task in tasks:
        verify_fixed_task.verify(task)
        if reuse is not None and task.get('prompt_version')!='fixed-performance-v1':raise ValueError('fixed reuse cannot relabel legacy workloads')
        if flat and task.get('prompt_version')!='fixed-performance-v1':raise ValueError('fixed comparison workload version differs')
        if task.get('prompt_version')=='fixed-performance-v1' and origin.get('fixed_prompt_version')!='fixed-performance-v1':
            raise ValueError('performance workload differs from explicit generator version')
        if (task.get('suite')!=suite or task.get('target_input_tokens')!=n or task.get('concurrency')!=concurrency or
                task.get('cache_designation')!=cache or task.get('output_tokens')!=cap or task.get('repetition')!=repetition):
            raise ValueError('fixed qualification task shape differs')
    router=read('router-qualification.json');mapping=routing_map.load_verified(root/'routing-map-source',router['lock_sha256'],router['router_executable_sha256'])
    placement=routing_map.select(mapping,concurrency,repetition)
    if plan.get('placement')!=placement or plan.get('mapping_sha256')!=mapping['mapping_sha256']:
        raise ValueError('fixed qualification frozen placement differs')
    attributed=verify_fixed_attribution.verify(root/'fixed-attribution')
    verify_fixed_attribution.verify_cache_profile(root/'fixed-attribution',read('profile-binding.json'))
    observer=(read('profile-binding.json').get('delta') or {}).get('worker_observation')
    worker_contract=read('fixed-attribution/attribution-contract.json')
    if observer is not None:
        expected_worker_context=verify_worker_observations.context_record(root)
        if (worker_contract.get('worker_observation_schema')!='worker-target-attribution/v1' or
                read('fixed-attribution/worker-context.json')!=expected_worker_context):
            raise ValueError('fixed worker attribution does not match global launch and physical evidence')
    elif 'worker_observation_schema' in worker_contract:
        raise ValueError('fixed worker attribution was not declared in the profile')
    for row in placement:
        task='s'+str(row['task_session'])
        if (inputs['session_ids'].get(task)!=row['session_header'] or
                attributed['session_routes'].get(row['session_header'])!=[row['endpoint'],row['rank']]):
            raise ValueError('fixed qualification actual routing differs from frozen placement')
    rows=[read('fixed-attribution/requests/measured/'+t['id']+'.json') for t in tasks]
    start=min(r['start_ns'] for r in rows);end=max(r['end_ns'] for r in rows)
    if scored:
        import verify_kernel_warmup
        verify_kernel_warmup.verify(root/'kernel-warmup',tasks,manifest,config,mapping,start)
    sampling=verify_telemetry.verify_window(root,start,end,config['expected_ranks'],
        types={**verify_telemetry.PILOT_TYPES,'vllm:request_success_total':'counter'})
    active=read('profile-binding.json')['record']['profile_delta'].get('speculative_config')
    speculative=[];selected=0
    for sample in verify_telemetry.iter_lines(root,'telemetry'):
        if not start<=sample['monotonic_ns']<=end:continue
        selected+=1
        for endpoint,ranks in config['expected_ranks'].items():
            engine_counters.snapshot(sample['engine_metrics'][endpoint],ranks)
            if active is not None:speculation_evidence.snapshot(sample['engine_metrics'][endpoint],ranks,active['num_speculative_tokens'])
    if not selected:raise ValueError('fixed qualification has no direct engine work sample')
    if active is not None:
        for command in read('launch-plan.json')['commands']:
            argv=command['argv']
            if '--per-request-spec-decode-metrics' not in argv or argv[argv.index('--per-request-spec-decode-metrics')+1]!='detailed':
                raise ValueError('fixed qualification lacks declared detailed speculative metrics')
        boundary_contract=read('fixed-attribution/attribution-contract.json').get('speculative_contract')
        if boundary_contract!={'max_spec_tokens':active['num_speculative_tokens'],
                'placement':{'s'+str(row['task_session']):[row['endpoint'],row['rank']] for row in placement}}:
            raise ValueError('fixed speculative qualification lacks bound per-rank accounting')
        speculative=speculative_population(rows,active['num_speculative_tokens'])
    outcome={'status':'passed','cell_id':ident,'qualification_cell_id':cell['id'],'repetition':repetition,'profile_sha256':expected,'performance_adoption_eligible':False,
        'scope':('one frozen-configuration performance repetition; full three-repetition cell and matched comparison acceptance remain separate' if scored else 'one common-settings anchor repetition; full anchor coverage, greedy control and final scored profiles remain pending' if anchor else 'one unscored registered tuning repetition; profile selection, full telemetry/overhead and scored acceptance remain pending'),
        **({'scored_performance_evidence':True} if scored else {})}
    if read('fixed-qualification-result.json')!=outcome:raise ValueError('fixed qualification completion record differs')
    runtime.verify_evidence_manifest(root)
    target_observations=[p['target_observation_accounting'] for p in attributed['populations']
        if p['phase']=='measured' and 'target_observation_accounting' in p]
    return {**outcome,'requests':concurrency,'valid_tokens':sum(r['completion_tokens'] for r in rows),
        'request_window_sampling':sampling,'attribution_status':attributed['status'],'startup_status':startup['status'],
        'detailed_speculative_observations':speculative,'verified_speculative_token_count':None,
        'target_verification_observations':target_observations,
        'verified_speculative_token_count_reason':('successful_target_draft_positions_reported_separately; bonus_positions_not_counted'
            if target_observations and active is not None else 'not_exported_by_pinned_surface' if active is not None else 'not_applicable_no_speculation'),
        'remaining_acceptance':('all three fresh repetitions, matched coverage and independent descriptive metric derivation; no quality, capacity or optimization claim' if scored else 'required repetitions/drift, optimization integrity/quality and paired tuning adoption; no capacity or allocator/kernel causal claim' if tasks[0]['prompt_version']=='fixed-performance-v1' else 'matched workload overhead, complete required telemetry including verified target-token/indexer/host occupancy sources, quality and paired tuning adoption rules')}
