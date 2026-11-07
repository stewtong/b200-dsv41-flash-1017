"""Recompute an unscored sampler arm and its matched before/on/after control."""
import hashlib,json,math,os
from pathlib import Path
import campaign,routing_map,runtime,verify_engine_profile,verify_fixed_attribution,verify_fixed_task,verify_telemetry
import verify_worker_observations

SCOPE='incremental full sampler cost over essential safety; unscored and no profile/scored acceptance'
TOTAL_SCOPE='incremental worker observer and full host sampler cost over essential safety; unscored and no profile/scored acceptance'


def essential_window(root,start,end,ranks):
    rows=verify_telemetry.lines(root,'essential-engine');previous=-1;selected=[]
    control=json.loads((Path(root)/'sampler-control.json').read_text())
    for row in rows:
        a=row.get('started_ns');z=row.get('monotonic_ns');limit=row.get('deadline_ns')
        if (type(a) is not int or type(z) is not int or type(limit) is not int or
                not previous<a<=z<limit<=a+8_000_000_000 or row.get('control_cell_id')!=control['cell_id']):raise ValueError('essential sampler clocks differ')
        previous=z
        verify_telemetry.verify_metrics(row['engine_metrics'],ranks,
            {**verify_telemetry.PILOT_TYPES,'vllm:request_success_total':'counter'})
        if start-15_000_000_000<=z<=end:selected.append(z)
    if (not selected or selected[0]>start or end-selected[-1]>=15_000_000_000 or
            any(b-a>=15_000_000_000 for a,b in zip(selected,selected[1:]))):
        raise ValueError('essential engine freshness missing across request window')
    # A nominal reduced-instrument arm containing full sampler work is invalid.
    if any(start<=r['monotonic_ns']<=end for r in verify_telemetry.lines(root,'telemetry')):
        raise ValueError('full sampler contaminated essential overhead arm')
    return {'status':'essential_engine_freshness_verified','samples':len(selected),
            'scope':'15second safety freshness only; not1Hz measurement acceptance'}


def verify(directory,*,completion_required=True):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda path:json.loads((root/path).read_text())
    startup=verify_engine_profile.verify(root);config=read('run-config.json');manifest=read('manifest.json')
    matrix=(root/'fixed-matrix.json').read_bytes()
    if hashlib.sha256(matrix).hexdigest()!=manifest.get('matrix_sha256'):raise ValueError('sampler matrix differs')
    canonical={c['id']:c for c in campaign.expand(json.loads(matrix))};cell=canonical.get(config.get('cell_id'),{})
    registered=next((r for r in manifest['cells'] if r['id']==config.get('cell_id')),None)
    if (config.get('execution_phase')!='qualification' or cell!=registered or cell.get('suite')!='sampler_overhead' or
            cell.get('layout')!=config.get('layout') or cell.get('repetitions')!=3 or
            cell.get('parameters',{}).get('input')!=8000 or cell['parameters'].get('concurrency')!=8 or
            not isinstance(config.get('tasks_files'),list) or len(config['tasks_files'])!=3 or
            read('profile-binding.json')['record']['profile_delta'].get('speculative_config') is not None):
        raise ValueError('sampler arm manifest, profile or population differs')
    arm=cell['parameters']['arm'];mode='full' if arm=='challenger' else 'essential'
    if arm not in ('control_before','challenger','control_after'):raise ValueError('sampler arm unknown')
    control=read('sampler-control.json');state=read('execution-state.json');applied=read('sampler-control-applied.json')
    contract=config.get('overhead_contract');total=contract=='total-worker-and-host/v1'
    observer=(read('profile-binding.json').get('delta') or {}).get('worker_observation')
    if (contract not in (None,'total-worker-and-host/v1') or total and ((mode=='full')!=(observer is not None)) or
            not total and observer is not None):raise ValueError('sampler worker observation measurement contract differs')
    scope=TOTAL_SCOPE if total else SCOPE
    if (control.get('schema')!=('bounded-sampler-overhead-control/v2' if total else 'bounded-sampler-overhead-control/v1') or control.get('cell_id')!=cell['id'] or
            control.get('arm')!=arm or control.get('mode')!=mode or control.get('safety_watchdog_enabled') is not True or
            control.get('scope')!=('incremental worker observer and full host sampler cost over essential safety; unscored control only' if total else 'incremental full sampler cost over essential safety; unscored control only') or
            total and control.get('measurement_contract')!=contract or
            applied.get('cell_id')!=cell['id'] or applied.get('mode')!=mode or
            state.get('phase')!='qualification' or state.get('run_id')!=config.get('run_id') or
            not control['selected_ns']<control['until_ns']<=state['deadline_ns']-120_000_000_000 or
            not read('startup-timing.json')['semantic_ready_ns']<=control['selected_ns']):
        raise ValueError('sampler mode lacks bounded qualified-session binding')
    if ((runtime.process_alive(state['driver_identity']) and state['driver_identity']!=runtime.process_identity(os.getpid())) or runtime.process_alive(state['watchdog_identity']) or
            any(runtime.process_alive(p) for p in state.get('owned_processes',[])) or
            not read('cleanup-result.json').get('success')):
        raise ValueError('sampler arm ownership or cleanup remains unresolved')
    router=read('router-qualification.json');mapping=routing_map.load_verified(root/'routing-map-source',router['lock_sha256'],router['router_executable_sha256'])
    actual_router=(root/'router.log').read_bytes();checked=[];last_end=control['selected_ns'];last_route=0;identities=[]
    for repetition in range(3):
        ident='samplerqual-'+hashlib.sha256(json.dumps([cell['id'],startup['profile_sha256'],repetition],separators=(',',':')).encode()).hexdigest()
        identities.append(ident);child=root/'cells'/ident
        load=lambda path:json.loads((child/path).read_text())
        source=child/'fixed-input-source';runtime.verify_evidence_manifest(source)
        plan=load('fixed-qualification-plan.json');placement=routing_map.select(mapping,8,repetition)
        declared={k:v for k,v in config.items() if k!='tasks_files'};declared.update(repetition=repetition,tasks_file=config['tasks_files'][repetition])
        if (load('run-config.json')!=declared or plan.get('cell_id')!=cell['id'] or plan.get('repetition')!=repetition or
                plan.get('attempt_cell_id')!=ident or plan.get('profile_sha256')!=startup['profile_sha256'] or
                plan.get('mapping_sha256')!=mapping['mapping_sha256'] or plan.get('placement')!=placement or
                plan.get('source_manifest_sha256')!=hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest()):
            raise ValueError('sampler repetition input or placement binding differs')
        for name in ('manifest.json','checkpoint-receipt.json','image-qualification.json'):
            if (source/name).read_bytes()!=(root/name).read_bytes():raise ValueError('sampler installed input dependency differs')
        generation=load('fixed-input-source/prompt-result.json');origin=load('fixed-input-source/run-config.json')
        expected_source=f"core-{config['layout']}-i8000-c8-cold-o2048"
        if (generation.get('status')!='passed' or generation.get('cleanup')!='passed' or generation.get('tasks')!=8 or
                generation.get('task_family')!='fixed' or generation.get('cell_id')!=expected_source or
                generation.get('repetition')!=repetition or origin.get('cell_id')!=expected_source or
                origin.get('image')!=config['image'] or origin.get('layout')!=config['layout'] or
                origin.get('repetition')!=repetition or origin.get('task_family')!='fixed'):
            raise ValueError('sampler installed generation source differs')
        tasks=load('fixed-input-source/fixed-tasks.json');inputs=load('fixed-attribution/requests/measured/fixed-inputs.json')
        if tasks!=inputs['tasks'] or len(tasks)!=8:raise ValueError('sampler transmitted task population differs')
        for task in tasks:
            verify_fixed_task.verify(task)
            if (task['suite']!='core' or task['target_input_tokens']!=8000 or task['concurrency']!=8 or
                    task['cache_designation']!='cold' or task['output_tokens']!=2048 or task['repetition']!=repetition):
                raise ValueError('sampler task shape differs')
        attributed=verify_fixed_attribution.verify(child/'fixed-attribution')
        verify_fixed_attribution.verify_cache_profile(child/'fixed-attribution',read('profile-binding.json'))
        worker_contract=load('fixed-attribution/attribution-contract.json')
        if observer is not None:
            if (worker_contract.get('worker_observation_schema')!='worker-target-attribution/v1' or
                    load('fixed-attribution/worker-context.json')!=verify_worker_observations.context_record(root)):
                raise ValueError('sampler worker attribution differs from global launch and physical evidence')
        elif 'worker_observation_schema' in worker_contract:
            raise ValueError('undeclared sampler worker attribution')
        if any(attributed['session_routes'].get(p['session_header'])!=[p['endpoint'],p['rank']] for p in placement):
            raise ValueError('sampler actual native route differs')
        span=load('fixed-attribution/routing-span.json');a=span['start_byte'];z=span['end_byte']
        if not last_route<=a<=z<=len(actual_router) or actual_router[a:z]!=(child/'fixed-attribution/routing.log').read_bytes():
            raise ValueError('sampler route span differs from shared log')
        last_route=z
        rows=[load('fixed-attribution/requests/measured/'+t['id']+'.json') for t in tasks]
        start=min(r['start_ns'] for r in rows);end=max(r['end_ns'] for r in rows)
        if not last_end<=start<end<control['until_ns']:raise ValueError('sampler windows overlap or exceed control deadline')
        last_end=end
        coverage=(verify_telemetry.verify_window(root,start,end,config['expected_ranks'],
            types={**verify_telemetry.PILOT_TYPES,'vllm:request_success_total':'counter'}) if mode=='full'
            else essential_window(root,start,end,config['expected_ranks']))
        tokens=sum(r['completion_tokens'] for r in rows);seconds=(end-start)/1e9
        check={'status':'passed','cell_id':ident,'control_cell_id':cell['id'],'repetition':repetition,
            'profile_sha256':startup['profile_sha256'],'arm':arm,'mode':mode,'requests':8,'valid_tokens':tokens,
            'window_seconds':seconds,'valid_tokens_per_second':tokens/seconds,'coverage':coverage,
            'input_body_sha256':[hashlib.sha256(json.dumps(t['body'],sort_keys=True,separators=(',',':')).encode()).hexdigest() for t in tasks],
            'performance_adoption_eligible':False,'scope':scope}
        if completion_required and load('sampler-overhead-repetition-result.json')!=check:
            raise ValueError('sampler repetition completion differs')
        checked.append(check)
    if {p.name for p in (root/'cells').iterdir()}!=set(identities):raise ValueError('sampler repetition directories differ')
    result={'status':'passed','cells':checked,'scope':scope}
    if completion_required and read('sampler-overhead-result.json')!=result:raise ValueError('sampler arm completion differs')
    runtime.verify_evidence_manifest(root)
    return result


def compare(before,on,after):
    roots=[Path(p) for p in (before,on,after)];arms=[verify(p) for p in roots]
    if [[r['arm'] for r in a['cells']] for a in arms]!=[['control_before']*3,['challenger']*3,['control_after']*3]:
        raise ValueError('matched sampler control arms differ')
    configs=[json.loads((p/'run-config.json').read_text()) for p in roots]
    owners=[json.loads((p/'execution-state.json').read_text()) for p in roots]
    contracts=[c.get('overhead_contract') for c in configs]
    if len(set(contracts))!=1 or contracts[0] not in (None,'total-worker-and-host/v1'):
        raise ValueError('sampler comparison measurement contracts differ')
    total=contracts[0]=='total-worker-and-host/v1'
    if (len({json.dumps(o['driver_identity'],sort_keys=True) for o in owners})!=3 or
            any(configs[0].get(k)!=c.get(k) for c in configs for k in (('layout','image','toolkit_manifest') if total else ('layout','image','toolkit_manifest','profile_id'))) or
            any((roots[0]/'manifest.json').read_bytes()!=(p/'manifest.json').read_bytes() for p in roots)):
        raise ValueError('sampler arms lack distinct fresh owners or matched dependencies')
    if total:
        bindings=[json.loads((p/'profile-binding.json').read_text()) for p in roots]
        deltas=[b['record']['profile_delta'] for b in bindings]
        engine=[{k:v for k,v in delta.items() if k!='worker_observation'} for delta in deltas]
        if (json.dumps(engine[0],sort_keys=True)!=json.dumps(engine[1],sort_keys=True) or
                json.dumps(engine[0],sort_keys=True)!=json.dumps(engine[2],sort_keys=True) or
                'worker_observation' in deltas[0] or 'worker_observation' in deltas[2] or
                not deltas[1].get('worker_observation') or
                bindings[0]['profile_sha256']!=bindings[2]['profile_sha256'] or
                bindings[1]['profile_sha256']==bindings[0]['profile_sha256'] or
                bindings[1]['record']['control_profile_sha256']!=bindings[0]['profile_sha256']):
            raise ValueError('total overhead profiles differ beyond declared instrumentation')
    all_containers=[cid for o in owners for cid in o['container_ids']]
    if not all_containers or len(set(all_containers))!=len(all_containers):raise ValueError('sampler arms reused engine containers')
    for previous,current in zip(roots,roots[1:]):
        closed=json.loads((previous/'cleanup-result.json').read_text())['monotonic_ns']
        opened=json.loads((current/'startup-timing.json').read_text())['launch_started_ns']
        if (json.loads((previous/'execution-state.json').read_text())['driver_identity']['boot_id']!=
                json.loads((current/'execution-state.json').read_text())['driver_identity']['boot_id'] or closed>opened):
            raise ValueError('sampler comparison owners overlap or change clock domain')
    observations=[]
    for repetition in range(3):
        a,b,c=[arm['cells'][repetition] for arm in arms]
        if (any(r['input_body_sha256']!=a['input_body_sha256'] for r in (b,c)) or
                not total and any(r['profile_sha256']!=a['profile_sha256'] for r in (b,c)) or
                total and [r['profile_sha256'] for r in (a,b,c)]!=[b['profile_sha256'] for b in bindings]):
            raise ValueError('sampler control changed profile or transmitted workload')
        baseline=math.sqrt(a['valid_tokens_per_second']*c['valid_tokens_per_second'])
        drift=c['valid_tokens_per_second']/a['valid_tokens_per_second']-1
        observations.append({'repetition':repetition,'full_to_essential_throughput_ratio':b['valid_tokens_per_second']/baseline,
            'incremental_overhead_fraction':1-b['valid_tokens_per_second']/baseline,
            'closing_control_drift_fraction':drift,'within_five_percent_control_drift':abs(drift)<=.05})
    return {'status':'matched_sampler_observations','repetitions':observations,
        'attribution_qualified':all(r['within_five_percent_control_drift'] for r in observations),
        'scope':(TOTAL_SCOPE if total else SCOPE)+'; common safety collector cost not isolated; three paired burst observations, no saturation claim'}
