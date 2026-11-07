"""Capture one preregistered tuning repetition in an owned candidate session.

Each repetition is a separate ledger item. This does not accept a scored cell,
select a winning profile or replace the three-repetition tuning requirement.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import fixed_input_source
import attempts,campaign,fixed_attribution,fixed_burst,launcher,node_pilot,node_smoke
import pilot_registry,profile_registry,routing_map,runtime,verify_fixed_task

DEFAULT_ROOT=Path('/data/hf-cache/dsv41-fixed-qualification-attempts')
METHOD_FILES=('model_clients.py','verify_model_clients.py','isolated_sampler.py','fixed_input_reuse.py','fixed_input_source.py','node_fixed_qualification.py','verify_fixed_qualification.py','fixed_attribution.py',
    'verify_fixed_attribution.py','engine_counters.py','fixed_session.py','fixed_burst.py',
    'stream.py','pilot_burst.py','verify_fixed_response.py','node_smoke.py','supervisor.py','watchdog.py','telemetry.py',
    'speculation_evidence.py','launcher.py','profile_registry.py','profile_overrides.py',
    'verify_fixed_session.py','verify_fixed_capture.py','verify_fixed_task.py','verify_telemetry.py','routing_map.py',
    'measurement_launch.py','measurement_worker_extension.py','worker_observations.py','allocator_observations.py','verify_worker_observations.py','verify_engine_profile.py')


def identity(cell,profile,repetition):
    return 'fixedqual-'+hashlib.sha256(json.dumps([cell,profile,repetition],separators=(',',':')).encode()).hexdigest()


def method_fingerprint(config,prepared):
    hypothesis=config.get('hypothesis')
    if hypothesis is not None and (not isinstance(hypothesis,str) or not 12<=len(hypothesis)<=500 or '\n' in hypothesis or '\r' in hypothesis):
        raise ValueError('invalid fixed qualification hypothesis')
    source=Path(__file__).resolve().parent
    payload={'cell_id':prepared['cell']['id'],'repetition':config['repetition'],'profile_sha256':prepared['binding']['profile_sha256'],
        'task_sha256':hashlib.sha256(prepared['task_bytes']).hexdigest(),'image':config['image'],
        'model_revision':prepared['manifest']['runtime']['model_revision'],'hypothesis':hypothesis,
        'fixed_client_workers':config.get('fixed_client_workers'),
        'method_files':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in METHOD_FILES}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(config,allowed_root='/data'):
    launcher.validate_run_id(config.get('run_id'))
    phase=config.get('execution_phase');scored=phase=='fixed-performance'
    if phase not in ('qualification','fixed-performance'):raise ValueError('fixed capture requires an explicit execution phase')
    workers=config.get('fixed_client_workers')
    if workers is not None and (not scored or type(workers) is not int or workers!=1):
        raise ValueError('scored client isolation requires one explicit measured client process')
    manifest=node_smoke.read(config['manifest']);binding=profile_registry.resolve(config,manifest)
    matrix_name='comparison-matrix.json' if scored else 'matrix.json'
    if (Path(config['matrix']).read_bytes()!=(campaign.HERE/matrix_name).read_bytes() or
            hashlib.sha256(Path(config['matrix']).read_bytes()).hexdigest()!=manifest['matrix_sha256']):
        raise ValueError('fixed tuning matrix differs from source or manifest')
    known={row['id']:row for row in manifest['cells']};cell=known.get(config.get('cell_id'),{})
    repetition=config.get('repetition')
    allowed=('core','warm_prefix_panel') if scored else ('tuning_trial','tuning_extra','anchor','anchor_greedy')
    if (cell.get('suite') not in allowed or cell.get('layout')!=config['layout'] or
            type(repetition) is not int or not 0<=repetition<3 or cell.get('repetitions')!=3):
        raise ValueError('fixed tuning repetition differs from declared trial')
    anchor=cell['suite'] in ('anchor','anchor_greedy')
    if scored:
        campaign.validate_phase(manifest,'fixed-performance',cell_ids=[cell['id']])
        if cell['id'] not in binding['record']['fixed_cell_ids']:raise ValueError('cell is outside frozen fixed configuration')
    elif anchor:
        if binding['delta']!={} or cell['id'] not in binding['record']['deciding_cell_ids']:
            raise ValueError('common anchor is not bound to its unchanged registered starting profile')
    else:
        registered=manifest['runtime'].get('qualification_profiles',{}).get(config.get('trial_profile_id'))
        profile_registry.validate_record(registered,manifest)
        arm=cell['parameters']['arm']
        desired=registered['profile_sha256'] if arm=='challenger' else registered['control_profile_sha256'] if arm in ('control_before','control_after') else None
        if (registered['layout']!=config['layout'] or cell['id'] not in registered['deciding_cell_ids'] or
                desired!=binding['profile_sha256']):
            raise ValueError('trial arm does not select its preregistered control or candidate')
    active=profile_registry.expected_runtime(binding,config['layout'])['speculative_config']
    if active is not None:
        for command in binding['record']['template']['commands']:
            argv=command['argv']
            if '--per-request-spec-decode-metrics' not in argv or argv[argv.index('--per-request-spec-decode-metrics')+1]!='detailed':
                raise ValueError('speculative tuning requires ordered detailed request metrics')
    flat=anchor or scored
    n=cell['input_tokens'] if flat else cell['parameters']['input']
    concurrency=cell['concurrency'] if flat else cell['parameters']['concurrency']
    inputs=fixed_input_source.load(config,n,concurrency,repetition,allowed_root,**({'suite':cell['suite'],'cache':cell['cache'],'output_tokens':cell['output_tokens']} if flat else {}))
    if flat and any(t.get('prompt_version')!='fixed-performance-v1' for t in inputs['tasks']):raise ValueError('new fixed comparisons require fixed-performance-v1')
    node_pilot.validate_rank_contract(config,cell)
    timeout=config.get('request_timeout_seconds')
    if type(timeout) not in (int,float) or not 0<timeout<=7200:raise ValueError('invalid tuning request deadline')
    return {'manifest':manifest,'binding':binding,'cell':cell,'scored_performance':scored,**inputs,
        'attempt_cell_id':identity(cell['id'],binding['profile_sha256'],repetition)}


def result_scope(prepared):
    if prepared.get('scored_performance'):
        return 'one frozen-configuration performance repetition; full three-repetition cell and matched comparison acceptance remain separate'
    return ('one common-settings anchor repetition; full anchor coverage, greedy control and final scored profiles remain pending' if prepared['cell']['suite'] in ('anchor','anchor_greedy') else 'one unscored registered tuning repetition; profile selection, full telemetry/overhead and scored acceptance remain pending')


def capture_budget(config,prepared):
    """Budget every serialized request phase and native accounting boundary."""
    phases=2 if prepared['tasks'][0]['cache_designation']=='warm_prefix' else 1
    return phases*(config['request_timeout_seconds']+2*fixed_attribution.BOUNDARY_SECONDS)


def run(config):
    if config.get('execution_phase')=='fixed-performance':
        raise ValueError('scored performance requires the shared block path with excluded disjoint warmup')
    _,router,seconds=node_smoke.validate(config);prepared=prepare(config)
    if capture_budget(config,prepared)+4320>seconds:raise ValueError('one tuning repetition and cleanup do not fit session')
    mapping=routing_map.load_verified(config['routing_map_evidence'],router['lock_sha256'],router['router_executable_sha256'])
    placement=routing_map.select(mapping,len(prepared['tasks']),config['repetition'])
    expected={f'http://127.0.0.1:{port}/metrics':list(range(mapping['dp'])) for port in mapping['ports']}
    if expected!=config['expected_ranks']:raise ValueError('frozen native map geometry differs from current engines')
    owner=runtime.process_identity(os.getpid());root=Path(config['run_dir'])
    method=method_fingerprint(config,prepared)
    with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):
        runtime.live_preflight(config['run_id'],mapping['ports']+[31100])
    attempt=attempts.begin(DEFAULT_ROOT,prepared['attempt_cell_id'],retry=config.get('retry',False),retry_reason=config.get('retry_reason'),method_sha256=method)
    runtime.atomic_json(attempt/'context.json',{'run_dir':str(root),'driver_identity':owner,'cell_id':prepared['cell']['id'],
        'profile_sha256':prepared['binding']['profile_sha256'],'repetition':config['repetition'],'method_sha256':method})
    def capture(session,shared,endpoint):
        capture_prepared(session,shared,endpoint,config,prepared,mapping,placement,
                         shared/'router.log',shared/'cache-config.json')
    try:
        node_smoke.run(config,capture)
        runtime.atomic_json(root/'fixed-qualification-result.json',{'status':'passed','cell_id':prepared['attempt_cell_id'],
            'qualification_cell_id':prepared['cell']['id'],'repetition':config['repetition'],
            'profile_sha256':prepared['binding']['profile_sha256'],
            'performance_adoption_eligible':False,
            'scope':result_scope(prepared),**({'scored_performance_evidence':True} if prepared.get('scored_performance') else {})})
        runtime.seal_evidence(root)
        import verify_fixed_qualification
        checked=verify_fixed_qualification.verify(root)
        pilot_registry.finish(attempt,root/'fixed-qualification-result.json',checked)
        return checked
    except BaseException as exc:
        pilot_registry.fail(attempt,exc)
        raise


def capture_prepared(session,directory,endpoint,config,prepared,mapping,placement,router_log,cache_config):
    """Capture one distinct repetition inside an already owned engine session."""
    shared=Path(directory);source=prepared['source'];runtime.verify_evidence_manifest(source)
    if hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest()!=prepared['source_manifest_sha256'] or (source/'fixed-tasks.json').read_bytes()!=prepared['task_bytes']:
        raise ValueError('fixed tuning source changed after reservation')
    shutil.copytree(source,shared/'fixed-input-source');runtime.verify_evidence_manifest(shared/'fixed-input-source')
    if 'input_lineage' in prepared:
        proof,files=prepared['input_lineage'];lineage=shared/'fixed-input-reuse-proof';lineage.mkdir()
        runtime.atomic_json(lineage/'lineage.json',proof)
        for name,data in files.items():runtime.atomic_bytes(lineage/name,data)
    route_source=Path(config['routing_map_evidence']);runtime.verify_evidence_manifest(route_source)
    shutil.copytree(route_source,shared/'routing-map-source');runtime.verify_evidence_manifest(shared/'routing-map-source')
    runtime.atomic_json(shared/'fixed-qualification-plan.json',{'cell_id':prepared['cell']['id'],
        'attempt_cell_id':prepared['attempt_cell_id'],'repetition':config['repetition'],
        'profile_sha256':prepared['binding']['profile_sha256'],'source_manifest_sha256':prepared['source_manifest_sha256'],
        'placement':placement,'mapping_sha256':mapping['mapping_sha256']})
    runtime.atomic_bytes(shared/'fixed-matrix.json',Path(config['matrix']).read_bytes())
    ids={'s'+str(row['task_session']):row['session_header'] for row in placement}
    namespace=fixed_burst.namespace(config['run_id'],prepared['cell']['id'],config['repetition'],prepared['binding']['profile_sha256'])
    if time.monotonic_ns()+int((capture_budget(config,prepared)+120)*1e9)>session.deadline_ns:
        raise runtime.RuntimeErrorClosed('fixed repetition and cleanup do not fit remaining session')
    active=profile_registry.expected_runtime(prepared['binding'],config['layout'])['speculative_config']
    extra={} if active is None else {'speculative_contract':{
        'max_spec_tokens':active['num_speculative_tokens'],
        'placement':{'s'+str(row['task_session']):[row['endpoint'],row['rank']] for row in placement}}}
    if (prepared['binding'].get('delta') or {}).get('worker_observation') is not None:
        extra['worker_root']=session.root
    if config.get('fixed_client_workers') is not None:
        cpus=node_smoke.read(session.root/'cpu-partition.json')['client_cpus']
        extra.update(client_workers=config['fixed_client_workers'],client_cpus=cpus)
    observed=fixed_attribution.capture(session,endpoint,prepared['tasks'],shared/'fixed-attribution',config['request_timeout_seconds'],
        node_pilot.clock_domain(session.run_id),namespace,ids,config['expected_ranks'],Path(router_log),node_smoke.read(cache_config)['block_size'],**extra)
    for row in placement:
        if observed['session_routes'][row['session_header']]!=[row['endpoint'],row['rank']]:raise ValueError('actual native selection differs from frozen placement')


def recover(attempt):
    import pilot_recover,verify_fixed_qualification
    return pilot_recover.recover(attempt,attempts_root=DEFAULT_ROOT,
        result_name='fixed-qualification-result.json',verifier=verify_fixed_qualification.verify)
