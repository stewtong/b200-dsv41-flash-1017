"""Owned unscored one/two/one-client comparison at one declared workload shape.

This neither accepts a scored cell nor establishes a universal client ceiling.
Every probe retains exact inputs, native routing, telemetry and child identities.
"""
import hashlib,json,os,time
from pathlib import Path
import attempts,fixed_attribution,fixed_burst,fixed_input_source,launcher
import cache_source_proof
import node_pilot,node_smoke,pilot_registry,profile_registry,routing_map,runtime

DEFAULT_ROOT=Path('/data/hf-cache/dsv41-model-client-attempts')
ARMS=(('before',1),('two',2),('after',1))


def identity(layout,profile,n,concurrency):
    return 'clientqual-'+hashlib.sha256(json.dumps([layout,profile,n,concurrency],separators=(',',':')).encode()).hexdigest()


def prepare(config,allowed_root='/data'):
    launcher.validate_run_id(config.get('run_id'))
    manifest=node_smoke.read(config['manifest']);binding=profile_registry.resolve(config,manifest)
    n=config.get('client_input_tokens');c=config.get('client_concurrency');files=config.get('tasks_files')
    timeout=config.get('request_timeout_seconds')
    if (config.get('execution_phase')!='qualification' or type(n) is not int or n not in (8000,128000,512000,1000000) or
            type(c) is not int or c not in (8,32,128) or not isinstance(files,list) or len(files)!=3 or len(set(files))!=3 or
            type(timeout) not in (int,float) or not 0<timeout<=7200 or
            binding.get('delta',{}).get('enable_prefix_caching') is not False or
            profile_registry.expected_runtime(binding,config['layout'])['speculative_config'] is not None):
        raise ValueError('model client check requires explicit cold NoSpec cache-off profile and three matched input repetitions')
    rows=[]
    cache_source_proof.verify(config['cache_source_evidence'],node_smoke.read(config['image_qualification'])['image_id'])
    for repetition,path in enumerate(files):
        child={**config,'tasks_file':path,'repetition':repetition}
        rows.append(fixed_input_source.load(child,n,c,repetition,allowed_root))
    ident=identity(config['layout'],binding['profile_sha256'],n,c)
    return {'manifest':manifest,'binding':binding,'inputs':rows,'identity':ident,'input_tokens':n,'concurrency':c},9*timeout


def run(config):
    prepared,budget=prepare(config);_,router,seconds=node_smoke.validate(config)
    if budget+4320>seconds:raise ValueError('nine model client bursts and shared startup/cleanup do not fit deadline')
    mapping=routing_map.load_verified(config['routing_map_evidence'],router['lock_sha256'],router['router_executable_sha256'])
    if {f'http://127.0.0.1:{p}/metrics':list(range(mapping['dp'])) for p in mapping['ports']}!=config['expected_ranks']:
        raise ValueError('model client native geometry differs')
    root=Path(config['run_dir']);owner=runtime.process_identity(os.getpid())
    dependencies={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    method=hashlib.sha256(json.dumps({'profile':prepared['binding']['profile_sha256'],
        'inputs':[hashlib.sha256(r['task_bytes']).hexdigest() for r in prepared['inputs']],
        'hypothesis':config.get('hypothesis'),'sources':dependencies},sort_keys=True).encode()).hexdigest()
    with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):runtime.live_preflight(config['run_id'],mapping['ports']+[31100])
    attempt=attempts.begin(DEFAULT_ROOT,prepared['identity'],retry=config.get('retry',False),
        retry_reason=config.get('retry_reason'),method_sha256=method)
    runtime.atomic_json(attempt/'context.json',{'run_dir':str(root),'driver_identity':owner,'cell_id':prepared['identity']})
    def capture(session,shared,endpoint):
        import shutil,verify_telemetry
        cpus=node_smoke.read(shared/'cpu-partition.json')['client_cpus']
        runtime.atomic_bytes(shared/'client-toolkit-MANIFEST.json',Path(config['toolkit_manifest']).read_bytes())
        shutil.copytree(config['cache_source_evidence'],shared/'cache-source-proof')
        cache_source_proof.verify(shared/'cache-source-proof',node_smoke.read(shared/'image-qualification.json')['image_id'])
        runtime.atomic_json(shared/'model-client-design.json',{'schema':'owned-model-client-check/v1',
            'identity':prepared['identity'],'input_tokens':prepared['input_tokens'],'concurrency':prepared['concurrency'],
            'repetitions':3,'arms':[{'name':a,'workers':w} for a,w in ARMS],
            'cpus':cpus,'prefix_caching':False,'output_tokens':2048,
            'scope':'one declared shape only; no global client ceiling, saturation or scored claim'})
        shutil.copytree(config['routing_map_evidence'],shared/'routing-map-source')
        runtime.verify_evidence_manifest(shared/'routing-map-source')
        completed=[]
        for repetition,source in enumerate(prepared['inputs']):
            original=shared/'inputs'/str(repetition);original.parent.mkdir(exist_ok=True)
            shutil.copytree(source['source'],original);runtime.verify_evidence_manifest(original)
            placement=routing_map.select(mapping,prepared['concurrency'],repetition)
            sessions={'s'+str(r['task_session']):r['session_header'] for r in placement}
            for arm,workers in ARMS:
                session.ensure_healthy()
                if time.monotonic_ns()+int((config['request_timeout_seconds']+120)*1e9)>session.deadline_ns:
                    raise runtime.RuntimeErrorClosed('client burst and cleanup do not fit remaining session')
                label='r'+str(repetition)+'-'+arm;dest=shared/'probes'/label
                namespace=fixed_burst.namespace(config['run_id'],prepared['identity']+'-'+arm,repetition,prepared['binding']['profile_sha256'])
                observed=fixed_attribution.capture(session,endpoint,source['tasks'],dest,config['request_timeout_seconds'],
                    node_pilot.clock_domain(session.run_id),namespace,sessions,config['expected_ranks'],shared/'router.log',
                    node_smoke.read(shared/'cache-config.json')['block_size'],client_workers=workers,client_cpus=cpus,prefix_caching=False)
                if any(observed['session_routes'][r['session_header']]!=[r['endpoint'],r['rank']] for r in placement):
                    raise ValueError('model client actual route differs from deterministic placement')
                raws=[node_smoke.read(dest/'requests/measured'/(t['id']+'.json'))['raw'] for t in source['tasks']]
                a=min(r['start_ns'] for r in raws);z=max(r['terminal_ns'] for r in raws)
                # This also requires a fresh opening/closing sample. A slower
                # sampler or missing rank cannot silently become a client gain.
                verify_telemetry.verify_window(shared,a,z,config['expected_ranks'],
                    types={**verify_telemetry.PILOT_TYPES,'vllm:request_success_total':'counter'})
                completed.append({'repetition':repetition,'arm':arm,'workers':workers,'path':'probes/'+label})
                runtime.atomic_json(shared/'completed-client-probes.json',completed)
    try:
        node_smoke.run(config,capture)
        runtime.seal_evidence(root)
        import verify_model_client_check
        checked=verify_model_client_check.verify(root,completion_required=False)
        runtime.atomic_json(root/'model-client-check-result.json',checked);runtime.seal_evidence(root)
        checked=verify_model_client_check.verify(root)
        pilot_registry.finish(attempt,root/'model-client-check-result.json',checked)
        return checked
    except BaseException as exc:
        pilot_registry.fail(attempt,exc);raise


def recover(attempt):
    import pilot_recover,verify_model_client_check
    return pilot_recover.recover(attempt,attempts_root=DEFAULT_ROOT,
        result_name='model-client-check-result.json',verifier=verify_model_client_check.verify)
