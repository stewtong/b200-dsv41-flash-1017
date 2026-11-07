"""Run one declared sampler arm with three distinct matched repetitions."""
import hashlib,json,os,time,shutil
from pathlib import Path
import attempts,fixed_input_source,node_fixed_qualification,node_smoke,pilot_registry
import profile_registry,routing_map,runtime

DEFAULT_ROOT=Path('/data/hf-cache/dsv41-sampler-overhead-attempts')


def identity(cell,profile,repetition):
    return 'samplerqual-'+hashlib.sha256(json.dumps([cell,profile,repetition],separators=(',',':')).encode()).hexdigest()


def prepare(config,allowed_root='/data'):
    manifest=node_smoke.read(config['manifest']);binding=profile_registry.resolve(config,manifest)
    cells=[row for row in manifest['cells'] if row['id']==config.get('cell_id')]
    files=config.get('tasks_files');timeout=config.get('request_timeout_seconds')
    if (config.get('execution_phase')!='qualification' or len(cells)!=1 or cells[0].get('suite')!='sampler_overhead' or
            cells[0].get('layout')!=config['layout'] or cells[0].get('repetitions')!=3 or
            cells[0].get('parameters',{}).get('input')!=8000 or cells[0]['parameters'].get('concurrency')!=8 or
            cells[0]['parameters'].get('arm') not in ('control_before','challenger','control_after') or
            not isinstance(files,list) or len(files)!=3 or len(set(files))!=3 or
            type(timeout) not in (int,float) or not 0<timeout<=1800 or
            profile_registry.expected_runtime(binding,config['layout'])['speculative_config'] is not None):
        raise ValueError('sampler control requires one declared NoSpec8K/c8 arm and three explicit repetitions')
    observer=(binding.get('delta') or {}).get('worker_observation')
    retry_repetitions=config.get('retry_repetitions')
    if retry_repetitions is not None:
        if (config.get('retry',False) or not isinstance(retry_repetitions,list) or not retry_repetitions or
                any(type(r) is not int or not 0<=r<3 for r in retry_repetitions) or
                len(set(retry_repetitions))!=len(retry_repetitions) or
                not isinstance(config.get('retry_reason'),str) or not 12<=len(config['retry_reason'].strip())<=500 or
                '\n' in config['retry_reason'] or '\r' in config['retry_reason']):
            raise ValueError('sampler mixed resume requires explicit repetition retry declarations')
    contract=config.get('overhead_contract')
    if contract not in (None,'total-worker-and-host/v1'):
        raise ValueError('unknown sampler overhead measurement contract')
    if contract=='total-worker-and-host/v1':
        if (cells[0]['parameters']['arm']=='challenger')!=(observer is not None):
            raise ValueError('total overhead requires worker observer only in the full arm')
    elif observer is not None:
        raise ValueError('worker observer requires explicit total overhead contract')
    rows=[]
    for repetition,path in enumerate(files):
        child={k:v for k,v in config.items() if k!='tasks_files'};child.update(repetition=repetition,tasks_file=path)
        inputs=fixed_input_source.load(child,8000,8,repetition,allowed_root)
        prepared={'manifest':manifest,'binding':binding,'cell':cells[0],**inputs,
            'attempt_cell_id':identity(cells[0]['id'],binding['profile_sha256'],repetition)}
        rows.append({'config':child,'prepared':prepared,'identity':prepared['attempt_cell_id']})
    return rows,3*timeout


def run(config):
    rows,budget=prepare(config);_,router,seconds=node_smoke.validate(config)
    if budget+4320>seconds:raise ValueError('sampler arm requests and shared overhead do not fit deadline')
    mapping=routing_map.load_verified(config['routing_map_evidence'],router['lock_sha256'],router['router_executable_sha256'])
    if {f'http://127.0.0.1:{p}/metrics':list(range(mapping['dp'])) for p in mapping['ports']}!=config['expected_ranks']:
        raise ValueError('sampler arm native geometry differs')
    root=Path(config['run_dir']);owner=runtime.process_identity(os.getpid())
    with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):runtime.live_preflight(config['run_id'],mapping['ports']+[31100])
    source=Path(__file__).parent
    dependencies={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
    specs=[{'cell_id':r['identity'],'method_sha256':hashlib.sha256(json.dumps({
        'input':hashlib.sha256(r['prepared']['task_bytes']).hexdigest(),'profile':r['prepared']['binding']['profile_sha256'],
        'cell':config['cell_id'],'sources':dependencies,'hypothesis':config.get('hypothesis')},sort_keys=True).encode()).hexdigest()} for r in rows]
    retry_args={'retry':config.get('retry',False),'retry_reason':config.get('retry_reason')}
    if config.get('retry_repetitions') is not None:
        retry_args={'retry_cells':{r['identity']:config['retry_reason'] for r in rows if r['config']['repetition'] in config['retry_repetitions']}}
    paths=attempts.begin_block(DEFAULT_ROOT,specs,config['run_id'],owner,**retry_args)
    for row,path in zip(rows,paths):
        row['attempt']=path
        runtime.atomic_json(path/'context.json',{'run_dir':str(root),'driver_identity':owner,'group_id':config['run_id'],
            'cell_artifact_dir':str(root/'cells'/row['identity'])})
    def capture(session,shared,endpoint):
        runtime.atomic_bytes(shared/'fixed-matrix.json',Path(config['matrix']).read_bytes())
        route_source=Path(config['routing_map_evidence']);runtime.verify_evidence_manifest(route_source)
        shutil.copytree(route_source,shared/'routing-map-source');runtime.verify_evidence_manifest(shared/'routing-map-source')
        # Full sampler qualification and physical startup precede this switch.
        limit=min(session.deadline_ns-120_000_000_000,time.monotonic_ns()+int((budget+90)*1e9))
        control=session.begin_sampler_control(config['cell_id'],limit)
        wait=min(limit,time.monotonic_ns()+20_000_000_000)
        while True:
            session.ensure_healthy();path=shared/'sampler-control-applied.json'
            applied=node_smoke.read(path) if path.exists() else {}
            if (applied.get('cell_id')==config['cell_id'] and applied.get('mode')==control['mode'] and
                    applied.get('completed_ns',0)>=control['selected_ns']):break
            if time.monotonic_ns()>=wait:raise runtime.RuntimeErrorClosed('sampler control did not acknowledge application')
            time.sleep(.1)
        for row in rows:
            session.ensure_healthy();dest=shared/'cells'/row['identity'];dest.mkdir(parents=True)
            runtime.atomic_json(row['attempt']/'dispatch-intent.json',{'cell_id':row['identity'],
                'group_id':config['run_id'],'driver_identity':owner,'monotonic_ns':time.monotonic_ns()})
            runtime.atomic_json(dest/'run-config.json',row['config'])
            placement=routing_map.select(mapping,8,row['config']['repetition'])
            node_fixed_qualification.capture_prepared(session,dest,endpoint,row['config'],row['prepared'],mapping,
                placement,shared/'router.log',shared/'cache-config.json')
            runtime.atomic_json(row['attempt']/'capture.json',{'status':'requests_captured_cleanup_pending','cell_artifact_dir':str(dest)})
            attempts.capture_pending(row['attempt'],['context.json','capture.json'],config['run_id'],owner)
    try:
        node_smoke.run(config,capture)
        import verify_sampler_overhead
        runtime.seal_evidence(root)
        checked=verify_sampler_overhead.verify(root,completion_required=False)
        for row,result in zip(rows,checked['cells']):
            runtime.atomic_json(root/'cells'/row['identity']/'sampler-overhead-repetition-result.json',result)
        runtime.atomic_json(root/'sampler-overhead-result.json',checked);runtime.seal_evidence(root)
        checked=verify_sampler_overhead.verify(root)
        for row,result in zip(rows,checked['cells']):
            pilot_registry.finish(row['attempt'],root/'cells'/row['identity']/'sampler-overhead-repetition-result.json',result)
        return checked
    except BaseException as exc:
        for path in paths:
            if node_smoke.read(path/'attempt.json')['status']=='running':pilot_registry.fail(path,exc)
        raise


def recover(directory):
    import pilot_block_recover,verify_sampler_overhead
    return pilot_block_recover.recover(directory,attempts_root=DEFAULT_ROOT,
        result_name='sampler-overhead-result.json',child_result_name='sampler-overhead-repetition-result.json',
        block_verifier=verify_sampler_overhead.verify)
