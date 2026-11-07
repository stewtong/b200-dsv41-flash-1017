"""Run distinct preregistered repetitions in one owned, bounded engine session."""
import hashlib,json,os,time
from pathlib import Path
import attempts,node_fixed_qualification as qualified,node_pilot_block,node_smoke
import pilot_registry,routing_map,runtime

FIELDS={'cell_id','tasks_file','request_timeout_seconds','repetition'}


def prepare(config,allowed_root='/data'):
    entries=config.get('cells')
    if (config.get('execution_phase') not in ('qualification','fixed-performance') or not isinstance(entries,list) or
            not 1<=len(entries)<=24):raise ValueError('one to twenty-four distinct qualification repetitions required')
    rows=[];seen=set();budget=0;group=None
    for entry in entries:
        if not isinstance(entry,dict) or set(entry)!=FIELDS:raise ValueError('qualification block entry fields differ')
        child={k:v for k,v in config.items() if k!='cells'};child.update(entry)
        prepared=qualified.prepare(child,allowed_root);ident=prepared['attempt_cell_id']
        if ident in seen:raise ValueError('duplicate qualification repetition')
        seen.add(ident)
        cell=prepared['cell'];params=cell.get('parameters',{})
        # The before/challenger/after arms require fresh engines. Combining
        # distinct seeds/shapes within one arm does not combine those arms.
        key=(('fixed-performance',None,None,prepared['binding']['profile_sha256']) if config['execution_phase']=='fixed-performance'
             else (cell['suite'],params.get('lever'),params.get('arm'),prepared['binding']['profile_sha256']))
        if group is None:group=key
        if key!=group:raise ValueError('qualification block mixes trial levers, arms or profiles')
        budget+=qualified.capture_budget(child,prepared)
        rows.append({'config':child,'prepared':prepared,'identity':ident})
    if config.get('rebaseline_cells') is not None:
        import tp4_rebaseline
        tp4_rebaseline.context(config,rows)
    return rows,budget


def method(config,prepared):
    source=Path(__file__).resolve().parent
    files=('node_fixed_block.py','verify_fixed_block.py','node_pilot_block.py','pilot_block_recover.py','kernel_warmup.py','verify_kernel_warmup.py')
    payload={'repetition_method':qualified.method_fingerprint(config,prepared),'kernel_warmup':config.get('kernel_warmup'),
             'block_files':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run(config):
    rows,budget=prepare(config)
    scored=config.get('execution_phase')=='fixed-performance'
    warmup=None
    if scored:
        import kernel_warmup
        warmup=kernel_warmup.prepare(config,rows[0]['prepared']['manifest'],
            [task for row in rows for task in row['prepared']['tasks']])
        budget+=warmup['budget_seconds']
    session_budget='fixed-performance-block' if scored else 'qualification-block'
    _,router,seconds=node_smoke.validate(config,session_budget=session_budget)
    if budget+4320>seconds:raise ValueError('qualification block and cleanup do not fit session')
    mapping=routing_map.load_verified(config['routing_map_evidence'],router['lock_sha256'],router['router_executable_sha256'])
    expected={f'http://127.0.0.1:{port}/metrics':list(range(mapping['dp'])) for port in mapping['ports']}
    if expected!=config['expected_ranks']:raise ValueError('qualification block native geometry differs')
    owner=runtime.process_identity(os.getpid());root=Path(config['run_dir'])
    with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):runtime.live_preflight(config['run_id'],mapping['ports']+[31100])
    specs=[{'cell_id':r['identity'],'method_sha256':method(r['config'],r['prepared'])} for r in rows]
    paths=attempts.begin_block(qualified.DEFAULT_ROOT,specs,config['run_id'],owner,
        retry=config.get('retry',False),retry_reason=config.get('retry_reason'),retry_cells=config.get('retry_cells'),
        rebaseline_cells=config.get('rebaseline_cells'))
    for row,path in zip(rows,paths):
        row['attempt']=path
        runtime.atomic_json(path/'context.json',{'run_dir':str(root),'driver_identity':owner,'group_id':config['run_id'],
            'cell_artifact_dir':str(root/'cells'/row['identity']),
            'qualification_cell_id':row['prepared']['cell']['id'],'repetition':row['config']['repetition']})
    def capture(session,shared,endpoint):
        if warmup is not None:
            kernel_warmup.run(session,shared,endpoint,config,rows[0]['prepared']['manifest'],warmup,mapping)
        for row in rows:
            session.ensure_healthy()
            runtime.atomic_json(row['attempt']/'dispatch-intent.json',{'cell_id':row['identity'],
                'group_id':config['run_id'],'driver_identity':owner,'monotonic_ns':time.monotonic_ns()})
            dest=shared/'cells'/row['identity'];dest.mkdir(parents=True);row['root']=dest
            runtime.atomic_json(dest/'run-config.json',row['config'])
            placement=routing_map.select(mapping,len(row['prepared']['tasks']),row['config']['repetition'])
            qualified.capture_prepared(session,dest,endpoint,row['config'],row['prepared'],mapping,placement,
                shared/'router.log',shared/'cache-config.json')
            runtime.atomic_json(row['attempt']/'capture.json',{'cell_artifact_dir':str(dest),'status':'requests_captured_cleanup_pending'})
            attempts.capture_pending(row['attempt'],['context.json','capture.json'],config['run_id'],owner)
    try:
        node_smoke.run(config,capture,session_budget=session_budget)
        import verify_fixed_qualification,verify_fixed_block
        child_roots=[]
        for row in rows:
            dest=row['root'];prepared=row['prepared'];child=row['config']
            node_pilot_block.project_shared(root,dest,owner,schema='shared-fixed-qualification-session/v1')
            runtime.atomic_json(dest/'fixed-qualification-result.json',{'status':'passed','cell_id':row['identity'],
                'qualification_cell_id':prepared['cell']['id'],'repetition':child['repetition'],
                'profile_sha256':prepared['binding']['profile_sha256'],'performance_adoption_eligible':False,
                'scope':qualified.result_scope(prepared),**({'scored_performance_evidence':True} if scored else {})})
            runtime.seal_evidence(dest);child_roots.append(dest)
        checked=verify_fixed_block.verify_children(child_roots)
        runtime.atomic_json(root/'fixed-qualification-block-result.json',{'status':'passed','cells':checked,
            'scope':('distinct frozen-configuration performance repetitions sharing one startup and cleanup; independent cell and matched comparison acceptance remain separate' if scored else 'distinct unscored registered repetitions sharing one startup and cleanup; no profile adoption or scored acceptance')})
        runtime.seal_evidence(root);verify_fixed_block.verify(root)
        for row,check in zip(rows,checked):pilot_registry.finish(row['attempt'],row['root']/'fixed-qualification-result.json',check)
        return {'status':'passed','cells':checked,'scope':'fixed-performance shared session' if scored else 'unscored shared qualification session'}
    except BaseException as exc:
        for path in paths:
            if node_smoke.read(path/'attempt.json')['status']=='running':pilot_registry.fail(path,exc)
        raise


def recover(directory):
    import pilot_block_recover,verify_fixed_block
    return pilot_block_recover.recover(directory,attempts_root=qualified.DEFAULT_ROOT,
        result_name='fixed-qualification-block-result.json',child_result_name='fixed-qualification-result.json',
        block_verifier=verify_fixed_block.verify)
