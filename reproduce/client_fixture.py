"""Bounded one/two-process HTTP client fixture; never model throughput evidence."""
import argparse,hashlib,json,os,re,resource,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urlsplit
import runtime,stream,verify_wire


def worker(config_path,digest):
    path=Path(config_path);data=path.read_bytes()
    if path.is_symlink() or path.parent.is_symlink() or path.name!='config.json' or hashlib.sha256(data).hexdigest()!=digest:raise ValueError('client worker config differs')
    c=json.loads(data);root=path.parent
    if c.get('schema')!='synthetic-client-worker/v1' or c.get('worker_source_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('client worker source or schema differs')
    url=urlsplit(c['url'])
    if (url.scheme!='http' or url.hostname!='127.0.0.1' or not url.port or url.path!='/v1/chat/completions' or
            url.username is not None or url.password is not None or url.query or url.fragment or
            type(c.get('threads')) is not int or not 1<=c['threads']<=128 or not isinstance(c.get('tasks'),list) or
            not 1<=len(c['tasks'])<=1024 or any(not isinstance(t,dict) or set(t)!={'id','session_id','body'} or
                not re.fullmatch(r'p[0-9]+',str(t.get('id'))) or not isinstance(t.get('body'),dict) or
                t.get('session_id')!='fixture-session-'+t['id'][1:] for t in c['tasks']) or
            len({t['id'] for t in c['tasks']})!=len(c['tasks'])):
        raise ValueError('client fixture must use explicit bounded loopback work')
    cpus=c.get('cpus')
    if cpus is not None:
        if not isinstance(cpus,list) or not cpus or any(type(n) is not int or n<0 for n in cpus):raise ValueError('client affinity invalid')
        os.sched_setaffinity(0,set(cpus))
        if os.sched_getaffinity(0)!=set(cpus):raise ValueError('client affinity read-back differs')
    elif c.get('node_owned') is True:raise ValueError('node fixture requires actual CPU affinity')
    runtime.atomic_json(root/'ready.json',{'pid':os.getpid(),'cpus':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None})
    gate=root/'start.json';wait=min(c['deadline_ns'],time.monotonic_ns()+15_000_000_000)
    while not gate.exists():
        if time.monotonic_ns()>=wait:raise RuntimeError('unreleased inert client deadline')
        if c['node_owned']:
            owner=Path(c['owner_root'])
            if (owner/'watchdog-failure.json').exists() or json.loads((owner/'execution-state.json').read_text()).get('stopping'):
                raise RuntimeError('owned client cancelled before gate')
        time.sleep(.02)
    start_record=json.loads(gate.read_text())
    if start_record!={'config_sha256':digest,'worker_pid':os.getpid()}:raise ValueError('client gate identity differs')
    if c['node_owned']:
        state=json.loads((Path(c['owner_root'])/'execution-state.json').read_text())
        identity=runtime.process_identity(os.getpid())
        if identity not in state['owned_processes'] or state.get('stopping') or not runtime.process_alive(state['driver_identity']):
            raise RuntimeError('client was not durably registered by live owner')
    before=resource.getrusage(resource.RUSAGE_SELF);started=time.monotonic_ns();cancel=stream.Cancellation();failed=False
    def one(task):
        timeout=min(c['request_timeout_seconds'],(c['deadline_ns']-time.monotonic_ns())/1e9)
        if timeout<=0:raise RuntimeError('client worker operational deadline')
        raw=stream.capture(c['url'],task['body'],task['id'],timeout,cancel,session_id=task['session_id'])
        runtime.atomic_json(root/(task['id']+'.json'),raw)
        try:
            verify_wire.verify_complete(raw);summary=stream.summarize_stream(raw)
            if summary['finish_reason']!='length' or summary['usage']!={'prompt_tokens':1,'completion_tokens':c['events'],'total_tokens':1+c['events']}:raise ValueError('fixture usage differs')
            text=''.join((choice.get('delta') or {}).get('content') or '' for e in raw['events'] if e['kind']=='event' for choice in e['data'].get('choices',[]))
            if text!='x'*(c['events']*c['piece_bytes']):raise ValueError('fixture stream population differs')
        except BaseException:
            cancel.cancel();raise
        return {'id':task['id'],'start_ns':raw['start_ns'],'end_ns':raw['terminal_ns'],
                'wire_bytes':raw['wire_byte_count'],'request_payload_bytes':raw['request_payload_bytes']}
    results=[]
    try:
        with ThreadPoolExecutor(max_workers=c['threads']) as pool:
            jobs=[pool.submit(one,t) for t in c['tasks']]
            for job in as_completed(jobs):results.append(job.result())
    except BaseException as exc:
        failed=True;runtime.atomic_json(root/'failure.json',{'type':type(exc).__name__,'completed_requests':len(results)});raise
    finally:
        after=resource.getrusage(resource.RUSAGE_SELF)
        runtime.atomic_json(root/'worker-resource.json',{'started_ns':started,'completed_ns':time.monotonic_ns(),
            'user_cpu_seconds':after.ru_utime-before.ru_utime,'system_cpu_seconds':after.ru_stime-before.ru_stime,
            'max_rss':after.ru_maxrss,'max_rss_unit':'KiB' if sys.platform.startswith('linux') else 'bytes',
            'failed':failed,'cpus':cpus,'scope':'whole worker capture/persistence CPU; synthetic loopback only'})
    runtime.atomic_json(root/'worker-result.json',{'status':'passed','requests':sorted(results,key=lambda r:r['id'])})
    if c['node_owned']:
        # Stay alive until the owner can atomically retire this exact child
        # behind the same lock used by watchdog ownership inspection.
        while not (root/'exit.json').exists():
            state=json.loads((Path(c['owner_root'])/'execution-state.json').read_text())
            if (time.monotonic_ns()>=c['deadline_ns'] or state.get('stopping') or
                    not runtime.process_alive(state['driver_identity'])):
                raise RuntimeError('owned client exit gate expired or owner stopped')
            time.sleep(.02)
        if json.loads((root/'exit.json').read_text())!={'config_sha256':digest,'worker_pid':os.getpid()}:
            raise ValueError('owned client exit gate differs')


def finish_owned_workers(session,children):
    """Retain successful exact-child release before removing live ownership."""
    with runtime.NodeLock(session.root/'cleanup.lock'):
        session.ensure_healthy();state=json.loads((session.root/'execution-state.json').read_text())
        if state.get('phase')!='client-fixture' or state.get('stopping'):
            raise RuntimeError('client retirement outside its active fixture phase')
        pending=[]
        for process,child,digest in children:
            identity=json.loads((child/'owned-identity.json').read_text())
            result=json.loads((child/'worker-result.json').read_text())
            if (identity not in state['owned_processes'] or identity['pid']!=process.pid or
                    process.poll() is not None or not runtime.process_alive(identity) or result.get('status')!='passed'):
                raise RuntimeError('client retirement lacks live ownership or terminal result')
            pending.append((process,child,digest,identity))
        for process,child,digest,identity in pending:
            runtime.atomic_json(child/'exit.json',{'config_sha256':digest,'worker_pid':process.pid})
        stop=time.monotonic()+10
        receipts=[]
        for process,child,digest,identity in pending:
            if process.wait(timeout=max(.1,stop-time.monotonic()))!=0 or runtime.process_alive(identity):
                raise RuntimeError('client retirement did not establish exact process death')
            receipt={'identity':identity,'worker_path':str(child.relative_to(session.root)),
                'config_sha256':digest,'result_sha256':hashlib.sha256((child/'worker-result.json').read_bytes()).hexdigest(),
                'exit_code':0,'completed_ns':time.monotonic_ns()}
            runtime.atomic_json(child/'release-receipt.json',receipt);receipts.append(receipt)
        for receipt in receipts:state['owned_processes'].remove(receipt['identity'])
        state.setdefault('completed_processes',[]).extend(receipts)
        runtime.atomic_json(session.root/'execution-state.json',state)


def capture(url,directory,*,workers=1,concurrency=8,requests=32,events=128,piece_bytes=32,
            input_bytes=32000,cpus=None,session=None,timeout=60):
    if (workers not in (1,2) or type(concurrency) is not int or not workers<=concurrency<=128 or
            type(requests) is not int or not concurrency<=requests<=1024 or
            type(events) is not int or not 1<=events<=2048 or type(piece_bytes) is not int or not 1<=piece_bytes<=1024 or
            type(input_bytes) is not int or not 1<=input_bytes<=8_000_000 or
            input_bytes*requests>128_000_000 or events*(piece_bytes+256)*requests>64_000_000 or
            type(timeout) not in (int,float) or not 0<timeout<=300):raise ValueError('bounded fixture geometry required')
    if cpus is not None and (len(set(cpus))!=len(cpus) or len(cpus)<workers):raise ValueError('fixture CPU partition too small')
    if session is not None and cpus is None:raise ValueError('owned node fixture needs declared CPU budget')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False);deadline=time.monotonic_ns()+int(timeout*1e9)
    tasks=[{'id':'p'+str(i),'session_id':'fixture-session-'+str(i),
        'body':{'model':'/model','messages':[{'role':'user','content':'a'*input_bytes}],
                'stream':True,'stream_options':{'include_usage':True},'max_tokens':events,'min_tokens':events,
                'ignore_eos':True,'seed':17+i,'temperature':1.0,'top_p':1.0,
                'chat_template_kwargs':{'thinking':False,'reasoning_effort':50}}} for i in range(requests)]
    runtime.atomic_json(root/'fixture-contract.json',{'schema':'synthetic-client-fixture/v1','url':url,'workers':workers,
        'concurrency':concurrency,'requests':requests,'events':events,'piece_bytes':piece_bytes,'input_bytes':input_bytes,
        'tasks':tasks,'cpus':cpus,'node_owned':session is not None,'deadline_ns':deadline,
        'scope':'synthetic response events and HTTP-body bytes; no model tokens, GPU capacity or SLO claim'})
    children=[];logs=[]
    try:
        for index in range(workers):
            child=root/('worker'+str(index));child.mkdir()
            config={'schema':'synthetic-client-worker/v1','url':url,'events':events,'piece_bytes':piece_bytes,
                'tasks':tasks[index::workers],'threads':concurrency//workers+(index<concurrency%workers),
                'cpus':cpus[index::workers] if cpus is not None else None,'node_owned':session is not None,
                'owner_root':str(session.root) if session is not None else None,'deadline_ns':deadline,
                'request_timeout_seconds':min(timeout,30),'worker_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            path=child/'config.json';runtime.atomic_json(path,config);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            log=(child/'worker.log').open('xb');logs.append(log)
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',str(path),digest],
                                     stdout=log,stderr=subprocess.STDOUT)
            children.append((process,child,digest))
            if session is not None:
                with runtime.NodeLock(session.root/'cleanup.lock'):
                    session.ensure_healthy();state=json.loads((session.root/'execution-state.json').read_text())
                    if state.get('stopping'):raise RuntimeError('cleanup before client registration')
                    identity=runtime.process_identity(process.pid);state['owned_processes'].append(identity)
                    runtime.atomic_json(session.root/'execution-state.json',state)
                    runtime.atomic_json(child/'owned-identity.json',identity)
        ready_deadline=min(deadline,time.monotonic_ns()+10_000_000_000)
        while not all((child/'ready.json').exists() for process,child,digest in children):
            if session is not None:session.ensure_healthy()
            if any(process.poll() is not None for process,child,digest in children) or time.monotonic_ns()>=ready_deadline:
                raise RuntimeError('client worker failed before gate')
            time.sleep(.02)
        for process,child,digest in children:
            if json.loads((child/'ready.json').read_text())['pid']!=process.pid:raise ValueError('client ready PID differs')
            runtime.atomic_json(child/'start.json',{'config_sha256':digest,'worker_pid':process.pid})
        def unfinished():
            return (not all((child/'worker-result.json').exists() for process,child,digest in children)
                    if session is not None else any(process.poll() is None for process,child,digest in children))
        while unfinished():
            if session is not None:session.ensure_healthy()
            if time.monotonic_ns()>=deadline or any(process.poll() not in (None,0) for process,child,digest in children):
                raise RuntimeError('client fixture failed or exceeded whole deadline')
            time.sleep(.02)
        if session is not None:finish_owned_workers(session,children)
        if any(process.poll()!=0 for process,child,digest in children):raise RuntimeError('client fixture worker nonzero')
        runtime.atomic_json(root/'fixture-result.json',{'status':'captured_verification_pending'})
        runtime.seal_evidence(root)
        import verify_client_fixture
        return verify_client_fixture.verify(root)
    except BaseException as exc:
        # Only these inert/registered CPU children are targeted, never a name selector.
        for process,child,digest in children:
            if process.poll() is None:process.terminate()
        stop=time.monotonic()+10
        for process,child,digest in children:
            try:process.wait(timeout=max(.1,stop-time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)
        runtime.atomic_json(root/'fixture-failure.json',{'type':type(exc).__name__,'scope':'failed fixture retained; no client ceiling acceptance'})
        raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('worker',));p.add_argument('config');p.add_argument('sha256')
    a=p.parse_args();worker(a.config,a.sha256)
