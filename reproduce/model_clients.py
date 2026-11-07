"""One/two independent HTTP clients for an owned, unscored model cross-check.

Captures identical prepared fixed workloads. This module cannot establish a
client ceiling, model capacity, matched overhead or scored acceptance alone.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

import pilot_burst
import runtime
import verify_fixed_response
import verify_fixed_task


def validate_url(url):
    u=urlsplit(url)
    if (u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port or
            u.path!='/v1/chat/completions' or u.username is not None or
            u.password is not None or u.query or u.fragment):
        raise ValueError('model clients require the declared node-local route')


def worker(path,digest):
    path=Path(path)
    if path.is_symlink() or path.parent.is_symlink() or path.name!='config.json':
        raise ValueError('model client config path differs')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('model client config hash differs')
    c=json.loads(raw);root=path.parent;validate_url(c['url'])
    if (c.get('schema')!='fixed-model-client-worker/v1' or
            c.get('worker_source_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
            not isinstance(c.get('tasks'),list) or not c['tasks'] or
            type(c.get('threads')) is not int or not 1<=c['threads']<=128 or
            type(c.get('deadline_ns')) is not int or c['deadline_ns']<=time.monotonic_ns()):
        raise ValueError('model client source, tasks or bounds differ')
    for task in c['tasks']:verify_fixed_task.verify(task)
    if c['cpus'] is not None:
        if not c['cpus'] or any(type(i) is not int or i<0 for i in c['cpus']):raise ValueError('model client CPU affinity differs')
        os.sched_setaffinity(0,set(c['cpus']))
        if os.sched_getaffinity(0)!=set(c['cpus']):raise ValueError('model client CPU affinity read-back differs')
    elif c['owner_root'] is not None:raise ValueError('owned model client lacks CPU budget')
    def healthy():
        if time.monotonic_ns()>=c['deadline_ns']:raise RuntimeError('model client deadline expired')
        if c['owner_root'] is not None:
            owner=Path(c['owner_root']);state=json.loads((owner/'execution-state.json').read_text())
            if ((owner/'watchdog-failure.json').exists() or state.get('stopping') or
                    not runtime.process_alive(state['driver_identity'])):
                raise RuntimeError('model client owner stopped')
    runtime.atomic_json(root/'ready.json',{'pid':os.getpid(),'cpus':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None})
    def gate(name):
        end=min(c['deadline_ns'],time.monotonic_ns()+15_000_000_000)
        while not (root/name).exists():
            healthy()
            if time.monotonic_ns()>=end:raise RuntimeError('model client gate expired')
            time.sleep(.02)
        if json.loads((root/name).read_text())!={'config_sha256':digest,'worker_pid':os.getpid()}:
            raise ValueError('model client gate identity differs')
    gate('start.json')
    if c['owner_root'] is not None:
        state=json.loads((Path(c['owner_root'])/'execution-state.json').read_text())
        if runtime.process_identity(os.getpid()) not in state['owned_processes']:
            raise RuntimeError('model client is not durably registered')
    class Session:
        def ensure_healthy(self):healthy()
        def abort(self,reason):runtime.atomic_json(root/'abort-request.json',{'reason':'model request transport or content failure'})
    before=resource.getrusage(resource.RUSAGE_SELF);started=time.monotonic_ns()
    by_id={t['id']:t for t in c['tasks']}
    def oracle(task,raw):
        return verify_fixed_response.evaluate(by_id[task['id']],raw,c['request_ids'][task['id']],c['session_ids'][task['id']])
    try:
        summary=pilot_burst.run(Session(),c['url'],c['tasks'],root/'capture',c['request_timeout_seconds'],
            c['clock_id'],oracle=oracle,request_ids=c['request_ids'],session_ids=c['session_ids'])
        if summary['valid_requests']!=len(c['tasks']):raise RuntimeError('model client response invalid')
        runtime.atomic_json(root/'worker-result.json',{'status':'passed','requests':len(c['tasks'])})
    except BaseException as exc:
        runtime.atomic_json(root/'failure.json',{'type':type(exc).__name__});raise
    finally:
        after=resource.getrusage(resource.RUSAGE_SELF)
        runtime.atomic_json(root/'worker-resource.json',{'started_ns':started,'completed_ns':time.monotonic_ns(),
            'user_cpu_seconds':after.ru_utime-before.ru_utime,'system_cpu_seconds':after.ru_stime-before.ru_stime,
            'max_rss':after.ru_maxrss,'max_rss_unit':'KiB' if sys.platform.startswith('linux') else 'bytes',
            'scope':'whole client capture, scoring and persistence; no model capacity or NIC ceiling claim'})
    if c['owner_root'] is not None:
        # Successful children remain live until the same owner retires them.
        while not (root/'exit.json').exists():healthy();time.sleep(.02)
        gate('exit.json')


def retire(session,children):
    with runtime.NodeLock(session.root/'cleanup.lock', wait_seconds=5):
        session.ensure_healthy();state=json.loads((session.root/'execution-state.json').read_text())
        if state.get('stopping') or state.get('run_id')!=session.run_id:
            raise RuntimeError('model client retirement outside active owner')
        pending=[]
        for process,root,digest in children:
            identity=json.loads((root/'owned-identity.json').read_text())
            c=json.loads((root/'config.json').read_text())
            if (c.get('schema')!='fixed-model-client-worker/v1' or c.get('owner_root')!=str(session.root) or
                    hashlib.sha256((root/'config.json').read_bytes()).hexdigest()!=digest or
                    identity not in state['owned_processes'] or identity['pid']!=process.pid or
                    process.poll() is not None or not runtime.process_alive(identity) or
                    json.loads((root/'worker-result.json').read_text()).get('status')!='passed'):
                raise RuntimeError('model client retirement identity or result differs')
            pending.append((process,root,digest,identity))
        for process,root,digest,identity in pending:
            runtime.atomic_json(root/'exit.json',{'config_sha256':digest,'worker_pid':process.pid})
        until=time.monotonic()+10;receipts=[]
        for process,root,digest,identity in pending:
            if process.wait(timeout=max(.1,until-time.monotonic()))!=0 or runtime.process_alive(identity):
                raise RuntimeError('model client did not stop with zero exit')
            receipt={'identity':identity,'worker_path':str(root.relative_to(session.root)),
                'config_sha256':digest,'result_sha256':hashlib.sha256((root/'worker-result.json').read_bytes()).hexdigest(),
                'exit_code':0,'completed_ns':time.monotonic_ns()}
            runtime.atomic_json(root/'release-receipt.json',receipt);receipts.append(receipt)
        for row in receipts:state['owned_processes'].remove(row['identity'])
        state.setdefault('completed_processes',[]).extend(receipts);runtime.atomic_json(session.root/'execution-state.json',state)


def capture(url,tasks,directory,request_ids,session_ids,clock_id,*,workers=1,cpus=None,session=None,timeout=1800):
    validate_url(url);ids={t['id'] for t in tasks}
    if (type(workers) is not int or workers not in (1,2) or not workers<=len(tasks)<=128 or
            len(ids)!=len(tasks) or set(request_ids)!=ids or set(session_ids)!=ids or
            len(set(request_ids.values()))!=len(tasks) or len(set(session_ids.values()))!=len(tasks) or
            any(not isinstance(x,str) or not x or not x.isascii() or any(not 33<=ord(ch)<=126 for ch in x)
                for mapping in (request_ids,session_ids) for x in mapping.values()) or
            not isinstance(clock_id,str) or not clock_id or
            len({t.get('prompt_version') for t in tasks})!=1 or
            type(timeout) not in (int,float) or not 0<timeout<=7200):
        raise ValueError('model client fixed-workload geometry differs')
    for task in tasks:
        verify_fixed_task.verify(task)
        if task['concurrency']!=len(tasks) or task['cache_designation'] not in ('cold','warm_prefix'):
            raise ValueError('model client capture requires one complete fixed burst')
    if len({t['cache_designation'] for t in tasks})!=1:
        raise ValueError('model client capture mixes cache designs')
    if cpus is not None and (len(set(cpus))!=len(cpus) or len(cpus)<workers or any(type(i) is not int or i<0 for i in cpus)):
        raise ValueError('model client CPU partition differs')
    root=Path(directory)
    if session is not None:
        if cpus is None or not root.resolve().is_relative_to(session.root.resolve()):raise ValueError('owned model clients require owned output and explicit CPU budget')
        session.ensure_healthy()
    deadline=time.monotonic_ns()+int(timeout*1e9)
    if session is not None:deadline=min(deadline,session.deadline_ns-120_000_000_000)
    if deadline<=time.monotonic_ns():raise ValueError('model clients lack cleanup time')
    root.mkdir(parents=True,exist_ok=False)
    contract={'schema':'fixed-model-clients/v1','url':url,'workers':workers,'tasks':tasks,'request_ids':request_ids,
        'session_ids':session_ids,'clock_id':clock_id,'cpus':cpus,'deadline_ns':deadline,
        'owner_root':str(session.root) if session is not None else None,
        'worker_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'raw fixed-model one/two-client capture only; launcher, CPU ceiling, paired comparison, telemetry and native routing verification remain separate'}
    runtime.atomic_json(root/'model-client-contract.json',contract);children=[];logs=[]
    runtime.atomic_bytes(root/'client-source.py',Path(__file__).read_bytes())
    try:
        for i in range(workers):
            child=root/('worker'+str(i));child.mkdir();selected=tasks[i::workers]
            c={**contract,'schema':'fixed-model-client-worker/v1','tasks':selected,
                'threads':len(selected),'request_ids':{t['id']:request_ids[t['id']] for t in selected},
                'session_ids':{t['id']:session_ids[t['id']] for t in selected},
                'cpus':cpus[i::workers] if cpus is not None else None,'request_timeout_seconds':timeout}
            path=child/'config.json';runtime.atomic_json(path,c);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            log=(child/'worker.log').open('xb');logs.append(log)
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',str(path),digest],stdout=log,stderr=subprocess.STDOUT)
            children.append((process,child,digest))
            if session is not None:
                with runtime.NodeLock(session.root/'cleanup.lock', wait_seconds=5):
                    session.ensure_healthy();state=json.loads((session.root/'execution-state.json').read_text())
                    if state.get('stopping'):raise RuntimeError('owner stopped before model client registration')
                    identity=runtime.process_identity(process.pid);state['owned_processes'].append(identity)
                    runtime.atomic_json(session.root/'execution-state.json',state);runtime.atomic_json(child/'owned-identity.json',identity)
        ready=min(deadline,time.monotonic_ns()+10_000_000_000)
        while not all((child/'ready.json').exists() for process,child,digest in children):
            if session is not None:session.ensure_healthy()
            if time.monotonic_ns()>=ready or any(p.poll() is not None for p,_,_ in children):raise RuntimeError('model client failed before gate')
            time.sleep(.02)
        for process,child,digest in children:
            if json.loads((child/'ready.json').read_text())['pid']!=process.pid:raise ValueError('model client ready identity differs')
            runtime.atomic_json(child/'start.json',{'config_sha256':digest,'worker_pid':process.pid})
        while not all((child/'worker-result.json').exists() for _,child,_ in children):
            if session is not None:session.ensure_healthy()
            if (time.monotonic_ns()>=deadline or any(p.poll() not in (None,0) or (child/'abort-request.json').exists() or (child/'failure.json').exists() for p,child,_ in children)):
                raise RuntimeError('model client capture failed or expired')
            time.sleep(.02)
        if session is not None:retire(session,children)
        else:
            for process,child,digest in children:
                if process.wait(timeout=10)!=0:raise RuntimeError('model client exited nonzero')
        for log in logs:log.close()
        runtime.seal_evidence(root)
        return {'status':'captured_verification_pending','workers':workers,'requests':len(tasks),'directory':str(root)}
    except BaseException as exc:
        runtime.atomic_json(root/'model-client-failure.json',{'type':type(exc).__name__})
        if session is not None:session.abort('owned model clients failed')
        for process,child,digest in children:
            if process.poll() is None:process.terminate()
        for process,child,digest in children:
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        raise
    finally:
        for log in logs:log.close()


def capture_fixed(session,url,tasks,directory,timeout,clock_id,namespace,session_ids,*,workers,cpus):
    """Retain subprocess proof and expose the same fixed-capture wire format.

    This performs one burst only. Engine/native route/cache/telemetry checks and
    the paired one/two-client experiment remain the owning launcher's job.
    """
    if not isinstance(namespace,str) or not re.fullmatch(r'fx[a-f0-9]{24}',namespace):
        raise ValueError('fixed model client namespace differs')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    ids={t['id']:namespace+'s'+str(t['session']) for t in tasks}
    # Fixtures may omit a node owner; a real session must provide its exact
    # owned root and CPU budget and is registered by capture itself.
    capture(url,tasks,root/'clients',ids,session_ids,clock_id,workers=workers,
            cpus=cpus,session=session,timeout=timeout)
    import verify_model_clients,verify_fixed_capture
    progress=session.ensure_healthy if session is not None else None
    clients=verify_model_clients.verify(root/'clients',progress=progress)
    records={}
    for path in (root/'clients').glob('worker*/capture/s[0-9]*.json'):
        if progress is not None:progress()
        if path.stem in records:raise ValueError('duplicate subprocess task capsule')
        records[path.stem]=json.loads(path.read_text())
        runtime.atomic_bytes(root/path.name,path.read_bytes())
    if set(records)!={t['id'] for t in tasks}:raise ValueError('subprocess fixed task coverage differs')
    report=verify_fixed_response.population(tasks,{k:v['raw'] for k,v in records.items()},ids,session_ids,clock_id,progress=progress)
    keys=('requests','valid_requests','valid_tokens','duration_seconds','valid_tokens_per_second','invalid_partial_tokens')
    if any(clients[k]!=report[k] for k in keys):raise ValueError('subprocess global denominator differs')
    runtime.atomic_json(root/'summary.json',{'status':'requests_passed_cleanup_pending','summary':{k:report[k] for k in keys}})
    runtime.atomic_json(root/'fixed-inputs.json',{'tasks':tasks,'request_ids':ids,'session_ids':session_ids,
        'clock_id':clock_id,'request_namespace':namespace})
    runtime.atomic_json(root/'fixed-verification.json',report)
    runtime.atomic_json(root/'fixed-capture-result.json',{'status':'requests_captured_cleanup_pending',
        'scope':'raw request capture only; no scored acceptance before full launcher verification'})
    runtime.seal_evidence(root);verify_fixed_capture.verify(root,progress=progress)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['worker']);p.add_argument('config');p.add_argument('sha256');a=p.parse_args();worker(a.config,a.sha256)
