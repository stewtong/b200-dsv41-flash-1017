"""Probe installed router affinity and rank headers using synthetic loopback backends."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import time
import launcher
import node_smoke
import qualify_router
import router_probe
import routing_map
import runtime
import supervisor


def run(config):
    if not os.environ.get('TMUX'):raise runtime.RuntimeErrorClosed('router probe requires tmux')
    run_id=config['run_id'];root=Path(config['run_dir']); layout=config['layout']
    if (root.parent!=Path('/data/hf-cache') or root.name!='dsv41-'+run_id or
        not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',run_id) or layout not in launcher.LAYOUTS):
        raise ValueError('invalid router probe identity')
    deadline=datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'))
    if deadline.tzinfo is None:raise ValueError('timezone required')
    seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
    if seconds<=300 or seconds>14400:raise ValueError('router probe deadline invalid')
    qpath=Path(config['router_qualification']);runtime.verify_evidence_manifest(qpath.parent.parent)
    q=json.loads(qpath.read_text()); router=Path(q['router_executable'])
    if (q.get('lock_sha256')!=node_smoke.ROUTER_LOCK_SHA or qualify_router.digest(router)!=q['router_executable_sha256'] or
        qualify_router.digest(qpath.parent/'router-help.txt')!=q['help_sha256']):
        raise ValueError('router qualification changed')
    current=qualify_router.versions(runtime.command([str(router.parent/'python'),'-m','pip','freeze','--all'],timeout=60))
    if current!=q['installed_packages']:raise ValueError('router environment changed')
    calibrate=config.get('calibrate_sessions',False)
    if type(calibrate) is not bool:raise ValueError('calibrate_sessions must be boolean')
    tp,dp,_=launcher.LAYOUTS[layout];ports=[(31000 if calibrate else 31050)+i for i in range(8//(tp*dp))];router_port=31100
    sessions=['session'+str(i) for i in range(6)];attempts=3; servers=[]
    try:
        with supervisor.Supervisor(runtime.DEFAULT_NODE_LOCK,root,run_id,time.monotonic_ns()+int(seconds*1e9),phase='router-probe') as session:
            runtime.atomic_json(root/'run-config.json',config)
            runtime.atomic_bytes(root/'router-qualification.json',qpath.read_bytes())
            runtime.atomic_bytes(root/'router-help.txt',(qpath.parent/'router-help.txt').read_bytes())
            limit=time.monotonic()+15
            while not (root/'watchdog-heartbeat.json').exists():
                session.ensure_healthy()
                if time.monotonic()>=limit:raise TimeoutError('watchdog startup timeout')
                time.sleep(.1)
            session.preflight(ports+[router_port])
            runtime.atomic_json(root/'retirement-preflight.json',runtime.verify_retirement(config))
            for port in ports:servers.append(router_probe.Backend(port))
            argv=[str(router),'--host','127.0.0.1','--port',str(router_port),'--worker-urls']
            argv += ['http://127.0.0.1:'+str(port) for port in ports]
            argv += ['--policy','consistent_hash','--intra-node-data-parallel-size',str(dp),
                     '--request-timeout-secs','60','--disable-retries']
            runtime.atomic_json(root/'router-argv.json',argv)
            session.start_router(argv)
            node_smoke.await_health('http://127.0.0.1:'+str(router_port)+'/health',session,min(time.monotonic()+60,session.deadline_ns/1e9-120))
            def all_records():
                records=[]
                for server in servers:
                    with server.lock:records.extend(server.records)
                return records
            frozen=None
            if calibrate:
                slots=128//(len(ports)*dp);counts=[0]*(len(ports)*dp)
                for index in range(4096):
                    session.ensure_healthy();name='grid'+str(index)
                    router_probe.request(router_port,name,name+'-r0')
                    matches=[r for r in all_records() if r['request_id']==name+'-r0']
                    if len(matches)!=1:raise runtime.RuntimeErrorClosed('calibration request was lost or retried')
                    counts[routing_map.placement(matches[0],ports,dp)]+=1
                    if min(counts)>=slots:break
                else:raise runtime.RuntimeErrorClosed('bounded session calibration did not cover all logical units')
                provisional=routing_map.build(all_records(),ports,dp,slots,require_recheck=False)
                for names in provisional['sessions'].values():
                    for name in names:
                        session.ensure_healthy();router_probe.request(router_port,name,name+'-r1')
                frozen=routing_map.build(all_records(),ports,dp,slots)
            else:
                for name in sessions:
                    for i in range(attempts):
                        session.ensure_healthy();router_probe.request(router_port,name,f'{name}-r{i}')
            diagnostic=router_probe.verify_no_retry(router_port,all_records)
            records=all_records();runtime.atomic_json(root/'arrivals.json',records)
            normal=[row for row in records if row['session_id']!='no-retry']
            if calibrate:
                result={'status':'synthetic_mapping_verified','logical_units':frozen['logical_units']}
                runtime.atomic_json(root/'routing-map.json',{'mapping':frozen,'router_lock_sha256':q['lock_sha256'],
                    'router_executable_sha256':q['router_executable_sha256'],
                    'arrivals_sha256':qualify_router.digest(root/'arrivals.json'),
                    'router_argv_sha256':qualify_router.digest(root/'router-argv.json')})
            else:
                result=router_probe.verify(normal,sessions,attempts,dp,ports)
            result['no_retry_diagnostic']=diagnostic
            runtime.atomic_json(root/'router-probe-observation.json',result)
    finally:
        for server in servers:server.close()
    runtime.atomic_json(root/'router-probe-result.json',{'status':'passed','layout':layout,'scope':'synthetic backend routing only; actual engine rank unverified'})
    runtime.seal_evidence(root)
