"""Qualify the synthetic client on its production loopback router path."""
from datetime import datetime,timezone
import hashlib,json,os,time,shutil
from pathlib import Path
import client_fixture,client_fixture_backend,cpu_partition,launcher,node_smoke,qualify_router,runtime,supervisor,toolkit_integrity
import verify_node_client_fixture

SHAPES=({'concurrency':8,'requests':32,'input_bytes':32000},
        {'concurrency':32,'requests':128,'input_bytes':32000},
        {'concurrency':128,'requests':256,'input_bytes':32000},
        {'concurrency':8,'requests':16,'input_bytes':4_000_000})


def run(config):
    if not os.environ.get('TMUX'):raise runtime.RuntimeErrorClosed('client fixture requires actual tmux')
    root=Path(config['run_dir']);run_id=config['run_id'];launcher.validate_run_id(run_id)
    if root.parent!=Path('/data/hf-cache') or root.name!='dsv41-'+run_id:raise ValueError('client fixture root differs')
    source=Path(__file__).resolve().parents[1]
    if (Path(config['toolkit_manifest'])!=source/'MANIFEST.json' or
            toolkit_integrity.verify(source/'MANIFEST.json')!=config['toolkit_manifest_sha256'] or
            hashlib.sha256(Path(config['plan']).read_bytes()).hexdigest()!=config['plan_sha256']):
        raise ValueError('client fixture source or plan pin differs')
    qpath=Path(config['router_qualification']);runtime.verify_evidence_manifest(qpath.parent.parent);q=json.loads(qpath.read_text())
    router=Path(q['router_executable'])
    installed=qualify_router.versions(runtime.command([str(router.parent/'python'),'-m','pip','freeze','--all'],timeout=60))
    if (q.get('lock_sha256')!=node_smoke.ROUTER_LOCK_SHA or qualify_router.digest(router)!=q['router_executable_sha256'] or
            installed!=q['installed_packages'] or installed.get('vllm-router')!='0.1.14'):
        raise ValueError('client fixture installed router differs')
    deadline=datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'));seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
    if deadline.tzinfo is None or not 2400<seconds<=14400:raise ValueError('client fixture bound must cover all probes and cleanup')
    disk=shutil.disk_usage('/data')
    if disk.free-10_000_000_000<disk.total*.15:raise RuntimeError('client fixture projected disk margin fails')
    old_affinity=os.sched_getaffinity(0);cpus=cpu_partition.discover();os.sched_setaffinity(0,set(cpus['client_cpus']))
    tp,dp,_=launcher.LAYOUTS[config['layout']];ports=list(range(31000,31000+8//(tp*dp)));servers=[];completed=[]
    try:
        with supervisor.Supervisor(runtime.DEFAULT_NODE_LOCK,root,run_id,time.monotonic_ns()+int(seconds*1e9),phase='client-fixture') as session:
            runtime.atomic_json(root/'run-config.json',config);runtime.atomic_json(root/'cpu-partition.json',cpus)
            runtime.atomic_json(root/'router-runtime-check.json',{'installed_packages':installed,'executable_sha256':qualify_router.digest(router)})
            runtime.atomic_json(root/'router-body-contract.json',verify_node_client_fixture.BODY_CONTRACT)
            runtime.atomic_json(root/'resource-projection.json',{'projected_additional_bytes':10_000_000_000,'disk_free_bytes':disk.free,'disk_total_bytes':disk.total})
            runtime.atomic_bytes(root/'router-qualification.json',qpath.read_bytes());runtime.atomic_bytes(root/'plan.md',Path(config['plan']).read_bytes())
            runtime.atomic_bytes(root/'toolkit-MANIFEST.json',(source/'MANIFEST.json').read_bytes())
            runtime.atomic_json(root/'fixture-design.json',{'shapes':list(SHAPES),'repetitions':3,
                'arms':[{'name':'before','workers':1},{'name':'two','workers':2},{'name':'after','workers':1}],
                'events':128,'piece_bytes':32,'probe_timeout_seconds':60,
                'scope':'CPU/client synthetic lower-bound observations; real-model second-client check still required'})
            limit=time.monotonic()+15
            while not (root/'watchdog-heartbeat.json').exists():
                session.ensure_healthy()
                if time.monotonic()>=limit:raise TimeoutError('client fixture watchdog startup')
                time.sleep(.1)
            session.preflight(ports+[31100]);runtime.atomic_json(root/'retirement-preflight.json',runtime.verify_retirement(config))
            os.sched_setaffinity(0,set(cpus['server_cpus']))
            for port in ports:servers.append(client_fixture_backend.Backend(port,128,32))
            os.sched_setaffinity(0,set(cpus['client_cpus']))
            argv=[str(router),'--host','127.0.0.1','--port','31100','--worker-urls',*['http://127.0.0.1:'+str(p) for p in ports],
                  '--policy','consistent_hash','--intra-node-data-parallel-size',str(dp),'--request-timeout-secs','90','--disable-retries']
            runtime.atomic_json(root/'router-argv.json',argv);router_process=session.start_router(argv)
            runtime.atomic_json(root/'router-identity.json',runtime.process_identity(router_process.pid))
            runtime.atomic_json(root/'router-affinity.json',{'pid':router_process.pid,'cpus':sorted(os.sched_getaffinity(router_process.pid))})
            node_smoke.await_health('http://127.0.0.1:31100/health',session,min(time.monotonic()+60,session.deadline_ns/1e9-120))
            def arrivals():
                rows=[]
                for server in servers:
                    with server.lock:rows.extend(server.records)
                return rows
            session_routes={}
            for shape_index,shape in enumerate(SHAPES):
                for repetition in range(3):
                    for arm,workers in (('before',1),('two',2),('after',1)):
                        session.ensure_healthy()
                        if time.monotonic_ns()+180_000_000_000>=session.deadline_ns:raise RuntimeError('client probe and cleanup do not fit')
                        before=arrivals();label='shape'+str(shape_index)+'-r'+str(repetition)+'-'+arm
                        report=client_fixture.capture('http://127.0.0.1:31100/v1/chat/completions',root/'probes'/label,
                            workers=workers,events=128,piece_bytes=32,cpus=cpus['client_cpus'],session=session,timeout=60,**shape)
                        after=arrivals();prior_ids={id(row) for row in before};selected=[row for row in after if id(row) not in prior_ids]
                        runtime.atomic_json(root/(label+'-arrivals.json'),selected)
                        # Reject a changed native contract at this first probe,
                        # retaining both hops before further fixture dispatch.
                        requests={};bodies={}
                        for folder in sorted((root/'probes'/label).glob('worker*')):
                            worker=json.loads((folder/'config.json').read_text())
                            for task in worker['tasks']:
                                requests[task['id']]=json.loads((folder/(task['id']+'.json')).read_text());bodies[task['id']]=task['body']
                        verify_node_client_fixture.audit_arrivals(requests,selected,ports,dp,cpus['server_cpus'],session_routes,bodies)
                        completed.append({'shape_index':shape_index,'repetition':repetition,'arm':arm,'path':'probes/'+label,
                            'arrivals_file':label+'-arrivals.json','observation':report})
                        runtime.atomic_json(root/'completed-probes.json',completed)
            # Prevent a backend thread from surviving outside the owned lifetime.
            for server in servers:server.close()
            servers=[]
        with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):
            processes=runtime.command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],timeout=5).strip()
            memory=runtime.command(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],timeout=5).splitlines()
            if processes or len(memory)!=8 or any(int(v.strip())>2048 for v in memory):raise RuntimeError('client fixture fresh idle proof failed')
            runtime.atomic_json(root/'fresh-idle-after-client-fixture.json',{'at':datetime.now(timezone.utc).isoformat(),
                'monotonic_ns':time.monotonic_ns(),'compute_processes':processes,'memory_used_mib':[int(v.strip()) for v in memory]})
        runtime.atomic_json(root/'client-fixture-result.json',{'status':'captured_verification_pending','probes':len(completed),
            'scope':'synthetic client/router observations only; no real-model ceiling, saturation, tuning or scored acceptance'})
        runtime.seal_evidence(root)
        return verify_node_client_fixture.verify(root)
    finally:
        for server in servers:server.close()
        os.sched_setaffinity(0,old_affinity)
