"""Time separate GPU fields under a bounded synthetic CUDA-graph control."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import runtime
import supervisor
import nvml_safety_probe

FIELDS=('index,ecc.errors.uncorrected.volatile.total','index,memory.used',
        'index,utilization.gpu','index,clocks.sm','direct_nvml_safety',
        'index,memory.used,utilization.gpu,utilization.memory,clocks.sm,power.draw,ecc.errors.uncorrected.volatile.total',
        'index,power.draw','index,utilization.memory')


def sampler_pressure_control(session,root,collector):
    """Exercise the actual sampler with real GPU pressure and fixture metrics.

    This proves instrument operation only. The synthetic rank labels cannot
    qualify a model's metric semantics, worker placement or workload overhead.
    """
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import verify_telemetry
    text=''.join('# TYPE '+name+' '+kind+'\n'+''.join(name+'{engine="'+str(i)+'"} 0\n' for i in range(8)) for name,kind in verify_telemetry.PILOT_TYPES.items()).encode()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/metrics':self.send_error(404);return
            self.send_response(200);self.send_header('Content-Length',str(len(text)));self.end_headers();self.wfile.write(text)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    url='http://127.0.0.1:'+str(server.server_port)+'/metrics'
    started=time.monotonic_ns()
    try:
        session.start_sampler([url],list(verify_telemetry.PILOT_TYPES),{url:list(range(8))},nvml_identity=collector)
        while time.monotonic_ns()-started<30_000_000_000:
            session.ensure_healthy();time.sleep(.1)
        session.sampler_stop.set();session.sampler.join(timeout=5)
        if session.sampler.is_alive():raise runtime.RuntimeErrorClosed('control sampler did not stop')
        session.ensure_healthy()
    finally:
        session.sampler_stop.set()
        if session.sampler is not None:session.sampler.join(timeout=5)
        server.shutdown();server.server_close();worker.join(timeout=2)
    rows=verify_telemetry.lines(root,'telemetry');timings=verify_telemetry.lines(root,'sampler-timing')
    stamps=[r['monotonic_ns'] for r in rows]
    if (len(rows)<30 or len(rows)!=len(timings) or stamps[-1]-stamps[0]<28_000_000_000 or
        max(b-a for a,b in zip(stamps,stamps[1:]))>1_100_000_000 or
        any(t['elapsed_seconds']>1 or t['start_interval_seconds']>1.1 for t in timings)):
        raise runtime.RuntimeErrorClosed('synthetic pressure sampler coverage failed')
    runtime.atomic_json(root/'pressure-sampler-control.json',{'status':'passed','samples':len(rows),
        'window_seconds':(stamps[-1]-stamps[0])/1e9,'max_completed_gap_seconds':max(b-a for a,b in zip(stamps,stamps[1:]))/1e9,
        'max_collection_seconds':max(t['elapsed_seconds'] for t in timings),'stats':session.sampler_stats,
        'metric_source':'synthetic loopback fixture with eight declared labels',
        'scope':'actual sampler under synthetic CUDA pressure; excludes live model ranks and matched workload overhead'})


def memory_headroom(raw, reserve_gib):
    if type(reserve_gib) is not int or reserve_gib not in (0,64,128): raise ValueError('unsupported GPU control reservation')
    rows=[]
    for line in raw.splitlines():
        fields=[int(x.strip()) for x in line.split(',')]
        if len(fields)!=3: raise ValueError('invalid GPU memory preflight')
        rows.append(fields)
    if len(rows)!=8 or {row[0] for row in rows}!=set(range(8)):
        raise ValueError('incomplete GPU memory preflight')
    if any(total<=0 or free<reserve_gib*1024+total*.2 for _,free,total in rows):
        raise runtime.RuntimeErrorClosed('GPU control reservation lacks 20 percent memory headroom')
    return rows


def read_live(root, identity, now_ns, max_age_ns=10_000_000_000, measured=False):
    data=json.loads((Path(root)/'nvml-live.json').read_text())
    return nvml_safety_probe.validate_live(data,identity,now_ns,max_age_ns=max_age_ns,measured=measured)


def freeze_collector(root, identity):
    runtime.atomic_json(Path(root)/'fault-injection-intent.json',{'kind':'freeze-owned-collector',
        'identity':identity,'monotonic_ns':time.monotonic_ns()})
    runtime.signal_owned_process(identity,signal.SIGSTOP)


def pause_shutdown_collector(root,identity,pause_seconds=25):
    """Resume this exact collector even if the driver enters failed cleanup."""
    if type(pause_seconds) not in (int,float) or not 0<pause_seconds<=25:
        raise ValueError('shutdown collector pause exceeds bounded control')
    root=Path(root);done=threading.Event();result={}
    runtime.atomic_json(root/'shutdown-pause-intent.json',{'kind':'pause-owned-collector-during-cleanup',
        'identity':identity,'pause_seconds':pause_seconds,'monotonic_ns':time.monotonic_ns()})
    # The non-daemon resumer is armed before SIGSTOP, so an exception after the
    # signal cannot leave an intentionally paused collector without its release.
    armed=threading.Event()
    def resume():
        armed.wait();done.wait(pause_seconds)
        try:
            runtime.signal_owned_process(identity,signal.SIGCONT)
            result.update(status='resumed',identity=identity,monotonic_ns=time.monotonic_ns())
        except Exception as exc:
            result.update(status='failed',error_type=type(exc).__name__,monotonic_ns=time.monotonic_ns())
        runtime.atomic_json(root/'shutdown-pause-release.json',result)
    worker=threading.Thread(target=resume,name='owned-collector-release',daemon=False);worker.start()
    try:runtime.signal_owned_process(identity,signal.SIGSTOP)
    except BaseException:
        done.set();raise
    finally:armed.set()
    return worker,result


def verify_shutdown_pause(root,identity):
    """Recompute the deliberate unknown interval and require final fresh proof."""
    root=Path(root);read=lambda name:json.loads((root/name).read_text())
    state=read('execution-state.json');intent=read('shutdown-pause-intent.json');release=read('shutdown-pause-release.json')
    proof=read('gpu-drain-proof.json');receipt=read('cleanup-result.json')
    pending=[json.loads(line) for path in root.glob('shutdown-pending-safety-*.jsonl') for line in path.read_text().splitlines()]
    if (intent.get('identity')!=identity or release.get('identity')!=identity or release.get('status')!='resumed' or
        not pending or any(row.get('gpu_safety_status')!='unknown_pending_owned_cleanup' or
            row.get('dispatch_stopped') is not True or row.get('cleanup_owner_identity')!=state.get('cleanup_owner_identity') or
            not state['cleanup_started_ns']<=row['monotonic_ns']<state['cleanup_deadline_ns'] for row in pending) or
        not any(intent['monotonic_ns']+10_000_000_000<=row['monotonic_ns']<release['monotonic_ns'] for row in pending) or
        proof['sample']['monotonic_ns']<release['monotonic_ns'] or receipt.get('success') is not True or
        not 0<receipt['elapsed_seconds']<=120 or receipt['monotonic_ns']>state['cleanup_deadline_ns'] or
        any((root/name).exists() for name in ('watchdog-failure.json','driver-failure.json','primary-failure.json','nvml-cadence-failure.json'))):
        raise runtime.RuntimeErrorClosed('shutdown pause did not demonstrate bounded recovery')
    # The independent pilot verifier recomputes exact stopped identities, raw
    # final GPU memory/process state and unchanged activation ECC/device set.
    import verify_pilot
    verify_pilot.verify_cleanup(root,read('run-config.json'))
    return {'status':'passed','pending_samples':len(pending),'scope':'bounded cleanup only; unknown GPU status never permits dispatch or measurement'}


def probe(field, timeout=3):
    if field not in FIELDS: raise ValueError('unapproved GPU query')
    started=time.monotonic_ns()
    try:
        argv=([sys.executable,'-B',str(Path(__file__).with_name('nvml_safety_probe.py'))] if field=='direct_nvml_safety' else
              ['nvidia-smi','--query-gpu='+field,'--format=csv,noheader,nounits'])
        p=subprocess.run(argv,
                         capture_output=True,text=True,timeout=timeout)
        elapsed=(time.monotonic_ns()-started)/1e9
        if field=='direct_nvml_safety':
            valid=False;data=None
            try:
                data=json.loads(p.stdout);devices=data['devices']
                valid=(p.returncode==0 and len(devices)==8 and {row['index'] for row in devices}==set(range(8)) and
                       len({row['uuid'] for row in devices})==8 and all(row['total_bytes']>0 and row['free_bytes']>=0 and
                       row['used_bytes']>=0 and row['reserved_bytes']>=0 and row['volatile_uncorrected_ecc']>=0 for row in devices))
            except (ValueError,KeyError,TypeError): pass
            return {'field':field,'elapsed_seconds':elapsed,'returncode':p.returncode,'valid':valid,'nvml':data}
        values=[[x.strip() for x in line.split(',')] for line in p.stdout.strip().splitlines()]
        valid=False
        try:
            valid=(p.returncode==0 and len(values)==8 and all(len(row)==len(field.split(',')) for row in values) and
                   {int(row[0]) for row in values}==set(range(8)) and
                   all(all(float(x)>=0 for x in row[1:]) for row in values))
        except ValueError: pass
        return {'field':field,'elapsed_seconds':elapsed,'returncode':p.returncode,'valid':valid,
                'rows':len(values),'values':values,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest()}
    except subprocess.TimeoutExpired:
        return {'field':field,'elapsed_seconds':(time.monotonic_ns()-started)/1e9,'timeout':True}


def shutdown_control(session,root,run_id,image,check_candidate):
    """Exercise four exact container stops; no model or CUDA workload is run."""
    ids=[]
    code="import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('SHUTDOWN_CONTROL_READY',flush=True); time.sleep(600)"
    for i in range(4):
        session.ensure_healthy();check_candidate()
        argv=['docker','run','-d','--restart','no','--network','none','--gpus',
              '"device='+str(2*i)+','+str(2*i+1)+'"','--name','dsv41-'+run_id+'-'+str(i),
              '--label','campaign=dsv41-1017','--label','run_id='+run_id,
              '--entrypoint','python3',image,'-B','-c',code]
        cid=session.create_start(argv);ids.append(cid)
    limit=min(time.monotonic()+20,session.deadline_ns/1e9-120)
    ready=set()
    while len(ready)!=4:
        session.ensure_healthy();check_candidate()
        for cid in ids:
            if cid not in ready and 'SHUTDOWN_CONTROL_READY' in runtime.command(['sudo','-n','docker','logs',cid],timeout=3):ready.add(cid)
        if len(ready)==4:break
        if time.monotonic()>=limit:raise TimeoutError('four-container shutdown control not ready')
        time.sleep(.1)
    runtime.atomic_json(root/'probe-observation.json',{'container_ids':ids,'term_ignored':True,
        'scope':'synthetic four-container SIGTERM-resistant shutdown; no model, busy CUDA or semantic qualification'})


def run(config):
    if not os.environ.get('TMUX'): raise runtime.RuntimeErrorClosed('GPU control requires tmux')
    run_id=config['run_id'];root=Path(config['run_dir']); image=config['image']
    fault=config.get('fault_mode','none')
    if fault not in ('none','freeze_collector','pause_collector_during_shutdown'): raise ValueError('invalid control fault mode')
    if (root.parent!=Path('/data/hf-cache') or root.name!='dsv41-'+run_id or
        not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',run_id)):
        raise ValueError('invalid GPU control identity')
    q=Path(config['image_qualification']);runtime.verify_evidence_manifest(q.parent.parent)
    image_q=json.loads(q.read_text())
    if image_q.get('image')!=image: raise ValueError('image qualification mismatch')
    installed=json.loads(runtime.command(['sudo','-n','docker','image','inspect',image]))
    if len(installed)!=1 or installed[0].get('Id')!=image_q.get('image_id') or image not in (installed[0].get('RepoDigests') or []):
        raise ValueError('installed image changed')
    deadline=datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'))
    if deadline.tzinfo is None: raise ValueError('timezone required')
    seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
    if seconds<=240 or seconds>1200: raise ValueError('control deadline invalid')
    shutdown_count=config.get('shutdown_container_count')
    sampler_control=config.get('sampler_pressure_control',False)
    if type(sampler_control) is not bool or sampler_control and (shutdown_count is not None or fault!='none' or config.get('safety_backend')!='nvml'):
        raise ValueError('sampler control requires fault-free CUDA pressure and NVML safety')
    pressure_count=config.get('pressure_container_count',1)
    if (type(pressure_count) is not int or pressure_count not in (1,4) or
        (pressure_count==4 and (shutdown_count is not None or fault not in ('none','pause_collector_during_shutdown') or config.get('safety_backend')!='nvml')) or
        (config.get('reserve_gib',0)==128 and pressure_count!=4)):
        raise ValueError('pressure control requires one or four declared disjoint GPU replicas')
    if fault=='pause_collector_during_shutdown' and (pressure_count!=4 or config.get('reserve_gib')!=128 or config.get('safety_backend')!='nvml'):
        raise ValueError('shutdown collector pause requires four128GiB replicas with NVML safety')
    if shutdown_count is not None and (type(shutdown_count) is not int or shutdown_count!=4 or
        fault!='none' or config.get('reserve_gib',0)!=0 or config.get('safety_backend')!='nvml'):
        raise ValueError('shutdown control requires four idle containers, NVML safety and no fault/reservation')
    resumer=None;resume_result=None
    with supervisor.Supervisor(runtime.DEFAULT_NODE_LOCK,root,run_id,time.monotonic_ns()+int(seconds*1e9),phase='gpu-control') as session:
        runtime.atomic_json(root/'run-config.json',config)
        limit=time.monotonic()+20
        while not (root/'watchdog-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic()>limit: raise TimeoutError('watchdog startup timeout')
            time.sleep(.1)
        session.preflight([])
        runtime.atomic_json(root/'retirement-preflight.json',runtime.verify_retirement(config))
        collector=session.start_nvml_collector()
        limit=time.monotonic()+20
        while not (root/'nvml-live.json').exists():
            session.ensure_healthy()
            if time.monotonic()>limit: raise TimeoutError('NVML candidate did not produce first sample')
            time.sleep(.1)
        initial=read_live(root,collector,time.monotonic_ns())
        backend=config.get('safety_backend','nvidia-smi')
        if backend not in ('nvidia-smi','nvml'): raise ValueError('invalid control safety backend')
        if backend=='nvml': session.activate_nvml_safety(collector)
        baseline={r['uuid']:r['volatile_uncorrected_ecc'] for r in initial['devices']}
        def check_candidate(measured=False):
            latest=read_live(root,collector,time.monotonic_ns(),max_age_ns=1_500_000_000 if measured else 10_000_000_000,measured=measured)
            current={r['uuid']:r['volatile_uncorrected_ecc'] for r in latest['devices']}
            if set(current)!=set(baseline) or any(current[k]>baseline[k] for k in current):
                raise runtime.RuntimeErrorClosed('NVML candidate device or ECC baseline changed')
            return latest
        if shutdown_count is not None:
            shutdown_control(session,root,run_id,image,check_candidate)
        else:
            reserve=config.get('reserve_gib',0)
            mem=runtime.command(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],timeout=8)
            runtime.atomic_json(root/'gpu-memory-preflight.json',{'reserve_gib_per_gpu':reserve,'gpu_mib':memory_headroom(mem,reserve)})
            script=Path(__file__).with_name('gpu_graph_control.py').resolve()
            ids=[]
            for replica in range(pressure_count):
                session.ensure_healthy();check_candidate()
                devices='all' if pressure_count==1 else '"device='+str(replica*2)+','+str(replica*2+1)+'"'
                argv=['docker','run','-d','--restart','no','--network','none','--gpus',devices,
                      '--name','dsv41-'+run_id+'-'+str(replica),'--label','campaign=dsv41-1017','--label','run_id='+run_id,
                      '--mount',f'type=bind,src={script},dst=/control.py,readonly','--entrypoint','python3',image,'-B','/control.py',str(reserve),str(8 if pressure_count==1 else 2)]
                ids.append(session.create_start(argv))
            limit=min(time.monotonic()+120,session.deadline_ns/1e9-120)
            rows=[]; index=0
            while time.monotonic()<limit:
                session.ensure_healthy()
                check_candidate()
                logs=[runtime.command(['sudo','-n','docker','logs',cid],timeout=5) for cid in ids]
                if all('GRAPH_CONTROL_READY' in value for value in logs): break
                row=probe(FIELDS[index%len(FIELDS)],timeout=2);row.update(phase='startup',repetition=index//len(FIELDS));rows.append(row)
                runtime.atomic_json(root/'probe-progress.json',{'probes':rows});index+=1
                time.sleep(1)
            else: raise TimeoutError('GPU graph control did not become ready')
            if fault=='freeze_collector':
                freeze_collector(root,collector)
                limit=time.monotonic()+25
                while time.monotonic()<limit:
                    session.ensure_healthy();time.sleep(.2)
                raise runtime.RuntimeErrorClosed('watchdog failed to stop on frozen NVML collector')
            measurement_started=time.monotonic_ns()
            runtime.atomic_json(root/'nvml-measurement-mode.json',{'scope':'synthetic cadence diagnostic'})
            limit=time.monotonic()+10
            while True:
                session.ensure_healthy();latest=read_live(root,collector,time.monotonic_ns())
                if latest.get('sample_started_ns',0)>=measurement_started: break
                if time.monotonic()>=limit: raise TimeoutError('NVML measured-mode sample timeout')
                time.sleep(.1)
            for repetition in range(3):
                for field in FIELDS:
                    session.ensure_healthy();check_candidate(measured=True);row=probe(field);row.update(phase='graph_capture',repetition=repetition);rows.append(row)
                    runtime.atomic_json(root/'probe-progress.json',{'probes':rows})
            limit=time.monotonic()+15
            while True:
                session.ensure_healthy();check_candidate(measured=True)
                samples=[]
                for path in sorted(root.glob('nvml-samples-*.jsonl')):
                    content=path.read_text();complete=content.rsplit('\n',1)[0]
                    for line in complete.splitlines():
                        sample=json.loads(line)
                        if sample.get('measurement_mode') and sample.get('sample_started_ns',0)>=measurement_started: samples.append(sample)
                if len(samples)>=10: break
                if time.monotonic()>=limit: raise TimeoutError('ten measured collector samples not observed')
                time.sleep(.1)
            final=check_candidate(measured=True)
            if sampler_control:sampler_pressure_control(session,root,collector)
            runtime.atomic_json(root/'probe-observation.json',{'probes':rows,'container_ids':ids,
                'persistent_nvml_candidate':{'identity':collector,'first_monotonic_ns':initial['monotonic_ns'],
                    'last_monotonic_ns':final['monotonic_ns'],'safety_backend':backend,
                    'measured_samples':len(samples),'max_measured_sample_seconds':max(x['sample_seconds'] for x in samples),
                    'max_measured_gap_seconds':max(x['sample_gap_seconds'] for x in samples),
                    'scope':'synthetic control only; not full-model qualification'},
                'reserve_gib_per_gpu':reserve,'pressure_container_count':pressure_count,
                'scope':'synthetic concurrent CUDA graph capture and declared reservation only; not model compilation or a qualified safety backend'})
            if fault=='pause_collector_during_shutdown':
                resumer,resume_result=pause_shutdown_collector(root,collector)
    if resumer is not None:
        resumer.join(timeout=30)
        if resumer.is_alive() or resume_result.get('status')!='resumed':raise runtime.RuntimeErrorClosed('owned collector release failed')
        runtime.atomic_json(root/'shutdown-pause-verification.json',verify_shutdown_pause(root,collector))
    runtime.atomic_json(root/'probe-result.json',{'status':'completed','cleanup':'passed',
        'scope':'diagnostic control; probe timeouts remain failures, not zero values'})
    runtime.seal_evidence(root)
