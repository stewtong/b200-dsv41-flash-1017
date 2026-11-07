"""Independent safety monitor primitives. No automatic restart or replacement driver."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time
from pathlib import Path
import runtime
import telemetry
import nvml_safety_probe
import nvml_drain

MAX_AGE_NS = 15_000_000_000

def cycle_deadline(tick, last_heartbeat):
    """Reserve two seconds for heartbeat persistence and scheduling jitter."""
    return min(tick+12, last_heartbeat+13) if last_heartbeat is not None else tick+12

def check(driver, host, engine, now_ns, engine_required, deadline_ns):
    reasons = []
    if now_ns >= deadline_ns: reasons.append('operational deadline')
    if not runtime.heartbeat_fresh(driver, now_ns, MAX_AGE_NS): reasons.append('driver heartbeat stale')
    if not runtime.heartbeat_fresh(host, now_ns, MAX_AGE_NS): reasons.append('host sample stale')
    if engine_required and not runtime.heartbeat_fresh(engine, now_ns, MAX_AGE_NS):
        reasons.append('engine sample stale')
    return reasons

def cleanup(run_dir, state, stop=runtime.stop_owned, stop_process=runtime.stop_owned_process, capture=runtime.capture_owned_logs, idle=runtime.verify_gpus_idle, deadline=None):
    """Stop only full, recorded, label-verified IDs; retain failures."""
    run_dir = Path(run_dir)
    cleanup_started = time.monotonic()
    deadline = min(cleanup_started + 120, deadline) if deadline is not None else cleanup_started + 120
    with runtime.NodeLock(run_dir / 'cleanup.lock', wait_seconds=max(0,deadline-time.monotonic())):
        receipt = run_dir/'cleanup-result.json'
        if receipt.exists():
            return read_json(receipt)
        # A create operation may have persisted a new ID while cleanup waited.
        state = read_json(run_dir / 'execution-state.json') if (run_dir / 'execution-state.json').exists() else state
        ids=state['container_ids']
        if (not isinstance(ids,list) or len(ids)>8 or len(ids)!=len(set(ids)) or
            any(not runtime.re_full_id(cid) for cid in ids)):
            raise runtime.RuntimeErrorClosed('invalid cleanup container identities')
        state['stopping'] = True
        state['ready'] = False
        state['cleanup_deadline_ns'] = int(deadline * 1e9)
        state['cleanup_started_ns']=int(cleanup_started*1e9)
        owner=next((p for p in (state.get('driver_identity'),state.get('watchdog_identity')) if p and p.get('pid')==os.getpid()),None)
        if state.get('shutdown_safety_schema')=='owned-cleanup-gpu-proof/v1' and (owner is None or not runtime.process_alive(owner)):
            raise runtime.RuntimeErrorClosed('cleanup owner incarnation is unverified')
        if owner is not None:state['cleanup_owner_identity']=owner
        use_nvml_drain=(state.get('safety_backend')=='nvml' and idle is runtime.verify_gpus_idle)
        if use_nvml_drain: nvml_drain.request(run_dir,state)
        runtime.atomic_json(run_dir/'execution-state.json',state)
        outcomes = []
        idle_probe_attempts = []
        collector = state.get('nvml_identity')
        def stop_identity(identity):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 2: raise runtime.RuntimeErrorClosed('cleanup deadline exhausted')
                stop_process(identity, timeout=min(18, remaining - 2))
                outcomes.append({'pid': identity['pid'], 'status': 'stopped'})
            except Exception as exc:
                outcomes.append({'pid': identity['pid'], 'status': 'failed', 'error_type': type(exc).__name__})
        for identity in state.get('owned_processes', []):
            if identity != collector: stop_identity(identity)
        def stop_container(cid):
            log_status = 'captured'
            capture_started = time.monotonic()
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 1: raise runtime.RuntimeErrorClosed('cleanup deadline exhausted')
                share = remaining / (4 * max(1,len(state['container_ids'])))
                capture(cid, state['run_id'], run_dir / ('container-' + cid + '-' + str(time.monotonic_ns()) + '.log'), timeout=min(15, share))
            except Exception:
                log_status = 'failed'
            capture_seconds = time.monotonic()-capture_started
            stop_started = time.monotonic()
            try:
                remaining = deadline - time.monotonic()
                reserve = 40 if collector else 10
                if remaining <= reserve + 1: raise runtime.RuntimeErrorClosed('cleanup deadline exhausted')
                share = remaining-reserve
                stop(cid, state['run_id'], timeout=min(90, share))
                return {'container_id': cid, 'status': 'stopped' if log_status == 'captured' else 'failed', 'log_status':log_status,
                                 'capture_seconds':capture_seconds,'stop_seconds':time.monotonic()-stop_started}
            except Exception as exc:
                outcome={'container_id': cid, 'status': 'failed', 'log_status':log_status, 'error_type': type(exc).__name__,
                         'capture_seconds':capture_seconds,'stop_seconds':time.monotonic()-stop_started}
                if isinstance(exc, runtime.CleanupStepTimeout): outcome['error_stage']=exc.stage
                return outcome
        if len(ids)>1:
            # One exclusive cleanup owner, independent exact-ID stop commands.
            # Parallel stop latency shares the global budget instead of dividing
            # an80-second stop allowance into four undersized20-second calls.
            with ThreadPoolExecutor(max_workers=min(8,len(ids))) as pool:
                outcomes.extend(pool.map(stop_container,ids))
        else:
            outcomes.extend(stop_container(cid) for cid in ids)
        if time.monotonic()>deadline:
            outcomes.append({'check':'cleanup_deadline','status':'failed','error_type':'CleanupDeadlineExceeded'})
        # Keep safety alive while stopped containers release their GPU processes.
        if collector and state['container_ids'] and all(x['status']=='stopped' for x in outcomes):
            # The driver can report containers stopped while their GPU contexts
            # are still draining. Keep the fixed global 120-second deadline and
            # reserve 20 seconds for collector release and terminal receipts.
            # A separate 20-second NVML wait discarded usable shared budget.
            settle_deadline=deadline-20
            if not use_nvml_drain:
                settle_deadline=min(time.monotonic()+20, settle_deadline)
            if use_nvml_drain:
                drain_started_ns=time.monotonic_ns()
                runtime.atomic_json(run_dir/'gpu-drain-budget.json',{
                    'schema':'remaining-shared-cleanup-deadline/v1',
                    'run_id':state['run_id'],'started_ns':drain_started_ns,
                    'drain_deadline_ns':int(settle_deadline*1e9),
                    'cleanup_deadline_ns':state['cleanup_deadline_ns'],
                    'collector_release_reserve_seconds':20,
                    'scope':'remaining shared cleanup budget; freshness, process count and memory checks unchanged'})
            def bounded(argv, timeout=30):
                remaining=min(timeout,3,settle_deadline-time.monotonic())
                if remaining<=0: raise runtime.RuntimeErrorClosed('GPU idle deadline exhausted')
                return runtime.command(argv,timeout=remaining)
            try:
                if use_nvml_drain:
                    proof=nvml_drain.wait_idle(run_dir,state,time.monotonic_ns(),settle_deadline)
                    idle_probe_attempts.append({'monotonic_ns':time.monotonic_ns(),'status':'idle',
                                                'source':proof['source'],'sample_ns':proof['sample']['monotonic_ns']})
                while not use_nvml_drain:
                    try:
                        idle(bounded)
                        idle_probe_attempts.append({'monotonic_ns':time.monotonic_ns(),'status':'idle'})
                        break
                    except (runtime.RuntimeErrorClosed, subprocess.TimeoutExpired) as exc:
                        idle_probe_attempts.append({'monotonic_ns':time.monotonic_ns(),'status':'unverified','error_type':type(exc).__name__})
                        transient = isinstance(exc, subprocess.TimeoutExpired) or str(exc) in ('GPU compute processes exist','GPU memory not near idle')
                        if not transient or time.monotonic()>=settle_deadline: raise
                        time.sleep(min(.5,max(0,settle_deadline-time.monotonic())))
            except Exception as exc:
                outcomes.append({'check':'gpu_idle','status':'failed','error_type':type(exc).__name__})
        for identity in state.get('owned_processes', []):
            if identity == collector:
                if all(x['status']=='stopped' for x in outcomes): stop_identity(identity)
                else: outcomes.append({'pid':identity['pid'],'status':'failed','error_type':'SafetyCollectorRetained'})
        result = {'monotonic_ns': time.monotonic_ns(), 'outcomes': outcomes, 'elapsed_seconds':time.monotonic()-cleanup_started,
                  'idle_probe_attempts':idle_probe_attempts,
                  'deadline_seconds':120,
                  'success': bool(outcomes) and all(x['status'] == 'stopped' for x in outcomes)}
        runtime.atomic_json(run_dir / 'cleanup-result.json', result)
        return result

def read_json(path):
    with open(path) as f:
        return json.load(f)

def inspect_ownership(run_dir, state, blocked_since, now, timeout=5):
    try:
        with runtime.NodeLock(Path(run_dir)/'cleanup.lock'):
            state = read_json(Path(run_dir)/'execution-state.json')
            if not state.get('stopping'):
                runtime.inspect_running_owned(state['container_ids'], state['run_id'], timeout=timeout)
                if any(not runtime.process_alive(p) for p in state.get('owned_processes', [])):
                    raise runtime.RuntimeErrorClosed('owned router or client process stopped')
            return state, None
    except runtime.RuntimeErrorClosed as exc:
        if str(exc) != 'campaign lock held': raise
        # Create/start and cleanup serialize on this lock. Continue safety
        # sampling, but bound how long ownership inspection can be delayed.
        if blocked_since is None: blocked_since = now
        current = read_json(Path(run_dir)/'execution-state.json')
        shutdown_deadline = current.get('cleanup_deadline_ns')
        if current.get('stopping') and type(shutdown_deadline) is int:
            if now*1e9 >= shutdown_deadline or now-blocked_since > 120:
                raise runtime.RuntimeErrorClosed('shutdown ownership inspection exceeded cleanup deadline')
            return current, blocked_since
        if now - blocked_since > 30:
            raise runtime.RuntimeErrorClosed('container ownership inspection blocked for over 30 seconds')
        return state, blocked_since

def verify_shutdown_gpu_idle(state, deadline, invoke=runtime.command, run_dir=None):
    """Permit collector exit only after exact containers and GPUs are idle."""
    if not state.get('stopping'): raise runtime.RuntimeErrorClosed('NVML safety collector unavailable')
    ids=state.get('container_ids')
    if not isinstance(ids,list) or len(set(ids))!=len(ids) or any(not runtime.re_full_id(cid) for cid in ids):
        raise runtime.RuntimeErrorClosed('invalid shutdown container identities')
    def bounded(argv, timeout=30):
        remaining=min(timeout,deadline-time.monotonic())
        if remaining<=0: raise runtime.RuntimeErrorClosed('shutdown safety deadline exceeded')
        return invoke(argv,timeout=remaining)
    for cid in ids:
        if runtime.inspect_owned(cid,state['run_id'],bounded,timeout=5).get('State',{}).get('Running') is not False:
            raise runtime.RuntimeErrorClosed('cannot release NVML safety while GPU container runs')
    if state.get('safety_backend')=='nvml' and run_dir is not None:
        return nvml_drain.read_proof(run_dir,state)
    runtime.verify_gpus_idle(bounded)


def verify_cleanup_idle(run_dir, state, invoke=runtime.command):
    if state.get('safety_backend')=='nvml':
        data=nvml_drain.read_proof(run_dir,state)
        runtime.atomic_json(Path(run_dir)/'gpu-idle-after-cleanup.json',{
            'monotonic_ns':time.monotonic_ns(),'idle':True,
            'source':'isolated-nvml-compute-and-memory','sample_ns':data['monotonic_ns']})
    else:
        runtime.verify_gpus_idle(invoke)
        runtime.atomic_json(Path(run_dir)/'gpu-idle-after-cleanup.json',{
            'monotonic_ns':time.monotonic_ns(),'idle':True})


def owned_cleanup_pending(run_dir,state,now_ns):
    """Only a live exact cleanup owner may await proof under its fixed budget."""
    root=Path(run_dir);started=state.get('cleanup_started_ns');deadline=state.get('cleanup_deadline_ns')
    owner=state.get('cleanup_owner_identity');collector=state.get('nvml_identity')
    if (state.get('stopping') is not True or state.get('ready') is not False or
        state.get('shutdown_safety_schema')!='owned-cleanup-gpu-proof/v1' or
        any(type(t) is not int for t in (started,deadline)) or not started<=now_ns<deadline or
        not 0<deadline-started<=120_000_001_000 or owner is None or collector is None or
        owner not in (state.get('driver_identity'),state.get('watchdog_identity')) or
        not runtime.process_alive(owner) or not runtime.process_alive(collector) or
        (root/'cleanup-result.json').exists()):return False
    if not (root/'nvml-drain-request.json').is_file():return False
    request=read_json(root/'nvml-drain-request.json')
    if (request.get('run_id')!=state['run_id'] or request.get('identity')!=collector or
        type(request.get('requested_ns')) is not int or not started<=request['requested_ns']<=now_ns):
        raise runtime.RuntimeErrorClosed('pending cleanup drain ownership differs')
    try:
        with runtime.NodeLock(root/'cleanup.lock'):return False
    except runtime.RuntimeErrorClosed as exc:
        if str(exc)!='campaign lock held':raise
        return True

def monitor(run_dir, interval=5):
    """Monitor a driver and host until told to stop or a failure triggers cleanup.

    Engine freshness must be supplied by a separately verified sampler once
    readiness is reached. A missing or malformed record fails closed.
    """
    run_dir = Path(run_dir)
    baseline, last_check, log_index, log_size, lock_blocked_since = None, None, 0, 0, None
    cycle_index, cycle_size, last_heartbeat = 0, 0, None
    pending_index,pending_size=0,0
    while True:
        state = read_json(run_dir / 'execution-state.json')
        if state.get('watchdog_stop') is True:
            return 0
        try:
            tick = time.monotonic()
            if last_check is not None and tick - last_check > 14:
                raise runtime.RuntimeErrorClosed('GPU safety sampling gap exceeds journal window')
            last_check = tick
            deadline = cycle_deadline(tick, last_heartbeat)
            nvml=None;pending=False;last_gpu_sample_ns=None
            if state.get('safety_backend')=='nvml':
                try:
                    current=read_json(run_dir/'nvml-live.json');last_gpu_sample_ns=current.get('monotonic_ns')
                    nvml=nvml_safety_probe.validate_live(current,state['nvml_identity'],time.monotonic_ns())
                except (runtime.RuntimeErrorClosed,FileNotFoundError) as exc:
                    state=read_json(run_dir/'execution-state.json')
                    if not state.get('stopping'): raise
                    if str(exc)=='NVML collector sample stale' and owned_cleanup_pending(run_dir,state,time.monotonic_ns()):
                        pending=True
                    elif (run_dir/'gpu-drain-proof.json').exists():
                        nvml=verify_shutdown_gpu_idle(state,deadline,run_dir=run_dir)
                    else:raise
            elif state.get('safety_backend','nvidia-smi')!='nvidia-smi':
                raise runtime.RuntimeErrorClosed('unknown safety backend')
            sample = telemetry.shutdown_pending_sample(state['safety_mount'],deadline,last_gpu_sample_ns) if pending else telemetry.safety_sample(state['safety_mount'], ecc_baseline=baseline,
                                             gpu_timeout=5 if state.get('ready') else 8, journal_window=14,
                                             deadline=deadline,nvml_data=nvml)
            if baseline is None and not pending: baseline = sample['uncorrected_ecc']
            log_index, log_size = runtime.append_sample(run_dir, 'safety', log_index, log_size, sample)
            host = sample['host']
            inspect_started=time.monotonic()
            remaining=min(5,deadline-inspect_started)
            if remaining <= 0: raise runtime.RuntimeErrorClosed('watchdog cycle deadline exceeded before ownership inspection')
            state, lock_blocked_since = inspect_ownership(run_dir, state, lock_blocked_since, inspect_started, timeout=remaining)
            inspect_done=time.monotonic()
            if inspect_done >= deadline: raise runtime.RuntimeErrorClosed('watchdog cycle deadline exceeded')
            driver = read_json(run_dir / 'driver-heartbeat.json')
            engine = read_json(run_dir / 'engine-heartbeat.json') if state.get('ready') else {}
            now = time.monotonic_ns()
            reasons = check(driver, host, engine, now, bool(state.get('ready')), state['deadline_ns'])
            cycle_index, cycle_size = runtime.append_sample(run_dir,'watchdog-cycle',cycle_index,cycle_size,
                {'monotonic_ns':now,'seconds':inspect_done-tick,'safety_probe_seconds':sample['probe_seconds'],
                 'ownership_seconds':inspect_done-inspect_started,'ownership_deferred':lock_blocked_since is not None,
                 'deadline_seconds':12})
            if pending:
                pending_index,pending_size=runtime.append_sample(run_dir,'shutdown-pending-safety',pending_index,pending_size,{
                    'monotonic_ns':now,'gpu_safety_status':'unknown_pending_owned_cleanup',
                    'last_gpu_sample_ns':last_gpu_sample_ns,'cleanup_deadline_ns':state['cleanup_deadline_ns'],
                    'cleanup_owner_identity':state['cleanup_owner_identity'],'dispatch_stopped':True})
            if time.monotonic() >= deadline: raise runtime.RuntimeErrorClosed('watchdog cycle deadline exceeded before heartbeat')
        except Exception as exc:
            reasons = ['safety check error: ' + type(exc).__name__ + ': ' + str(exc)]
        runtime.atomic_json(run_dir / 'watchdog-heartbeat.json', {'monotonic_ns': time.monotonic_ns(),
                                                                  'pid': os.getpid(), 'reasons': reasons,
                                                                  'gpu_safety_status':'failed' if reasons else sample.get('gpu_safety_status','fresh_verified_sample')})
        last_heartbeat = time.monotonic()
        if reasons:
            runtime.atomic_json(run_dir / 'watchdog-failure.json', {'monotonic_ns': time.monotonic_ns(), 'reasons': reasons})
            cleanup(run_dir, state)
            return 1
        time.sleep(max(0, interval - (time.monotonic() - tick)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    raise SystemExit(monitor(args.run_dir))
