"""Stage the pinned checkpoint one file at a time with independent safety monitoring."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import node_smoke
import qualify_router
import runtime
import verify_checkpoint

TARGET = Path('/data/hf-cache/DeepSeek-V4.1-Flash-1017')
MODEL = '1bae0e1088993b9b82de08f59de5be5a34e42170'
HUB_LOCK_SHA = '1367072f52ae6e0307237a90e2ed8bea8a4bb0d5c4ce5f8313c59a1796688dd4'
OWNER = Path('/data/hf-cache/dsv41-stage-owner.json')

def check_headroom(total, available, next_file_size):
    if total <= 0 or available < 0 or next_file_size < 0:
        raise ValueError('invalid disk projection')
    if available-next_file_size < total*.17 + 50_000_000_000:
        raise runtime.RuntimeErrorClosed('insufficient projected disk headroom')

def check_no_mounts(target, invoke=runtime.command):
    target = str(target)
    mounts = invoke(['findmnt','-rn','-o','TARGET'],timeout=10).splitlines()
    if any(x == target or x.startswith(target+'/') for x in mounts):
        raise runtime.RuntimeErrorClosed('checkpoint target contains a mount')

def reconcile_previous(marker=OWNER, invoke=runtime.command, alive=runtime.process_alive, allowed_root='/data'):
    if not marker.exists(): return
    old_root=Path(json.loads(marker.read_text())['run_dir']).resolve(strict=True)
    if not old_root.is_relative_to(Path(allowed_root).resolve()) or not re.fullmatch(r'dsv41-stage-[a-z0-9-]+',old_root.name):
        raise runtime.RuntimeErrorClosed('invalid previous staging ownership record')
    old=json.loads((old_root/'stage-state.json').read_text())
    for field in ('driver_identity','watchdog_identity','worker_identity'):
        identity=old.get(field)
        if identity and alive(identity): raise runtime.RuntimeErrorClosed('previous staging process remains alive')
    # A crash can occur between spawning a gate and persisting its identity.
    listing=invoke(['ps','-eo','pid=,args='],timeout=10)
    for line in listing.splitlines():
        fields=line.strip().split()
        if any(Path(fields[i]).name in ('stage_worker.py','stage_watchdog.py') and fields[i+1] == str(old_root)
               for i in range(1,len(fields)-1)):
            raise runtime.RuntimeErrorClosed('unrecorded previous staging process remains alive')

def run(config):
    if not os.environ.get('TMUX'): raise runtime.RuntimeErrorClosed('staging must run inside tmux')
    inv_path = Path(config['inventory'])
    if hashlib.sha256(inv_path.read_bytes()).hexdigest() != node_smoke.INVENTORY_SHA:
        raise ValueError('checkpoint inventory hash mismatch')
    inv = json.loads(inv_path.read_text())
    if inv['revision'] != MODEL or len(inv['files']) != 83: raise ValueError('checkpoint inventory mismatch')
    root = Path(config['run_dir'])
    if not root.is_absolute() or not root.parent.resolve(strict=True).is_relative_to(Path('/data')) or not re.fullmatch(r'dsv41-stage-[a-z0-9-]+', root.name):
        raise ValueError('invalid staging evidence directory')
    root = root.parent.resolve(strict=True)/root.name
    deadline = datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'))
    if deadline.tzinfo is None: raise ValueError('timezone required')
    seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0 or seconds > 86400: raise ValueError('staging deadline must be within 24 hours')
    hub_python = Path(config['hub_python'])
    if not hub_python.is_absolute() or not hub_python.is_file() or not hub_python.parent.parent.resolve(strict=True).is_relative_to(Path('/data')):
        raise ValueError('downloader must be isolated under /data')
    prefix = runtime.command([str(hub_python),'-c','import sys; print(sys.prefix); print(sys.base_prefix)'],timeout=20).splitlines()
    if len(prefix) != 2 or Path(prefix[0]).resolve() != hub_python.parent.parent.resolve() or prefix[0] == prefix[1]:
        raise ValueError('downloader interpreter is not in the expected virtual environment')
    lock = Path(config['hub_lock'])
    if hashlib.sha256(lock.read_bytes()).hexdigest() != HUB_LOCK_SHA: raise ValueError('downloader lock mismatch')
    installed = qualify_router.versions(runtime.command([str(hub_python),'-m','pip','freeze','--all'],timeout=30))
    expected = qualify_router.versions(lock.read_text())
    if any(installed.get(k) != v for k,v in expected.items()) or set(installed)-set(expected)-{'pip','setuptools','wheel'}:
        raise ValueError('downloader environment differs from lock')
    version = runtime.command([str(hub_python),'-c','import huggingface_hub; print(huggingface_hub.__version__)'],timeout=20).strip()
    if version != '1.33.0': raise ValueError('unexpected downloader version')
    with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):
        reconcile_previous()
        runtime.verify_retirement(config)
        runtime.verify_gpus_idle()
        if TARGET.is_symlink() or TARGET.parent.resolve(strict=True) != Path('/data/hf-cache'):
            raise runtime.RuntimeErrorClosed('unsafe checkpoint target')
        check_no_mounts(TARGET)
        root.mkdir(parents=True, exist_ok=False)
        TARGET.mkdir(exist_ok=True)
        state = {'driver_identity':runtime.process_identity(os.getpid()), 'worker_identity':None,
                 'deadline_ns':time.monotonic_ns()+int(seconds*1e9), 'stop_watchdog':False,
                 'revision':MODEL, 'completed':[]}
        runtime.atomic_json(root/'stage-state.json',state)
        runtime.atomic_json(OWNER,{'run_dir':str(root)})
        runtime.atomic_json(root/'stage-driver-heartbeat.json',{'monotonic_ns':time.monotonic_ns()})
        stop = threading.Event(); failed = threading.Event()
        log = (root/'watchdog.log').open('ab',buffering=0)
        guard = subprocess.Popen([os.sys.executable,'-B',str(Path(__file__).with_name('stage_watchdog.py')),str(root)],stdout=log,stderr=log,start_new_session=True)
        state['watchdog_identity'] = runtime.process_identity(guard.pid)
        runtime.atomic_json(root/'stage-state.json',state)
        def pulse():
            started=time.monotonic_ns()
            while not stop.is_set():
                try:
                    runtime.atomic_json(root/'stage-driver-heartbeat.json',{'monotonic_ns':time.monotonic_ns()})
                    if guard.poll() is not None: raise runtime.RuntimeErrorClosed('staging watchdog exited')
                    p=root/'stage-watchdog-heartbeat.json'
                    if p.exists():
                        hb=json.loads(p.read_text())
                        if hb.get('pid') != guard.pid or not runtime.heartbeat_fresh(hb,time.monotonic_ns()):
                            raise runtime.RuntimeErrorClosed('staging watchdog heartbeat stale')
                    elif time.monotonic_ns()-started > 15_000_000_000:
                        raise runtime.RuntimeErrorClosed('staging watchdog heartbeat missing')
                except Exception as exc:
                    failed.set()
                    runtime.atomic_json(root/'stage-driver-failure.json',{'type':type(exc).__name__,'message':str(exc)})
                    with runtime.NodeLock(root/'stage-worker.lock',wait_seconds=30):
                        current=json.loads((root/'stage-state.json').read_text())
                        if current.get('worker_identity'): runtime.stop_owned_process(current['worker_identity'])
                    return
                stop.wait(5)
        heartbeat=threading.Thread(target=pulse,daemon=True); heartbeat.start()
        receipt = None
        try:
            def check_safety():
                if failed.is_set() or time.monotonic_ns() >= state['deadline_ns'] or (root/'stage-watchdog-failure.json').exists():
                    raise runtime.RuntimeErrorClosed('staging safety stop')
            start=time.monotonic()+15
            while not (root/'stage-watchdog-heartbeat.json').exists():
                if failed.is_set() or time.monotonic() >= start: raise runtime.RuntimeErrorClosed('staging watchdog not ready')
                time.sleep(.1)
            for index,row in enumerate(inv['files']):
                check_safety()
                existing=TARGET.joinpath(row['path'])
                if os.path.lexists(existing):
                    verify_checkpoint.verify_file(row,TARGET,check_safety)
                    state['completed'].append(row['path']); runtime.atomic_json(root/'stage-state.json',state)
                    continue
                fs=os.statvfs(TARGET); available=fs.f_bavail*fs.f_frsize; total=fs.f_blocks*fs.f_frsize
                check_headroom(total,available,row['size'])
                gate='worker-%03d.ok'%index
                with runtime.NodeLock(root/'stage-worker.lock',wait_seconds=10):
                    if failed.is_set() or (root/'stage-watchdog-failure.json').exists(): raise runtime.RuntimeErrorClosed('staging stop before download')
                    state['current_file']=row['path']; runtime.atomic_json(root/'stage-state.json',state)
                    with (root/('worker-%03d.log'%index)).open('ab',buffering=0) as worker_log:
                        worker=subprocess.Popen([str(hub_python),'-B',str(Path(__file__).with_name('stage_worker.py')),
                                                 str(root),str(TARGET),MODEL,row['path'],gate],
                                                stdout=worker_log,stderr=worker_log,start_new_session=True)
                    state['worker_identity']=runtime.process_identity(worker.pid)
                    runtime.atomic_json(root/'stage-state.json',state)
                    runtime.atomic_json(root/gate,{'pid':worker.pid,'file':row['path']})
                while worker.poll() is None:
                    if failed.is_set() or (root/'stage-watchdog-failure.json').exists() or time.monotonic_ns() >= state['deadline_ns']:
                        with runtime.NodeLock(root/'stage-worker.lock',wait_seconds=30):
                            runtime.stop_owned_process(state['worker_identity'])
                        raise runtime.RuntimeErrorClosed('staging worker stopped by safety monitor')
                    time.sleep(1)
                if worker.returncode != 0:
                    runtime.atomic_json(root/'stage-download-failure.json',{'file':row['path'],'exit_code':worker.returncode})
                    raise runtime.RuntimeErrorClosed('checkpoint download failed; attempt retained')
                state['worker_identity']=None; runtime.atomic_json(root/'stage-state.json',state)
                verify_checkpoint.verify_file(row,TARGET,check_safety)
                state['completed'].append(row['path']); runtime.atomic_json(root/'stage-state.json',state)
            receipt=verify_checkpoint.verify(inv_path,TARGET,check_safety)
            receipt.update(inventory_sha256=node_smoke.INVENTORY_SHA,root=str(TARGET))
            check_safety()
        finally:
            stop.set(); heartbeat.join(timeout=10)
            with runtime.NodeLock(root/'stage-worker.lock',wait_seconds=30):
                current=json.loads((root/'stage-state.json').read_text())
                if current.get('worker_identity') and runtime.process_alive(current['worker_identity']):
                    runtime.stop_owned_process(current['worker_identity'])
                current['stop_watchdog']=True; runtime.atomic_json(root/'stage-state.json',current)
            try: guard.wait(timeout=20)
            finally: log.close()
        if receipt is not None:
            if guard.returncode != 0 or (root/'stage-watchdog-failure.json').exists() or failed.is_set():
                raise runtime.RuntimeErrorClosed('staging watchdog did not finish cleanly')
            runtime.atomic_json(root/'checkpoint-receipt.json',receipt)
            runtime.seal_evidence(root)
