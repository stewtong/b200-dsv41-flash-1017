"""Create one hash-locked dependency environment with a node watchdog."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import runtime

LOCKS={'downloader':('hub-linux-py312.lock','1367072f52ae6e0307237a90e2ed8bea8a4bb0d5c4ce5f8313c59a1796688dd4','install_downloader.sh','hub'),
       'router':('router-linux-py312.lock','af40d763ab7c538cb8fad58c16414dd2ea358b3520549eae90dac2ee2f337bfd','install_router.sh','router')}


def image_pull_command(image, root):
    if not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}',image):
        raise ValueError('immutable image required')
    return ['sudo','-n','env','DOCKER_CONFIG='+str(Path(root)/'docker-empty-config'),'/usr/bin/docker','pull',image]


def check_image_headroom(total, available):
    if total<=0 or available<0: raise ValueError('invalid filesystem capacity')
    if available-(9_769_310_188+150_000_000_000+50_000_000_000)<total*.20:
        raise runtime.RuntimeErrorClosed('insufficient projected image-pull headroom')


def run(config):
    if not os.environ.get('TMUX'): raise runtime.RuntimeErrorClosed('bootstrap requires tmux')
    kind=config['kind']; root=Path(config['run_dir']); base=Path(__file__).resolve().parents[1]
    if kind not in (*LOCKS,'image'): raise ValueError('invalid bootstrap kind')
    if root.parent != Path('/data/hf-cache') or not re.fullmatch(r'dsv41-bootstrap-[a-z0-9-]+',root.name):
        raise ValueError('invalid bootstrap run path')
    if os.path.lexists(root): raise ValueError('bootstrap destination already exists')
    expected=None; venv=None
    if kind == 'image':
        image=config['image']
        worker=image_pull_command(image,root)
    else:
        lock_name,expected,installer,prefix=LOCKS[kind]; venv=Path(config['venv_dir'])
        if venv.parent!=Path('/data/hf-cache') or not re.fullmatch(r'dsv41-'+prefix+r'-[a-z0-9-]+',venv.name):
            raise ValueError('invalid bootstrap environment path')
        if os.path.lexists(venv): raise ValueError('bootstrap environment already exists')
        lock=base/'configs'/lock_name; script=base/'reproduce'/installer
        if hashlib.sha256(lock.read_bytes()).hexdigest()!=expected: raise ValueError('dependency lock mismatch')
        worker=['/bin/bash',str(script),str(lock),str(venv)]
    deadline=datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'))
    if deadline.tzinfo is None: raise ValueError('timezone required')
    seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
    if seconds<=120 or seconds>(14400 if kind=='image' else 3600): raise ValueError('bootstrap deadline invalid')
    with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):
        runtime.verify_retirement(config)
        runtime.verify_gpus_idle()
        if kind=='image':
            docker_root=runtime.command(['sudo','-n','docker','info','--format','{{.DockerRootDir}}'],timeout=30).strip()
            if not docker_root.startswith('/') or os.stat(docker_root).st_dev != os.stat('/data').st_dev:
                raise runtime.RuntimeErrorClosed('Docker storage is not on the monitored filesystem')
            listing=runtime.command(['sudo','-n','docker','image','ls','--no-trunc','--digests','--format','{{json .}}'],timeout=30)
            repository,digest=image.split('@',1)
            if any(row.get('Repository')==repository and row.get('Digest')==digest for row in (json.loads(line) for line in listing.splitlines())):
                raise runtime.RuntimeErrorClosed('candidate image already present; qualify existing image')
            fs=os.statvfs('/data'); available=fs.f_bavail*fs.f_frsize; total=fs.f_blocks*fs.f_frsize
            check_image_headroom(total,available)
        root.mkdir()
        if kind=='image':
            (root/'docker-empty-config').mkdir(mode=0o700)
            runtime.atomic_json(root/'image-pull-projection.json',{'free_before':available,'filesystem_bytes':total,
                                'compressed_layer_bytes':9_769_310_188,'extraction_allowance_bytes':150_000_000_000,
                                'other_reserve_bytes':50_000_000_000,'minimum_free_fraction':.20,
                                'scope':'extraction allowance is an estimate, not a measured size'})
        state={'driver_identity':runtime.process_identity(os.getpid()),'worker_identity':None,
               'deadline_ns':time.monotonic_ns()+int(seconds*1e9),'watchdog_stop':False,
               'worker_privileged':kind=='image','minimum_disk_fraction':.20 if kind=='image' else .17}
        runtime.atomic_json(root/'bootstrap-state.json',state)
        runtime.atomic_json(root/'driver-heartbeat.json',{'monotonic_ns':time.monotonic_ns()})
        stop=threading.Event(); failed=threading.Event(); child=None
        log=(root/'watchdog.log').open('ab',buffering=0)
        guard=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('bootstrap_watchdog.py')),str(root)],
                               stdout=log,stderr=log,start_new_session=True)
        state['watchdog_identity']=runtime.process_identity(guard.pid); runtime.atomic_json(root/'bootstrap-state.json',state)
        def pulse():
            start=time.monotonic_ns()
            while not stop.is_set():
                try:
                    now=time.monotonic_ns(); runtime.atomic_json(root/'driver-heartbeat.json',{'monotonic_ns':now})
                    if guard.poll() is not None: raise runtime.RuntimeErrorClosed('bootstrap watchdog exited')
                    path=root/'watchdog-heartbeat.json'
                    if path.exists():
                        hb=json.loads(path.read_text())
                        if hb.get('pid')!=guard.pid or not runtime.heartbeat_fresh(hb,now): raise runtime.RuntimeErrorClosed('watchdog heartbeat stale')
                    elif now-start>15_000_000_000: raise runtime.RuntimeErrorClosed('watchdog heartbeat missing')
                except Exception as exc:
                    failed.set(); runtime.atomic_json(root/'driver-failure.json',{'type':type(exc).__name__,'message':str(exc)})
                    with runtime.NodeLock(root/'worker.lock',wait_seconds=30):
                        current=json.loads((root/'bootstrap-state.json').read_text())
                        if current.get('worker_identity') and runtime.process_alive(current['worker_identity']):
                            runtime.stop_owned_process_group(current['worker_identity'],privileged=bool(current.get('worker_privileged',False)))
                    return
                stop.wait(5)
        heartbeat=threading.Thread(target=pulse,daemon=True); heartbeat.start()
        success=False
        try:
            limit=time.monotonic()+15
            while not (root/'watchdog-heartbeat.json').exists():
                if failed.is_set() or time.monotonic()>limit: raise runtime.RuntimeErrorClosed('watchdog failed to start')
                time.sleep(.1)
            with runtime.NodeLock(root/'worker.lock',wait_seconds=10):
                if failed.is_set() or (root/'watchdog-failure.json').exists(): raise runtime.RuntimeErrorClosed('bootstrap safety stop')
                with (root/'installer.log').open('ab',buffering=0) as out:
                    child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('process_gate.py')),str(root)]+worker,stdout=out,stderr=out,start_new_session=True)
                identity=runtime.process_identity(child.pid)
                if os.getpgid(child.pid)!=child.pid or os.getsid(child.pid)!=child.pid: raise runtime.RuntimeErrorClosed('worker session identity invalid')
                state['worker_identity']=identity; runtime.atomic_json(root/'bootstrap-state.json',state)
                runtime.atomic_json(root/'router-start.ok',{'pid':child.pid})
            while child.poll() is None:
                if failed.is_set() or (root/'watchdog-failure.json').exists() or time.monotonic_ns()>=state['deadline_ns']:
                    raise runtime.RuntimeErrorClosed('bootstrap safety stop')
                time.sleep(.5)
            if child.returncode!=0: raise runtime.RuntimeErrorClosed('bootstrap worker failed')
            try: os.killpg(child.pid,0)
            except ProcessLookupError: pass
            else: raise runtime.RuntimeErrorClosed('installer left a process group behind')
            with runtime.NodeLock(root/'worker.lock',wait_seconds=10):
                current=json.loads((root/'bootstrap-state.json').read_text()); current['worker_identity']=None
                runtime.atomic_json(root/'bootstrap-state.json',current)
            success=True
        finally:
            stop.set(); heartbeat.join(timeout=10)
            with runtime.NodeLock(root/'worker.lock',wait_seconds=30):
                current=json.loads((root/'bootstrap-state.json').read_text())
                if current.get('worker_identity') and runtime.process_alive(current['worker_identity']):
                    runtime.stop_owned_process_group(current['worker_identity'],privileged=bool(current.get('worker_privileged',False)))
                current['watchdog_stop']=True; runtime.atomic_json(root/'bootstrap-state.json',current)
            try: guard.wait(timeout=20)
            finally: log.close()
        if not success or guard.returncode!=0 or failed.is_set() or (root/'watchdog-failure.json').exists():
            raise runtime.RuntimeErrorClosed('bootstrap did not finish cleanly')
        result={'status':'passed','kind':kind}
        if kind=='image':
            rows=json.loads(runtime.command(['sudo','-n','docker','image','inspect',image],timeout=30))
            if (len(rows)!=1 or image not in (rows[0].get('RepoDigests') or []) or rows[0].get('Architecture')!='amd64' or
                (rows[0].get('Config',{}).get('Labels') or {}).get('org.opencontainers.image.revision')!=config['source_revision']):
                raise runtime.RuntimeErrorClosed('pulled image digest, architecture or revision mismatch')
            result.update(image=image,image_id=rows[0].get('Id'),source_revision=config['source_revision'],
                          scope='image identity only; installed code and GPU serving unqualified')
        else:
            result.update(venv=str(venv),lock_sha256=expected)
        runtime.atomic_json(root/'bootstrap-result.json',result)
        runtime.seal_evidence(root)
