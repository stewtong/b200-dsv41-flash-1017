"""Qualify the installed router under the node lock and safety watchdog."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import qualify_router
import runtime
import supervisor

ROUTER_LOCK_SHA='af40d763ab7c538cb8fad58c16414dd2ea358b3520549eae90dac2ee2f337bfd'


def run(config):
    if not os.environ.get('TMUX'): raise runtime.RuntimeErrorClosed('router qualification requires tmux')
    root=Path(config['run_dir']); run_id=config['run_id']; venv=Path(config['router_venv'])
    if (root.parent!=Path('/data/hf-cache') or root.name!='dsv41-'+run_id or
        not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',run_id) or
        venv.parent!=Path('/data/hf-cache') or not re.fullmatch(r'dsv41-router-[a-z0-9-]+',venv.name)):
        raise ValueError('invalid router qualification paths')
    lock=Path(config['router_lock'])
    if hashlib.sha256(lock.read_bytes()).hexdigest()!=ROUTER_LOCK_SHA: raise ValueError('router lock mismatch')
    deadline=datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'))
    if deadline.tzinfo is None: raise ValueError('timezone required')
    seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
    if seconds<=120 or seconds>14400: raise ValueError('router qualification deadline invalid')
    with supervisor.Supervisor(runtime.DEFAULT_NODE_LOCK,root,run_id,time.monotonic_ns()+int(seconds*1e9),phase='router-qualification') as session:
        runtime.atomic_json(root/'run-config.json',config)
        limit=time.monotonic()+15
        while not (root/'watchdog-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic()>=limit: raise TimeoutError('watchdog startup timeout')
            time.sleep(.1)
        session.preflight([])
        runtime.atomic_json(root/'retirement-preflight.json',runtime.verify_retirement(config))
        qualify_router.qualify(venv,lock,ROUTER_LOCK_SHA,root/'qualification')
        session.ensure_healthy()
    runtime.atomic_json(root/'router-result.json',{'status':'passed','scope':'installed versions and flags; routing unqualified'})
    runtime.seal_evidence(root)
