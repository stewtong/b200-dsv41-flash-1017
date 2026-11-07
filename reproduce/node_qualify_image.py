"""Run installed image checks under the campaign safety supervisor."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import qualify_image
import runtime
import supervisor


def run(config):
    if not os.environ.get('TMUX'):
        raise runtime.RuntimeErrorClosed('image qualification requires tmux')
    run_id = config['run_id']
    root = Path(config['run_dir'])
    if (not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', run_id) or not root.is_absolute()
            or root.name != 'dsv41-' + run_id
            or not root.parent.resolve(strict=True).is_relative_to(Path('/data'))):
        raise ValueError('invalid run directory or ID')
    deadline = datetime.fromisoformat(config['deadline_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('deadline needs a timezone')
    seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 120 or seconds > 14400:
        raise ValueError('qualification deadline must allow cleanup within four hours')
    deadline_ns = time.monotonic_ns() + int(seconds * 1e9)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with supervisor.Supervisor(runtime.DEFAULT_NODE_LOCK, root, run_id, deadline_ns,
                               manifest_hash=config_hash, phase='image-qualification') as session:
        runtime.atomic_json(root/'run-config.json', config)
        wait_until = time.monotonic() + 15
        while not (root/'watchdog-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic() >= wait_until:
                raise TimeoutError('watchdog startup timeout')
            time.sleep(.1)
        session.preflight([])
        runtime.atomic_json(root/'retirement-preflight.json', runtime.verify_retirement(config))
        qualify_image.qualify(config['image'], config['source_revision'], run_id,
                              root/'qualification', session)
    runtime.atomic_json(root/'qualification-result.json', {'status':'passed', 'cleanup':'passed',
                        'scope':'installed code and flags; GPU serving unqualified'})
    runtime.seal_evidence(root)
