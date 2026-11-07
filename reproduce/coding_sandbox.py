"""Run bounded patch tests only in an exact-owned CPU container.

The caller holds a separate supervised scoring session after GPU serving
ownership is released. No candidate is compiled or executed in this process.
"""
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

import coding_patch
import runtime


def argv(image, run_id):
    import re
    if not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}', image):
        raise ValueError('immutable scorer image required')
    return ['docker', 'run', '-d', '--runtime', 'runc', '--network', 'none', '--ipc', 'private',
            '--restart', 'no', '--user', '65534:65534', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '512m', '--cpus', '1',
            '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=16777216',
            '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'NVIDIA_DRIVER_CAPABILITIES=none',
            '--env', 'DSV41_ISOLATED_CODING_SCORER=1',
            '--label', 'campaign=' + runtime.CAMPAIGN, '--label', 'run_id=' + run_id,
            '--entrypoint', 'sleep', image, 'infinity']


def verify_isolation(row):
    host = row.get('HostConfig', {}); config = row.get('Config', {})
    mounts = row.get('Mounts', [])
    if (any(m.get('Type') != 'tmpfs' or m.get('Destination') != '/tmp' for m in mounts) or
        config.get('User') != '65534:65534' or host.get('NetworkMode') != 'none' or
        host.get('IpcMode') != 'private' or host.get('Runtime') != 'runc' or
        host.get('ReadonlyRootfs') is not True or host.get('Privileged') is not False or
        host.get('PidsLimit') != 64 or host.get('Memory') != 512 * 1024 * 1024 or
        host.get('NanoCpus') != 1_000_000_000 or set(host.get('CapDrop', [])) != {'ALL'} or
        host.get('DeviceRequests') or host.get('Devices') or
        host.get('Binds') or not any(x in ('no-new-privileges', 'no-new-privileges=true') for x in host.get('SecurityOpt', []))):
        raise runtime.RuntimeErrorClosed('scorer container isolation mismatch')
    environment = dict(item.split('=', 1) for item in config.get('Env', []) if '=' in item)
    if (environment.get('NVIDIA_VISIBLE_DEVICES') != 'void' or
        environment.get('NVIDIA_DRIVER_CAPABILITIES') != 'none' or
        environment.get('DSV41_ISOLATED_CODING_SCORER') != '1' or
        set(host.get('Tmpfs', {})) != {'/tmp'}):
        raise runtime.RuntimeErrorClosed('scorer environment or tmpfs mismatch')
    options = set(host['Tmpfs']['/tmp'].split(','))
    if not {'rw', 'nosuid', 'nodev', 'noexec', 'size=16777216'}.issubset(options):
        raise runtime.RuntimeErrorClosed('scorer tmpfs options mismatch')
    return {'status': 'isolated_cpu_container', 'container_id': row['Id'],
            'image_id': row['Image'], 'scope': 'namespace, privilege, resource and device isolation; pure Python worker restrictions are additional'}


def create(session, image):
    state = json.loads((session.root / 'execution-state.json').read_text())
    if state.get('container_ids') or state.get('ready'):
        raise runtime.RuntimeErrorClosed('scoring requires a separate session without serving engines')
    cid = session.create_start(argv(image, session.run_id))
    inspected = runtime.inspect_owned(cid, session.run_id)
    proof = verify_isolation(inspected)
    retained = {key: inspected[key] for key in ('Id', 'Image', 'Mounts', 'HostConfig')}
    retained['Config'] = {'User': inspected['Config']['User'],
                          'Env': [item for item in inspected['Config'].get('Env', []) if item.split('=', 1)[0] in
                                  ('NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES', 'DSV41_ISOLATED_CODING_SCORER')]}
    runtime.atomic_json(session.root / 'scorer-container-inspect.json', retained)
    runtime.atomic_json(session.root / 'scorer-isolation.json', proof)
    return cid


def score(session, cid, task, patch, output):
    """A timeout aborts the whole exact-owned scorer, never the next task."""
    source = coding_patch.apply(task['source'], patch)
    verify_isolation(runtime.inspect_owned(cid, session.run_id))
    worker = Path(__file__).with_name('coding_score_worker.py').read_text()
    payload = json.dumps({'source': source, 'cases': task['cases']}).encode()
    if len(payload) > 256 * 1024: raise ValueError('scorer payload exceeds bound')
    session.ensure_healthy()
    started = time.monotonic_ns()
    proc = subprocess.Popen(['sudo', '-n', 'docker', 'exec', '-i', cid, 'python3', '-I', '-S', '-c', worker],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    data = bytearray(); selector = selectors.DefaultSelector(); sent = 0
    try:
        os.set_blocking(proc.stdin.fileno(), False)
        os.set_blocking(proc.stdout.fileno(), False)
        selector.register(proc.stdin, selectors.EVENT_WRITE)
        selector.register(proc.stdout, selectors.EVENT_READ)
        while True:
            session.ensure_healthy()
            if time.monotonic_ns() - started > 15_000_000_000:
                raise runtime.RuntimeErrorClosed('coding scorer wall deadline exceeded')
            events = selector.select(.1)
            if not events: continue
            done = False
            for key, mask in events:
                if key.fileobj is proc.stdin:
                    sent += os.write(proc.stdin.fileno(), payload[sent:sent + 8192])
                    if sent == len(payload):
                        selector.unregister(proc.stdin); proc.stdin.close()
                else:
                    chunk = os.read(proc.stdout.fileno(), 8192)
                    if not chunk: done = True; break
                    data.extend(chunk)
                    if len(data) > 65536: raise runtime.RuntimeErrorClosed('coding scorer output exceeded bound')
            if done: break
        code = proc.wait(timeout=1)
        result = {'schema': 'coding-sandbox-result/v1', 'task_id': task['id'],
                  'task_sha256': task['sha256'], 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(), 'container_id': cid,
                  'exit_code': code, 'elapsed_seconds': (time.monotonic_ns() - started) / 1e9,
                  'raw_output_sha256': hashlib.sha256(data).hexdigest(), 'passed': False}
        runtime.atomic_bytes(Path(output).with_suffix('.raw'), bytes(data))
        if code == 0:
            checked = json.loads(data)
            if (checked.get('schema') != 'coding-executable-score/v1' or
                checked.get('source_sha256') != result['source_sha256'] or
                [row.get('index') for row in checked.get('tests', [])] != list(range(len(task['cases']))) or
                any(type(row.get('passed')) is not bool for row in checked['tests']) or
                checked.get('passed') != all(row['passed'] for row in checked['tests']) or
                any(row['passed'] != ('actual' in row and json.dumps(row['actual'], sort_keys=True, allow_nan=False) ==
                    json.dumps(task['cases'][i]['expected'], sort_keys=True, allow_nan=False)) for i, row in enumerate(checked['tests']))):
                raise runtime.RuntimeErrorClosed('malformed executable score')
            result['passed'] = checked['passed']; result['tests'] = checked['tests']
        runtime.atomic_json(output, result)
        return result
    except BaseException:
        runtime.atomic_bytes(Path(output).with_suffix('.partial'), bytes(data))
        # Losing a docker-exec client does not prove its candidate stopped.
        session.abort('isolated coding scorer did not complete safely')
        raise
    finally:
        selector.close()
        if proc.poll() is None: proc.kill()
        proc.wait(timeout=2)
        if not proc.stdin.closed: proc.stdin.close()
        proc.stdout.close()
