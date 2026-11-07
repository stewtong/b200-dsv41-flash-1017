"""Verify raw executable outcomes without executing patches or importing the scorer."""
import hashlib
import json
import math
from pathlib import Path

import coding_panel
import coding_patch
import runtime
import verify_coding_capture
import verify_pilot


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def isolation(root):
    row = read(root / 'scorer-container-inspect.json'); config = row['Config']; host = row['HostConfig']
    owner = read(root / 'execution-state.json'); qualified = read(root / 'image-qualification.json')
    environment = dict(item.split('=', 1) for item in config['Env'])
    if (owner['container_ids'] != [row['Id']] or row['Image'] != qualified['image_id'] or
        config['User'] != '65534:65534' or environment != {'NVIDIA_VISIBLE_DEVICES': 'void', 'NVIDIA_DRIVER_CAPABILITIES': 'none', 'DSV41_ISOLATED_CODING_SCORER': '1'} or
        host['NetworkMode'] != 'none' or host['IpcMode'] != 'private' or host['Runtime'] != 'runc' or
        host['ReadonlyRootfs'] is not True or host['Privileged'] is not False or
        host['PidsLimit'] != 64 or host['Memory'] != 536870912 or host['NanoCpus'] != 1000000000 or
        set(host['CapDrop']) != {'ALL'} or host.get('DeviceRequests') or host.get('Devices') or host.get('Binds') or
        any(m.get('Type') != 'tmpfs' or m.get('Destination') != '/tmp' for m in row['Mounts']) or
        not any(option in ('no-new-privileges', 'no-new-privileges=true') for option in host['SecurityOpt']) or
        set(host['Tmpfs']) != {'/tmp'} or
        not {'rw', 'nosuid', 'nodev', 'noexec', 'size=16777216'}.issubset(host['Tmpfs']['/tmp'].split(','))):
        raise ValueError('raw CPU scorer isolation differs')
    return row['Id']


def outcome(root, path, task, patch, cid, worker_sha=None):
    row = read(path); raw_path = path.with_suffix('.raw'); source = coding_patch.apply(task['source'], patch)
    if (row.get('schema') != 'coding-sandbox-result/v1' or row.get('task_id') != task['id'] or
        row.get('task_sha256') != task['sha256'] or row.get('source_sha256') != hashlib.sha256(source.encode()).hexdigest() or
        row.get('container_id') != cid or row.get('raw_output_sha256') != digest(raw_path) or
        type(row.get('exit_code')) is not int or type(row.get('passed')) is not bool or
        type(row.get('elapsed_seconds')) not in (int, float) or not math.isfinite(row['elapsed_seconds']) or not 0 < row['elapsed_seconds'] <= 16 or
        (worker_sha is not None and row.get('worker_sha256') != worker_sha)):
        raise ValueError('coding executable source, ownership, timing or raw binding differs')
    if row['exit_code'] != 0:
        if row['passed'] or 'tests' in row: raise ValueError('failed executable was counted correct')
        return False
    raw = read(raw_path); cases = raw.get('tests', [])
    if raw.get('schema') != 'coding-executable-score/v1' or raw.get('source_sha256') != row['source_sha256'] or len(cases) != len(task['cases']):
        raise ValueError('raw executable test population differs')
    verdicts = []
    for i, case in enumerate(cases):
        correct = ('actual' in case and json.dumps(case['actual'], sort_keys=True, allow_nan=False) == json.dumps(task['cases'][i]['expected'], sort_keys=True, allow_nan=False))
        if case.get('index') != i or type(case.get('passed')) is not bool or case['passed'] != correct:
            raise ValueError('raw executable answer differs from expected test value')
        verdicts.append(correct)
    passed = all(verdicts)
    if raw.get('passed') != passed or row['passed'] != passed or row.get('tests') != cases:
        raise ValueError('coding executable verdict differs')
    return passed


def control(root):
    root = Path(root); runtime.verify_evidence_manifest(root)
    config = read(root / 'run-config.json'); verify_pilot.verify_cleanup(root, config)
    cid = isolation(root); task = read(root / 'control-task.json')
    if task != coding_panel.tasks()[0]: raise ValueError('scorer control task changed')
    worker_sha = read(root / 'correct.json').get('worker_sha256')
    if not isinstance(worker_sha, str) or len(worker_sha) != 64: raise ValueError('scorer worker hash missing')
    actual = []
    for name in ('correct', 'wrong', 'blocked_import', 'cpu_limit'):
        path = root / (name + '.json'); row = read(path)
        passed = outcome(root, path, task, (root / (name + '.diff')).read_text(), cid, worker_sha)
        if passed != (name == 'correct') or (name == 'wrong' and row['exit_code'] != 0) or (name in ('blocked_import', 'cpu_limit') and row['exit_code'] == 0):
            raise ValueError('scorer intentional fault did not produce required outcome')
        actual.append({'name': name, 'passed': passed, 'exit_code': row['exit_code']})
    claimed = read(root / 'coding-sandbox-control.json')
    if claimed.get('status') != 'passed' or claimed.get('controls') != actual: raise ValueError('scorer control summary differs')
    return {'status': 'passed', 'worker_sha256': worker_sha, 'controls': actual,
            'scope': 'raw isolated CPU scoring control; no model or benchmark acceptance'}


def verify(root, capture_run, qualification):
    root = Path(root); capture_run = Path(capture_run); qualification = Path(qualification)
    runtime.verify_evidence_manifest(root); runtime.verify_evidence_manifest(capture_run)
    control_check = control(qualification); config = read(root / 'run-config.json')
    verify_pilot.verify_cleanup(root, config); cid = isolation(root)
    captures = verify_coding_capture.verify(capture_run / 'coding-panel')
    if ((root / 'capture-source-SHA256SUMS').read_bytes() != (capture_run / 'SHA256SUMS').read_bytes() or
        (root / 'scorer-qualification-SHA256SUMS').read_bytes() != (qualification / 'SHA256SUMS').read_bytes() or
        read(root / 'panel.json') != coding_panel.tasks()):
        raise ValueError('scoring source provenance differs')
    correct = 0
    for task in coding_panel.tasks():
        dest = root / task['id']; source = capture_run / 'coding-panel' / task['id']
        conversation = read(source / 'conversation.json'); row = read(dest / 'score.json')
        if read(dest / 'conversation.json') != conversation: raise ValueError('scoring conversation changed')
        if not conversation['conversation_valid']:
            if row != {'task_id': task['id'], 'passed': False, 'reason': 'invalid_conversation', 'executable': False}:
                raise ValueError('invalid conversation was not retained as incorrect')
        else:
            if (dest / 'candidate.diff').read_bytes() != (source / 'candidate.diff').read_bytes(): raise ValueError('scored patch differs from returned patch')
            correct += outcome(root, dest / 'score.json', task, (dest / 'candidate.diff').read_text(), cid, control_check['worker_sha256'])
    result = {'tasks': 100, 'correct': correct, 'tool_valid': captures['tool_valid'], 'conversation_valid': captures['conversation_valid'],
              'source_manifest_sha256': digest(capture_run / 'SHA256SUMS')}
    claimed = read(root / 'coding-score-summary.json')
    if any(claimed.get(k) != v for k, v in result.items()): raise ValueError('coding correctness denominator or summary differs')
    result['scope'] = 'independently recomputed synthetic panel correctness; profile and cap qualification remain separate'
    return result
