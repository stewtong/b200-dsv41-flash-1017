"""Owned CPU-only executable scoring after serving cleanup, plus a fault control."""
import difflib
import hashlib
import json
from pathlib import Path
import time

import coding_panel
import coding_sandbox
import node_smoke
import runtime
import supervisor
import verify_coding_capture
import verify_pilot
import verify_coding_score


def patch_for(source):
    return ''.join(difflib.unified_diff(coding_panel.SOURCE.splitlines(True), source.splitlines(True),
                                       'a/solution.py', 'b/solution.py'))


def start(config, required_seconds):
    image, _, seconds = node_smoke.validate(config)
    if seconds < required_seconds: raise ValueError('CPU scoring and cleanup do not fit deadline')
    return image, seconds


def control(config):
    """Intentional wrong-code/import/CPU-limit trials, never benchmark cells."""
    image, seconds = start(config, 180)
    root = Path(config['run_dir']); task = coding_panel.tasks()[0]
    sources = {
        'correct': 'def solve(value):\n    items = value["items"]\n    k = value["offset"] % len(items) if items else 0\n    return items[k:] + items[:k]\n',
        'wrong': 'def solve(value):\n    return []\n',
        'blocked_import': 'import os\ndef solve(value):\n    return os.listdir("/")\n',
        'cpu_limit': 'def solve(value):\n    while True:\n        pass\n',
    }
    results = []
    with supervisor.Supervisor(node_smoke.NODE_LOCK, root, config['run_id'],
                               time.monotonic_ns() + int(seconds * 1e9), phase='coding-sandbox-control') as session:
        runtime.atomic_json(root / 'run-config.json', config)
        runtime.atomic_json(root / 'control-task.json', task)
        for key, name in [('manifest', 'manifest.json'), ('image_qualification', 'image-qualification.json')]:
            runtime.atomic_bytes(root / name, Path(config[key]).read_bytes())
        limit = time.monotonic() + 15
        while not (root / 'watchdog-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic() >= limit: raise TimeoutError('CPU scorer watchdog startup timeout')
            time.sleep(.1)
        session.preflight([])
        runtime.atomic_json(root / 'retirement-preflight.json', runtime.verify_retirement(config))
        cid = coding_sandbox.create(session, config['image'])
        if runtime.inspect_owned(cid, config['run_id'])['Image'] != image['image_id']:
            raise runtime.RuntimeErrorClosed('scorer image differs from qualification')
        for name, source in sources.items():
            runtime.atomic_bytes(root / (name + '.diff'), patch_for(source).encode())
            result = coding_sandbox.score(session, cid, task, patch_for(source), root / (name + '.json'))
            if result['passed'] != (name == 'correct') or (name == 'wrong' and result['exit_code'] != 0) or (name in ('blocked_import', 'cpu_limit') and result['exit_code'] == 0):
                raise runtime.RuntimeErrorClosed('executable scorer fault control did not distinguish the expected outcome')
            results.append({'name': name, 'passed': result['passed'], 'exit_code': result['exit_code']})
    verify_pilot.verify_cleanup(root, config)
    runtime.atomic_json(root / 'coding-sandbox-control.json', {'status': 'passed', 'controls': results,
                        'scope': 'owned isolated CPU scoring and deliberate fault outcomes only; no model or benchmark acceptance'})
    runtime.seal_evidence(root)
    verify_coding_score.control(root)
    return {'status': 'passed', 'controls': results}


def run(config):
    source = Path(config['capture_run']).resolve(strict=True)
    if source.parent != Path('/data/hf-cache') or not source.name.startswith('dsv41-'):
        raise ValueError('scoring source must be an owned campaign run')
    runtime.verify_evidence_manifest(source)
    old = node_smoke.read(source / 'run-config.json')
    verify_pilot.verify_cleanup(source, old)
    if old.get('image') != config['image'] or old.get('layout') != config['layout']:
        raise ValueError('scorer source image or layout differs')
    capture = source / 'coding-panel'; checked = verify_coding_capture.verify(capture)
    image, seconds = start(config, 100 * 15 + 120)
    qualification = Path(config['scorer_qualification']).resolve(strict=True)
    runtime.verify_evidence_manifest(qualification)
    verify_coding_score.control(qualification)
    control_result = node_smoke.read(qualification / 'coding-sandbox-control.json')
    control_config = node_smoke.read(qualification / 'run-config.json')
    correct = node_smoke.read(qualification / 'correct.json')
    if (control_result.get('status') != 'passed' or control_config.get('image') != config['image'] or
        correct.get('worker_sha256') != hashlib.sha256(Path(__file__).with_name('coding_score_worker.py').read_bytes()).hexdigest()):
        raise ValueError('scorer control does not qualify this image and worker')
    root = Path(config['run_dir']); rows = []
    with supervisor.Supervisor(node_smoke.NODE_LOCK, root, config['run_id'],
                               time.monotonic_ns() + int(seconds * 1e9), phase='coding-executable-scoring') as session:
        runtime.atomic_json(root / 'run-config.json', config)
        for key, name in [('manifest', 'manifest.json'), ('image_qualification', 'image-qualification.json')]:
            runtime.atomic_bytes(root / name, Path(config[key]).read_bytes())
        runtime.atomic_bytes(root / 'scorer-qualification-SHA256SUMS', (qualification / 'SHA256SUMS').read_bytes())
        runtime.atomic_bytes(root / 'capture-source-SHA256SUMS', (source / 'SHA256SUMS').read_bytes())
        runtime.atomic_json(root / 'capture-verification.json', checked)
        runtime.atomic_json(root / 'panel.json', coding_panel.tasks())
        limit = time.monotonic() + 15
        while not (root / 'watchdog-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic() >= limit: raise TimeoutError('CPU scorer watchdog startup timeout')
            time.sleep(.1)
        session.preflight([])
        runtime.atomic_json(root / 'retirement-preflight.json', runtime.verify_retirement(config))
        cid = coding_sandbox.create(session, config['image'])
        if runtime.inspect_owned(cid, config['run_id'])['Image'] != image['image_id']:
            raise runtime.RuntimeErrorClosed('scorer image differs from qualification')
        for task in coding_panel.tasks():
            session.ensure_healthy()
            dest = root / task['id']; dest.mkdir()
            conversation = node_smoke.read(capture / task['id'] / 'conversation.json')
            runtime.atomic_json(dest / 'conversation.json', conversation)
            if not conversation['conversation_valid']:
                result = {'task_id': task['id'], 'passed': False, 'reason': 'invalid_conversation', 'executable': False}
                runtime.atomic_json(dest / 'score.json', result)
            else:
                patch = (capture / task['id'] / 'candidate.diff').read_text()
                runtime.atomic_bytes(dest / 'candidate.diff', patch.encode())
                result = coding_sandbox.score(session, cid, task, patch, dest / 'score.json')
            rows.append(result)
    runtime.verify_evidence_manifest(source)
    verify_pilot.verify_cleanup(root, config)
    result = {'tasks': 100, 'correct': sum(row['passed'] for row in rows),
              'tool_valid': checked['tool_valid'], 'conversation_valid': checked['conversation_valid'],
              'source_manifest_sha256': hashlib.sha256((source / 'SHA256SUMS').read_bytes()).hexdigest(),
              'scope': 'fixed-panel executable correctness; publication requires independent verification and qualified source profiles/caps'}
    runtime.atomic_json(root / 'coding-score-summary.json', result)
    runtime.seal_evidence(root)
    verify_coding_score.verify(root, source, qualification)
    return result
