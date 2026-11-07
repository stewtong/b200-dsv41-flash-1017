"""Verify an isolated router installation against its frozen package lock."""
import hashlib
import re
from pathlib import Path
import runtime

FLAGS = ('--worker-urls', '--host', '--port', '--policy', '--intra-node-data-parallel-size',
         '--request-timeout-secs', '--disable-retries')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize(name):
    return re.sub(r'[-_.]+', '-', name).lower()

def versions(text):
    pairs = re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;\\]+)', text, flags=re.M)
    result = {}
    for name, version in pairs:
        key = normalize(name)
        if key in result: raise ValueError('duplicate package in lock or environment')
        result[key] = version
    return result

def qualify(venv, lock, expected_lock_sha256, out, invoke=runtime.command):
    venv, lock = Path(venv), Path(lock)
    if digest(lock) != expected_lock_sha256: raise ValueError('router lock hash mismatch')
    expected = versions(lock.read_text())
    if expected.get('vllm-router') != '0.1.14': raise ValueError('router version missing from lock')
    python, router = venv/'bin/python', venv/'bin/vllm-router'
    invoke([str(python),'-m','pip','check'], timeout=60)
    frozen = invoke([str(python),'-m','pip','freeze','--all'], timeout=60)
    actual = versions(frozen)
    if any(actual.get(name) != version for name, version in expected.items()):
        raise ValueError('installed router packages differ from lock')
    extra = sorted(set(actual) - set(expected) - {'pip', 'setuptools', 'wheel'})
    if extra: raise ValueError('unexpected installed packages: ' + ', '.join(extra))
    help_text = invoke([str(router),'--help'], timeout=60)
    missing = [flag for flag in FLAGS if flag not in help_text]
    if missing: raise ValueError('router flags missing: ' + ', '.join(missing))
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    runtime.atomic_bytes(out/'router-help.txt',help_text.encode())
    runtime.atomic_bytes(out/'freeze.txt',frozen.encode())
    record = {'lock_sha256':expected_lock_sha256, 'router_executable':str(router.resolve()),
              'router_executable_sha256':digest(router), 'installed_packages':actual,
              'help_sha256':hashlib.sha256(help_text.encode()).hexdigest(),
              'scope':'installed dependency versions and flags; rank routing remains unqualified'}
    runtime.atomic_json(out/'qualification.json', record)
    return record
