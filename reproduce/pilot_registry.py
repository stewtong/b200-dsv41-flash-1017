"""Persist pilot attempts before launch and retain incomplete attempts for reconciliation."""
import hashlib
import json
from pathlib import Path
import attempts
import runtime

DEFAULT_ROOT=Path('/data/hf-cache/dsv41-pilot-attempts')
METHOD_FILES=('launcher.py','node_pilot.py','node_smoke.py','pilot_burst.py','stream.py','workload.py',
              'vllm_prompt.py','cache_evidence.py','supervisor.py','watchdog.py','telemetry.py','faults.py',
              'runtime.py','readiness.py','verify_pilot.py','toolkit_integrity.py','qualify_router.py','check_fix.py','process_gate.py',
              'prompt_reuse.py')


def begin(config,task_sha,root=DEFAULT_ROOT,identity=runtime.process_identity):
    root=Path(root)
    if type(config.get('retry',False)) is not bool: raise ValueError('retry must be boolean')
    hypothesis=config.get('hypothesis')
    if hypothesis is not None and (not isinstance(hypothesis,str) or not hypothesis.strip() or len(hypothesis)>500 or '\n' in hypothesis or '\r' in hypothesis):
        raise ValueError('invalid hypothesis')
    if root.is_symlink():raise runtime.RuntimeErrorClosed('pilot attempt root is a symlink')
    method_sha=method_fingerprint(config,task_sha)
    attempt=attempts.begin(root,config['cell_id'],retry=config.get('retry',False),
                           retry_reason=config.get('retry_reason'),method_sha256=method_sha)
    payload=json.dumps(config,sort_keys=True,separators=(',',':')).encode()
    runtime.atomic_json(attempt/'context.json',{'run_dir':config['run_dir'],'task_sha256':task_sha,
                        'config_sha256':hashlib.sha256(payload).hexdigest(),'method_sha256':method_sha,'driver_identity':identity(__import__('os').getpid())})
    return attempt


def fail(attempt,exc):
    runtime.atomic_json(Path(attempt)/'failure.json',{'type':type(exc).__name__,'message':str(exc),
                        'status':'requires_reconciliation'})


def finish(attempt,result_path,verification):
    attempt=Path(attempt);data=Path(result_path).read_bytes()
    result=json.loads(data)
    if (result.get('status') not in ('passed','invalid') or verification.get('status')!=result.get('status') or
        verification.get('cell_id')!=result.get('cell_id')):
        raise runtime.RuntimeErrorClosed('pilot result was not independently verified')
    runtime.atomic_bytes(attempt/'outcome.json',data)
    runtime.atomic_json(attempt/'verification.json',verification)
    return attempts.finish(attempt,result['status'],['context.json','outcome.json','verification.json'])


def method_fingerprint(config,task_sha):
    manifest=json.loads(Path(config['manifest']).read_text()); runtime_config=manifest['runtime']
    toolkit_sha=runtime_config.get('toolkit_manifest_sha256')
    if not isinstance(toolkit_sha,str) or len(toolkit_sha)!=64 or any(c not in '0123456789abcdef' for c in toolkit_sha):
        raise ValueError('pilot method requires the complete toolkit fingerprint')
    source=Path(__file__).resolve().parent
    method={'cell_id':config['cell_id'],'task_sha256':task_sha,'image':config['image'],
            'source_revision':config['source_revision'],
            'profile_sha256':runtime_config['starting_profiles'][config['layout']],
            'toolkit_manifest_sha256':toolkit_sha,
            'hypothesis':config.get('hypothesis'),
            'method_files':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in METHOD_FILES}}
    return hashlib.sha256(json.dumps(method,sort_keys=True,separators=(',',':')).encode()).hexdigest()
