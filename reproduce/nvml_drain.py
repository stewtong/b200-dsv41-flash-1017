"""Require a fresh post-stop process and memory proof from the isolated collector."""
import json
from pathlib import Path
import time
import nvml_safety_probe
import runtime


def request(root, state):
    if state.get('stopping') is not True or not state.get('nvml_identity'):
        raise runtime.RuntimeErrorClosed('NVML drain requires stopping owned work')
    runtime.atomic_json(Path(root)/'nvml-drain-request.json',{
        'run_id':state['run_id'],'identity':state['nvml_identity'],
        'requested_ns':time.monotonic_ns()})


def validate_idle(data, after_ns, now_ns):
    nvml_safety_probe.validate_sample(data,now_ns)
    if data['sample_started_ns']<after_ns:
        raise runtime.RuntimeErrorClosed('NVML drain sample predates stopped containers')
    counts=[row.get('compute_process_count') for row in data['devices']]
    if any(type(n) is not int or n<0 for n in counts):
        raise runtime.RuntimeErrorClosed('NVML drain compute-process coverage incomplete')
    if any(counts): raise runtime.RuntimeErrorClosed('GPU compute processes exist')
    if any(row['used_bytes']>2048*1024**2 for row in data['devices']):
        raise runtime.RuntimeErrorClosed('GPU memory not near idle')
    return data


def wait_idle(root, state, after_ns, deadline):
    root=Path(root);attempts=[]
    while time.monotonic()<deadline:
        try:
            data=json.loads((root/'nvml-live.json').read_text())
            nvml_safety_probe.validate_live(data,state['nvml_identity'],time.monotonic_ns())
            validate_idle(data,after_ns,time.monotonic_ns())
            validate_ecc(data,state)
            proof={'run_id':state['run_id'],'identity':state['nvml_identity'],
                   'after_stopped_ns':after_ns,'observed_ns':time.monotonic_ns(),
                   'source':'isolated-nvml-compute-and-memory','sample':data}
            runtime.atomic_json(root/'gpu-drain-proof.json',proof)
            return proof
        except (FileNotFoundError,runtime.RuntimeErrorClosed) as exc:
            reason=str(exc) if isinstance(exc,runtime.RuntimeErrorClosed) else 'sample missing'
            attempts.append({'monotonic_ns':time.monotonic_ns(),'reason':reason})
            runtime.atomic_json(root/'nvml-drain-observations.json',{'attempts':attempts})
            if reason not in ('sample missing','NVML collector sample stale',
                              'NVML drain sample predates stopped containers',
                              'GPU compute processes exist','GPU memory not near idle'):
                raise
            time.sleep(min(.1,max(0,deadline-time.monotonic())))
    raise runtime.RuntimeErrorClosed('NVML GPU idle deadline exhausted')


def read_proof(root, state):
    root=Path(root)
    proof=json.loads((root/'gpu-drain-proof.json').read_text())
    request=json.loads((root/'nvml-drain-request.json').read_text())
    if (state.get('stopping') is not True or proof.get('run_id')!=state.get('run_id') or
        proof.get('identity')!=state.get('nvml_identity') or
        request.get('run_id')!=state.get('run_id') or request.get('identity')!=state.get('nvml_identity') or
        type(request.get('requested_ns')) is not int or type(proof.get('after_stopped_ns')) is not int or
        request['requested_ns']>proof['after_stopped_ns'] or
        proof.get('source')!='isolated-nvml-compute-and-memory' or
        proof.get('sample',{}).get('pid')!=state['nvml_identity']['pid']):
        raise runtime.RuntimeErrorClosed('NVML drain proof ownership mismatch')
    data=validate_idle(proof['sample'],proof['after_stopped_ns'],time.monotonic_ns())
    validate_ecc(data,state)
    return data


def validate_ecc(data,state):
    baseline=state.get('nvml_ecc_baseline')
    if state.get('shutdown_safety_schema')=='owned-cleanup-gpu-proof/v1' and baseline is None:
        raise runtime.RuntimeErrorClosed('shutdown ECC baseline missing')
    if baseline is not None and baseline!={str(d['index']):d['volatile_uncorrected_ecc'] for d in data['devices']}:
        raise runtime.RuntimeErrorClosed('NVML ECC counters changed during shutdown')
