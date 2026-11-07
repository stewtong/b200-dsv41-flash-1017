"""Give the existing measurement sampler its own interpreter and owned lifecycle.

This changes scheduling isolation, not telemetry fields, cadence or acceptance.
The independent safety watchdog and NVML collector remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import runtime


class ProcessSampler:
    def __init__(self,child,identity):
        self.child=child
        self.identity=identity

    def is_alive(self):
        return self.child.poll() is None

    def join(self,timeout=None):
        try:self.child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:pass


def start(owner,endpoints,required_names,expected_ranks,server_cpus,nvml_identity):
    root=owner.root
    with runtime.NodeLock(root/'cleanup.lock'):
        owner.ensure_healthy()
        state=json.loads((root/'execution-state.json').read_text())
        if state.get('stopping') or (root/'cleanup-result.json').exists():
            raise runtime.RuntimeErrorClosed('cleanup has already started')
        if (root/'isolated-sampler-contract.json').exists():
            raise runtime.RuntimeErrorClosed('isolated sampler already registered')
        script=Path(__file__).resolve()
        contract={'schema':'owned-isolated-full-sampler/v1','driver_identity':state['driver_identity'],
            'deadline_ns':state['deadline_ns'],'safety_mount':state['safety_mount'],
            'endpoints':endpoints,'required_names':required_names,'expected_ranks':expected_ranks,
            'server_cpus':server_cpus,'nvml_identity':nvml_identity,
            'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
        runtime.atomic_json(root/'isolated-sampler-contract.json',contract)
        with (root/'isolated-sampler.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-B',str(script),'--run-dir',str(root)],
                stdout=log,stderr=log,start_new_session=True)
        identity=runtime.process_identity(child.pid)
        state['owned_processes'].append(identity)
        state['measurement_sampler_identity']=identity
        runtime.atomic_json(root/'execution-state.json',state)
        owner.ensure_healthy()
        runtime.atomic_json(root/'isolated-sampler-start.ok',{'identity':identity,
            'contract_sha256':hashlib.sha256((root/'isolated-sampler-contract.json').read_bytes()).hexdigest()})
    return ProcessSampler(child,identity)


def run(root):
    """Run only behind an exact-identity, source-bound, parent-live start gate."""
    import os
    import supervisor
    root=Path(root)
    contract_path=root/'isolated-sampler-contract.json'
    contract=json.loads(contract_path.read_text())
    if (contract.get('schema')!='owned-isolated-full-sampler/v1' or
        contract.get('source_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise runtime.RuntimeErrorClosed('isolated sampler source or schema differs')
    identity=runtime.process_identity(os.getpid())
    gate_deadline=time.monotonic()+10
    while True:
        state=json.loads((root/'execution-state.json').read_text())
        if (state.get('stopping') or not runtime.process_alive(contract['driver_identity']) or
            state['driver_identity']!=contract['driver_identity']):
            raise runtime.RuntimeErrorClosed('isolated sampler parent stopped before gate')
        gate=root/'isolated-sampler-start.ok'
        if gate.exists():
            receipt=json.loads(gate.read_text())
            if (receipt.get('identity')!=identity or
                receipt.get('contract_sha256')!=hashlib.sha256(contract_path.read_bytes()).hexdigest() or
                state.get('measurement_sampler_identity')!=identity or identity not in state['owned_processes']):
                raise runtime.RuntimeErrorClosed('isolated sampler start ownership differs')
            break
        if time.monotonic()>=gate_deadline:
            raise runtime.RuntimeErrorClosed('isolated sampler start gate expired')
        time.sleep(.05)
    # Reuse the exact existing collector and its raw/timing checks. This instance
    # does not enter a node session, start a watchdog or acquire node ownership.
    worker=supervisor.Supervisor(root/'unused-sampler.lock',root,state['run_id'],
        contract['deadline_ns'],contract['safety_mount'])
    worker.external_sampler=True
    worker.start_sampler(contract['endpoints'],contract['required_names'],contract['expected_ranks'],
        contract['server_cpus'],contract['nvml_identity'])
    try:
        while True:
            state=json.loads((root/'execution-state.json').read_text())
            if state.get('stopping'):break
            if (not runtime.process_alive(contract['driver_identity']) or
                time.monotonic_ns()>=contract['deadline_ns']):
                raise runtime.RuntimeErrorClosed('isolated sampler parent died or deadline expired')
            if worker.failed.is_set() or not worker.sampler.is_alive():
                raise runtime.RuntimeErrorClosed('isolated measurement collection failed')
            runtime.atomic_json(root/'isolated-sampler-stats.json',{'identity':identity,'stats':worker.sampler_stats})
            time.sleep(.05)
    finally:
        worker.sampler_stop.set();worker.sampler.join(10)
        if worker.sampler.is_alive():raise runtime.RuntimeErrorClosed('isolated sampler thread did not stop')
        runtime.atomic_json(root/'isolated-sampler-stats.json',{'identity':identity,'stats':worker.sampler_stats})
    if worker.failed.is_set():raise runtime.RuntimeErrorClosed('isolated sampler collection failed')
    return 0


def main(root):
    try:return run(root)
    except Exception as exc:
        path=Path(root)/'sampler-failure.json'
        # Keep the first actual collector error rather than replacing it with
        # the process wrapper's subsequent observation of that failure.
        if not path.exists():runtime.atomic_json(path,{'type':type(exc).__name__,'message':str(exc)})
        return 1


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--run-dir':raise SystemExit('usage: isolated_sampler.py --run-dir DIRECTORY')
    raise SystemExit(main(sys.argv[2]))
