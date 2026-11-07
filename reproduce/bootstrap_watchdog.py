"""Watch a bounded package bootstrap and stop its recorded process group on failure."""
import json
from pathlib import Path
import sys
import time
import runtime
import telemetry


def main(root):
    root=Path(root); baseline=None; last=None; index=0; size=0
    while True:
        state=json.loads((root/'bootstrap-state.json').read_text())
        if state.get('watchdog_stop'): return 0
        try:
            tick=time.monotonic()
            if last is not None and tick-last>6: raise runtime.RuntimeErrorClosed('safety sample gap')
            last=tick
            sample=telemetry.safety_sample('/data',ecc_baseline=baseline)
            runtime.verify_gpus_idle()
            if baseline is None: baseline=sample['uncorrected_ecc']
            index,size=runtime.append_sample(root,'safety',index,size,sample)
            minimum=state.get('minimum_disk_fraction',.17)
            if not isinstance(minimum,(int,float)) or not .17<=minimum<=.95: raise runtime.RuntimeErrorClosed('invalid disk threshold')
            if sample['host']['free_disk_fraction']<minimum: raise runtime.RuntimeErrorClosed('bootstrap disk margin crossed')
            heartbeat=json.loads((root/'driver-heartbeat.json').read_text())
            now=time.monotonic_ns()
            if not runtime.heartbeat_fresh(heartbeat,now) or not runtime.process_alive(state['driver_identity']):
                raise runtime.RuntimeErrorClosed('driver unavailable')
            if now>=state['deadline_ns']: raise runtime.RuntimeErrorClosed('bootstrap deadline expired')
            runtime.atomic_json(root/'watchdog-heartbeat.json',{'pid':__import__('os').getpid(),'monotonic_ns':now})
        except Exception as exc:
            runtime.atomic_json(root/'watchdog-failure.json',{'type':type(exc).__name__,'message':str(exc)})
            with runtime.NodeLock(root/'worker.lock',wait_seconds=30):
                state=json.loads((root/'bootstrap-state.json').read_text())
                if state.get('worker_identity'):
                    runtime.stop_owned_process_group(state['worker_identity'],privileged=bool(state.get('worker_privileged',False)))
            return 1
        time.sleep(max(0,5-(time.monotonic()-tick)))

if __name__=='__main__': raise SystemExit(main(sys.argv[1]))
