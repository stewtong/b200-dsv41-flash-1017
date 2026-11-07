"""Independent staging monitor; stop only the recorded download process."""
import json
from pathlib import Path
import sys
import time
import runtime
import telemetry

def main(root):
    root = Path(root)
    baseline, last_tick, log_index, log_size = None, None, 0, 0
    while True:
        state = json.loads((root/'stage-state.json').read_text())
        if state.get('stop_watchdog'): return 0
        try:
            tick = time.monotonic()
            if last_tick is not None and tick - last_tick > 6:
                raise runtime.RuntimeErrorClosed('staging safety sample gap exceeds journal window')
            last_tick = tick
            sample = telemetry.safety_sample('/data', ecc_baseline=baseline)
            runtime.verify_gpus_idle()
            if baseline is None: baseline = sample['uncorrected_ecc']
            log_index, log_size = runtime.append_sample(root, 'stage-safety', log_index, log_size, sample)
            hb = json.loads((root/'stage-driver-heartbeat.json').read_text())
            now = time.monotonic_ns()
            if not runtime.heartbeat_fresh(hb, now) or not runtime.process_alive(state['driver_identity']):
                raise runtime.RuntimeErrorClosed('staging driver unavailable')
            if now >= state['deadline_ns']: raise runtime.RuntimeErrorClosed('staging deadline expired')
            if sample['host']['free_disk_fraction'] < 0.17:
                raise runtime.RuntimeErrorClosed('disk staging margin crossed')
            runtime.atomic_json(root/'stage-watchdog-heartbeat.json', {'monotonic_ns':now, 'pid':__import__('os').getpid()})
        except Exception as exc:
            runtime.atomic_json(root/'stage-watchdog-failure.json', {'type':type(exc).__name__, 'message':str(exc)})
            with runtime.NodeLock(root/'stage-worker.lock', wait_seconds=30):
                state = json.loads((root/'stage-state.json').read_text())
                if state.get('worker_identity'):
                    runtime.stop_owned_process(state['worker_identity'], timeout=20)
            return 1
        time.sleep(max(0, 5 - (time.monotonic()-tick)))

if __name__ == '__main__': raise SystemExit(main(sys.argv[1]))
