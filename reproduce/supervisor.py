"""Launch-session ownership and watchdog integration for node-side work."""
import json
import hashlib
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import runtime
import faults
import telemetry
import watchdog
import nvml_safety_probe

# Leave collection-time headroom while retaining the required maximum gap.
# This is an instrument setting, whose workload overhead still needs controls.
SAMPLER_PERIOD_SECONDS = 0.5
SAMPLER_MAX_GAP_NS = 1_100_000_000


def check_completed_sample(previous_ns, current_ns):
    if type(current_ns) is not int or current_ns < 0:
        raise runtime.RuntimeErrorClosed('telemetry sample clock invalid')
    if previous_ns is not None and not 0 < current_ns-previous_ns <= SAMPLER_MAX_GAP_NS:
        raise runtime.RuntimeErrorClosed('telemetry completed-sample gap exceeds 1.1 seconds')

class Supervisor:
    """Hold exclusive node ownership until cleanup completes.

    This class is a lifecycle component; it does not yet implement the complete
    benchmark protocol or establish image qualification by itself.
    """
    def __init__(self, node_lock, run_dir, run_id, deadline_ns, safety_mount='/data', manifest_hash=None, phase='smoke'):
        self.root = Path(run_dir)
        self.run_id = run_id
        self.deadline_ns = deadline_ns
        self.safety_mount = safety_mount
        self.manifest_hash = manifest_hash
        self.phase = phase
        self.lock = runtime.NodeLock(node_lock)
        self.failed = threading.Event()
        self.stop_event = threading.Event()
        self.monitor = None
        self.pulse = None
        self.log = None
        self.preflight_done = False
        self.sampler_stop = threading.Event()
        self.sampler = None
        self.sampler_control = None
        self.external_sampler = False
        self.sampler_process = None
        self.sampler_stats = {'count':0, 'total_seconds':0.0, 'thread_cpu_seconds':0.0, 'max_seconds':0.0, 'max_interval_seconds':0.0}

    @property
    def sampler_stats(self):
        if self.sampler_process is not None:
            path=self.root/'isolated-sampler-stats.json'
            if path.exists():
                row=json.loads(path.read_text())
                if row['identity']!=self.sampler_process.identity:
                    raise runtime.RuntimeErrorClosed('isolated sampler stats identity differs')
                return row['stats']
        return self._sampler_stats

    @sampler_stats.setter
    def sampler_stats(self,value):
        self._sampler_stats=value

    def __enter__(self):
        self.lock.__enter__()
        try:
            self.root.mkdir(parents=True, exist_ok=False)
            if self.deadline_ns <= time.monotonic_ns():
                raise runtime.RuntimeErrorClosed('operational deadline already expired')
            state = {'run_id':self.run_id, 'driver_pid':os.getpid(), 'driver_identity':runtime.process_identity(os.getpid()),
                     'container_ids':[], 'owned_processes':[], 'ready':False, 'stopping':False, 'watchdog_stop':False,
                     'safety_mount':self.safety_mount, 'deadline_ns':self.deadline_ns,
                     'manifest_sha256':self.manifest_hash, 'phase':self.phase,
                     'started_utc':datetime.now(timezone.utc).isoformat()}
            runtime.atomic_json(self.root/'execution-state.json', state)
            runtime.atomic_json(self.root/'driver-heartbeat.json', {'monotonic_ns':time.monotonic_ns()})
            self.log = (self.root/'watchdog.log').open('ab', buffering=0)
            self.monitor = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('watchdog.py')),
                                             '--run-dir', str(self.root)], stdout=self.log, stderr=self.log,
                                            start_new_session=True)
            state['watchdog_identity'] = runtime.process_identity(self.monitor.pid)
            runtime.atomic_json(self.root/'execution-state.json', state)
            self.pulse = threading.Thread(target=self._pulse, daemon=True)
            self.pulse.start()
            return self
        except BaseException:
            if self.monitor is not None and self.monitor.poll() is None:
                self.monitor.terminate()
                self.monitor.wait(timeout=10)
            if self.log is not None: self.log.close()
            self.lock.__exit__(None, None, None)
            raise

    def _pulse(self):
        start = time.monotonic_ns()
        last_log_poll = None
        while not self.stop_event.is_set():
            try:
                now = time.monotonic_ns()
                runtime.atomic_json(self.root/'driver-heartbeat.json', {'monotonic_ns':now})
                if self.monitor.poll() is not None:
                    raise runtime.RuntimeErrorClosed('watchdog exited')
                state = json.loads((self.root/'execution-state.json').read_text())
                ids = state.get('container_ids', [])
                if not state.get('stopping') and ids:
                    if last_log_poll is not None and now-last_log_poll > 9_000_000_000:
                        raise runtime.RuntimeErrorClosed('container log safety check gap')
                    last_log_poll = now
                    faults.check(ids)
                else:
                    last_log_poll = None
                heartbeat = self.root/'watchdog-heartbeat.json'
                if heartbeat.exists():
                    data = json.loads(heartbeat.read_text())
                    if data.get('pid') != self.monitor.pid or not runtime.heartbeat_fresh(data, time.monotonic_ns()) or data.get('reasons'):
                        raise runtime.RuntimeErrorClosed('watchdog heartbeat invalid')
                elif now - start > 15_000_000_000:
                    raise runtime.RuntimeErrorClosed('watchdog heartbeat missing')
            except Exception as exc:
                self.failed.set()
                runtime.atomic_json(self.root/'driver-failure.json', {'type':type(exc).__name__, 'message':str(exc)})
                try:
                    state = json.loads((self.root/'execution-state.json').read_text())
                    watchdog.cleanup(self.root, state)
                except Exception as cleanup_exc:
                    runtime.atomic_json(self.root/'driver-cleanup-failure.json', {'type':type(cleanup_exc).__name__})
                return
            self.stop_event.wait(5)

    def ensure_healthy(self):
        if (self.root/'sampler-failure.json').exists():
            raise runtime.RuntimeErrorClosed('measurement sampler failed; original record retained')
        if self.sampler_process is not None and not self.sampler_process.is_alive():
            raise runtime.RuntimeErrorClosed('isolated measurement sampler exited')
        if (self.root/'nvml-cadence-failure.json').exists():
            raise runtime.RuntimeErrorClosed('measured NVML cadence failed; safety collector remains active')
        if self.failed.is_set() or (self.root/'watchdog-failure.json').exists() or time.monotonic_ns() >= self.deadline_ns:
            raise runtime.RuntimeErrorClosed('supervisor safety condition failed')

    def preflight(self, ports):
        self.ensure_healthy()
        hb = json.loads((self.root/'watchdog-heartbeat.json').read_text())
        if hb.get('pid') != self.monitor.pid or hb.get('reasons') or not runtime.heartbeat_fresh(hb, time.monotonic_ns()):
            raise runtime.RuntimeErrorClosed('watchdog heartbeat is not ready')
        sample = runtime.live_preflight(self.run_id, ports)
        runtime.atomic_json(self.root/'preflight.json', {'monotonic_ns':time.monotonic_ns(), 'safety':sample,
                                                         'ports':ports, 'run_id':self.run_id})
        self.preflight_done = True

    def create_start(self, argv):
        self.ensure_healthy()
        if not self.preflight_done:
            raise runtime.RuntimeErrorClosed('node preflight not passed')
        cid = runtime.create_and_start(argv, self.run_id, self.root)
        self.ensure_healthy()
        return cid

    def start_router(self, argv):
        self.ensure_healthy()
        if not self.preflight_done or not argv or Path(argv[0]).name != 'vllm-router':
            raise runtime.RuntimeErrorClosed('invalid router start or missing preflight')
        with runtime.NodeLock(self.root/'cleanup.lock', wait_seconds=5):
            self.ensure_healthy()
            state = json.loads((self.root/'execution-state.json').read_text())
            if state.get('stopping') or (self.root/'cleanup-result.json').exists():
                raise runtime.RuntimeErrorClosed('cleanup has already started')
            with (self.root/'router.log').open('ab', buffering=0) as log:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('process_gate.py')),
                                          str(self.root)] + argv, stdout=log, stderr=log, start_new_session=True)
            try:
                identity = runtime.process_identity(child.pid)
                state['owned_processes'].append(identity)
                runtime.atomic_json(self.root/'execution-state.json', state)
                self.ensure_healthy()
                runtime.atomic_json(self.root/'router-start.ok', {'pid':child.pid})
                return child
            except BaseException:
                # The gate expires even if identity capture fails.
                raise

    def start_nvml_collector(self):
        """Start one exact recorded candidate collector behind its own gate."""
        self.ensure_healthy()
        if not self.preflight_done: raise runtime.RuntimeErrorClosed('missing preflight')
        gate=self.root/'nvml-gate'
        with runtime.NodeLock(self.root/'cleanup.lock', wait_seconds=5):
            self.ensure_healthy()
            state=json.loads((self.root/'execution-state.json').read_text())
            if state.get('stopping') or (self.root/'cleanup-result.json').exists():
                raise runtime.RuntimeErrorClosed('cleanup has already started')
            gate.mkdir()
            script=Path(__file__).with_name('nvml_safety_probe.py')
            with (self.root/'nvml-collector.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('process_gate.py')),
                    str(gate),sys.executable,'-B',str(script),'--loop',str(self.root)],stdout=log,stderr=log,start_new_session=True)
            identity=runtime.process_identity(child.pid)
            state['owned_processes'].append(identity);runtime.atomic_json(self.root/'execution-state.json',state)
            self.ensure_healthy();runtime.atomic_json(gate/'router-start.ok',{'pid':child.pid})
            return identity

    def start_input_worker(self, cid):
        """Register a single preparation child before permitting container exec."""
        self.ensure_healthy()
        if self.phase != 'prompt-generation' or not self.preflight_done:
            raise runtime.RuntimeErrorClosed('input worker requires preparation preflight')
        gate = self.root/'input-worker-gate'
        with runtime.NodeLock(self.root/'cleanup.lock', wait_seconds=5):
            self.ensure_healthy()
            state = json.loads((self.root/'execution-state.json').read_text())
            if (state.get('stopping') or (self.root/'cleanup-result.json').exists() or
                    state.get('container_ids') != [cid]):
                raise runtime.RuntimeErrorClosed('input worker requires exact live ownership')
            row = runtime.inspect_owned(cid, self.run_id)
            if row.get('State', {}).get('Running') is not True:
                raise runtime.RuntimeErrorClosed('input container is not running')
            gate.mkdir()
            argv = ['sudo', '-n', 'docker', 'exec', cid, 'python3', '-B',
                    '/campaign/input_panel_worker.py', '/input-spec.json', '/output']
            with (self.root/'input-worker.log').open('ab', buffering=0) as log:
                child = subprocess.Popen([sys.executable, '-B',
                    str(Path(__file__).with_name('process_gate.py')), str(gate)] + argv,
                    stdout=log, stderr=log, start_new_session=True)
            identity = runtime.process_identity(child.pid)
            state['owned_processes'].append(identity)
            runtime.atomic_json(self.root/'execution-state.json', state)
            runtime.atomic_json(self.root/'input-worker-identity.json', identity)
            self.ensure_healthy()
            runtime.atomic_json(gate/'router-start.ok', {'pid': child.pid})
            return child

    def activate_nvml_safety(self, identity):
        self.ensure_healthy()
        data=json.loads((self.root/'nvml-live.json').read_text())
        nvml_safety_probe.validate_live(data,identity,time.monotonic_ns())
        with runtime.NodeLock(self.root/'cleanup.lock', wait_seconds=5):
            self.ensure_healthy()
            state=json.loads((self.root/'execution-state.json').read_text())
            if state.get('stopping') or identity not in state.get('owned_processes',[]):
                raise runtime.RuntimeErrorClosed('NVML collector ownership missing')
            state['safety_backend']='nvml';state['nvml_identity']=identity
            runtime.atomic_json(self.root/'nvml-initial-safety.json',data)
            state['nvml_ecc_baseline']={str(d['index']):d['volatile_uncorrected_ecc'] for d in data['devices']}
            state['shutdown_safety_schema']='owned-cleanup-gpu-proof/v1'
            runtime.atomic_json(self.root/'execution-state.json',state)
        runtime.atomic_json(self.root/'safety-backend.json',{'backend':'isolated-nvml-v2',
            'identity':identity,'scope':'fresh memory and ECC safety; performance fields remain required in measured sampling'})

    def begin_sampler_control(self,cell_id,until_ns):
        """Select one declared overhead arm; required safety never switches off."""
        self.ensure_healthy()
        state=json.loads((self.root/'execution-state.json').read_text())
        config=json.loads((self.root/'run-config.json').read_text())
        manifest=json.loads((self.root/'manifest.json').read_text())
        cells=[row for row in manifest['cells'] if row['id']==cell_id]
        if (self.phase!='qualification' or self.sampler_control is not None or self.sampler is None or
                not state.get('ready') or state.get('stopping') or config.get('cell_id')!=cell_id or
                len(cells)!=1 or cells[0].get('suite')!='sampler_overhead' or
                cells[0].get('layout')!=config.get('layout') or cells[0].get('repetitions')!=3 or
                cells[0].get('parameters',{}).get('input')!=8000 or cells[0]['parameters'].get('concurrency')!=8 or
                cells[0]['parameters'].get('arm') not in ('control_before','challenger','control_after') or
                type(until_ns) is not int or not time.monotonic_ns()<until_ns<=self.deadline_ns-120_000_000_000):
            raise runtime.RuntimeErrorClosed('sampler overhead control contract or deadline differs')
        mode='full' if cells[0]['parameters']['arm']=='challenger' else 'essential'
        control={'schema':'bounded-sampler-overhead-control/v1','cell_id':cell_id,
            'arm':cells[0]['parameters']['arm'],'mode':mode,'until_ns':until_ns,
            'selected_ns':time.monotonic_ns(),'safety_watchdog_enabled':True,
            'scope':'incremental full sampler cost over essential safety; unscored control only'}
        contract=config.get('overhead_contract')
        if contract not in (None,'total-worker-and-host/v1'):
            raise runtime.RuntimeErrorClosed('unknown sampler overhead measurement contract')
        if contract=='total-worker-and-host/v1':
            binding=json.loads((self.root/'profile-binding.json').read_text())
            if (mode=='full')!=((binding.get('delta') or {}).get('worker_observation') is not None):
                raise runtime.RuntimeErrorClosed('total overhead worker observation arm differs')
            control.update(schema='bounded-sampler-overhead-control/v2',measurement_contract=contract,
                scope='incremental worker observer and full host sampler cost over essential safety; unscored control only')
        runtime.atomic_json(self.root/'sampler-control.json',control)
        self.sampler_control=control
        return control

    def start_sampler(self, endpoints, required_names, expected_ranks=None, server_cpus=None, nvml_identity=None, isolated=False):
        """Collect raw samples; this does not prove per-rank metric semantics."""
        self.ensure_healthy()
        if self.sampler is not None: raise runtime.RuntimeErrorClosed('sampler already started')
        if type(isolated) is not bool:raise ValueError('isolated sampler flag must be boolean')
        if isolated:
            import isolated_sampler
            self.sampler_process=isolated_sampler.start(self,endpoints,required_names,expected_ranks,server_cpus,nvml_identity)
            self.sampler=self.sampler_process
            return
        def run():
            index, size, baseline = 0, 0, None
            timing_index, timing_size, previous_started = 0, 0, None
            previous_completed_ns = None
            previous_mode='full';essential_index,essential_size=0,0
            try:
                while not self.sampler_stop.is_set():
                    started = time.monotonic()
                    started_ns = time.monotonic_ns()
                    cpu_started=time.thread_time()
                    control=self.sampler_control
                    if control is not None and started_ns>=control['until_ns']:
                        raise runtime.RuntimeErrorClosed('sampler overhead control deadline expired')
                    mode=control['mode'] if control is not None else 'full'
                    if mode!=previous_mode:
                        previous_started=None;previous_completed_ns=None;previous_mode=mode
                    if mode=='essential':
                        sample=telemetry.essential_engine_sample(endpoints,required_names,expected_ranks)
                        if previous_completed_ns is not None and not 0<sample['monotonic_ns']-previous_completed_ns<15_000_000_000:
                            raise runtime.RuntimeErrorClosed('essential engine completion freshness failed')
                        previous_completed_ns=sample['monotonic_ns']
                        sample['control_cell_id']=control['cell_id']
                        essential_index,essential_size=runtime.append_sample(self.root,'essential-engine',essential_index,essential_size,sample)
                        runtime.atomic_json(self.root/'engine-heartbeat.json',{'monotonic_ns':sample['monotonic_ns'],
                            'mode':'essential','control_cell_id':control['cell_id']})
                        runtime.atomic_json(self.root/'sampler-control-applied.json',{'cell_id':control['cell_id'],
                            'mode':'essential','completed_ns':sample['monotonic_ns']})
                        self.sampler_stop.wait(max(0,5-(time.monotonic()-started)))
                        continue
                    interval = started-previous_started if previous_started is not None else 0.0
                    if interval > 1.1: raise runtime.RuntimeErrorClosed('telemetry sampler missed 1 Hz start cadence')
                    previous_started = started
                    sample = telemetry.sample(endpoints, required_names, self.safety_mount, ecc_baseline=baseline, expected_ranks=expected_ranks, server_cpus=server_cpus, nvml_identity=nvml_identity, run_dir=self.root)
                    if self.external_sampler and json.loads((self.root/'execution-state.json').read_text()).get('stopping'):
                        runtime.atomic_json(self.root/'isolated-sampler-shutdown-observation.json',sample)
                        break
                    if control is not None:
                        runtime.atomic_json(self.root/'sampler-control-applied.json',{'cell_id':control['cell_id'],
                            'mode':'full','completed_ns':sample['monotonic_ns']})
                    if baseline is None: baseline = sample['uncorrected_ecc']
                    # Retain the offending raw observation before failing closed.
                    index, size = runtime.append_sample(self.root, 'telemetry', index, size, sample)
                    runtime.atomic_json(self.root/'engine-heartbeat.json', {'monotonic_ns':time.monotonic_ns()})
                    elapsed = time.monotonic() - started
                    timing_index, timing_size = runtime.append_sample(self.root,'sampler-timing',timing_index,timing_size,
                        {'monotonic_ns':time.monotonic_ns(),'elapsed_seconds':elapsed,'thread_cpu_seconds':time.thread_time()-cpu_started,'start_interval_seconds':interval,
                         'collection_started_ns':started_ns,'sample_completed_ns':sample['monotonic_ns'],
                         'target_period_seconds':SAMPLER_PERIOD_SECONDS,'probe_seconds':sample['collection_seconds']})
                    check_completed_sample(previous_completed_ns,sample['monotonic_ns'])
                    previous_completed_ns = sample['monotonic_ns']
                    elapsed = time.monotonic() - started
                    if elapsed > 1.0: raise runtime.RuntimeErrorClosed('telemetry sampler exceeded one-second cycle')
                    observed=sample.get('worker_observations',{})
                    if observed.get('status')=='physical_worker_observation_sample':
                        runtime.atomic_json(self.root/'worker-observation-last-sampled.json',
                            {'observed_at_ns':observed['observed_at_ns'],'sample_completed_ns':sample['monotonic_ns']})
                        elapsed=time.monotonic()-started
                        if elapsed>1.0:raise runtime.RuntimeErrorClosed('worker observation acknowledgment exceeded one-second cycle')
                    old=self.sampler_stats
                    self.sampler_stats={'count':old['count']+1,'total_seconds':old['total_seconds']+elapsed,
                                        'thread_cpu_seconds':old['thread_cpu_seconds']+time.thread_time()-cpu_started,
                                        'max_seconds':max(old['max_seconds'],elapsed),
                                        'max_interval_seconds':max(old['max_interval_seconds'],interval)}
                    self.sampler_stop.wait(max(0, SAMPLER_PERIOD_SECONDS - (time.monotonic() - started)))
            except Exception as exc:
                self.failed.set()
                runtime.atomic_json(self.root/'sampler-failure.json', {'type':type(exc).__name__, 'message':str(exc)})
                if self.external_sampler:return
                try:
                    state = json.loads((self.root/'execution-state.json').read_text())
                    watchdog.cleanup(self.root, state)
                except Exception as cleanup_exc:
                    runtime.atomic_json(self.root/'sampler-cleanup-failure.json', {'type':type(cleanup_exc).__name__})
        self.sampler = threading.Thread(target=run, daemon=True)
        self.sampler.start()

    def mark_ready(self, evidence_files):
        self.ensure_healthy()
        hb = json.loads((self.root/'engine-heartbeat.json').read_text())
        if not runtime.heartbeat_fresh(hb, time.monotonic_ns()):
            raise runtime.RuntimeErrorClosed('engine telemetry is stale')
        with runtime.NodeLock(self.root/'cleanup.lock', wait_seconds=5):
            self.ensure_healthy()
            state = json.loads((self.root/'execution-state.json').read_text())
            if state.get('stopping') or (self.root/'cleanup-result.json').exists():
                raise runtime.RuntimeErrorClosed('cleanup has already started')
            if len(evidence_files) != len(state['container_ids']) + 1:
                raise runtime.RuntimeErrorClosed('missing backend or router readiness evidence')
            identities = set()
            for name in evidence_files:
                path = Path(name)
                if path.parent != self.root:
                    raise runtime.RuntimeErrorClosed('readiness evidence outside run directory')
                evidence = json.loads(path.read_text())
                if not evidence.get('passed') or not evidence.get('raw', {}).get('complete') or evidence['raw'].get('http_status') != 200:
                    raise runtime.RuntimeErrorClosed('semantic readiness failed')
                identities.add(evidence.get('identity'))
            if identities != set(state['container_ids']) | {'router'}:
                raise runtime.RuntimeErrorClosed('readiness identity coverage mismatch')
            state['ready'] = True
            state['readiness_evidence'] = [str(p) for p in evidence_files]
            runtime.atomic_json(self.root/'execution-state.json', state)

    def abort(self, reason):
        """Stop dispatch and owned work; a fresh qualified session is required."""
        if not reason: raise ValueError('abort reason required')
        self.failed.set()
        with runtime.NodeLock(self.root/'cleanup.lock', wait_seconds=120):
            state = json.loads((self.root/'execution-state.json').read_text())
            state['ready'] = False
            state['stopping'] = True
            runtime.atomic_json(self.root/'execution-state.json', state)
            runtime.atomic_json(self.root/'abort.json', {'reason':reason, 'monotonic_ns':time.monotonic_ns(),
                                                         'disposition':'recovery_required'})
        self.sampler_stop.set()
        result = watchdog.cleanup(self.root, state)
        if not result['success']: raise runtime.RuntimeErrorClosed('abort cleanup failed')
        watchdog.verify_cleanup_idle(self.root,state)
        return result

    def __exit__(self, kind, value, tb):
        cleanup_deadline=time.monotonic()+120
        def remaining(limit):
            seconds=min(limit,cleanup_deadline-time.monotonic())
            if seconds<=0: raise runtime.RuntimeErrorClosed('total cleanup deadline exhausted')
            return seconds
        cleanup_error = None
        primary_record_error = None
        if value is not None:
            try:
                runtime.atomic_json(self.root/'primary-failure.json',{
                    'type':type(value).__name__,'message_sha256':hashlib.sha256(str(value).encode()).hexdigest(),
                    'frames':[{'file':Path(frame.filename).name,'line':frame.lineno,'function':frame.name}
                              for frame in traceback.extract_tb(tb)],
                    'scope':'original exception fingerprint; consult screened driver traceback for text'})
            except Exception as exc:
                primary_record_error = exc
        try:
            with runtime.NodeLock(self.root/'cleanup.lock', wait_seconds=remaining(120)):
                state = json.loads((self.root/'execution-state.json').read_text())
                state['ready'] = False
                state['stopping'] = True
                runtime.atomic_json(self.root/'execution-state.json', state)
            self.sampler_stop.set()
            if self.sampler is not None: self.sampler.join(timeout=remaining(10))
            if self.sampler is not None and self.sampler.is_alive():
                raise runtime.RuntimeErrorClosed('telemetry sampler did not stop')
            state = json.loads((self.root/'execution-state.json').read_text())
            if state['container_ids'] or state['owned_processes']:
                result = watchdog.cleanup(self.root, state, deadline=cleanup_deadline)
                if not result['success']: raise runtime.RuntimeErrorClosed('cleanup failed')
                def bounded(argv,timeout=30): return runtime.command(argv,timeout=remaining(timeout))
                watchdog.verify_cleanup_idle(self.root,state,bounded)
            state = json.loads((self.root/'execution-state.json').read_text())
            state['watchdog_stop'] = True
            runtime.atomic_json(self.root/'execution-state.json', state)
            self.stop_event.set()
            if self.monitor is not None:
                try:
                    if self.monitor.wait(timeout=remaining(10)) != 0:
                        raise runtime.RuntimeErrorClosed('watchdog exited unsuccessfully')
                except subprocess.TimeoutExpired: raise runtime.RuntimeErrorClosed('watchdog did not stop')
            if self.failed.is_set() or (self.root/'watchdog-failure.json').exists() or (self.root/'sampler-failure.json').exists():
                raise runtime.RuntimeErrorClosed('supervisor failed during session')
        except Exception as exc:
            cleanup_error = exc
        finally:
            self.stop_event.set()
            if self.pulse is not None: self.pulse.join(timeout=max(0,min(10,cleanup_deadline-time.monotonic())))
            if self.log is not None: self.log.close()
            self.lock.__exit__(kind, value, tb)
        if cleanup_error is not None:
            raise cleanup_error
        if primary_record_error is not None:
            raise runtime.RuntimeErrorClosed('primary failure evidence could not be persisted') from primary_record_error
