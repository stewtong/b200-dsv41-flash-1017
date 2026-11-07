from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import supervisor

class FakeWatchdog:
    pid = 42
    def poll(self): return None
    def wait(self, timeout): return 0

class SupervisorTests(unittest.TestCase):
    def sampler_fixture(self,root,costs):
        import threading
        import json
        clock=[10.0];rows={};s=supervisor.Supervisor(root/'lock',root,'fixture',10**15)
        class Stop:
            def is_set(self):return len(rows.get('telemetry',[]))>=len(costs)
            def wait(self,seconds):clock[0]+=seconds
        class Thread:
            def __init__(self,target,daemon):self.target=target
            def start(self):self.target()
        def sample(*a,**kw):
            cost=costs[len(rows.get('telemetry',[]))];clock[0]+=cost
            return {'monotonic_ns':round(clock[0]*1e9),'uncorrected_ecc':{},'collection_seconds':{'total':cost}}
        def append(root,prefix,index,size,row):rows.setdefault(prefix,[]).append(row);return index,size
        s.sampler_stop=Stop()
        with patch.object(s,'ensure_healthy'),patch.object(supervisor.threading,'Thread',Thread), \
             patch.object(supervisor.time,'monotonic',side_effect=lambda:clock[0]), \
             patch.object(supervisor.time,'monotonic_ns',side_effect=lambda:round(clock[0]*1e9)), \
             patch.object(supervisor.telemetry,'sample',side_effect=sample), \
             patch.object(supervisor.runtime,'append_sample',side_effect=append), \
             patch.object(supervisor.watchdog,'cleanup') as cleanup:
            s.start_sampler(['fixture'],['fixture'])
        return s,rows,cleanup.call_count

    def test_variable_collection_cost_keeps_completed_sample_cadence(self):
        with tempfile.TemporaryDirectory() as d:
            s,rows,cleanup=self.sampler_fixture(Path(d),[.27,.55,.28,.48,.29])
            self.assertFalse(s.failed.is_set());self.assertEqual(cleanup,0)
            stamps=[r['monotonic_ns'] for r in rows['telemetry']]
            self.assertLessEqual(max(b-a for a,b in zip(stamps,stamps[1:])),1_100_000_000)
            self.assertTrue(all(r['target_period_seconds']==.5 for r in rows['sampler-timing']))

    def test_completed_gap_stops_sampler_even_with_on_time_starts(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'execution-state.json').write_text('{}')
            s,rows,cleanup=self.sampler_fixture(root,[.05,.8])
            self.assertTrue(s.failed.is_set());self.assertEqual(cleanup,1)
            self.assertEqual(len(rows['telemetry']),2)
            self.assertEqual(rows['sampler-timing'][1]['start_interval_seconds'],.5)
            self.assertIn('completed-sample gap',json.loads((root/'sampler-failure.json').read_text())['message'])

    def test_primary_failure_is_preserved_without_exception_text(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            with patch.object(supervisor.runtime,'process_identity',return_value={'pid':1,'start_ticks':1,'boot_id':'fixture'}), \
                 patch.object(supervisor.subprocess,'Popen',return_value=FakeWatchdog()):
                with self.assertRaises(ValueError):
                    with supervisor.Supervisor(Path(d)/'lock',root,'run1',time.monotonic_ns()+30_000_000_000):
                        raise ValueError('sensitive fixture text')
            raw=(root/'primary-failure.json').read_text();self.assertNotIn('sensitive fixture text',raw)
            self.assertEqual(json.loads(raw)['type'],'ValueError')

    def test_primary_evidence_write_failure_does_not_skip_cleanup(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';original=supervisor.runtime.atomic_json
            def atomic(path,value):
                if Path(path).name=='primary-failure.json': raise OSError('fixture')
                return original(path,value)
            with patch.object(supervisor.runtime,'process_identity',return_value={'pid':1,'start_ticks':1,'boot_id':'fixture'}), \
                 patch.object(supervisor.runtime,'atomic_json',side_effect=atomic), \
                 patch.object(supervisor.subprocess,'Popen',return_value=FakeWatchdog()):
                with self.assertRaisesRegex(supervisor.runtime.RuntimeErrorClosed,'could not be persisted'):
                    with supervisor.Supervisor(Path(d)/'lock',root,'run1',time.monotonic_ns()+30_000_000_000):
                        raise ValueError('fixture')
            self.assertTrue(json.loads((root/'execution-state.json').read_text())['watchdog_stop'])

    def test_nvml_collector_is_recorded_before_its_gate_opens(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir()
            supervisor.runtime.atomic_json(root/'execution-state.json',{'owned_processes':[],'stopping':False})
            s=supervisor.Supervisor(Path(d)/'lock',root,'run1',time.monotonic_ns()+30_000_000_000);s.preflight_done=True
            identity={'pid':42,'start_ticks':1,'boot_id':'fixture'}
            with patch.object(supervisor.runtime,'process_identity',return_value=identity), patch.object(supervisor.subprocess,'Popen',return_value=FakeWatchdog()):
                self.assertEqual(s.start_nvml_collector(),identity)
            self.assertEqual(json.loads((root/'execution-state.json').read_text())['owned_processes'],[identity])
            self.assertEqual(json.loads((root/'nvml-gate/router-start.ok').read_text())['pid'],42)
            self.assertFalse((root/'router-start.ok').exists())

    def test_nvml_safety_requires_recorded_collector(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();identity={'pid':42,'start_ticks':1,'boot_id':'fixture'}
            supervisor.runtime.atomic_json(root/'execution-state.json',{'owned_processes':[],'stopping':False})
            supervisor.runtime.atomic_json(root/'nvml-live.json',{'pid':42,'devices':[
                {'index':i,'uuid':'GPU-'+str(i),'volatile_uncorrected_ecc':0} for i in range(8)]})
            s=supervisor.Supervisor(Path(d)/'lock',root,'run1',time.monotonic_ns()+30_000_000_000)
            with patch.object(supervisor.nvml_safety_probe,'validate_live'):
                with self.assertRaisesRegex(supervisor.runtime.RuntimeErrorClosed,'ownership'):
                    s.activate_nvml_safety(identity)
                supervisor.runtime.atomic_json(root/'execution-state.json',{'owned_processes':[identity],'stopping':False})
                s.activate_nvml_safety(identity)
            state=json.loads((root/'execution-state.json').read_text())
            self.assertEqual(state['safety_backend'],'nvml')
            self.assertEqual(state['nvml_identity'],identity)
            self.assertEqual(state['nvml_ecc_baseline'],dict.fromkeys(map(str,range(8)),0))
            self.assertEqual(json.loads((root/'nvml-initial-safety.json').read_text())['pid'],42)

    def test_context_persists_state_and_requests_watchdog_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'run'
            with patch.object(supervisor.runtime, 'process_identity', return_value={'pid':1,'start_ticks':1,'boot_id':'fixture'}), \
                 patch.object(supervisor.subprocess, 'Popen', return_value=FakeWatchdog()):
                with supervisor.Supervisor(Path(d)/'lock', root, 'run1', time.monotonic_ns()+30_000_000_000) as s:
                    s.ensure_healthy()
                    with self.assertRaises(supervisor.runtime.RuntimeErrorClosed): s.create_start([])
                import json
                self.assertTrue(json.loads((root/'execution-state.json').read_text())['watchdog_stop'])

    def test_dead_watchdog_stops_dispatch_and_invokes_cleanup(self):
        class Dead(FakeWatchdog):
            def poll(self): return 1
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'run'
            with patch.object(supervisor.runtime, 'process_identity', return_value={'pid':1,'start_ticks':1,'boot_id':'fixture'}), \
                 patch.object(supervisor.subprocess, 'Popen', return_value=Dead()), \
                 patch.object(supervisor.watchdog, 'cleanup', return_value={'success':True}) as cleanup:
                with self.assertRaises(supervisor.runtime.RuntimeErrorClosed):
                    with supervisor.Supervisor(Path(d)/'lock', root, 'run1', time.monotonic_ns()+30_000_000_000) as s:
                        self.assertTrue(s.failed.wait(2))
                        with self.assertRaises(supervisor.runtime.RuntimeErrorClosed): s.ensure_healthy()
                cleanup.assert_called()

    def test_log_fault_stops_dispatch_and_invokes_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            with patch.object(supervisor.runtime, 'process_identity', return_value={'pid':1,'start_ticks':1,'boot_id':'fixture'}), \
                 patch.object(supervisor.subprocess, 'Popen', return_value=FakeWatchdog()), \
                 patch.object(supervisor.faults, 'check', side_effect=supervisor.runtime.RuntimeErrorClosed('GPU fault')), \
                 patch.object(supervisor.runtime, 'verify_gpus_idle', return_value=True), \
                 patch.object(supervisor.watchdog, 'cleanup', return_value={'success':True}) as cleanup:
                with self.assertRaises(supervisor.runtime.RuntimeErrorClosed):
                    with supervisor.Supervisor(Path(d)/'lock',root,'run1',time.monotonic_ns()+30_000_000_000) as session:
                        import json
                        state=json.loads((root/'execution-state.json').read_text())
                        state['container_ids']=['a'*64]
                        supervisor.runtime.atomic_json(root/'execution-state.json',state)
                        self.assertTrue(session.failed.wait(7))
                        session.ensure_healthy()
                cleanup.assert_called()

    def test_ready_rejects_missing_semantic_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'run'
            with patch.object(supervisor.runtime, 'process_identity', return_value={'pid':1,'start_ticks':1,'boot_id':'fixture'}), \
                 patch.object(supervisor.subprocess, 'Popen', return_value=FakeWatchdog()):
                with supervisor.Supervisor(Path(d)/'lock', root, 'run1', time.monotonic_ns()+30_000_000_000) as s:
                    supervisor.runtime.atomic_json(root/'engine-heartbeat.json', {'monotonic_ns':time.monotonic_ns()})
                    with self.assertRaises(supervisor.runtime.RuntimeErrorClosed): s.mark_ready([])

    def test_abort_blocks_new_dispatch_and_records_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            with patch.object(supervisor.runtime, 'process_identity', return_value={'pid':1,'start_ticks':1,'boot_id':'fixture'}), \
                 patch.object(supervisor.subprocess, 'Popen', return_value=FakeWatchdog()), \
                 patch.object(supervisor.watchdog, 'cleanup', return_value={'success':True}), \
                 patch.object(supervisor.runtime, 'verify_gpus_idle', return_value=True):
                with self.assertRaises(supervisor.runtime.RuntimeErrorClosed):
                    with supervisor.Supervisor(Path(d)/'lock', root, 'run1', time.monotonic_ns()+30_000_000_000) as s:
                        s.abort('fixture timeout')
                        with self.assertRaises(supervisor.runtime.RuntimeErrorClosed): s.ensure_healthy()
                import json
                self.assertEqual(json.loads((root/'abort.json').read_text())['disposition'], 'recovery_required')
