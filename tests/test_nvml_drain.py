import ctypes
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import nvml_drain as drain
import nvml_safety_probe as nvml
import runtime
import watchdog
import verify_pilot


def sample(stamp=None, count=0, memory=0):
    stamp=time.monotonic_ns() if stamp is None else stamp
    return {'pid':42,'sample_started_ns':stamp,'monotonic_ns':stamp,
            'sample_seconds':.001,'sample_gap_seconds':.5,'measurement_mode':False,
            'devices':[{'index':i,'uuid':'GPU-'+str(i),'total_bytes':200*1024**3,
                        'used_bytes':memory,'reserved_bytes':1024**3,'free_bytes':199*1024**3-memory,
                        'volatile_uncorrected_ecc':0,'compute_process_count':count} for i in range(8)]}


class NvmlDrainTests(unittest.TestCase):
    def test_only_fresh_complete_post_stop_zero_process_low_memory_sample_passes(self):
        now=time.monotonic_ns();data=sample(now)
        self.assertEqual(drain.validate_idle(data,now,now),data)
        cases=[(sample(now,count=1),'compute processes'),
               (sample(now,memory=3*1024**3),'memory not near idle'),
               (sample(now-11_000_000_000),'stale'),
               (sample(now-1),'predates')]
        missing=sample(now);del missing['devices'][0]['compute_process_count']
        cases.append((missing,'coverage incomplete'))
        for value,error in cases:
            with self.subTest(error=error),self.assertRaisesRegex(runtime.RuntimeErrorClosed,error):
                drain.validate_idle(value,now,now)

    def test_null_buffer_process_count_contract_rejects_unavailable_and_inconsistent_results(self):
        collector=object.__new__(nvml.Collector)
        for status,count,valid in ((0,0,True),(7,3,True),(0,3,False),(7,0,False),(6,0,False)):
            class Lib:
                def nvmlDeviceGetComputeRunningProcesses_v3(self,handle,pointer,buffer):
                    self.assert_null=buffer is None;pointer._obj.value=count;return status
            collector.lib=Lib();timings=[]
            if valid:self.assertEqual(collector.process_count(ctypes.c_void_p(),timings),count)
            else:
                with self.assertRaises(RuntimeError):collector.process_count(ctypes.c_void_p(),timings)
            self.assertTrue(collector.lib.assert_null);self.assertEqual(timings[0]['status'],status)

    def test_compute_queries_require_explicit_shutdown_and_do_not_enter_measured_window(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            runtime.atomic_json(root/'execution-state.json',{'stopping':False})
            runtime.atomic_json(root/'nvml-drain-request.json',{})
            self.assertFalse(nvml.drain_active(root))
            runtime.atomic_json(root/'execution-state.json',{'stopping':True})
            self.assertTrue(nvml.drain_active(root));self.assertFalse(nvml.measurement_active(root))

    def test_actual_cleanup_waits_for_post_stop_sample_and_never_calls_gpu_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);collector={'pid':42,'start_ticks':1,'boot_id':'fixture'}
            state={'run_id':'mine','container_ids':['d'*64],'owned_processes':[collector],
                   'nvml_identity':collector,'safety_backend':'nvml'}
            stopped=threading.Event();published=threading.Event();released=[]
            runtime.atomic_json(root/'nvml-live.json',sample(count=2,memory=32*1024**3))
            def publish():
                stopped.wait(2);time.sleep(.02)
                runtime.atomic_json(root/'nvml-live.json',sample());published.set()
            writer=threading.Thread(target=publish);writer.start()
            def release(identity,**kwargs):
                self.assertTrue(published.is_set());released.append(identity)
            with patch.object(runtime,'process_alive',return_value=True), \
                 patch.object(runtime,'command',side_effect=AssertionError('unexpected GPU CLI')):
                result=watchdog.cleanup(root,state,stop=lambda *_a,**_k:stopped.set(),
                       capture=lambda *_a,**_k:None,stop_process=release)
                self.assertTrue(result['success']);self.assertEqual(released,[collector])
                state=json.loads((root/'execution-state.json').read_text())
                watchdog.verify_cleanup_idle(root,state)
                self.assertTrue(json.loads((root/'gpu-idle-after-cleanup.json').read_text())['idle'])
                state['watchdog_stop']=True;runtime.atomic_json(root/'execution-state.json',state)
                verify_pilot.verify_cleanup(root,{'run_id':'mine'})
                proof=json.loads((root/'gpu-drain-proof.json').read_text())
                self.assertGreaterEqual(proof['sample']['sample_started_ns'],proof['after_stopped_ns'])
                proof['sample']['devices'][0]['compute_process_count']=1
                runtime.atomic_json(root/'gpu-drain-proof.json',proof)
                with self.assertRaisesRegex(ValueError,'raw NVML drain proof'):verify_pilot.verify_cleanup(root,{'run_id':'mine'})
                proof['sample']['devices'][0]['compute_process_count']=0
                proof['identity']['pid']=43;runtime.atomic_json(root/'gpu-drain-proof.json',proof)
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'ownership mismatch'):drain.read_proof(root,state)
            writer.join(2);self.assertFalse(writer.is_alive())

    def test_unknown_state_exhausts_bounded_wait_without_releasing_collector(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);collector={'pid':42};state={'run_id':'mine','nvml_identity':collector,'stopping':True}
            runtime.atomic_json(root/'nvml-live.json',sample(count=1))
            with patch.object(runtime,'process_alive',return_value=True):
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'deadline exhausted'):
                    drain.wait_idle(root,state,time.monotonic_ns(),time.monotonic()+.03)
            self.assertFalse((root/'gpu-drain-proof.json').exists())
            self.assertTrue((root/'nvml-drain-observations.json').exists())

    def test_late_context_drain_uses_remaining_shared_budget_without_extending_deadline(self):
        # The retained live incident's first complete idle sample arrived about
        # 58 seconds after shutdown began, after 20-second container stops.
        for window, expected_success in ((120, True), (45, False)):
            with self.subTest(window=window), tempfile.TemporaryDirectory() as d:
                root=Path(d);clock=[1000.0];released=[]
                collector={'pid':42,'start_ticks':1,'boot_id':'fixture'}
                state={'run_id':'mine','container_ids':['d'*64],'owned_processes':[collector],
                       'nvml_identity':collector,'safety_backend':'nvml'}
                def current(count):return sample(int(clock[0]*1e9),count=count,memory=0 if count==0 else 32*1024**3)
                def stop(*_a,**_k):
                    clock[0]+=20;runtime.atomic_json(root/'nvml-live.json',current(1))
                def sleep(seconds):
                    clock[0]+=seconds
                    runtime.atomic_json(root/'nvml-live.json',current(0 if clock[0]>=1058 else 1))
                def release(identity,**_k):released.append(identity);clock[0]+=1
                with patch.object(time,'monotonic',side_effect=lambda:clock[0]), \
                     patch.object(time,'monotonic_ns',side_effect=lambda:int(clock[0]*1e9)), \
                     patch.object(time,'sleep',side_effect=sleep), \
                     patch.object(runtime,'process_alive',return_value=True), \
                     patch.object(runtime,'command',side_effect=AssertionError('unexpected GPU CLI')):
                    result=watchdog.cleanup(root,state,stop=stop,capture=lambda *_a,**_k:None,
                                            stop_process=release,deadline=1000+window)
                    self.assertEqual(result['success'],expected_success)
                    self.assertEqual(bool(released),expected_success)
                    self.assertLessEqual(result['elapsed_seconds'],window)
                    budget=json.loads((root/'gpu-drain-budget.json').read_text())
                    self.assertAlmostEqual(budget['drain_deadline_ns']/1e9,1000+window-20)
                    if expected_success:
                        state=json.loads((root/'execution-state.json').read_text());state['watchdog_stop']=True
                        runtime.atomic_json(root/'execution-state.json',state)
                        watchdog.verify_cleanup_idle(root,state)
                        verify_pilot.verify_cleanup(root,{'run_id':'mine'})
                        budget['drain_deadline_ns']+=1_000_000_000
                        runtime.atomic_json(root/'gpu-drain-budget.json',budget)
                        with self.assertRaisesRegex(ValueError,'shared cleanup contract'):
                            verify_pilot.verify_cleanup(root,{'run_id':'mine'})
