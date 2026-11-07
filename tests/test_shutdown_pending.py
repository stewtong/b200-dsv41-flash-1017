import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import nvml_drain,runtime,telemetry,watchdog
from test_nvml_drain import sample


class ShutdownPendingTests(unittest.TestCase):
    def state(self):
        now=time.monotonic_ns();owner={'pid':42,'start_ticks':1,'boot_id':'fixture'};collector=dict(owner,pid=44)
        return {'run_id':'mine','stopping':True,'ready':False,'safety_backend':'nvml','safety_mount':'/fixture',
            'container_ids':['a'*64],'owned_processes':[collector],'driver_identity':owner,
            'watchdog_identity':dict(owner,pid=43),'nvml_identity':collector,'deadline_ns':now+120_000_000_000,
            'cleanup_owner_identity':owner,'cleanup_started_ns':now-5_000_000_000,'cleanup_deadline_ns':now+115_000_000_000,
            'shutdown_safety_schema':'owned-cleanup-gpu-proof/v1'}

    def test_pending_requires_exact_live_owner_held_lock_matching_request_and_original_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=self.state();now=time.monotonic_ns()
            runtime.atomic_json(root/'nvml-drain-request.json',{'run_id':'mine','identity':state['nvml_identity'],'requested_ns':now})
            with patch.object(runtime,'process_alive',return_value=True),runtime.NodeLock(root/'cleanup.lock'):
                self.assertTrue(watchdog.owned_cleanup_pending(root,state,time.monotonic_ns()))
                for key,value in [('stopping',False),('ready',True),('cleanup_deadline_ns',now-1),
                    ('cleanup_deadline_ns',now+200_000_000_000),('cleanup_owner_identity',{'pid':999})]:
                    with self.subTest(key=key):
                        wrong=dict(state);wrong[key]=value;self.assertFalse(watchdog.owned_cleanup_pending(root,wrong,time.monotonic_ns()))
                with patch.object(runtime,'process_alive',return_value=False):self.assertFalse(watchdog.owned_cleanup_pending(root,state,time.monotonic_ns()))
            with patch.object(runtime,'process_alive',return_value=True):self.assertFalse(watchdog.owned_cleanup_pending(root,state,time.monotonic_ns()))

    def test_actual_monitor_reports_unknown_gpu_during_owned_cleanup_without_fabricating_a_sample(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=self.state();runtime.atomic_json(root/'execution-state.json',state)
            runtime.atomic_json(root/'driver-heartbeat.json',{'monotonic_ns':time.monotonic_ns()})
            stale=sample(time.monotonic_ns()-11_000_000_000);stale['pid']=44;runtime.atomic_json(root/'nvml-live.json',stale)
            runtime.atomic_json(root/'nvml-drain-request.json',{'run_id':'mine','identity':state['nvml_identity'],'requested_ns':time.monotonic_ns()})
            result=[]
            def host(_):return {'monotonic_ns':time.monotonic_ns(),'available_memory_fraction':.5,'free_disk_fraction':.5}
            original_pending=telemetry.shutdown_pending_sample
            with patch.object(runtime,'process_alive',return_value=True),patch.object(runtime,'safety_sample',side_effect=host), \
                 patch.object(telemetry,'shutdown_pending_sample',side_effect=lambda mount,deadline,last:original_pending(mount,deadline,last,invoke=lambda *_a,**_k:'')),runtime.NodeLock(root/'cleanup.lock'):
                worker=threading.Thread(target=lambda:result.append(watchdog.monitor(root,interval=.01)));worker.start()
                try:
                    limit=time.monotonic()+2
                    while not (root/'watchdog-heartbeat.json').exists() and time.monotonic()<limit:time.sleep(.01)
                    self.assertFalse((root/'watchdog-failure.json').exists())
                    heartbeat=json.loads((root/'watchdog-heartbeat.json').read_text())
                    self.assertEqual(heartbeat['gpu_safety_status'],'unknown_pending_owned_cleanup')
                    saved=[json.loads(line) for p in root.glob('safety-*.jsonl') for line in p.read_text().splitlines()]
                    self.assertIsNone(saved[0]['uncorrected_ecc']);self.assertIsNone(saved[0]['nvml_safety'])
                finally:
                    state['watchdog_stop']=True;runtime.atomic_json(root/'execution-state.json',state);worker.join(2)
                self.assertFalse(worker.is_alive());self.assertEqual(result,[0])

    def test_final_idle_proof_must_preserve_ecc_baseline(self):
        data=sample();state={'shutdown_safety_schema':'owned-cleanup-gpu-proof/v1','nvml_ecc_baseline':dict.fromkeys(map(str,range(8)),0)}
        nvml_drain.validate_ecc(data,state)
        data['devices'][3]['volatile_uncorrected_ecc']=1
        with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'ECC counters changed'):nvml_drain.validate_ecc(data,state)

    def test_monitor_still_fails_closed_outside_exact_cleanup_and_for_host_faults(self):
        for fault in ('active','expired','dead_owner','missing_request','host_failure','kernel_xid'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                root=Path(d);state=self.state();now=time.monotonic_ns()
                if fault=='active':state['stopping']=False
                if fault=='expired':state['cleanup_deadline_ns']=now-1
                runtime.atomic_json(root/'execution-state.json',state)
                runtime.atomic_json(root/'driver-heartbeat.json',{'monotonic_ns':now})
                stale=sample(now-11_000_000_000);stale['pid']=44;runtime.atomic_json(root/'nvml-live.json',stale)
                if fault!='missing_request':runtime.atomic_json(root/'nvml-drain-request.json',{'run_id':'mine','identity':state['nvml_identity'],'requested_ns':now})
                def alive(identity):return not(fault=='dead_owner' and identity==state['driver_identity'])
                def host(_):
                    if fault=='host_failure':raise runtime.RuntimeErrorClosed('memory or disk safety threshold crossed')
                    return {'monotonic_ns':time.monotonic_ns(),'available_memory_fraction':.5,'free_disk_fraction':.5}
                original=telemetry.shutdown_pending_sample
                with patch.object(runtime,'process_alive',side_effect=alive),patch.object(runtime,'safety_sample',side_effect=host), \
                     patch.object(telemetry,'shutdown_pending_sample',side_effect=lambda m,d,l:original(m,d,l,invoke=lambda *_a,**_k:'Xid 79' if fault=='kernel_xid' else '')), \
                     patch.object(watchdog,'cleanup',return_value={'success':False}),runtime.NodeLock(root/'cleanup.lock'):
                    self.assertEqual(watchdog.monitor(root,interval=.001),1)
                self.assertTrue(json.loads((root/'watchdog-failure.json').read_text())['reasons'])
                self.assertEqual(json.loads((root/'watchdog-heartbeat.json').read_text())['gpu_safety_status'],'failed')


if __name__=='__main__':unittest.main()
