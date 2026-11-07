import subprocess
import json
import ctypes
import tempfile
import signal
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_gpu_probe
import nvml_safety_probe


class GpuProbeTests(unittest.TestCase):
    def test_records_success_and_timeout_distinctly(self):
        field=node_gpu_probe.FIELDS[0]
        with patch.object(node_gpu_probe.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'0, 0\n1, 0\n','')):
            result=node_gpu_probe.probe(field)
        self.assertEqual(result['rows'],2)
        self.assertEqual(result['returncode'],0)
        self.assertFalse(result['valid'])
        self.assertIn('stdout_sha256',result)
        with patch.object(node_gpu_probe.subprocess,'run',return_value=subprocess.CompletedProcess([],0,''.join(f'{i}, 0\n' for i in range(8)),'')):
            self.assertTrue(node_gpu_probe.probe(field)['valid'])
        with patch.object(node_gpu_probe.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture',3)):
            result=node_gpu_probe.probe(field)
        self.assertTrue(result['timeout'])
        self.assertNotIn('rows',result)
        with self.assertRaises(ValueError):node_gpu_probe.probe('unapproved')

    def test_nvml_probe_requires_eight_distinct_devices(self):
        devices=[{'index':i,'uuid':'GPU-'+str(i),'total_bytes':100,'free_bytes':80,'used_bytes':20,'reserved_bytes':0,'volatile_uncorrected_ecc':0} for i in range(8)]
        output=json.dumps({'devices':devices,'call_timings':[]})
        with patch.object(node_gpu_probe.subprocess,'run',return_value=subprocess.CompletedProcess([],0,output,'')):
            self.assertTrue(node_gpu_probe.probe('direct_nvml_safety')['valid'])
        devices[7]['uuid']=devices[0]['uuid']
        with patch.object(node_gpu_probe.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps({'devices':devices}),'')):
            self.assertFalse(node_gpu_probe.probe('direct_nvml_safety')['valid'])

    def test_nvml_v2_struct_version(self):
        self.assertEqual(ctypes.sizeof(nvml_safety_probe.MemoryV2),40)
        self.assertEqual(ctypes.sizeof(nvml_safety_probe.MemoryV2) | (2 << 24),0x02000028)
        self.assertLess(nvml_safety_probe.PERIOD_SECONDS,1)

    def test_memory_control_requires_headroom(self):
        good='\n'.join(f'{i}, 170000, 180000' for i in range(8))
        self.assertEqual(len(node_gpu_probe.memory_headroom(good,64)),8)
        bad='\n'.join(f'{i}, 70000, 180000' for i in range(8))
        with self.assertRaises(node_gpu_probe.runtime.RuntimeErrorClosed):node_gpu_probe.memory_headroom(bad,64)
        with self.assertRaises(ValueError):node_gpu_probe.memory_headroom(good,256)
        high='\n'.join(f'{i}, 182000, 183359' for i in range(8))
        self.assertEqual(len(node_gpu_probe.memory_headroom(high,128)),8)
        tight='\n'.join(f'{i}, 166000, 180000' for i in range(8))
        with self.assertRaises(node_gpu_probe.runtime.RuntimeErrorClosed):node_gpu_probe.memory_headroom(tight,128)

    def test_live_collector_requires_fresh_complete_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'nvml-live.json';identity={'pid':123}
            rows=[{'index':i,'uuid':'GPU-'+str(i),'total_bytes':100,'free_bytes':80,'used_bytes':10,'reserved_bytes':10,'volatile_uncorrected_ecc':0} for i in range(8)]
            p.write_text(json.dumps({'pid':123,'sample_started_ns':100,'monotonic_ns':100,'sample_seconds':.2,'sample_gap_seconds':0,'devices':rows}))
            with patch.object(node_gpu_probe.runtime,'process_alive',return_value=True):
                self.assertEqual(len(node_gpu_probe.read_live(d,identity,110,max_age_ns=20)['devices']),8)
                with self.assertRaisesRegex(node_gpu_probe.runtime.RuntimeErrorClosed,'stale'):
                    node_gpu_probe.read_live(d,identity,121,max_age_ns=20)
                rows.pop();p.write_text(json.dumps({'pid':123,'sample_started_ns':100,'monotonic_ns':100,'sample_seconds':.2,'sample_gap_seconds':0,'devices':rows}))
                with self.assertRaises(node_gpu_probe.runtime.RuntimeErrorClosed):node_gpu_probe.read_live(d,identity,110,max_age_ns=20)

    def test_fault_injection_targets_recorded_identity(self):
        with tempfile.TemporaryDirectory() as d:
            identity={'pid':123,'start_ticks':1,'boot_id':'fixture'}
            with patch.object(node_gpu_probe.runtime,'signal_owned_process') as send:
                node_gpu_probe.freeze_collector(d,identity)
            send.assert_called_once_with(identity,signal.SIGSTOP)
            intent=json.loads((Path(d)/'fault-injection-intent.json').read_text())
            self.assertEqual(intent['identity'],identity)

    def test_shutdown_pause_has_exact_bounded_non_daemon_release(self):
        with tempfile.TemporaryDirectory() as d:
            identity={'pid':123,'start_ticks':1,'boot_id':'fixture'}
            with patch.object(node_gpu_probe.runtime,'signal_owned_process') as send:
                worker,result=node_gpu_probe.pause_shutdown_collector(d,identity,.02)
                self.assertFalse(worker.daemon);worker.join(1)
            self.assertFalse(worker.is_alive());self.assertEqual(result['status'],'resumed')
            self.assertEqual(send.call_args_list,[unittest.mock.call(identity,signal.SIGSTOP),unittest.mock.call(identity,signal.SIGCONT)])
            release=json.loads((Path(d)/'shutdown-pause-release.json').read_text())
            self.assertEqual(release['identity'],identity)
            for seconds in (0,26,True):
                with self.assertRaises(ValueError):node_gpu_probe.pause_shutdown_collector(d,identity,seconds)
