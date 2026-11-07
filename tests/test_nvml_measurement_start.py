"""A fresh safety-only cycle cannot acknowledge measured-mode startup."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_smoke as smoke


class NvmlMeasurementStartTests(unittest.TestCase):
    def sample(self,measured=True,start=120,gap=.5):
        return {'pid':42,'sample_started_ns':start,'monotonic_ns':start+1,
                'sample_seconds':.01,'sample_gap_seconds':gap,'measurement_mode':measured,
                'devices':[{'index':i,'uuid':'GPU-'+str(i),'total_bytes':100,
                            'free_bytes':80,'used_bytes':10,'reserved_bytes':10,
                            'volatile_uncorrected_ecc':0} for i in range(8)]}

    def test_fresh_safety_cycle_is_not_a_mode_acknowledgment(self):
        # Collector can read the mode before publication, then timestamp its
        # safety-only cycle after the owner's request timestamp.
        first=self.sample(False);second=self.sample(True,start=130)
        self.assertGreaterEqual(first['sample_started_ns'],100)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);session=SimpleNamespace(ensure_healthy=lambda:None)
            with patch.object(smoke,'read',side_effect=[first,second]), \
                 patch.object(smoke.time,'monotonic_ns',return_value=140), \
                 patch.object(smoke.time,'sleep') as sleep, \
                 patch.object(smoke.nvml_safety_probe.runtime,'process_alive',return_value=True):
                self.assertEqual(smoke.await_nvml_measurement(session,root,{'pid':42},100),second)
                sleep.assert_called_once_with(.1)
            receipt=json.loads((root/'nvml-measurement-ready.json').read_text())
            self.assertEqual(receipt,{'identity':{'pid':42},'requested_after_ns':100,'sample':second})

    def test_old_measured_cycle_does_not_acknowledge_new_request(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(smoke,'read',side_effect=[self.sample(start=80),self.sample(start=120)]), \
                 patch.object(smoke.time,'monotonic_ns',return_value=140), \
                 patch.object(smoke.time,'sleep') as sleep, \
                 patch.object(smoke.nvml_safety_probe.runtime,'process_alive',return_value=True):
                smoke.await_nvml_measurement(SimpleNamespace(ensure_healthy=lambda:None),d,{'pid':42},100)
                sleep.assert_called_once_with(.1)

    def test_bad_measured_sample_or_collector_does_not_create_receipt(self):
        for sample,live in [(self.sample(gap=1.2),True),(self.sample(),False)]:
            with self.subTest(live=live,gap=sample['sample_gap_seconds']),tempfile.TemporaryDirectory() as d:
                with patch.object(smoke,'read',return_value=sample), \
                     patch.object(smoke.time,'monotonic_ns',return_value=140), \
                     patch.object(smoke.nvml_safety_probe.runtime,'process_alive',return_value=live):
                    with self.assertRaises(smoke.runtime.RuntimeErrorClosed):
                        smoke.await_nvml_measurement(SimpleNamespace(ensure_healthy=lambda:None),d,{'pid':42},100)
                self.assertFalse((Path(d)/'nvml-measurement-ready.json').exists())

    def test_missing_mode_ack_times_out_with_existing_source_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sample=self.sample(False);original=copy.deepcopy(sample)
            with patch.object(smoke,'read',return_value=sample), \
                 patch.object(smoke.time,'monotonic',side_effect=[0,11]):
                with self.assertRaisesRegex(TimeoutError,'measured-mode'):
                    smoke.await_nvml_measurement(SimpleNamespace(ensure_healthy=lambda:None),root,{'pid':42},100)
            self.assertEqual(sample,original)
            self.assertFalse((root/'nvml-measurement-ready.json').exists())
