import sys
import json
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import nvml_safety_probe as nvml


class NvmlSafetyTests(unittest.TestCase):
    def test_measured_window_excludes_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertFalse(nvml.measurement_active(root))
            (root/'nvml-measurement-mode.json').write_text('{}')
            (root/'execution-state.json').write_text(json.dumps({'stopping':False}))
            self.assertTrue(nvml.measurement_active(root))
            (root/'execution-state.json').write_text(json.dumps({'stopping':True}))
            self.assertFalse(nvml.measurement_active(root))

    def test_slow_sample_remains_available_for_safety_during_and_after_measurement(self):
        for stopping in (False,True):
            with self.subTest(stopping=stopping), tempfile.TemporaryDirectory() as d:
                root=Path(d)
                (root/'nvml-measurement-mode.json').write_text('{}')
                (root/'execution-state.json').write_text(json.dumps({'stopping':False}))
                real_sleep=time.sleep
                class FakeCollector:
                    setup_timings=[]
                    def read(self):
                        real_sleep(1.01)
                        if stopping: (root/'execution-state.json').write_text(json.dumps({'stopping':True}))
                        return {'devices':[]}
                    def read_measurement(self):return self.read()
                    def close(self): pass
                with patch.object(nvml,'Collector',FakeCollector), patch.object(nvml.time,'sleep',side_effect=StopIteration):
                    with self.assertRaises(StopIteration): nvml.loop(root)
                self.assertTrue((root/'nvml-live.json').exists())
                self.assertEqual((root/'nvml-cadence-failure.json').exists(),not stopping)

    def test_startup_allows_slow_fresh_sample_but_measured_mode_rejects_it(self):
        rows=[{'index':i,'uuid':'GPU-'+str(i),'total_bytes':100,'free_bytes':80,'used_bytes':10,
               'reserved_bytes':10,'volatile_uncorrected_ecc':0} for i in range(8)]
        data={'pid':42,'sample_started_ns':100,'monotonic_ns':110,'sample_seconds':1.27,
              'sample_gap_seconds':1.3,'measurement_mode':True,'devices':rows}
        with patch.object(nvml.runtime,'process_alive',return_value=True):
            self.assertEqual(nvml.validate_live(data,{'pid':42},120,max_age_ns=30),data)
            with self.assertRaisesRegex(nvml.runtime.RuntimeErrorClosed,'cadence'):
                nvml.validate_live(data,{'pid':42},120,max_age_ns=30,measured=True)
            with self.assertRaisesRegex(nvml.runtime.RuntimeErrorClosed,'stale'):
                nvml.validate_live(data,{'pid':42},131,max_age_ns=30)
            data['sample_seconds']=float('nan')
            with self.assertRaisesRegex(nvml.runtime.RuntimeErrorClosed,'timing'):
                nvml.validate_live(data,{'pid':42},120,max_age_ns=30)
            data['sample_seconds']=.2
            rows.pop()
            with self.assertRaisesRegex(nvml.runtime.RuntimeErrorClosed,'coverage'):
                nvml.validate_live(data,{'pid':42},120,max_age_ns=30)
