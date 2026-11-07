import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import process_metrics


class ProcessMetricsTests(unittest.TestCase):
    def test_collects_and_preserves_raw_rank_uncertainty(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'123';root.mkdir()
            fields=['S']+['0']*19; fields[11]='12'; fields[12]='7'; fields[19]='987'
            (root/'stat').write_text('123 (worker name) '+' '.join(fields))
            (root/'status').write_text('VmRSS:\t4321 kB\nCpus_allowed_list:\t8-15,80-87\nNSpid:\t123 12\n')
            invoke=lambda *a,**k:'GPU-abcdef-1234, 123, 100\n'
            result=process_metrics.capture(invoke,d)
            self.assertEqual(result['processes']['123']['cpu_ticks'],19)
            self.assertEqual(result['processes']['123']['rss_kib'],4321)
            self.assertEqual(result['processes']['123']['namespace_pid'],12)
            self.assertIn('not established',result['scope'])
            def forbidden(*a,**k):raise AssertionError('native table must avoid the GPU CLI')
            self.assertEqual(process_metrics.capture(forbidden,d,gpu_processes=[{'gpu_uuid':'GPU-abcdef-1234','pid':123,'gpu_memory_mib':100}])['processes'],result['processes'])
            with self.assertRaisesRegex(ValueError,'overlaps'):
                process_metrics.capture(invoke,d,server_cpus={8,9})
            (root/'status').write_text('Cpus_allowed_list:\t8-15\n')
            with self.assertRaisesRegex(ValueError,'RSS'):process_metrics.capture(invoke,d)

    def test_missing_or_bad_processes_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'no GPU'):
            process_metrics.capture(lambda *a,**k:'')
        with self.assertRaisesRegex(ValueError,'invalid GPU'):
            process_metrics.capture(lambda *a,**k:'GPU-f, not-a-pid, 100')
