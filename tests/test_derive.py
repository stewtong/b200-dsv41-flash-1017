import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('derive', Path(__file__).resolve().parents[1] / 'reproduce/derive.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)

class DerivationTests(unittest.TestCase):
    def test_failed_request_extends_makespan_but_not_numerator(self):
        rows = [dict(request_id='a', start_ns=0, end_ns=4_000_000_000, completion_tokens=12, valid=True, clock_id='a'),
                dict(request_id='b', start_ns=1_000_000_000, end_ns=6_000_000_000, completion_tokens=5, valid=False, clock_id='a')]
        self.assertEqual(d.burst(rows), {'duration_seconds':6, 'valid_tokens':12, 'invalid_partial_tokens':5,
                                       'valid_tokens_per_second':2, 'valid_requests':1, 'requests':2})
        rows[1]['clock_id'] = 'b'
        with self.assertRaises(ValueError): d.burst(rows)
    def test_window_and_missing_fields(self):
        self.assertEqual(d.stream_window(1_000_000_000, 3_000_000_000, 3), 1)
        self.assertIsNone(d.stream_window(1, 1, 1))
        with self.assertRaises(ValueError): d.burst([{'request_id':'a'}])
    def test_rejects_noninteger_usage_and_clock_values(self):
        row=dict(request_id='a',start_ns=0,end_ns=1,completion_tokens=1,valid=True,clock_id='clock')
        for key,value in [('completion_tokens',1.5),('completion_tokens',True),('start_ns',0.5),('clock_id','')]:
            bad=dict(row);bad[key]=value
            with self.assertRaises(ValueError):d.burst([bad])

    def test_undersampled_bucket_cannot_pass(self):
        self.assertEqual(d.slo_bucket([1] * 19, 3, 1, 1, False)['verdict'], 'inconclusive')
        self.assertEqual(d.slo_bucket([1] * 20, 3, 1, 1, False)['verdict'], 'passed')
        self.assertEqual(d.slo_bucket([1] * 19, 3, 0.9, 1, False)['verdict'], 'failed')
        self.assertEqual(d.slo_bucket([1] * 20, 3, 1, 1, None)['verdict'], 'inconclusive')

if __name__ == '__main__': unittest.main()
