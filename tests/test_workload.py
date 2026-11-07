from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import workload as w

class WorkloadTests(unittest.TestCase):
    def test_refined_bracket_preserves_the_exact_crossing_and_tie_rule(self):
        # The established contract selects the first count at/above target or
        # its immediate predecessor, including plateaus in token counts.
        for fn in (lambda n:10+7*n,lambda n:10+n*n,lambda n:10+(n//5)*7,
                   lambda n:10+int(n**.5)*9,lambda n:10+n+(n//13)*7):
            for target in (13,27,51,109,313):
                with self.subTest(target=target,fn=fn):
                    cross=next(n for n in range(100000) if fn(n)>=target)
                    expected=min((max(0,cross-1),cross),key=lambda n:(abs(fn(n)-target),n))
                    result=w.calibrate(str,lambda text:fn(int(text)),target,1,tolerance=1)
                    self.assertEqual(result['filler_units'],expected)

    def test_refinement_reduces_fully_tokenized_work_for_long_linear_filler(self):
        measured=[]
        def count(text):
            n=int(text);measured.append(37+9*n);return measured[-1]
        result=w.calibrate(str,count,1_000_000,512)
        self.assertEqual(result['filler_units'],111107)
        self.assertLess(sum(measured),6_000_000)
        self.assertEqual(result['input_tokens'],1_000_000)

    def test_seeds_match_arms_and_differ_by_session(self):
        self.assertEqual(w.seed_for('core', 1000, 8, 1, 2), w.seed_for('core', 1000, 8, 1, 2))
        self.assertNotEqual(w.seed_for('core', 1000, 8, 1, 2), w.seed_for('core', 1000, 8, 1, 3))
    def test_counts_rendered_overhead_and_output_room(self):
        result = w.calibrate(lambda n:'sys'+('x'*n), len, 1000, 128)
        self.assertEqual(result['input_tokens'], 1000)
        self.assertEqual(result['filler_units'], 997)
        with self.assertRaises(ValueError): w.calibrate(lambda n:'x'*n, len, 1_048_000, 1024)
        with self.assertRaises(ValueError): w.calibrate(lambda n:'x'*(n*100), len, 1050, 1)
    def test_explicit_sampling_and_reasoning_contract(self):
        body = w.request_body('/model', [{'role':'user','content':'x'}], 1000, 128, 4)
        self.assertEqual(body['chat_template_kwargs'], {'thinking':False, 'reasoning_effort':50})
        self.assertEqual((body['temperature'], body['top_p'], body['min_tokens']), (1.0, 1.0, 128))
        self.assertTrue(body['ignore_eos'])
        with self.assertRaises(ValueError): w.request_body('/model', [{'role':'user','content':'x'}], 1000, 128, 4, effort=101)
