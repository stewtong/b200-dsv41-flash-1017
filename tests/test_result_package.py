"""Synthetic aggregate fixtures exercise publication verification, not inference."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import verify_result_package as verify


class ResultPackageTest(unittest.TestCase):
    def supplement_rows(self):
        return [dict(output_tokens=o, layout=layout, cache=cache, concurrency=c,
                     verdict='accepted', repetitions=3, requests=3*c,
                     burst_output_tokens_per_second_min=1,
                     burst_output_tokens_per_second_p50=2,
                     burst_output_tokens_per_second_max=3)
                for o in (512, 2048) for layout in verify.LAYOUTS
                for cache, clients in [('cold', (1, 2, 4, 8, 16, 32, 64, 128)),
                                       ('warm_prefix', (1, 8, 32))] for c in clients]

    def write_supplement(self, root, rows):
        with (root/'comparison.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root/'summary.json').write_text(json.dumps(dict(supplement_accepted_cells=44,
            supplement_required_cells=44, supplement_repetitions=132,
            supplement_requests=3552, supplement_complete=True)))

    def test_supplement_complete_and_duplicate_omission_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.supplement_rows()
            self.write_supplement(root, rows)
            self.assertEqual(verify.supplement(root)['requests'], 3552)
            rows[-1] = rows[-2]
            self.write_supplement(root, rows)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                verify.supplement(root)

    def tuning_data(self):
        experiments = []
        for name, lengths in [('dspark', (8000, 64000)), ('graph', (8000, 64000)),
                              ('prefill', (128000, 256000))]:
            pairs = []
            for length in lengths:
                for concurrency in (1, 8):
                    metric = dict(baseline_repetitions=[2, 3, 4], candidate_repetitions=[1, 2, 3],
                                  paired_ratios=[.5, 2/3, .75], median_ratio=2/3)
                    pairs.append(dict(input_tokens=length, concurrency=concurrency,
                        metrics={key: copy.deepcopy(metric) for key in
                                 ('ttft', 'end_to_end', 'throughput', 'decode_tpot')}))
            qualification = dict(dspark=38, graph=58, prefill=32)[name]
            experiments.append(dict(experiment=name, status='inconclusive', pairs=pairs,
                measured_requests=108, verified_repetitions=24, accepted_repetition_attempts=24,
                total_requests_reserved=108+qualification, qualification_requests_reserved=qualification))
        return dict(experiments=experiments, measured_repetitions=72, measured_requests=324,
                    total_requests_reserved=452, qualification_requests_reserved=128)

    def test_tuning_arithmetic_and_changed_ratio_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.tuning_data()
            (root/'comparisons.json').write_text(json.dumps(data))
            self.assertEqual(verify.tuning(root)['pairs'], 12)
            data['experiments'][0]['pairs'][0]['metrics']['ttft']['median_ratio'] = .1
            (root/'comparisons.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'aggregate'):
                verify.tuning(root)


if __name__ == '__main__':
    unittest.main()
