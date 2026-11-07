"""Client exit, missing ranks and old zero snapshots cannot prove cancellation."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import request_absence


class AbsenceTests(unittest.TestCase):
    expected = {'http://127.0.0.1:31000/metrics': [0, 1]}
    def snapshot(self, start, running=0, waiting=0, frontend=0):
        text = ''.join(f'# TYPE {metric} gauge\n' for metric in request_absence.GAUGES)
        text += ''.join(f'{metric}{{engine="{rank}"}} {value if rank == 0 else 0}\n'
                        for metric, value in zip(request_absence.GAUGES, (running, waiting)) for rank in (0, 1))
        return {'clock_id': 'fixture', 'started_ns': start, 'monotonic_ns': start + 100_000_000,
                'metrics': {'http://127.0.0.1:31000/metrics': text},
                'loads': {'http://127.0.0.1:31000/load': {'server_load': frontend}}}
    def proof(self, after=None, busy=None):
        return request_absence.verify(self.snapshot(0), busy or self.snapshot(1_000_000_000, running=1, frontend=1),
                after if after is not None else [self.snapshot(4_000_000_000), self.snapshot(5_100_000_000)], self.expected,
                cancellation_ns=1_200_000_000, client_terminal_ns=2_000_000_000, clock_id='fixture')
    def test_all_rank_live_transition_and_separated_zero_observations(self):
        self.assertEqual(self.proof()['status'], 'exclusive_request_work_absent')
    def test_frontend_released_but_engine_or_queue_still_working_fails(self):
        for running, waiting in ((1, 0), (0, 1)):
            with self.assertRaisesRegex(ValueError, 'work remains'):
                self.proof(after=[self.snapshot(4_000_000_000, running=running, waiting=waiting), self.snapshot(5_100_000_000)])
        with self.assertRaisesRegex(ValueError, 'transition unproven'): self.proof(busy=self.snapshot(1_000_000_000))
    def test_missing_rank_or_type_stale_snapshot_and_boolean_load_fail(self):
        cases = []
        row = self.snapshot(4_000_000_000); row['metrics'][next(iter(self.expected))] = row['metrics'][next(iter(self.expected))].replace('vllm:num_requests_waiting{engine="1"} 0\n', ''); cases.append(row)
        row = self.snapshot(4_000_000_000); row['loads']['http://127.0.0.1:31000/load']['server_load'] = False; cases.append(row)
        row = self.snapshot(4_000_000_000); row['metrics'][next(iter(self.expected))] = row['metrics'][next(iter(self.expected))].replace(' gauge', ' counter'); cases.append(row)
        row = self.snapshot(4_000_000_000); row['clock_id'] = 'other'; cases.append(row)
        row = self.snapshot(4_000_000_000); row['metrics'][next(iter(self.expected))] += 'vllm:num_requests_running 1\n'; cases.append(row)
        cases.append(self.snapshot(3_000_000_000))
        for row in cases:
            with self.assertRaises(ValueError): self.proof(after=[row, self.snapshot(5_100_000_000)])


if __name__ == '__main__': unittest.main()
