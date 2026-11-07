import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import rank_attribution


class RankAttributionTests(unittest.TestCase):
    def args(self):
        return ['(Worker_DP0_EP0 pid=12) info\n(Worker_DP1_EP1 pid=13) info',
                [101, 102], {'101': {'namespace_pid': 12, 'start_ticks': 5},
                             '102': {'namespace_pid': 13, 'start_ticks': 6}},
                [{'pid': 101, 'gpu_uuid': 'GPU-a'}, {'pid': 102, 'gpu_uuid': 'GPU-b'}],
                [{'uuid': 'GPU-a', 'index': 0}, {'uuid': 'GPU-b', 'index': 1}], [0, 1]]

    def test_exact_join_preserves_process_incarnation(self):
        result = rank_attribution.map_dp_workers(*self.args())
        self.assertEqual([(x['dp_rank'], x['host_pid'], x['gpu_index']) for x in result['workers']],
                         [(0, 101, 0), (1, 102, 1)])

    def test_unowned_missing_or_restarted_worker_is_rejected(self):
        for mutate in (lambda a: a[1].pop(),
                       lambda a: a[2]['102'].update(namespace_pid=12),
                       lambda a: a[3].pop(),
                       lambda a: a.__setitem__(0, a[0]+'\n(Worker_DP0_EP0 pid=99) info'),
                       lambda a: a[3][1].update(gpu_uuid='GPU-a')):
            a = self.args(); mutate(a)
            with self.assertRaises(ValueError): rank_attribution.map_dp_workers(*a)

    def test_tp_shards_require_exact_installed_names_and_physical_join(self):
        a=self.args();a[0]='(Worker_TP0 pid=12) info\n(Worker_TP1 pid=13) info'
        rows=rank_attribution.map_tp_workers(*a)['workers']
        self.assertEqual([x['tp_rank'] for x in rows],[0,1])
        for mutate in (lambda a:a.__setitem__(0,a[0].replace('Worker_TP1','Worker_DP1_EP1')),
                       lambda a:a[3].pop(),
                       lambda a:a.__setitem__(0,a[0]+'\n(Worker_TP0 pid=99) info')):
            bad=self.args();bad[0]=a[0];mutate(bad)
            with self.assertRaises(ValueError):rank_attribution.map_tp_workers(*bad)
