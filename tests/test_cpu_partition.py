import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import cpu_partition


class CpuPartitionTests(unittest.TestCase):
    def test_reserves_whole_cores_on_each_node(self):
        rows = ['# CPU,Core,Socket,Node']
        for node in range(2):
            for core in range(6):
                for thread in range(2):
                    cpu = node * 12 + core * 2 + thread
                    rows.append(f'{cpu},{node * 6 + core},{node},{node}')
        result = cpu_partition.partition('\n'.join(rows), set(range(24)))
        self.assertEqual(result['client_cpus'], list(range(8)) + list(range(12, 20)))
        self.assertEqual(set(result['server_cpus']), set(range(24)) - set(result['client_cpus']))
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            cpu_partition.partition('\n'.join(rows), set(range(23)))

    def test_rejects_one_numa_node(self):
        with self.assertRaisesRegex(ValueError, 'two NUMA'):
            cpu_partition.partition('0,0,0,0\n1,0,0,0', {0, 1})
