"""Panel identity, exact inert tools and host-side non-executing patch checks."""
import copy
import difflib
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import coding_panel
import coding_patch
import coding_sandbox
import runtime


class CodingPanelTests(unittest.TestCase):
    def test_fixed_population_and_edge_oracles(self):
        panel = coding_panel.tasks()
        self.assertEqual(coding_panel.validate_panel(panel)['tasks'], 100)
        self.assertEqual(len({row['sha256'] for row in panel}), 100)
        self.assertEqual(len({row['id'] for row in panel}), 100)
        self.assertTrue(all(len(row['cases']) == 12 for row in panel))
        self.assertEqual(coding_panel.answer('intervals', [[1, 3], [3, 4], [8, 8]]), [[1, 4], [8, 8]])
        self.assertEqual(coding_panel.answer('histogram', {'items': [-1, 0, 2], 'width': 2}), [[-2, 1], [0, 1], [2, 1]])
        self.assertEqual(coding_panel.answer('reachable', {'start': 'a', 'edges': [['a', 'b'], ['b', 'a'], ['z', 'q']]}), ['a', 'b'])
        self.assertEqual(coding_panel.answer('paths', ['/../../a//./b/../']), ['/a'])
        changed = copy.deepcopy(panel); changed[0]['cases'][0]['expected'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'fixed version'): coding_panel.validate_panel(changed)

    def test_nested_tool_arguments_multiple_calls_and_duplicates(self):
        task = coding_panel.tasks()[0]
        def calls(turn):
            return [{'id': f'call-{i}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
                    for i, (name, args) in enumerate(coding_panel.expected_calls(task, turn))]
        self.assertEqual(len(coding_panel.tool_results(task, 0, calls(0))), 1)
        self.assertEqual(len(coding_panel.tool_results(task, 1, list(reversed(calls(1))))), 2)
        for modify in ('case', 'bool', 'duplicate', 'population', 'id'):
            bad = calls(1)
            if modify == 'case': bad[0]['function']['arguments'] = bad[0]['function']['arguments'].replace('marker_', 'Marker_')
            elif modify == 'bool': bad[0]['function']['arguments'] = bad[0]['function']['arguments'].replace('"offset": 0', '"offset": false')
            elif modify == 'duplicate': bad[0]['function']['arguments'] = bad[0]['function']['arguments'].replace('"offset": 0', '"offset": 9, "offset": 0')
            elif modify == 'population': bad.pop()
            else: bad[1]['id'] = bad[0]['id']
            with self.assertRaises(ValueError): coding_panel.tool_results(task, 1, bad)

    def test_unified_patch_is_applied_as_data_and_rejects_escape_or_wrong_context(self):
        source = coding_panel.SOURCE
        candidate = 'def solve(value):\n    return value\n'
        patch = ''.join(difflib.unified_diff(source.splitlines(True), candidate.splitlines(True), 'a/solution.py', 'b/solution.py'))
        self.assertEqual(coding_patch.apply(source, patch), candidate)
        for bad in (patch.replace('b/solution.py', 'b/../../outside'), patch.replace('TODO', 'WRONG'),
                    patch.replace('@@ -1,2 +1,2 @@', '@@ -1,3 +1,2 @@'), patch + '--- a/other.py\n'):
            with self.assertRaises(ValueError): coding_patch.apply(source, bad)
        # A dangerous expression is only a string here, never evaluated.
        dangerous = candidate.replace('return value', 'return __import__("os").system("false")')
        patch = ''.join(difflib.unified_diff(source.splitlines(True), dangerous.splitlines(True), 'solution.py', 'solution.py'))
        self.assertEqual(coding_patch.apply(source, patch), dangerous)

    def test_cpu_scorer_isolation_rejects_mount_devices_privilege_or_weak_limits(self):
        row = {'Id': 'a' * 64, 'Image': 'sha256:' + 'b' * 64, 'Mounts': [],
               'Config': {'User': '65534:65534', 'Env': ['NVIDIA_VISIBLE_DEVICES=void', 'NVIDIA_DRIVER_CAPABILITIES=none', 'DSV41_ISOLATED_CODING_SCORER=1']},
               'HostConfig': {'NetworkMode': 'none', 'IpcMode': 'private', 'Runtime': 'runc',
                              'ReadonlyRootfs': True, 'Privileged': False, 'PidsLimit': 64,
                              'Memory': 512 * 1024 * 1024, 'NanoCpus': 1_000_000_000,
                              'CapDrop': ['ALL'], 'DeviceRequests': None, 'Devices': [],
                              'SecurityOpt': ['no-new-privileges'],
                              'Tmpfs': {'/tmp': 'rw,nosuid,nodev,noexec,size=16777216'}}}
        self.assertEqual(coding_sandbox.verify_isolation(row)['status'], 'isolated_cpu_container')
        for key, value in [('NetworkMode', 'host'), ('Privileged', True), ('Runtime', 'nvidia'),
                           ('Memory', 0), ('DeviceRequests', [{'Count': -1}]), ('Binds', ['/data:/data']),
                           ('Tmpfs', {'/tmp': 'rw,size=16777216'})]:
            bad = copy.deepcopy(row); bad['HostConfig'][key] = value
            with self.assertRaises(runtime.RuntimeErrorClosed): coding_sandbox.verify_isolation(bad)
        bad = copy.deepcopy(row); bad['Mounts'] = [{'Type': 'bind', 'Destination': '/model'}]
        with self.assertRaises(runtime.RuntimeErrorClosed): coding_sandbox.verify_isolation(bad)
        command = coding_sandbox.argv('vllm/vllm-openai@sha256:' + 'a' * 64, 'fixture')
        self.assertNotIn('--gpus', command)
        self.assertIn('runc', command)

    def test_worker_fails_before_reading_or_executing_outside_scorer_environment(self):
        worker = Path(coding_panel.__file__).with_name('coding_score_worker.py')
        result = subprocess.run([sys.executable, '-I', '-S', str(worker)], input=b'{}', capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'container scorer environment required', result.stderr)


if __name__ == '__main__': unittest.main()
