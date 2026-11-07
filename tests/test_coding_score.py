"""Independent scorer checks against declared synthetic raw records."""
import copy
import difflib
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import campaign
import coding_panel
import runtime
import verify_coding_score


class CodingScoreTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir(); cid = 'a' * 64; image = 'sha256:' + 'b' * 64; task = coding_panel.tasks()[0]
        write = lambda name, obj: runtime.atomic_json(root / name, obj)
        write('run-config.json', {'run_id': 'fixture', 'evidence_class': 'synthetic raw fixture'})
        write('execution-state.json', {'run_id': 'fixture', 'container_ids': [cid], 'owned_processes': [], 'stopping': True, 'watchdog_stop': True})
        write('cleanup-result.json', {'success': True, 'deadline_seconds': 120, 'elapsed_seconds': 1,
                                     'outcomes': [{'container_id': cid, 'status': 'stopped', 'log_status': 'captured'}]})
        write('gpu-idle-after-cleanup.json', {'idle': True})
        write('image-qualification.json', {'image_id': image})
        write('control-task.json', task)
        inspected = {'Id': cid, 'Image': image, 'Mounts': [],
                     'Config': {'User': '65534:65534', 'Env': ['NVIDIA_VISIBLE_DEVICES=void', 'NVIDIA_DRIVER_CAPABILITIES=none', 'DSV41_ISOLATED_CODING_SCORER=1']},
                     'HostConfig': {'NetworkMode': 'none', 'IpcMode': 'private', 'Runtime': 'runc',
                                    'ReadonlyRootfs': True, 'Privileged': False, 'PidsLimit': 64,
                                    'Memory': 536870912, 'NanoCpus': 1000000000, 'CapDrop': ['ALL'],
                                    'DeviceRequests': None, 'Devices': [], 'SecurityOpt': ['no-new-privileges'],
                                    'Tmpfs': {'/tmp': 'rw,nosuid,nodev,noexec,size=16777216'}}}
        write('scorer-container-inspect.json', inspected)
        controls = []
        for name, code in [('correct', 0), ('wrong', 0), ('blocked_import', 1), ('cpu_limit', 137)]:
            # No source in this fixture is executed. Expected values are supplied
            # as raw oracle inputs for the independent verifier's arithmetic.
            source = 'def solve(value):\n    return value\n# fixture-' + name + '\n'
            diff = ''.join(difflib.unified_diff(task['source'].splitlines(True), source.splitlines(True), 'a/solution.py', 'b/solution.py'))
            (root / (name + '.diff')).write_text(diff)
            source_sha = hashlib.sha256(source.encode()).hexdigest(); tests = []
            for i, case in enumerate(task['cases']):
                actual = case['expected'] if name == 'correct' else []
                tests.append({'index': i, 'actual': actual, 'passed': actual == case['expected']})
            raw = (json.dumps({'schema': 'coding-executable-score/v1', 'source_sha256': source_sha,
                               'tests': tests, 'passed': all(row['passed'] for row in tests)}).encode() if code == 0 else b'declared intentional failure\n')
            (root / (name + '.raw')).write_bytes(raw)
            result = {'schema': 'coding-sandbox-result/v1', 'task_id': task['id'], 'task_sha256': task['sha256'],
                      'source_sha256': source_sha, 'worker_sha256': 'c' * 64, 'container_id': cid,
                      'exit_code': code, 'elapsed_seconds': 1, 'raw_output_sha256': hashlib.sha256(raw).hexdigest(), 'passed': name == 'correct'}
            if code == 0: result['tests'] = tests
            write(name + '.json', result)
            controls.append({'name': name, 'passed': name == 'correct', 'exit_code': code})
        write('coding-sandbox-control.json', {'status': 'passed', 'controls': controls})
        runtime.seal_evidence(root)

    def test_control_cli_recomputes_answers_and_rejects_resealed_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'control'; self.fixture(root)
            output = io.StringIO()
            with patch.object(sys, 'argv', ['campaign.py', 'verify-coding-score-control', '--run-dir', str(root)]), redirect_stdout(output):
                campaign.main()
            self.assertEqual(json.loads(output.getvalue())['status'], 'passed')
            raw_path = root / 'correct.raw'; original = raw_path.read_bytes(); raw = json.loads(original)
            raw['tests'][0]['actual'] = 'tampered'
            raw_path.write_text(json.dumps(raw)); result_path = root / 'correct.json'; result = json.loads(result_path.read_text())
            result['raw_output_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest(); result_path.write_text(json.dumps(result))
            runtime.seal_evidence(root)
            with self.assertRaisesRegex(ValueError, 'expected test value'): verify_coding_score.control(root)

    def test_raw_privilege_change_or_failed_executable_cannot_be_counted_correct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'control'; self.fixture(root)
            path = root / 'scorer-container-inspect.json'; original = path.read_text(); row = json.loads(original)
            row['HostConfig']['Privileged'] = True; path.write_text(json.dumps(row)); runtime.seal_evidence(root)
            with self.assertRaisesRegex(ValueError, 'isolation'): verify_coding_score.control(root)
            path.write_text(original)
            result_path = root / 'cpu_limit.json'; row = json.loads(result_path.read_text()); row['passed'] = True; result_path.write_text(json.dumps(row)); runtime.seal_evidence(root)
            with self.assertRaisesRegex(ValueError, 'counted correct'): verify_coding_score.control(root)


if __name__ == '__main__': unittest.main()
