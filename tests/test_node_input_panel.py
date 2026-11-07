from pathlib import Path
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_input_panel as panel
import runtime
import supervisor
from test_input_panel_worker import spec


class InputPanelLifecycleTests(unittest.TestCase):
    def test_initialization_task_and_intertask_bounds_are_external(self):
        data = spec()
        job = data['jobs'][0]
        began = time.monotonic_ns()
        init = {'phase': 'tokenizer_initializing', 'started_ns': began}
        panel.check_progress(init, data, began, began+100_000_000_000)
        with self.assertRaises(TimeoutError):
            panel.check_progress(init, data, began, began+111_000_000_000)
        task = dict(phase='task_started', started_ns=began, run_id=job['run_id'],
                    cell_id=job['cell_id'], repetition=job['repetition'], index=0,
                    task_timeout_seconds=job['task_timeout_seconds'])
        with self.assertRaises(TimeoutError):
            panel.check_progress(task, data, began, began+111_000_000_000)
        task['task_timeout_seconds'] = 120
        with self.assertRaises(ValueError):
            panel.check_progress(task, data, began, began+1)
        committed = dict(phase='task_committed', finished_ns=began, run_id=job['run_id'], index=0)
        with self.assertRaises(TimeoutError):
            panel.check_progress(committed, data, began, began+11_000_000_000)
        committed['finished_ns'] = began+2
        with self.assertRaises(ValueError):
            panel.check_progress(committed, data, began, began+1)

    def test_worker_failure_or_missing_result_cannot_pass(self):
        class Session:
            def ensure_healthy(self): pass
        class Child:
            returncode = 1
            def poll(self): return self.returncode
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(runtime.RuntimeErrorClosed):
                panel.monitor_worker(Session(), Child(), Path(d), spec(), time.monotonic_ns())
            Child.returncode = 0
            with self.assertRaises(FileNotFoundError):
                panel.monitor_worker(Session(), Child(), Path(d), spec(), time.monotonic_ns())

    def test_preparation_child_identity_is_saved_before_exec_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            cid = 'a'*64
            runtime.atomic_json(root/'execution-state.json', {
                'owned_processes': [], 'container_ids': [cid], 'stopping': False})
            session = supervisor.Supervisor(root/'node.lock', root, 'input-fixture',
                                           time.monotonic_ns()+600_000_000_000, phase='prompt-generation')
            session.preflight_done = True
            identity = {'pid': 123, 'start_ticks': 1, 'boot_id': 'fixture'}
            class Child:
                pid = 123
            def create(argv, **kwargs):
                self.assertFalse((root/'input-worker-gate/router-start.ok').exists())
                self.assertEqual(json.loads((root/'execution-state.json').read_text())['owned_processes'], [])
                self.assertIn('/campaign/input_panel_worker.py', argv)
                self.assertIn('-B', argv)
                return Child()
            with patch.object(supervisor.runtime, 'inspect_owned', return_value={'State': {'Running': True}}), \
                 patch.object(supervisor.runtime, 'process_identity', return_value=identity), \
                 patch.object(supervisor.subprocess, 'Popen', side_effect=create):
                self.assertEqual(session.start_input_worker(cid).pid, 123)
            self.assertEqual(json.loads((root/'execution-state.json').read_text())['owned_processes'], [identity])
            self.assertEqual(json.loads((root/'input-worker-gate/router-start.ok').read_text()), {'pid': 123})
            self.assertFalse((root/'router-start.ok').exists())

    def test_preparation_worker_rejects_stopping_wrong_container_or_phase_before_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            session = supervisor.Supervisor(root/'lock', root, 'input-fixture',
                                           time.monotonic_ns()+600_000_000_000, phase='prompt-generation')
            session.preflight_done = True
            for state in ({'container_ids': ['a'*64], 'stopping': True},
                          {'container_ids': ['b'*64], 'stopping': False}):
                runtime.atomic_json(root/'execution-state.json', state)
                with patch.object(supervisor.subprocess, 'Popen') as spawn, self.assertRaises(runtime.RuntimeErrorClosed):
                    session.start_input_worker('a'*64)
                spawn.assert_not_called()
            session.phase = 'fixed-performance'
            with self.assertRaises(runtime.RuntimeErrorClosed):
                session.start_input_worker('a'*64)

    def test_declared_jobs_match_source_matrix_and_reject_regeneration(self):
        import campaign
        data = spec()
        del data['deadline_ns']
        cfg = {'input_panel': data, 'matrix': str(campaign.HERE/'comparison-matrix.json'),
               'task_family': 'fixed', 'layout': '2xtp4',
               'fixed_prompt_version': 'fixed-performance-v1', 'cell_id': data['jobs'][0]['cell_id']}
        deadline = time.monotonic_ns()+600_000_000_000
        # The existing scientific prompt version is used without aliases.
        import fixed_workload
        cfg['fixed_prompt_version'] = fixed_workload.PERFORMANCE_VERSION
        self.assertEqual(panel.validate_panel(cfg, deadline)['deadline_ns'], deadline)
        data['jobs'][0]['retained_tasks'] = {'0': {'path': '/retained', 'sha256': '0'*64}}
        with self.assertRaisesRegex(ValueError, 'linked separately'):
            panel.validate_panel(cfg, deadline)


if __name__ == '__main__':
    unittest.main()
