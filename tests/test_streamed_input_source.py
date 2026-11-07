from pathlib import Path
import copy
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_input_source
import fixed_workload
import input_panel_worker as worker
import runtime
import streamed_input_source as source
import verify_fixed_task
from test_input_panel_worker import spec
from test_vllm_prompt import FakeTokenizer


class StreamedInputSourceTests(unittest.TestCase):
    def fixture(self, root):
        data = spec()
        out = root/'output'
        out.mkdir()
        worker.run(data, out, FakeTokenizer(), tokenizer_initializations=1)
        identity = {'pid': 1, 'start_ticks': 1, 'boot_id': 'fixture'}
        cfg = {'input_panel': {k: v for k, v in data.items() if k != 'deadline_ns'},
               'run_id': 'fixture', 'layout': '2xtp4', 'image': 'fixture',
               'task_family': 'fixed', 'fixed_prompt_version': fixed_workload.PERFORMANCE_VERSION}
        toolkit = root/'toolkit-MANIFEST.json'
        runtime.atomic_json(toolkit, {'reproduce/input_panel_worker.py': worker.sha(worker.__file__),
             'reproduce/fixed_workload.py': worker.sha(fixed_workload.__file__),
             'reproduce/verify_fixed_task.py': worker.sha(verify_fixed_task.__file__)})
        cfg['toolkit_manifest'] = str(toolkit)
        runtime.atomic_json(root/'input-spec.json', data)
        runtime.atomic_json(root/'execution-state.json', {'run_id': 'fixture', 'driver_identity': identity,
             'watchdog_identity': identity, 'owned_processes': [identity], 'stopping': True, 'watchdog_stop': True})
        runtime.atomic_json(root/'input-worker-identity.json', identity)
        runtime.atomic_json(root/'cleanup-result.json', {'success': True})
        runtime.atomic_json(root/'gpu-idle-after-cleanup.json', {'idle': True})
        runtime.atomic_json(root/'manifest.json', {'runtime': {'toolkit_manifest_sha256': worker.sha(toolkit)}})
        runtime.atomic_json(root/'checkpoint-receipt.json', {'fixture': 'checkpoint'})
        runtime.atomic_json(root/'image-qualification.json', {'fixture': 'image'})
        for key, name in [('manifest', 'manifest.json'), ('checkpoint_receipt', 'checkpoint-receipt.json'),
                          ('image_qualification', 'image-qualification.json')]:
            cfg[key] = str(root/name)
        with patch.object(source.runtime, 'process_identity', return_value=identity), \
             patch.object(source.runtime, 'process_alive', return_value=False):
            rows = source.project(root, data, cfg)
        return Path(rows[0]['source']), cfg, data, identity

    def test_sealed_population_passes_actual_input_loader_without_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            population, cfg, data, _ = self.fixture(root)
            job = data['jobs'][0]
            cfg.update(tasks_file=str(population/'fixed-tasks.json'), cell_id=job['cell_id'], repetition=0)
            with patch.object(fixed_workload, 'build', side_effect=AssertionError('no regeneration')):
                loaded = fixed_input_source.load(cfg, 8000, 2, 0, allowed_root=root)
            self.assertEqual(len(loaded['tasks']), 2)
            self.assertEqual(loaded['task_bytes'], (root/'output'/job['run_id']/'fixed-tasks.json').read_bytes())
            self.assertFalse(source.verify(population)['scientific_acceptance'])

    def test_resealed_changed_checkpoint_source_cleanup_or_population_is_rejected(self):
        mutations = [('task-000-generation.json', 'index', 1),
                     ('task-000-generation.json', 'source_generator_sha256', '0'*64),
                     ('cleanup-result.json', 'success', False),
                     ('worker-result.json', 'tokenizer_initializations', 2),
                     ('population-result.json', 'new_tasks', 1),
                     ('lineage.json', 'worker_source_sha256', '0'*64)]
        for name, key, value in mutations:
            with self.subTest(name=name, key=key), tempfile.TemporaryDirectory() as d:
                population, _, _, _ = self.fixture(Path(d).resolve())
                path = population/'streamed-generation'/name
                row = source.read(path)
                row[key] = value
                runtime.atomic_json(path, row)
                runtime.seal_evidence(population)
                with self.assertRaises(ValueError):
                    source.verify(population)

    def test_extra_checkpoint_cannot_hide_in_a_resealed_source(self):
        with tempfile.TemporaryDirectory() as d:
            population, _, _, _ = self.fixture(Path(d).resolve())
            (population/'streamed-generation/extra.json').write_text('{}')
            runtime.seal_evidence(population)
            with self.assertRaisesRegex(ValueError, 'complete proof'):
                source.verify(population)


if __name__ == '__main__':
    unittest.main()
