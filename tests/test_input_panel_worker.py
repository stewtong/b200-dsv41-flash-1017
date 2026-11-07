from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import fixed_workload
import input_panel_worker as worker
import prompt_identity
import verify_fixed_task
from test_vllm_prompt import FakeTokenizer


def spec(n=8000, cache='cold', shapes=((2, 0),)):
    suite = 'core' if cache == 'cold' else 'warm_prefix_panel'
    output = 512 if n == 1000000 else 2048
    return {'schema': worker.SCHEMA, 'input_tokens': n, 'output_tokens': output, 'cache': cache,
            'prompt_version': fixed_workload.PERFORMANCE_VERSION, 'block_size': 128 if cache == 'warm_prefix' else None,
            'jobs': [{'run_id': prompt_identity.child(n, c, rep, cache, '20261104t0555z'),
                      'cell_id': f'{suite}-2xtp4-i{n}-c{c}-{cache}-o{output}', 'concurrency': c,
                      'repetition': rep, 'task_timeout_seconds': 110, 'retained_tasks': {}} for c, rep in shapes],
            'deadline_ns': time.monotonic_ns() + 600 * 10**9, 'cleanup_seconds': 120,
            'initialization_timeout_seconds': 110, 'generator_sha256': worker.sha(fixed_workload.__file__),
            'verifier_sha256': worker.sha(verify_fixed_task.__file__)}


class InputPanelWorkerTests(unittest.TestCase):
    def test_streaming_population_is_exact_original_task_array_bytes(self):
        data = spec(shapes=((2, 0), (1, 1)))
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            result = worker.run(data, root, tokenizer)
            self.assertEqual(len(result), 2)
            for job in data['jobs']:
                expected = [fixed_workload.build(tokenizer, '/model', 8000, 2048, 'core',
                            job['concurrency'], job['repetition'], i, 'cold', None,
                            prompt_version=fixed_workload.PERFORMANCE_VERSION) for i in range(job['concurrency'])]
                actual = root / job['run_id'] / 'fixed-tasks.json'
                self.assertEqual(actual.read_bytes(), (json.dumps(expected, sort_keys=True, indent=2) + '\n').encode())
            self.assertEqual(json.loads((root / 'worker-result.json').read_text())['tokenizer_initializations'], 0)

    def test_interruption_keeps_checkpoints_and_never_publishes_partial_population(self):
        data = spec()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            def fail_second(*args, **kwargs):
                if args[7] == 1:
                    raise RuntimeError('intentional fixture interruption')
                return fixed_workload.build(*args, **kwargs)
            with self.assertRaises(RuntimeError):
                worker.run(data, root, FakeTokenizer(), builder=fail_second)
            child = root / data['jobs'][0]['run_id']
            self.assertTrue((child / 'task-000.json').is_file())
            self.assertTrue((child / 'task-000-generation.json').is_file())
            self.assertFalse((child / 'fixed-tasks.json').exists())
            self.assertFalse((root / 'worker-result.json').exists())
            self.assertTrue(list(child.glob('fixed-tasks.json.tmp.*')))

    def test_retained_task_is_verified_and_reused_without_generation(self):
        data = spec()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(strict=True)
            old = base / 'retained'
            out = base / 'new'
            old.mkdir()
            out.mkdir()
            task = fixed_workload.build(FakeTokenizer(), '/model', 8000, 2048, 'core', 2, 0, 0,
                    'cold', None, prompt_version=fixed_workload.PERFORMANCE_VERSION)
            source = old / 'task.json'
            source.write_text(json.dumps(task))
            raw = source.read_bytes()
            data['jobs'][0]['retained_tasks'] = {'0': {'path': str(source), 'sha256': worker.sha(source)}}
            generated = []
            def build(*args, **kwargs):
                generated.append(args[7])
                return fixed_workload.build(*args, **kwargs)
            worker.run(data, out, FakeTokenizer(), retained_root=old, builder=build)
            self.assertEqual(generated, [1])
            self.assertEqual(source.read_bytes(), raw)
            result = json.loads((out / 'worker-result.json').read_text())
            self.assertEqual((result['new_tasks'], result['reused_tasks']), (1, 1))

    def test_tampered_retained_task_or_wrong_source_does_not_generate(self):
        data = spec()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            old = root / 'retained'
            out = root / 'out'
            old.mkdir()
            out.mkdir()
            source = old / 'task.json'
            source.write_text('{}')
            data['jobs'][0]['retained_tasks'] = {'0': {'path': str(source), 'sha256': '0' * 64}}
            with self.assertRaises(ValueError):
                worker.run(data, out, FakeTokenizer(), retained_root=old,
                           builder=lambda *a, **k: self.fail('must not regenerate retained work'))

    def test_deadline_admission_precedes_next_build(self):
        data = spec()
        data['deadline_ns'] = time.monotonic_ns() + 150 * 10**9
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(TimeoutError):
            worker.run(data, Path(directory).resolve(strict=True), FakeTokenizer(),
                       builder=lambda *a, **k: self.fail('inadmissible task must not start'))

    def test_source_change_and_unknown_population_rejected(self):
        data = spec()
        data['generator_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            worker.validate(data)
        data = spec()
        data['jobs'][0]['run_id'] = 'x' * 49
        with self.assertRaises(ValueError):
            worker.validate(data)
        data = spec()
        data['jobs'].append(copy.deepcopy(data['jobs'][0]))
        with self.assertRaises(ValueError):
            worker.validate(data)

    def test_actual_cli_loads_once_per_panel_with_cold_warm_and_long_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            package = root / 'vllm/tokenizers'
            package.mkdir(parents=True)
            (root / 'vllm/__init__.py').write_text('')
            (package / '__init__.py').write_text('')
            (package / 'deepseek_v41.py').write_text('''import os
class DeepseekV41Tokenizer:
 @classmethod
 def from_pretrained(cls,*a,**k):
  with open(os.environ['FIXTURE_LOAD_LOG'],'a') as f:f.write('loaded\\n')
  return cls()
 def apply_chat_template(self,messages,**kwargs):return '|'.join(x['role']+':'+x['content'] for x in messages)
 def encode(self,text,add_special_tokens=False):return list(text)
''')
            for n, cache, shapes in [(8000, 'cold', ((2, 0), (1, 1))),
                                      (128000, 'warm_prefix', ((1, 0),)),
                                      (1000000, 'cold', ((1, 2),))]:
                data = spec(n, cache, shapes)
                path = root / f'spec-{n}-{cache}.json'
                out = root / f'output-{n}-{cache}'
                out.mkdir()
                path.write_text(json.dumps(data))
                loads = root / f'loads-{n}-{cache}.txt'
                env = dict(os.environ, PYTHONPATH=str(root), FIXTURE_LOAD_LOG=str(loads))
                actual = subprocess.run([sys.executable, '-B', worker.__file__, str(path), str(out), '--model', '/fixture'],
                                        env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(actual.returncode, 0, actual.stderr)
                self.assertEqual(loads.read_text(), 'loaded\n')
                for job in data['jobs']:
                    expected = [fixed_workload.build(FakeTokenizer(), '/model', n, data['output_tokens'],
                                'core' if cache == 'cold' else 'warm_prefix_panel', job['concurrency'],
                                job['repetition'], i, cache, data['block_size'],
                                prompt_version=fixed_workload.PERFORMANCE_VERSION) for i in range(job['concurrency'])]
                    self.assertEqual((out / job['run_id'] / 'fixed-tasks.json').read_bytes(),
                                     (json.dumps(expected, sort_keys=True, indent=2) + '\n').encode())
                summary = json.loads((out / 'worker-result.json').read_text())
                self.assertEqual(summary['tokenizer_initializations'], 1)
                self.assertLessEqual(summary['maximum_buffered_task_input_tokens'], 8_000_000)


if __name__ == '__main__':
    unittest.main()
