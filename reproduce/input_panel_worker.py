"""Generate one input panel with one tokenizer and bounded task checkpoints.

This worker runs only inside the owned CPU preparation container. The existing
fixed_workload.build and tokenizer verifier define the scientific task bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import comparison_scope
import fixed_workload
import launcher
import runtime
import verify_fixed_task

SCHEMA = 'streamed-fixed-input-panel/v1'
MAX_BUFFERED_INPUT_TOKENS = 8_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate(spec):
    required = {'schema', 'input_tokens', 'output_tokens', 'cache', 'prompt_version', 'block_size',
                'jobs', 'deadline_ns', 'cleanup_seconds', 'initialization_timeout_seconds',
                'generator_sha256', 'verifier_sha256'}
    if not isinstance(spec, dict) or set(spec) != required or spec['schema'] != SCHEMA:
        raise ValueError('exact streamed input-panel specification required')
    cache = spec['cache']
    ns = comparison_scope.INPUTS if cache == 'cold' else comparison_scope.WARM_INPUTS
    cs = comparison_scope.CONCURRENCIES if cache == 'cold' else comparison_scope.WARM_CONCURRENCIES
    if cache not in ('cold', 'warm_prefix') or type(spec['input_tokens']) is not int or spec['input_tokens'] not in ns:
        raise ValueError('unknown preparation panel')
    if spec['output_tokens'] != (512 if spec['input_tokens'] == 1000000 else 2048) or spec['prompt_version'] != fixed_workload.PERFORMANCE_VERSION:
        raise ValueError('scientific output or prompt contract differs')
    if spec['input_tokens'] > MAX_BUFFERED_INPUT_TOKENS:
        raise ValueError('one task exceeds existing memory cap')
    if cache == 'warm_prefix' and (type(spec['block_size']) is not int or spec['block_size'] <= 0):
        raise ValueError('warm construction needs the measured cache block size')
    if cache == 'cold' and spec['block_size'] is not None:
        raise ValueError('cold construction cannot invent warm cache metadata')
    if sha(Path(fixed_workload.__file__)) != spec['generator_sha256'] or sha(Path(verify_fixed_task.__file__)) != spec['verifier_sha256']:
        raise ValueError('original generation or verification source differs')
    if (type(spec['deadline_ns']) is not int or not time.monotonic_ns() < spec['deadline_ns'] <= time.monotonic_ns() + 14400 * 10**9 or
            type(spec['cleanup_seconds']) is not int or spec['cleanup_seconds'] != 120 or
            type(spec['initialization_timeout_seconds']) is not int or not 1 <= spec['initialization_timeout_seconds'] <= 120):
        raise ValueError('bounded deadline and existing cleanup allowance required')
    jobs = spec['jobs']
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= (24 if cache == 'cold' else 9):
        raise ValueError('bounded source-population list required')
    identities = set()
    shapes = set()
    suite = 'core' if cache == 'cold' else 'warm_prefix_panel'
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {'run_id', 'cell_id', 'concurrency', 'repetition', 'task_timeout_seconds', 'retained_tasks'}:
            raise ValueError('exact source population declaration required')
        launcher.validate_run_id(job['run_id'])
        c, rep = job['concurrency'], job['repetition']
        if type(c) is not int or c not in cs or type(rep) is not int or rep not in range(3):
            raise ValueError('population outside the original matrix')
        expected_cell = f'{suite}-2xtp4-i{spec["input_tokens"]}-c{c}-{cache}-o{spec["output_tokens"]}'
        if job['cell_id'] != expected_cell or job['run_id'] in identities or (c, rep) in shapes:
            raise ValueError('source cell identity, alias or shape differs')
        identities.add(job['run_id'])
        shapes.add((c, rep))
        if type(job['task_timeout_seconds']) is not int or not 1 <= job['task_timeout_seconds'] <= 120:
            raise ValueError('task bound must remain explicitly bounded')
        retained = job['retained_tasks']
        if not isinstance(retained, dict) or any(str(i) not in map(str, range(c)) for i in retained):
            raise ValueError('invalid retained task indices')
        for row in retained.values():
            if not isinstance(row, dict) or set(row) != {'path', 'sha256'} or not isinstance(row['path'], str):
                raise ValueError('retained task requires exact path and fingerprint')
    return jobs


def _retained(row, root):
    path = Path(row['path'])
    base = Path(root).resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(base):
        raise ValueError('retained task outside read-only source root')
    if sha(path) != row['sha256']:
        raise ValueError('retained task bytes changed')
    return json.loads(path.read_bytes())


def validate_output(output):
    output = Path(output)
    if output.is_symlink() or output.resolve(strict=True) != output or not output.is_dir():
        raise ValueError('existing exact output directory required')
    return output


def run(spec, output, tokenizer, *, retained_root=None, builder=fixed_workload.build,
        verifier=verify_fixed_task.verify, tokenizer_initializations=0):
    jobs = validate(spec)
    output = validate_output(output)
    suite = 'core' if spec['cache'] == 'cold' else 'warm_prefix_panel'
    result = []
    for job in jobs:
        population = output / job['run_id']
        population.mkdir()
        target = population / 'fixed-tasks.json'
        temporary = target.with_name(target.name + '.tmp.' + str(os.getpid()))
        new_tasks = 0
        reused = 0
        with temporary.open('x') as stream:
            stream.write('[\n')
            for index in range(job['concurrency']):
                if time.monotonic_ns() + int((job['task_timeout_seconds'] + spec['cleanup_seconds']) * 1e9) > spec['deadline_ns']:
                    raise TimeoutError('next task and cleanup do not fit fixed panel deadline')
                started = time.monotonic_ns()
                runtime.atomic_json(output / 'worker-progress.json', {
                    'phase': 'task_started', 'run_id': job['run_id'], 'cell_id': job['cell_id'],
                    'repetition': job['repetition'], 'index': index, 'started_ns': started,
                    'task_timeout_seconds': job['task_timeout_seconds'],
                    'completed_populations': len(result)})
                saved = job['retained_tasks'].get(str(index))
                if saved is not None:
                    if retained_root is None:
                        raise ValueError('retained tasks require an explicit read-only source root')
                    task = _retained(saved, retained_root)
                    reused += 1
                else:
                    task = builder(tokenizer, '/model', spec['input_tokens'], spec['output_tokens'], suite,
                                   job['concurrency'], job['repetition'], index, spec['cache'],
                                   spec['block_size'], prompt_version=spec['prompt_version'])
                    new_tasks += 1
                verifier(task, tokenizer)
                if any(task[k] != v for k, v in {'suite': suite, 'session': index, 'concurrency': job['concurrency'],
                        'repetition': job['repetition'], 'target_input_tokens': spec['input_tokens'],
                        'output_tokens': spec['output_tokens'], 'cache_designation': spec['cache'],
                        'prompt_version': spec['prompt_version']}.items()):
                    raise ValueError('task checkpoint dimensions differ')
                if time.monotonic_ns() - started > job['task_timeout_seconds'] * 1e9:
                    raise TimeoutError('task generation exceeded original bound')
                task_file = population / f'task-{index:03d}.json'
                runtime.atomic_json(task_file, task)
                # Match runtime.atomic_json(list_of_tasks) byte-for-byte while
                # retaining only one task. No aggregate task list is allocated.
                if index:
                    stream.write(',\n')
                encoded = json.dumps(task, sort_keys=True, indent=2)
                stream.write('\n'.join('  ' + line for line in encoded.split('\n')))
                stream.flush()
                os.fsync(stream.fileno())
                runtime.atomic_json(population / f'task-{index:03d}-generation.json', {
                    'status': 'original_tokenizer_task_verified', 'task_sha256': sha(task_file),
                    'source_generator_sha256': spec['generator_sha256'],
                    'source_verifier_sha256': spec['verifier_sha256'],
                    'cell_id': job['cell_id'], 'repetition': job['repetition'], 'index': index,
                    'started_ns': started, 'finished_ns': time.monotonic_ns(), 'reused': saved is not None})
                runtime.atomic_json(output / 'worker-progress.json', {
                    'phase': 'task_committed', 'run_id': job['run_id'], 'index': index,
                    'finished_ns': time.monotonic_ns(), 'completed_populations': len(result)})
                del task, encoded
            stream.write('\n]\n')
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise ValueError('population output already exists')
        os.replace(temporary, target)
        row = {'run_id': job['run_id'], 'cell_id': job['cell_id'], 'repetition': job['repetition'],
               'tasks': job['concurrency'], 'tasks_sha256': sha(target), 'new_tasks': new_tasks,
               'reused_tasks': reused}
        runtime.atomic_json(population / 'population-result.json', row)
        result.append(row)
        runtime.atomic_json(output / 'completed-populations.json', result)
    runtime.atomic_json(output / 'worker-result.json', {
        'status': 'all_declared_tasks_generated_and_tokenizer_verified', 'populations': result,
        'new_tasks': sum(r['new_tasks'] for r in result), 'reused_tasks': sum(r['reused_tasks'] for r in result),
        'tokenizer_initializations': tokenizer_initializations, 'maximum_buffered_task_input_tokens': spec['input_tokens'],
        'existing_memory_cap_tokens': MAX_BUFFERED_INPUT_TOKENS,
        'scientific_acceptance': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', default='/model')
    parser.add_argument('--retained-root', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    validate(spec)
    validate_output(args.output)
    started = time.monotonic_ns()
    runtime.atomic_json(args.output / 'worker-progress.json', {'phase': 'tokenizer_initializing', 'started_ns': started})
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    tokenizer = DeepseekV41Tokenizer.from_pretrained(args.model, local_files_only=True)
    if time.monotonic_ns() - started > spec['initialization_timeout_seconds'] * 1e9:
        raise TimeoutError('tokenizer initialization exceeded declared bound')
    run(spec, args.output, tokenizer, retained_root=args.retained_root, tokenizer_initializations=1)


if __name__ == '__main__':
    main()
