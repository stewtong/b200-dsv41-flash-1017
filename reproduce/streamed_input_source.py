"""Project and audit explicit persistent-tokenizer input lineage after cleanup."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import input_panel_worker
import runtime
import verify_fixed_task

METHOD = 'persistent-tokenizer-stream/v1'


def read(path):
    return json.loads(Path(path).read_bytes())


def project(root, spec, config):
    root = Path(root)
    state = read(root/'execution-state.json')
    cleanup = read(root/'cleanup-result.json')
    idle = read(root/'gpu-idle-after-cleanup.json')
    result = read(root/'output/worker-result.json')
    if (not state['stopping'] or not state['watchdog_stop'] or cleanup.get('success') is not True or
            idle.get('idle') is not True or state['driver_identity'] != runtime.process_identity(os.getpid()) or
            any(runtime.process_alive(p) for p in [state['watchdog_identity'], *state['owned_processes']]) or
            result.get('tokenizer_initializations') != 1):
        raise ValueError('input projection requires exact dead writers and passed cleanup')
    output = root/'populations'
    output.mkdir()
    rows = []
    for job in spec['jobs']:
        source = root/'output'/job['run_id']
        target = output/('dsv41-'+job['run_id'])
        target.mkdir()
        child = dict(config, cell_id=job['cell_id'], repetition=job['repetition'],
                     generation_method=METHOD)
        # This is a new explicit generator artifact, never a fabricated result
        # for the original failed parent or an original node-prompt child.
        runtime.atomic_json(target/'run-config.json', child)
        for name in ('manifest.json', 'checkpoint-receipt.json', 'image-qualification.json'):
            runtime.atomic_bytes(target/name, (root/name).read_bytes())
        runtime.atomic_json(target/'prompt-result.json', {
            'status': 'passed', 'cleanup': 'passed', 'cell_id': job['cell_id'],
            'tasks': job['concurrency'], 'task_family': 'fixed', 'repetition': job['repetition'],
            'generation_method': METHOD, 'scope': 'persistent installed tokenizer; no inference acceptance'})
        proof = target/'streamed-generation'
        proof.mkdir()
        runtime.atomic_bytes(proof/'toolkit-MANIFEST.json', Path(config['toolkit_manifest']).read_bytes())
        for name in ('execution-state.json', 'cleanup-result.json', 'gpu-idle-after-cleanup.json',
                     'input-spec.json', 'input-worker-identity.json'):
            runtime.atomic_bytes(proof/name, (root/name).read_bytes())
        runtime.atomic_bytes(proof/'worker-result.json', (root/'output/worker-result.json').read_bytes())
        for file in source.iterdir():
            if file.is_symlink() or not file.is_file() or '.tmp.' in file.name:
                raise ValueError('population has an unfinished or unsafe checkpoint')
            dest = target/'fixed-tasks.json' if file.name == 'fixed-tasks.json' else proof/file.name
            shutil.copyfile(file, dest)
        runtime.atomic_json(proof/'lineage.json', {
            'schema': METHOD, 'run_id': config['run_id'], 'population_run_id': job['run_id'],
            'worker_source_sha256': input_panel_worker.sha(Path(input_panel_worker.__file__)),
            'generator_sha256': spec['generator_sha256'], 'verifier_sha256': spec['verifier_sha256'],
            'ownership_ids_are_scientific_ids': False, 'scientific_acceptance': False})
        runtime.seal_evidence(target)
        checked = verify(target)
        rows.append({'source': str(target), 'source_seal_sha256': input_panel_worker.sha(target/'SHA256SUMS'),
                     **checked})
    return rows


def verify(source):
    """Audit checkpoints without reinitializing an encoder or regenerating tasks."""
    source = Path(source)
    runtime.verify_evidence_manifest(source)
    cfg = read(source/'run-config.json')
    generation = read(source/'prompt-result.json')
    proof = source/'streamed-generation'
    lineage = read(proof/'lineage.json')
    spec = read(proof/'input-spec.json')
    state = read(proof/'execution-state.json')
    worker = read(proof/'worker-result.json')
    toolkit = read(proof/'toolkit-MANIFEST.json')
    manifest = read(source/'manifest.json')
    if (input_panel_worker.sha(proof/'toolkit-MANIFEST.json') != manifest['runtime']['toolkit_manifest_sha256'] or
            toolkit.get('reproduce/input_panel_worker.py') != lineage.get('worker_source_sha256') or
            toolkit.get('reproduce/fixed_workload.py') != lineage.get('generator_sha256') or
            toolkit.get('reproduce/verify_fixed_task.py') != lineage.get('verifier_sha256') or
            spec.get('generator_sha256') != lineage.get('generator_sha256') or
            spec.get('verifier_sha256') != lineage.get('verifier_sha256')):
        raise ValueError('persistent generator sources differ from the bound toolkit')
    selected = [j for j in spec['jobs'] if j['run_id'] == lineage.get('population_run_id')]
    if (len(selected) != 1 or lineage.get('schema') != METHOD or cfg.get('generation_method') != METHOD or
            generation.get('generation_method') != METHOD or lineage.get('scientific_acceptance') is not False):
        raise ValueError('persistent generation lineage or selected population differs')
    job = selected[0]
    if (state.get('run_id') != cfg['run_id'] or lineage.get('run_id') != cfg['run_id'] or
            state.get('stopping') is not True or state.get('watchdog_stop') is not True or
            read(proof/'cleanup-result.json').get('success') is not True or
            read(proof/'gpu-idle-after-cleanup.json').get('idle') is not True or
            read(proof/'input-worker-identity.json') not in state.get('owned_processes', []) or
            worker.get('status') != 'all_declared_tasks_generated_and_tokenizer_verified' or
            worker.get('tokenizer_initializations') != 1 or worker.get('scientific_acceptance') is not False or
            cfg['input_panel'] != {k: v for k, v in spec.items() if k != 'deadline_ns'} or
            cfg['cell_id'] != job['cell_id'] or cfg['repetition'] != job['repetition'] or
            generation.get('status') != 'passed' or generation.get('cleanup') != 'passed' or
            generation.get('tasks') != job['concurrency'] or job['retained_tasks']):
        raise ValueError('persistent generation ownership, cleanup or dimensions differ')
    population = read(proof/'population-result.json')
    if population not in worker['populations'] or any(population.get(k) != v for k, v in {
            'run_id': job['run_id'], 'cell_id': job['cell_id'], 'repetition': job['repetition'],
            'tasks': job['concurrency'], 'new_tasks': job['concurrency'], 'reused_tasks': 0}.items()):
        raise ValueError('persistent population completion differs')
    digest = hashlib.sha256(b'[\n')
    last = 0
    expected = {'lineage.json', 'execution-state.json', 'cleanup-result.json', 'gpu-idle-after-cleanup.json',
                'input-spec.json', 'input-worker-identity.json', 'worker-result.json', 'population-result.json',
                'toolkit-MANIFEST.json'}
    for index in range(job['concurrency']):
        task_name, receipt_name = f'task-{index:03d}.json', f'task-{index:03d}-generation.json'
        expected.update((task_name, receipt_name))
        task = read(proof/task_name)
        receipt = read(proof/receipt_name)
        verify_fixed_task.verify(task)
        if (receipt.get('status') != 'original_tokenizer_task_verified' or receipt.get('reused') is not False or
                receipt.get('task_sha256') != input_panel_worker.sha(proof/task_name) or
                receipt.get('source_generator_sha256') != spec['generator_sha256'] or
                receipt.get('source_verifier_sha256') != spec['verifier_sha256'] or
                receipt.get('cell_id') != job['cell_id'] or receipt.get('repetition') != job['repetition'] or
                receipt.get('index') != index or not last <= receipt['started_ns'] <= receipt['finished_ns'] <= spec['deadline_ns'] or
                receipt['finished_ns']-receipt['started_ns'] > job['task_timeout_seconds']*10**9 or
                task['concurrency'] != job['concurrency'] or task['repetition'] != job['repetition'] or
                task['session'] != index or task['target_input_tokens'] != spec['input_tokens'] or
                task['output_tokens'] != spec['output_tokens'] or task['cache_designation'] != spec['cache']):
            raise ValueError('persistent task checkpoint or original-generation binding differs')
        last = receipt['finished_ns']
        if index:
            digest.update(b',\n')
        digest.update('\n'.join('  '+line for line in json.dumps(task, sort_keys=True, indent=2).split('\n')).encode())
    digest.update(b'\n]\n')
    if ({p.name for p in proof.iterdir()} != expected or
            digest.hexdigest() != population['tasks_sha256'] or
            input_panel_worker.sha(source/'fixed-tasks.json') != digest.hexdigest()):
        raise ValueError('persistent task-array bytes or complete proof population differ')
    runtime.verify_evidence_manifest(source)
    return {'cell_id': job['cell_id'], 'repetition': job['repetition'], 'tasks': job['concurrency'],
            'generation_method': METHOD, 'tasks_sha256': digest.hexdigest(), 'scientific_acceptance': False}
