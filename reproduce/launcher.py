#!/usr/bin/env python3
"""Render candidate launch arguments. This module never starts containers."""
import argparse
import hashlib
import json
import re
from pathlib import PurePosixPath

LAYOUTS = {
    '2xtp4': (4, 1, False),
    '4xdep2': (1, 2, True),
    '2xdep4': (1, 4, True),
    'dep8': (1, 8, True),
    'tp8': (8, 1, False),
    '4xtp2': (2, 1, False),
}
BACKENDS = {'2xtp4': 'FLASHINFER_MLA_SPARSE_DSV41',
            '4xtp2': 'FLASHINFER_MLA_SPARSE_DSV41',
            'tp8': 'FLASHINFER_MLA_SPARSE_DSV41',
            '4xdep2': 'FLASHINFER_MLA_SPARSE_DSV41',
            '2xdep4': 'FLASHINFER_MLA_SPARSE_DSV41',
            'dep8': 'FLASHMLA_MEGA_ATTN_DSV41'}
SEQ_LIMITS = {'2xtp4':64, '4xdep2':32, '2xdep4':64, 'dep8':128, 'tp8':128, '4xtp2':8}

def validate_run_id(run_id):
    if not isinstance(run_id,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',run_id):
        raise ValueError('invalid run ID')


def render(layout, image, run_id, model_dir, cache_dir, server_cpus=None, overrides=None):
    if layout not in LAYOUTS:
        raise ValueError('unknown layout')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*@sha256:[0-9a-f]{64}', image):
        # The image name can contain registry and repository separators.
        if not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}', image):
            raise ValueError('image must be pinned by sha256 digest')
    validate_run_id(run_id)
    for p in (model_dir, cache_dir):
        if not PurePosixPath(p).is_absolute() or ',' in p or '\n' in p or '..' in PurePosixPath(p).parts:
            raise ValueError('mount paths must be absolute and contain no comma or parent traversal')
    if server_cpus is not None and not re.fullmatch(r'\d+(,\d+)*', server_cpus):
        raise ValueError('invalid server CPU list')
    tp, dp, expert = LAYOUTS[layout]
    width = tp * dp
    groups = [list(range(i, i + width)) for i in range(0, 8, width)]
    commands = []
    for i, gpus in enumerate(groups):
        port = 31000 + i
        cmd = ['docker', 'run', '-d', '--restart', 'no', '--name', f'dsv41-{layout}-{i}',
               '--label', 'campaign=dsv41-1017', '--label', f'run_id={run_id}',
               '--gpus', '"device=' + ','.join(map(str, gpus)) + '"',
               '--ipc=host', '-p', f'127.0.0.1:{port}:8000',
               '--mount', f'type=bind,src={model_dir},dst=/model,readonly',
               '--mount', f'type=bind,src={cache_dir},dst=/root/.cache/vllm',
               '-e', 'VLLM_ENGINE_READY_TIMEOUT_S=3600', '-e', 'VLLM_USE_V2_MODEL_RUNNER=1',
               '-e', 'VLLM_USE_RUST_FRONTEND=1', image,
               '--model', '/model', '--host', '0.0.0.0', '--port', '8000', '--language-model-only',
               '--tokenizer-mode', 'deepseek_v41', '--tensor-parallel-size', str(tp),
               '--kv-cache-dtype', 'fp8_ds_mla' if BACKENDS[layout]=='FLASHMLA_MEGA_ATTN_DSV41' else 'fp8', '--max-model-len', '1048576', '--max-num-seqs', str(SEQ_LIMITS[layout]),
               '--max-num-batched-tokens', '8192', '--enable-chunked-prefill', '--gpu-memory-utilization', '0.90', '--enable-prefix-caching',
               '--reasoning-parser', 'deepseek_v41', '--tool-call-parser', 'deepseek_v41', '--enable-auto-tool-choice',
               '--compilation-config', '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}', '--engram-config',
               '{"cpu_offload":true,"embedding_across_dp":true}' if expert else '{"cpu_offload":true}',
               '--attention-config', json.dumps({'backend': BACKENDS[layout], 'indexer_kv_dtype':'mxfp4', 'indexer_sparse_logits':True}, separators=(',', ':'))]
        if server_cpus is not None:
            cmd[3:3] = ['--cpuset-cpus', server_cpus]
        if expert:
            cmd += ['--data-parallel-size', str(dp), '--data-parallel-size-local', str(dp),
                    '--enable-expert-parallel', '--kernel-config',
                    '{"enable_flashinfer_autotune":true,"moe_backend":"deep_gemm_mega_moe"}']
        commands.append({'gpu_ids': gpus, 'host_port': port, 'argv': cmd})
    router = ['vllm-router', '--host', '127.0.0.1', '--port', '31100',
              '--worker-urls'] + [f'http://127.0.0.1:{c["host_port"]}' for c in commands] + [
                  '--policy', 'consistent_hash', '--intra-node-data-parallel-size', str(dp),
                  '--request-timeout-secs', '14400', '--disable-retries']
    plan = {'status': 'candidate_unqualified', 'layout': layout, 'run_id': run_id,
            'limitations': ['render only; no processes started', 'flags, image and router require source and GPU qualification',
                            'no readiness, ownership, watchdog or cleanup implementation'],
            'commands': commands, 'router_argv': router}
    if overrides is not None:
        import profile_overrides
        return profile_overrides.apply(plan,overrides)
    return plan

def profile_sha256(layout,overrides=None):
    """Hash the complete candidate command structure using fixed path and identity placeholders."""
    plan = render(layout, 'example/image@sha256:' + '0'*64, 'profile', '/model-source', '/cache-source', '0,1',overrides=overrides)
    plan['cpu_policy'] = 'two-numa-four-physical-cores-per-node-v1'
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', required=True, choices=LAYOUTS)
    parser.add_argument('--image', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--overrides-file', help='Unqualified JSON engine candidate; starts no containers')
    a = parser.parse_args()
    overrides=json.loads(__import__('pathlib').Path(a.overrides_file).read_text()) if a.overrides_file else None
    print(json.dumps(render(a.layout, a.image, a.run_id, a.model_dir, a.cache_dir,overrides=overrides), indent=2))

if __name__ == '__main__':
    main()
