"""Verify retained installed configuration and worker cache-backend selections."""
import json
import re


def verify(text, backend, tp, dp, packed_format=None, expert=False, kv_dtype=None,expected_spec=None,expected_moe='deep_gemm_mega_moe'):
    configs = []
    for line in text.splitlines():
        if 'Launching Rust frontend:' in line and '--args-json ' in line:
            value, _ = json.JSONDecoder().raw_decode(line.split('--args-json ', 1)[1])
            configs.append(value)
    if len(configs) != 1:
        raise ValueError('exactly one installed configuration dump required')
    config = configs[0]
    actual_spec=config.get('speculative_config')
    if expected_spec is None and actual_spec is not None:
        raise ValueError('pilot candidate unexpectedly enables speculation')
    if expected_spec is not None:
        if (not isinstance(actual_spec,dict) or actual_spec.get('method')!='dspark' or
            actual_spec.get('rejection_sample_method')!='block' or
            any(re.search(r'golden|synthetic|forced.accept|fake.accept',key,re.I) for key in actual_spec) or
            any(key not in actual_spec or type(actual_spec[key]) is not type(value) or actual_spec[key]!=value for key,value in expected_spec.items())):
            raise ValueError('installed speculative configuration differs from registered real-block candidate')
    if kv_dtype is not None and config.get('kv_cache_dtype') != kv_dtype:
        raise ValueError('installed KV dtype differs from candidate')
    selected = config.get('attention_config', {})
    if not re.search(r'(?:\.|^)' + re.escape(backend) + r'(?::|$)', str(selected.get('backend'))):
        raise ValueError('installed attention backend differs from candidate')
    if selected.get('indexer_kv_dtype') != 'mxfp4':
        raise ValueError('installed indexer dtype differs from candidate')
    sizes={'tensor_parallel_size':tp,'data_parallel_size':dp,'data_parallel_size_local':dp}
    # The frontend serializes non-default scalar arguments only. Use the
    # worker's actual distributed world size as well; never fill absent args
    # from a guessed runtime default.
    worlds={int(x) for x in re.findall(r'world_size=(\d+)\s+rank=',text)}
    if (worlds != {tp*dp} or any(key in config and config[key] != expected
                                for key,expected in sizes.items())):
        raise ValueError('installed parallelism differs from candidate')
    if expert and (config.get('enable_expert_parallel') is not True or
                   config.get('kernel_config', {}).get('moe_backend') != expected_moe):
        raise ValueError('installed expert backend differs from candidate')
    workers = {}
    for name, size, backends in re.findall(
            r'\((Worker_[^\s()]+)\s+pid=\d+\).*?Setting kv cache block size to (\d+) for ([^\n]+?) backend\(s\)', text):
        observation = (int(size), backends.split('/')[0])
        if name in workers and workers[name] != observation:
            raise ValueError('worker backend changed')
        workers[name] = observation
    if len(workers) != tp * dp or any(size <= 0 or chosen != backend for size, chosen in workers.values()):
        raise ValueError('worker backend coverage incomplete or conflicting')
    formats = set(re.findall(r"Using DeepSeek's ([a-z0-9_]+) KV cache format", text))
    if packed_format is not None and formats != {packed_format}:
        raise ValueError('resolved packed cache format differs from candidate')
    return {'status': 'configured_backend_confirmed', 'backend': backend,
            'configured_kv_dtype':config.get('kv_cache_dtype'), 'workers': sorted(workers), 'resolved_packed_formats': sorted(formats),
            'kernel_execution_profiled': False,
            'scope': 'installed configuration and worker cache-backend selection; completed semantic request is separate evidence, not a kernel trace'}
