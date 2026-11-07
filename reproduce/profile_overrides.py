"""Render registered engine candidates without changing ownership or GPU groups.

Field names come from the pinned installed config classes. Rendering never
establishes installed compatibility, fit, quality or performance qualification.
"""
import copy
import json
import math
import re

ALLOWED={'speculative_config','compilation_config','attention_backend','kv_cache_dtype',
         'engram_config','kernel_config','gpu_memory_utilization','max_num_batched_tokens',
         'max_num_seqs','prefill_schedule_interval','per_request_spec_decode_metrics','router_timeout_seconds',
         'hisparse_host_pool_gib','worker_observation','enable_prefix_caching'}


def integer(value,maximum=None,minimum=1):
    return type(value) is int and value>=minimum and (maximum is None or value<=maximum)


def validate(delta):
    if not isinstance(delta,dict) or set(delta)-ALLOWED:raise ValueError('unknown candidate profile field')
    json.dumps(delta,allow_nan=False)
    if 'enable_prefix_caching' in delta and type(delta['enable_prefix_caching']) is not bool:
        raise ValueError('prefix caching override must be boolean')
    if 'worker_observation' in delta:
        import measurement_launch
        measurement_launch.validate(delta['worker_observation'])
    for key,maximum in [('max_num_batched_tokens',1048576),('max_num_seqs',128),
                        ('prefill_schedule_interval',None),('router_timeout_seconds',None)]:
        if key in delta and not integer(delta[key],maximum):raise ValueError('invalid candidate integer')
    if 'gpu_memory_utilization' in delta:
        value=delta['gpu_memory_utilization']
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=1:raise ValueError('invalid GPU memory fraction')
    if 'hisparse_host_pool_gib' in delta and (type(delta['hisparse_host_pool_gib']) is not int or delta['hisparse_host_pool_gib']!=32):
        raise ValueError('registered HiSparse challenger uses a 32 GiB host pool')
    if 'attention_backend' in delta and not re.fullmatch(r'[A-Z][A-Z0-9_]+',str(delta['attention_backend'])):
        raise ValueError('invalid candidate attention backend')
    if 'kv_cache_dtype' in delta and delta['kv_cache_dtype'] not in ('fp8','fp8_ds_mla','nvfp4_ds_mla'):
        raise ValueError('unsupported candidate KV precision')
    if 'per_request_spec_decode_metrics' in delta and delta['per_request_spec_decode_metrics'] not in ('none','summary','detailed'):
        raise ValueError('invalid speculative metric level')
    spec=delta.get('speculative_config')
    if spec is not None:
        fields={'method','num_speculative_tokens','rejection_sample_method','enable_adaptive_verification',
                'draft_sample_method','num_speculative_tokens_per_batch_size','model'}
        if not isinstance(spec,dict):raise ValueError('invalid speculative config')
        if any(re.search(r'golden|synthetic|forced.accept|fake.accept',k,re.I) for k in spec):
            raise ValueError('synthetic acceptance override forbidden')
        if (set(spec)-fields or spec.get('method')!='dspark' or spec.get('rejection_sample_method')!='block' or
            not integer(spec.get('num_speculative_tokens'),64) or spec.get('model') not in (None,'/model')):
            raise ValueError('DSpark candidate requires registered fields, the pinned checkpoint and real block rejection')
        if 'enable_adaptive_verification' in spec and type(spec['enable_adaptive_verification']) is not bool:
            raise ValueError('invalid adaptive verification switch')
        if 'draft_sample_method' in spec and spec['draft_sample_method'] not in ('greedy','probabilistic'):
            raise ValueError('invalid draft sampling method')
        ranges=spec.get('num_speculative_tokens_per_batch_size')
        if ranges is not None:
            if not isinstance(ranges,list) or not ranges:raise ValueError('invalid draft-length ranges')
            last=0
            for row in ranges:
                if (not isinstance(row,(list,tuple)) or len(row)!=3 or
                    not integer(row[0],128) or not integer(row[1],128) or not row[0]<=row[1] or row[0]<=last or
                    not integer(row[2],spec['num_speculative_tokens'])):raise ValueError('invalid or overlapping draft-length ranges')
                last=row[1]
    if 'engram_config' in delta:
        value=delta['engram_config']
        if (not isinstance(value,dict) or not value or set(value)-{'cpu_offload','embedding_across_dp','use_thp'} or
            any(type(v) is not bool for v in value.values())):raise ValueError('invalid registered Engram candidate')
    if 'kernel_config' in delta:
        value=delta['kernel_config'];fields={'moe_backend','monokernel','sparse_indexer_topk_backend','enable_flashinfer_autotune'}
        if not isinstance(value,dict) or not value or set(value)-fields:raise ValueError('unknown candidate kernel field')
        for k,v in value.items():
            if k in ('monokernel','enable_flashinfer_autotune'):
                if type(v) is not bool:raise ValueError('invalid kernel switch')
            elif not isinstance(v,str) or not re.fullmatch(r'[a-z0-9_]+',v):raise ValueError('invalid kernel backend')
    if 'compilation_config' in delta:
        value=delta['compilation_config']
        fields={'mode','cudagraph_mode','cudagraph_capture_sizes','max_cudagraph_capture_size',
                'cudagraph_decoder_replay','decoder_replay_cudagraph_capture_sizes','decoder_replay_trim_threshold'}
        if not isinstance(value,dict) or not value or set(value)-fields:raise ValueError('unknown candidate compilation field')
        for k,v in value.items():
            if k in ('mode','cudagraph_mode'):
                if not isinstance(v,str) or not re.fullmatch(r'[A-Z][A-Z0-9_]+',v):raise ValueError('invalid compilation mode')
            elif k.endswith('capture_sizes'):
                if not isinstance(v,list) or any(not integer(n,1048576) for n in v) or sorted(set(v))!=v:raise ValueError('invalid graph capture sizes')
            elif k=='cudagraph_decoder_replay':
                if type(v) is not bool:raise ValueError('invalid graph replay switch')
            elif not integer(v,1048576,0 if k=='decoder_replay_trim_threshold' else 1):raise ValueError('invalid graph capture bound')


def set_option(argv,flag,value,default=None):
    if argv.count(flag)>1:raise ValueError('ambiguous renderer flag')
    if flag in argv:
        i=argv.index(flag)+1
        if type(value) in (int,float) and float(argv[i])==value:return
        argv[i]=str(value)
    elif value!=default:argv.extend([flag,str(value)])


def merge_json(argv,flag,delta):
    current=json.loads(argv[argv.index(flag)+1]) if flag in argv else {}
    current.update(delta);set_option(argv,flag,json.dumps(current,separators=(',',':')))


def apply(plan,delta):
    validate(delta)
    # JSON persistence may reorder object keys. Render every candidate from the
    # same canonical order so its exact argv and profile hash survive read-back.
    delta=json.loads(json.dumps(delta,allow_nan=False,sort_keys=True,separators=(',',':')))
    result=copy.deepcopy(plan)
    if 'hisparse_host_pool_gib' in delta and plan.get('layout') not in ('4xdep2','2xdep4'):
        raise ValueError('registered host-tier challenger applies only to DEP2 and DEP4')
    for command in result['commands']:
        argv=command['argv']
        if 'enable_prefix_caching' in delta:
            flags=('--enable-prefix-caching','--disable-prefix-caching')
            if sum(argv.count(flag) for flag in flags)!=1:raise ValueError('prefix caching flag is ambiguous')
            index=next(i for i,value in enumerate(argv) if value in flags)
            argv[index]='--enable-prefix-caching' if delta['enable_prefix_caching'] else '--disable-prefix-caching'
        for key in ('max_num_batched_tokens','max_num_seqs','gpu_memory_utilization'):
            if key in delta:set_option(argv,'--'+key.replace('_','-'),delta[key])
        if 'prefill_schedule_interval' in delta:set_option(argv,'--prefill-schedule-interval',delta['prefill_schedule_interval'],1)
        if 'per_request_spec_decode_metrics' in delta:set_option(argv,'--per-request-spec-decode-metrics',delta['per_request_spec_decode_metrics'],'none')
        if 'speculative_config' in delta and delta['speculative_config'] is None and '--speculative-config' in argv:
            index=argv.index('--speculative-config');del argv[index:index+2]
        if delta.get('speculative_config') is not None:
            set_option(argv,'--speculative-config',json.dumps(delta['speculative_config'],separators=(',',':')))
        for key in ('compilation_config','engram_config','kernel_config'):
            if key in delta:merge_json(argv,'--'+key.replace('_','-'),delta[key])
        if 'attention_backend' in delta:
            merge_json(argv,'--attention-config',{'backend':delta['attention_backend']})
            if delta['attention_backend']=='FLASHMLA_MEGA_ATTN_DSV41' and 'kv_cache_dtype' not in delta:
                set_option(argv,'--kv-cache-dtype','fp8_ds_mla')
        if 'kv_cache_dtype' in delta:set_option(argv,'--kv-cache-dtype',delta['kv_cache_dtype'])
        if 'hisparse_host_pool_gib' in delta:
            set_option(argv,'--kv-transfer-config',json.dumps({'kv_connector':'HiSparseConnector','kv_role':'kv_both',
                'kv_connector_extra_config':{'host_pool_gib':delta['hisparse_host_pool_gib']}},separators=(',',':')))
            merge_json(argv,'--attention-config',{'hisparse_config':{}})
    if 'router_timeout_seconds' in delta:set_option(result['router_argv'],'--request-timeout-secs',delta['router_timeout_seconds'])
    if 'worker_observation' in delta:
        import measurement_launch
        result=measurement_launch.template(result,delta['worker_observation'])
    return result
