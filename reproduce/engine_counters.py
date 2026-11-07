"""Audit pinned engine work counters without treating cached tokens as compute.

Request phase durations are sums of overlapping per-request observations. They
are not node wall time and cannot establish isolated prefill/decode throughput.
"""
import math
import re

LOGGER_SHA256='a17291f044251f1586457b6bd1542f98660724dd07a426d61ce852186b95b9e1'
REASONS=('stop','length','abort','error','repetition')
SOURCES=('local_compute','local_cache_hit','external_kv_transfer')
COUNTERS=('prompt_tokens','prompt_tokens_cached','generation_tokens','num_preemptions','corrupted_requests')
HISTOGRAMS=('request_prompt_tokens','request_generation_tokens','request_prefill_time_seconds','request_decode_time_seconds')


class UnsettledPopulationError(ValueError):
    """A complete scrape caught counter updates between a request's fields."""


def contract(ranks):
    result={}
    for rank in ranks:
        for name in COUNTERS:result[('vllm:'+name+'_total',str(rank),None)]=('counter',True)
        for source in SOURCES:result[('vllm:prompt_tokens_by_source_total',str(rank),source)]=('counter',True)
        for reason in REASONS:result[('vllm:request_success_total',str(rank),reason)]=('counter',True)
        for name in HISTOGRAMS:
            for suffix in ('sum','count'):
                result[('vllm:'+name+'_'+suffix,str(rank),None)]=('histogram',suffix=='count' or name.endswith('_tokens'))
    return result


def snapshot(text,ranks):
    if (not isinstance(text,str) or not ranks or len(set(map(str,ranks)))!=len(ranks) or
            any(not str(r).isdigit() for r in ranks)):
        raise ValueError('explicit unique engine ranks required')
    expected=contract(ranks);names={k[0] for k in expected};types={};values={}
    for line in text.splitlines():
        match=re.fullmatch(r'# TYPE (\S+) (\S+)',line)
        if match:
            name,kind=match.groups()
            if name in types and types[name]!=kind:raise ValueError('conflicting engine counter type')
            types[name]=kind
        if line.startswith('#') or not line.strip():continue
        name=line.split('{',1)[0].split(' ',1)[0]
        if name not in names:continue
        match=re.fullmatch(r'(\S+?)\{([^}]*)\}\s+(\S+)',line)
        if not match:raise ValueError('malformed required engine counter')
        name,labels,number=match.groups()
        pairs=re.findall(r'(\w+)="([^"\\]*)"',labels)
        if ','.join(k+'="'+v+'"' for k,v in pairs)!=labels or len({k for k,v in pairs})!=len(pairs):
            raise ValueError('malformed or duplicate engine counter labels')
        labels=dict(pairs);dimension='source' if name=='vllm:prompt_tokens_by_source_total' else 'finished_reason' if name=='vllm:request_success_total' else None
        if set(labels)!={'model_name','engine'}|({dimension} if dimension else set()) or labels['model_name']!='/model':
            raise ValueError('unexpected engine counter labels or model')
        key=(name,labels['engine'],labels.get(dimension))
        if key not in expected or key in values:raise ValueError('unknown or duplicate engine counter series')
        value=float(number)
        if not math.isfinite(value) or value<0 or expected[key][1] and not value.is_integer():
            raise ValueError('invalid engine counter value')
        base=name.removesuffix('_total') if expected[key][0]=='counter' else name.rsplit('_',1)[0]
        if types.get(base)!=expected[key][0]:raise ValueError('engine counter type missing or wrong')
        values[key]=int(value) if expected[key][1] else value
    if set(values)!=set(expected):raise ValueError('missing engine counter population')
    return values


def differences(before,after,ranks):
    if set(before)!=set(ranks) or set(after)!=set(ranks):raise ValueError('engine counter endpoints differ')
    rows=[]
    for endpoint,engines in sorted(ranks.items()):
        old=snapshot(before[endpoint],engines);new=snapshot(after[endpoint],engines)
        delta={key:new[key]-old[key] for key in old}
        if any(v<0 for v in delta.values()):raise ValueError('engine counter reset or clock population mismatch')
        for rank in engines:
            rank=str(rank)
            get=lambda name,dimension=None:delta[('vllm:'+name,rank,dimension)]
            sources={source:get('prompt_tokens_by_source_total',source) for source in SOURCES}
            finished={reason:get('request_success_total',reason) for reason in REASONS}
            counts={name:get(name+'_count') for name in HISTOGRAMS}
            if (get('prompt_tokens_total')!=sum(sources.values()) or
                    get('prompt_tokens_cached_total')!=sources['local_cache_hit']+sources['external_kv_transfer'] or
                    set(counts.values())!={sum(finished.values())}):
                raise UnsettledPopulationError('engine counter source or completed-request population is not settled')
            rows.append({'endpoint':endpoint,'rank':rank,'prompt_tokens_by_source':sources,
                'prompt_tokens_total':get('prompt_tokens_total'),'generation_tokens':get('generation_tokens_total'),
                'finished_by_reason':finished,'successfully_finished_requests':finished['stop']+finished['length'],
                'preemptions':get('num_preemptions_total'),'corrupted_requests':get('corrupted_requests_total'),
                'finished_request_prompt_tokens':get('request_prompt_tokens_sum'),
                'finished_request_generation_tokens':get('request_generation_tokens_sum'),
                'request_prefill_seconds_sum':get('request_prefill_time_seconds_sum'),
                'request_decode_seconds_sum':get('request_decode_time_seconds_sum')})
    return {'status':'pinned_engine_counter_deltas_verified','logger_source_sha256':LOGGER_SHA256,'ranks':rows,
        'scope':'aggregate engine work and completed-request phases; includes failed-request work, excludes isolated phase throughput, exact queued IDs, indexer/host occupancy and complete scored telemetry qualification'}
