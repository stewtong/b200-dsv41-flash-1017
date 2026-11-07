"""Compare per-engine cache counters without treating missing series as zero."""
import math
import re

LINE = re.compile(r'^(vllm:prefix_cache_(?:queries|hits)_total)\{([^}]*)\}\s+([^\s]+)')
ENGINE = re.compile(r'(?:^|,)\s*engine="([^"]+)"(?:,|$)')

def counters(metrics, expected_ranks):
    result = {}
    if set(metrics) != set(expected_ranks): raise ValueError('endpoint coverage mismatch')
    if any(not ranks or len({str(r) for r in ranks}) != len(ranks) for ranks in expected_ranks.values()):
        raise ValueError('missing or duplicate expected ranks')
    for endpoint, text in metrics.items():
        for line in text.splitlines():
            match = LINE.match(line)
            if not match: continue
            rank = ENGINE.search(match.group(2))
            if rank is None: raise ValueError('cache counter missing engine label')
            key = (endpoint, rank.group(1), match.group(1))
            value = float(match.group(3))
            if not math.isfinite(value) or value < 0 or key in result: raise ValueError('invalid or duplicate counter')
            result[key] = value
    expected = {(url, str(rank), 'vllm:prefix_cache_'+kind+'_total')
                for url,ranks in expected_ranks.items() for rank in ranks for kind in ('queries','hits')}
    if set(result) != expected: raise ValueError('missing or unexpected per-rank cache counters')
    return result

def differences(before, after, expected_ranks):
    a, b = counters(before, expected_ranks), counters(after, expected_ranks)
    out = []
    for url, ranks in expected_ranks.items():
        for rank in ranks:
            q = (url,str(rank),'vllm:prefix_cache_queries_total')
            h = (url,str(rank),'vllm:prefix_cache_hits_total')
            queries, hits = b[q]-a[q], b[h]-a[h]
            if queries < 0 or hits < 0 or hits > queries: raise ValueError('reset or inconsistent cache counters')
            out.append({'endpoint':url,'rank':str(rank),'queries':queries,'hits':hits})
    return out

def difference(before, after, expected_ranks):
    out = differences(before, after, expected_ranks)
    active = [r for r in out if r['queries'] > 0]
    if len(active) != 1: raise ValueError('cannot attribute request to exactly one rank')
    return {'ranks':out, 'selected':active[0], 'hit_fraction':active[0]['hits']/active[0]['queries']}


def has_cache_metric(metrics):
    """Fallback is allowed only when every endpoint lacks the metric entirely."""
    return any(line.lstrip().startswith('vllm:cache_config_info') or
               re.match(r'^\s*#\s*(?:HELP|TYPE|UNIT)\s+vllm:cache_config_info(?:\s|$)',line)
               for text in metrics.values() for line in text.splitlines())


def cache_block_size(metrics):
    """Read a consistent enabled-cache block size from each live engine endpoint."""
    if not metrics: raise ValueError('cache configuration endpoints missing')
    found={}
    for endpoint,text in metrics.items():
        sizes=[]
        for line in text.splitlines():
            if not line.startswith('vllm:cache_config_info{'): continue
            match=re.fullmatch(r'vllm:cache_config_info\{([^}]*)\}\s+1(?:\.0+)?',line.strip())
            if not match: raise ValueError('invalid cache configuration metric')
            pairs=re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="([^"\\]*)"',match.group(1))
            labels=dict(pairs)
            if len(labels)!=len(pairs) or labels.get('enable_prefix_caching','').lower()!='true':
                raise ValueError('prefix caching not verified')
            value=labels.get('block_size','')
            if not value.isdigit() or int(value)<=0: raise ValueError('cache block size missing')
            sizes.append(int(value))
        if not sizes or len(set(sizes))!=1: raise ValueError('missing or inconsistent cache block size')
        found[endpoint]=sizes[0]
    if len(set(found.values()))!=1: raise ValueError('cache block size differs across engines')
    return {'block_size':next(iter(found.values())),'endpoints':found,
            'scope':'live cache configuration; per-request cache hits require counter evidence'}


def cache_block_size_from_logs(logs, workers_per_engine):
    """Read the explicit setting from each worker's retained startup log."""
    if not logs or not isinstance(workers_per_engine,int) or workers_per_engine<=0:
        raise ValueError('startup log coverage missing')
    found={}
    pattern=re.compile(r'\((Worker_[^\s()]+)\s+pid=\d+\).*?Setting kv cache block size to (\d+) for ')
    for container,text in logs.items():
        if not re.search(r"non-default args:.*'enable_prefix_caching': True",text):
            raise ValueError('prefix-caching startup declaration missing')
        workers={}
        for line in text.splitlines():
            match=pattern.search(line)
            if not match: continue
            name,size=match.group(1),int(match.group(2))
            if size<=0 or (name in workers and workers[name]!=size):
                raise ValueError('invalid or inconsistent worker cache block size')
            workers[name]=size
        if len(workers)!=workers_per_engine or len(set(workers.values()))!=1:
            raise ValueError('worker cache block-size coverage incomplete')
        found[container]=next(iter(workers.values()))
    if len(set(found.values()))!=1: raise ValueError('cache block size differs across engines')
    return {'block_size':next(iter(found.values())),'containers':found,'source':'retained startup logs',
            'scope':'startup declarations from distinct workers; per-request cache hits require counter evidence'}
