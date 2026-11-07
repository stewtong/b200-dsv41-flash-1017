"""Derive burst metrics from independently scored request records."""
import math

def burst(rows):
    if not rows:
        raise ValueError('empty burst')
    ids = set()
    for row in rows:
        for key in ('request_id', 'start_ns', 'end_ns', 'completion_tokens', 'valid', 'clock_id'):
            if key not in row: raise ValueError('missing ' + key)
        if not isinstance(row['request_id'],str) or not row['request_id'] or not isinstance(row['clock_id'],str) or not row['clock_id']:
            raise ValueError('invalid request or clock identity')
        if row['request_id'] in ids: raise ValueError('duplicate request')
        ids.add(row['request_id'])
        if (any(type(row[k]) is not int or row[k] < 0 for k in ('start_ns','end_ns')) or row['start_ns'] > row['end_ns'] or
            (row['completion_tokens'] is not None and (type(row['completion_tokens']) is not int or row['completion_tokens'] < 0))):
            raise ValueError('invalid clock or token count')
        if not isinstance(row['valid'], bool): raise ValueError('validity must be boolean')
        if row['valid'] and row['completion_tokens'] is None: raise ValueError('valid request missing token count')
    if len({r['clock_id'] for r in rows}) != 1:
        raise ValueError('mixed clock domains')
    duration = (max(r['end_ns'] for r in rows) - min(r['start_ns'] for r in rows)) / 1e9
    if not math.isfinite(duration) or duration <= 0: raise ValueError('invalid makespan')
    valid_tokens = sum(r['completion_tokens'] for r in rows if r['valid'])
    invalid = [r['completion_tokens'] for r in rows if not r['valid']]
    partial_tokens = None if any(x is None for x in invalid) else sum(invalid)
    return {'duration_seconds': duration, 'valid_tokens': valid_tokens,
            'invalid_partial_tokens': partial_tokens, 'valid_tokens_per_second': valid_tokens / duration,
            'valid_requests': sum(r['valid'] for r in rows), 'requests': len(rows)}

def stream_window(first_ns, last_ns, tokens):
    if first_ns < 0 or last_ns < first_ns or tokens < 0: raise ValueError('invalid stream window')
    if tokens <= 1: return None
    return (last_ns - first_ns) / 1e9 / (tokens - 1)

def slo_bucket(ttfts, threshold_seconds, completion_fraction, rate_fraction, queue_growth):
    """One bucket's SLO disposition; p95 requires at least 20 observations."""
    if (not math.isfinite(threshold_seconds) or threshold_seconds <= 0 or
        not math.isfinite(completion_fraction) or not 0 <= completion_fraction <= 1 or
        not math.isfinite(rate_fraction) or rate_fraction < 0 or queue_growth not in (True, False, None)):
        raise ValueError('invalid SLO inputs')
    if any(not math.isfinite(x) or x < 0 for x in ttfts): raise ValueError('invalid latency')
    if completion_fraction < 0.99 or rate_fraction < 0.90 or queue_growth is True:
        return {'verdict': 'failed', 'reason': 'completion, rate or queue rule'}
    if queue_growth is None or len(ttfts) < 20:
        return {'verdict': 'inconclusive', 'reason': 'missing queue coverage or fewer than 20 observations'}
    # Nearest-rank percentile; the convention must match the frozen manifest.
    value = sorted(ttfts)[math.ceil(0.95 * len(ttfts)) - 1]
    return {'verdict': 'passed' if value < threshold_seconds else 'failed', 'p95_seconds': value}
