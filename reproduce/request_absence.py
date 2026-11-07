"""Validate all-rank work transitions for one exclusive cancellation control.

Frontend body release alone is insufficient. Require the same complete
scheduler population to transition from one running request to zero work.
"""
import math
import re

GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting')
SERIES = re.compile(r'^(vllm:num_requests_(?:running|waiting))\{([^}]*)\}\s+(\S+)\s*$')
ENGINE = re.compile(r'(?:^|,)\s*engine="([^"]+)"(?:,|$)')


def population(snapshot, expected):
    if (set(snapshot.get('metrics', {})) != set(expected) or
        set(snapshot.get('loads', {})) != {url.removesuffix('/metrics') + '/load' for url in expected}):
        raise ValueError('work-absence endpoint coverage incomplete')
    started = snapshot.get('started_ns'); ended = snapshot.get('monotonic_ns')
    if type(started) is not int or type(ended) is not int or not 0 <= started <= ended or ended - started > 2_500_000_000:
        raise ValueError('work-absence observation clock or collection bound invalid')
    totals = {'running': 0, 'waiting': 0, 'frontend': 0}
    for url, ranks in expected.items():
        if not ranks or len(set(map(str, ranks))) != len(ranks): raise ValueError('ambiguous expected work ranks')
        text = snapshot['metrics'][url]; rows = {}; types = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[:2] == ['#', 'TYPE'] and parts[2] in GAUGES:
                if parts[2] in types: raise ValueError('duplicate work metric type')
                types[parts[2]] = parts[3]
            match = SERIES.fullmatch(line)
            if not match and re.match(r'^vllm:num_requests_(?:running|waiting)(?:\s|\{)', line):
                raise ValueError('work metric has malformed or unlabelled population')
            if not match: continue
            rank = ENGINE.search(match[2])
            if rank is None: raise ValueError('work metric rank absent')
            key = (match[1], rank[1])
            value = float(match[3])
            if key in rows or not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError('invalid or duplicate running/queued work value')
            rows[key] = int(value)
        required = {(metric, str(rank)) for metric in GAUGES for rank in ranks}
        if set(rows) != required or types != dict.fromkeys(GAUGES, 'gauge'):
            raise ValueError('work metric type or all-rank population differs')
        totals['running'] += sum(value for (metric, _), value in rows.items() if metric == GAUGES[0])
        totals['waiting'] += sum(value for (metric, _), value in rows.items() if metric == GAUGES[1])
        load = snapshot['loads'][url.removesuffix('/metrics') + '/load']
        if set(load) != {'server_load'} or type(load['server_load']) is not int or load['server_load'] < 0:
            raise ValueError('frontend load evidence invalid')
        totals['frontend'] += load['server_load']
    return totals


def verify(before, busy, after, expected, *, client_terminal_ns, cancellation_ns, clock_id):
    if type(client_terminal_ns) is not int or type(cancellation_ns) is not int or cancellation_ns > client_terminal_ns:
        raise ValueError('client/cancellation timestamps invalid')
    if len(after) < 2: raise ValueError('two post-cancellation observations required')
    snapshots = [before, busy] + after
    if not clock_id or any(row.get('clock_id') != clock_id for row in snapshots):
        raise ValueError('work-absence clock domains differ')
    values = [population(row, expected) for row in snapshots]
    zero = {'running': 0, 'waiting': 0, 'frontend': 0}
    if values[0] != zero or values[1] != {'running': 1, 'waiting': 0, 'frontend': 1}:
        raise ValueError('exclusive zero-to-one live-work transition unproven')
    if (before['monotonic_ns'] > busy['started_ns'] or busy['monotonic_ns'] > cancellation_ns or
        after[0]['started_ns'] < client_terminal_ns + 1_500_000_000 or
        any(b['started_ns'] < a['monotonic_ns'] + 1_000_000_000 for a, b in zip(after, after[1:]))):
        raise ValueError('work-absence transition ordering or observation separation invalid')
    if any(value != zero for value in values[2:]):
        raise ValueError('owned frontend or scheduler work remains after client exit')
    return {'status': 'exclusive_request_work_absent', 'post_cancel_observations': len(after),
            'expected_ranks': sum(len(ranks) for ranks in expected.values()),
            'scope': 'same complete frontend/scheduler population transitioned zero to one to zero; request provenance, cancellation stage and engine health remain separate'}
