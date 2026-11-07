"""Verify public result checksums, coverage and available aggregate arithmetic."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import runtime
import verify_comparison_tables


LAYOUTS = {'2xtp4', '4xdep2', '2xdep4', 'dep8'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12),
            'aggregate differs from published values')


def supplement(root):
    summary = json.loads((root / 'summary.json').read_text())
    with (root / 'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(str(output), layout, cache, str(c))
                for output in (512, 2048) for layout in LAYOUTS
                for cache, clients in [('cold', (1, 2, 4, 8, 16, 32, 64, 128)),
                                       ('warm_prefix', (1, 8, 32))]
                for c in clients}
    keys = [(x['output_tokens'], x['layout'], x['cache'], x['concurrency']) for x in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'supplement coverage differs')
    for x in rows:
        require(x['verdict'] == 'accepted' and int(x['repetitions']) == 3 and
                int(x['requests']) == 3 * int(x['concurrency']), 'invalid repetition denominator')
        lo, median, hi = [float(x['burst_output_tokens_per_second_' + k]) for k in ('min', 'p50', 'max')]
        require(all(math.isfinite(v) for v in (lo, median, hi)) and 0 < lo <= median <= hi,
                'invalid rate spread')
    new = [x for x in rows if x['output_tokens'] == '2048']
    require(len(new) == summary['supplement_accepted_cells'] == summary['supplement_required_cells'] == 44,
            'supplement summary coverage differs')
    require(sum(int(x['repetitions']) for x in new) == summary['supplement_repetitions'] == 132 and
            sum(int(x['requests']) for x in new) == summary['supplement_requests'] == 3552 and
            summary['supplement_complete'], 'supplement totals differ')
    baseline_path = root.parent / 'fixed-comparison' / 'tables' / 'cells.csv'
    if baseline_path.exists():
        with baseline_path.open() as stream:
            base = {(x['layout'], x['cache'], x['concurrency']): x
                    for x in csv.DictReader(stream) if x['input_tokens'] == '1000000'}
        for x in rows:
            if x['output_tokens'] != '512':
                continue
            old = base[x['layout'], x['cache'], x['concurrency']]
            for short, column in [('p50', 'burst_rate_p50'), ('min', 'burst_rate_min'), ('max', 'burst_rate_max')]:
                near(x['burst_output_tokens_per_second_' + short], old[column])
    return {'cells': 44, 'repetitions': 132, 'requests': 3552,
            'scope': 'Aggregate coverage, spread and baseline joins only; new raw-stream audit remains private.'}


def tuning(root):
    data = json.loads((root / 'comparisons.json').read_text())
    seen = set()
    repetitions = requests = reserved = qualification = 0
    for experiment in data['experiments']:
        name = experiment['experiment']
        require(name not in seen and name in {'dspark', 'graph', 'prefill'}, 'duplicate or unknown experiment')
        seen.add(name)
        require(experiment['status'] in {'inconclusive', 'rejected', 'accepted'}, 'unknown disposition')
        lengths = (128000, 256000) if name == 'prefill' else (8000, 64000)
        pairs = experiment['pairs']
        keys = [(x['input_tokens'], x['concurrency']) for x in pairs]
        require(len(keys) == 4 and len(set(keys)) == 4 and
                set(keys) == {(length, c) for length in lengths for c in (1, 8)}, 'missing matched workload')
        for pair in pairs:
            require(set(pair['metrics']) == {'ttft', 'end_to_end', 'throughput', 'decode_tpot'},
                    'missing performance metric')
            for metric in pair['metrics'].values():
                b, c = metric['baseline_repetitions'], metric['candidate_repetitions']
                require(len(b) == len(c) == len(metric['paired_ratios']) == 3 and
                        all(math.isfinite(x) and x > 0 for x in b + c), 'invalid repetition values')
                near(metric['median_ratio'], statistics.median(c) / statistics.median(b))
                for base, candidate, ratio in zip(b, c, metric['paired_ratios']):
                    near(ratio, candidate / base)
        count = sum(6 * pair['concurrency'] for pair in pairs)
        require(experiment['measured_requests'] == count == 108 and
                experiment['verified_repetitions'] == experiment['accepted_repetition_attempts'] == 24,
                'tuning measurement denominator differs')
        require(experiment['total_requests_reserved'] == count + experiment['qualification_requests_reserved'],
                'reserved request denominator differs')
        repetitions += 24
        requests += count
        reserved += experiment['total_requests_reserved']
        qualification += experiment['qualification_requests_reserved']
    require(seen == {'dspark', 'graph', 'prefill'}, 'incomplete experiment set')
    require(repetitions == data['measured_repetitions'] == 72 and
            requests == data['measured_requests'] == 324 and
            reserved == data['total_requests_reserved'] == 452 and
            qualification == data['qualification_requests_reserved'] == 128, 'tuning summary differs')
    return {'pairs': 12, 'repetitions': repetitions, 'requests': requests,
            'scope': 'Published metric arithmetic and denominators only; original decision and raw acceptance audits remain private.'}


def verify(root):
    runtime.verify_evidence_manifest(root)
    if root.name == 'fixed-comparison':
        return verify_comparison_tables.verify(root / 'tables')
    if root.name == 'context-1m-output2048':
        return supplement(root)
    if root.name == 'tuning-adaptation':
        return tuning(root)
    raise ValueError('unknown result package')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory)))
