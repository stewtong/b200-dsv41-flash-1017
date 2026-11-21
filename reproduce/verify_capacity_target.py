"""Verify the capacity-at-latency-target tables: identities, derived arithmetic and verdict consistency."""
import csv
import json
import math
from pathlib import Path

MEASURED_SECONDS = 360
TARGET_COMPLETION, TARGET_FIRST_OUTPUT = 30, 10
IMAGE = 'vllm/vllm-openai@sha256:aeb580d55ee3dcccd7b5c78d7871326f367bee5cfd8ca8e411c253d7983a0046'
KV_TOKENS = 40819149
DRAFT_TOKENS = 5
PARTS = {'capacity ladder': 14, 'midpoint': 14, 'K5 vs NoSpec': 18, 'dep8 K5': 18}
NODE_CONFIGURATION = {'A': '2xtp4 K5', 'B': 'dep8 K5'}
UNIT_IDENTITY = {'A': {('2xtp4', 'K5')}, 'A+': {('2xtp4', 'K5'), ('2xtp4', 'NoSpec')}, 'B': {('dep8', 'K5')}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(left, right, label, abs_tol):
    left, right = float(left), float(right)
    require(math.isfinite(left) and math.isfinite(right) and abs(left - right) <= abs_tol, label)


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value == '':
        return None
    result = float(value)
    require(math.isfinite(result), 'nonfinite value in table')
    return result


def flag(value, label):
    require(value in ('True', 'False'), label)
    return value == 'True'


def check_unit_counts(row, turns):
    """The published quantiles must agree with the published turn counts (one turn of interpolation slack)."""
    conversations, scheduled = int(row['measured_conversations']), int(row['scheduled_measured_turns'])
    completed, within = int(row['completed_turns']), int(row['turns_within_30s'])
    require(scheduled == conversations * turns, 'scheduled turns differ from conversations times turns')
    require(0 <= within <= completed <= scheduled, 'turn counts are inconsistent')
    p95, first = number(row['completion_p95_s']), number(row['first_output_p95_s'])
    require((p95 is None) == (completed < 0.95 * scheduled), 'p95 availability differs from completed turns')
    if p95 is not None:
        if p95 <= TARGET_COMPLETION:
            require(within >= math.ceil(0.95 * scheduled) - 1, 'p95 under target contradicts turns within 30 s')
        else:
            require(within <= math.floor(0.95 * scheduled) + 1, 'p95 over target contradicts turns within 30 s')
    quantile_pass = p95 is not None and first is not None and p95 <= TARGET_COMPLETION and first <= TARGET_FIRST_OUTPUT
    expected = (quantile_pass and completed == scheduled and flag(row['drain_ok'], 'drain status missing')
                and row['early_overload'] == '')
    require(flag(row['unit_pass'], 'unit verdict missing') == expected,
            'unit verdict differs from its quantiles, counts, drain and early stop')


def verify(root):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text())
    config = json.loads((root / 'configuration.json').read_text())
    arms, node = rows(root / 'tables/arms.csv'), rows(root / 'tables/node-cells.csv')
    paired = rows(root / 'tables/k5-nospec-paired.csv')
    turns = {w: config['workloads'][w]['turns'] for w in ('W1', 'W2')}

    require(summary['schema'] == 'capacity-at-latency-target-public/v1' and
            summary['target'] == {'completion_p95_seconds_max': TARGET_COMPLETION, 'first_output_p95_seconds_max': TARGET_FIRST_OUTPUT},
            'summary scope differs')
    rule = config['pass_rule']
    require(config['image_digest'] == IMAGE and config['measurement']['measured_arrival_seconds'] == MEASURED_SECONDS and
            rule['completion_p95_seconds_max'] == TARGET_COMPLETION and rule['first_output_p95_seconds_max'] == TARGET_FIRST_OUTPUT,
            'configuration differs')
    kv = [e['gpu_kv_cache_tokens_per_engine_core'] for name, c in config['configurations'].items()
          if name.startswith('2xtp4') for e in c['engines']]
    require(kv == [KV_TOKENS] * 3, 'K5 and NoSpec KV capacity differ from the stated value')
    require(any(name.startswith('dep8') for name in config['configurations']), 'dep8 configuration missing')

    # Windows: two unit rows each, one study part, identical inputs, identity matching the window.
    windows = {}
    for row in arms:
        windows.setdefault((row['window'], row['workload'], row['rate_label_per_stream'], row['repeat']), []).append(row)
    require(len(arms) == summary['unit_rows'] == 2 * len(windows) and len(windows) == summary['windows'] == 64,
            'window coverage differs')
    parts = {}
    for (window, _, _, _), units in windows.items():
        require(len(units) == 2 and len({u['unit'] for u in units}) == 2 and len({u['study_part'] for u in units}) == 1,
                'window units differ')
        require({(u['layout'], u['speculation']) for u in units} <= UNIT_IDENTITY[window], 'unit identity differs from window')
        require(len({(u['measured_conversations'], u['input_tokens_per_second']) for u in units}) == 1,
                'paired units received different inputs')
        parts[units[0]['study_part']] = parts.get(units[0]['study_part'], 0) + 1
    require(parts == PARTS == summary['windows_by_part'], 'study parts differ')

    for row in arms:
        require(flag(row['cell_usable'], 'usability missing'), 'unusable window published as a result')
        require(row['measured_arrivals_per_second'] == str(round(int(row['measured_conversations']) / MEASURED_SECONDS, 4)),
                'measured arrivals differ')
        check_unit_counts(row, turns[row['workload']])
        if row['speculation'] == 'NoSpec':
            require(row['acceptance_length'] == row['draft_token_acceptance_rate'] == '', 'NoSpec reports acceptance')
        else:
            near(row['acceptance_length'], 1 + DRAFT_TOKENS * float(row['draft_token_acceptance_rate']),
                 'acceptance length differs from draft acceptance rate', 2e-3)

    # Node verdicts: exactly the windows of the capacity ladder, midpoint and dep8 parts.
    node_keys = [(r['window'], r['workload'], r['rate_label_per_stream'], r['repeat']) for r in node]
    require(len(node) == summary['node_verdicts'] == 46 and len(set(node_keys)) == 46 and
            set(node_keys) == {k for k in windows if k[0] in NODE_CONFIGURATION}, 'node coverage differs')
    for row, key in zip(node, node_keys):
        units = windows[key]
        require(row['configuration'] == NODE_CONFIGURATION[key[0]] and
                {f"{u['layout']} {u['speculation']}" for u in units} == {row['configuration']}, 'node configuration differs')
        require(flag(row['usable'], 'usability missing'), 'unusable node verdict published')
        conversations = sum(int(u['measured_conversations']) for u in units)
        require(int(row['measured_conversations_node']) == conversations and
                row['measured_arrivals_per_second_node'] == str(round(conversations / MEASURED_SECONDS, 4)), 'node arrivals differ')
        near(row['input_tokens_per_second_node'], sum(float(u['input_tokens_per_second']) for u in units), 'node input rate differs', 2.0)
        p95, passed = number(row['node_completion_p95_s']), flag(row['pass'], 'node verdict missing')
        if row['configuration'] == '2xtp4 K5':
            require(row['dep8_pooled_first_output_p95_s'] == row['dep8_pooled_drained'] == '', 'pooled dep8 fields on 2xtp4')
            require(passed == all(u['unit_pass'] == 'True' for u in units), '2xtp4 node verdict differs from replicas')
        else:
            first = number(row['dep8_pooled_first_output_p95_s'])
            drained = flag(row['dep8_pooled_drained'], 'pooled drain missing')
            require(passed == (p95 is not None and first is not None and p95 <= TARGET_COMPLETION and
                               first <= TARGET_FIRST_OUTPUT and drained), 'dep8 pooled verdict differs')
        if passed:
            require(p95 is not None and p95 <= TARGET_COMPLETION, 'passing node exceeds the target')

    def loads(window, workload):
        grouped = {}
        for row in node:
            if row['window'] == window and row['workload'] == workload:
                grouped.setdefault(float(row['measured_arrivals_per_second_node']), []).append(row['pass'] == 'True')
        return grouped

    for workload in ('W1', 'W2'):
        passing = [load for load, verdicts in loads('A', workload).items() if all(verdicts)]
        require(passing, 'no passing load')
        require(max(passing) == summary['highest_all_passing_measured_arrivals_per_node'][workload],
                'stated limit differs from verdicts')
    dep8_any = any(all(v) for w in ('W1', 'W2') for v in loads('B', w).values())
    require(bool(loads('B', 'W1')) and bool(loads('B', 'W2')), 'dep8 verdicts missing')
    require(dep8_any is False and summary['dep8_any_load_all_runs_pass'] is False, 'dep8 verdict summary differs')

    # NoSpec limit: the lowest tested load failed on every run, so no limit is stated.
    for workload in ('W1', 'W2'):
        nospec = [r for r in arms if r['window'] == 'A+' and r['workload'] == workload and r['speculation'] == 'NoSpec']
        lowest = min(float(r['rate_label_per_stream']) for r in nospec)
        require(all(r['unit_pass'] == 'False' for r in nospec if float(r['rate_label_per_stream']) == lowest),
                'NoSpec passed at the lowest tested load')
    require(summary['nospec_limit'] is None, 'NoSpec limit stated without a passing load')

    require(sorted(r['workload'] for r in paired) == ['W1', 'W2'], 'paired coverage differs')
    for row in paired:
        for metric in ('completion_time_ratio', 'decode_rate_ratio'):
            low, mid, high = (number(row[f'{metric}_{s}']) for s in ('ci95_low', 'geomean', 'ci95_high'))
            require(0 < low <= mid <= high, 'paired interval does not contain its estimate')
        scheduled = int(row['scheduled_measured_turns'])
        require(int(row['conversations']) * turns[row['workload']] == scheduled and int(row['matched_turns']) <= scheduled and
                int(row['k5_turns_within_30s']) <= scheduled and int(row['nospec_turns_within_30s']) <= scheduled, 'paired counts differ')

    report = (root / 'report.md').read_text()
    require('measured arrivals' in report and 'NoSpec missed the 30 s target at the lowest load' in report and
            'The mechanism behind the input-token match is not measured' in report and 'not a task-quality measurement' in report and
            'no fresh-node replication' in report and 'not attributed to random tokens alone' in report and '—' not in report,
            'report qualifications differ')
    return {'status': 'passed', 'windows': len(windows), 'unit_rows': len(arms), 'node_verdicts': len(node),
            'highest_all_passing_measured_arrivals_per_node': summary['highest_all_passing_measured_arrivals_per_node'],
            'scope': 'Table identities, derived arithmetic and verdict consistency, including count-quantile agreement '
                     'to within one turn (a p95 within one turn of the 30 s boundary cannot be checked from counts); '
                     'node p95 values and private raw records are not recomputed.'}
