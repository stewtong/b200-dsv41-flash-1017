"""Recompute sanitized timing/usage tables; raw transport acceptance is separate."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import comparison_scope
import runtime


def close(actual, expected):
    if expected is None:
        if actual != '':
            raise ValueError('undefined metric is populated')
    elif not math.isfinite(float(actual)) or not math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('derived metric differs from retained request timing or usage')


def quantiles(values):
    values = sorted(v for v in values if v is not None)
    result = dict(n=len(values), p50=statistics.median(values) if values else None,
                  min=min(values) if values else None, max=max(values) if values else None,
                  p95_nearest_rank=None)
    if len(values) >= 20:
        result['p95_nearest_rank'] = values[math.ceil(.95 * len(values)) - 1]
    return result


def verify(directory):
    directory = Path(directory).resolve(strict=True)
    runtime.verify_evidence_manifest(directory)
    if {p.name for p in directory.iterdir()} != {'cells.csv', 'repetitions.csv', 'requests.csv', 'summary.json', 'SHA256SUMS'}:
        raise ValueError('table file set differs')
    summary = json.loads((directory / 'summary.json').read_text())
    matrix_path = Path(__file__).with_name('comparison-matrix.json')
    expected = {c['id']: c for c in comparison_scope.expand(json.loads(matrix_path.read_text()))}
    if (summary.get('schema') != 'fixed-comparison-results/v1' or
            summary.get('scope_id') != comparison_scope.SCOPE_ID or
            summary.get('matrix_sha256') != hashlib.sha256(matrix_path.read_bytes()).hexdigest()):
        raise ValueError('scope, schema or matrix differs')
    if (summary.get('required_cells'), summary.get('required_cold_cells'), summary.get('required_warm_cells')) != (336, 288, 48):
        raise ValueError('required denominators differ')
    profile_root = Path(__file__).resolve().parents[1] / 'configs/frozen-profiles'
    profile_hashes = {layout: json.loads((profile_root / (layout+'.json')).read_text())['profile_sha256']
                      for layout in comparison_scope.LAYOUTS}
    def rows(name):
        with (directory / name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    cells = rows('cells.csv'); repetitions = rows('repetitions.csv'); requests = rows('requests.csv')
    cell_map = {c['id']: c for c in cells}
    if len(cells) != 336 or set(cell_map) != set(expected) or len(repetitions) != 1008:
        raise ValueError('complete cell/repetition rows required')
    def shape(row):
        base = expected.get(row['id'])
        if base is None or any(str(base[k]) != row[k] for k in
                ('suite', 'layout', 'input_tokens', 'concurrency', 'cache', 'output_tokens')):
            raise ValueError('row shape differs from declared matrix')
    populations = {}
    for row in requests:
        shape(row)
        key = (row['id'], int(row['repetition']))
        populations.setdefault(key, []).append(row)
    seen = set(); observed = {}; burst_rates = {}; contracts = {}; seals = set()
    metrics = ('first_output_seconds', 'first_visible_text_seconds', 'completion_seconds',
               'decode_window_tokens_per_second', 'end_to_end_output_tokens_per_second')
    for row in repetitions:
        shape(row); key = (row['id'], int(row['repetition']))
        if key in seen or key[1] not in (0, 1, 2):
            raise ValueError('duplicate or invalid repetition')
        seen.add(key); population = populations.pop(key, [])
        if row['observed'] == 'False':
            identity_fields = {'id','suite','layout','input_tokens','concurrency','cache','output_tokens','repetition','observed'}
            if population or any(value for k, value in row.items() if k not in identity_fields):
                raise ValueError('missing repetition contains measurements')
            continue
        if row['observed'] != 'True' or len(population) != int(row['concurrency']):
            raise ValueError('observed request population differs')
        if {int(r['request_ordinal']) for r in population} != set(range(len(population))):
            raise ValueError('request identities differ')
        if (row['profile_sha256'] != profile_hashes[row['layout']] or
                any(not re.fullmatch('[a-f0-9]{64}', row[k]) for k in ('source_seal','task_contract_sha256'))):
            raise ValueError('frozen profile or source fingerprint differs')
        contracts[key] = row['task_contract_sha256']; seals.add(row['source_seal'])
        values = {k: [] for k in metrics}; starts = []; ends = []; tokens = 0
        for request in population:
            if request['valid'] != 'True' or int(request['completion_tokens']) != int(row['output_tokens']):
                raise ValueError('accepted timing row lacks complete valid output')
            target = int(request['target_input_tokens']); prompt = int(request['prompt_tokens'])
            if (target != int(row['input_tokens']) or abs(prompt - target) > target * .0025 or
                    prompt != int(request['expected_input_tokens'])):
                raise ValueError('input usage outside declared calibration')
            start = int(request['dispatch_ns']); first = int(request['first_output_ns'])
            visible = int(request['first_visible_text_ns']); last = int(request['last_output_ns']); end = int(request['terminal_ns'])
            if not 0 <= start <= first <= visible <= last <= end:
                raise ValueError('request clocks are out of order')
            n = int(request['completion_tokens']); duration = (end - start) / 1e9
            measured = dict(first_output_seconds=(first-start)/1e9, first_visible_text_seconds=(visible-start)/1e9,
                completion_seconds=duration, decode_window_tokens_per_second=(n-1)/((last-first)/1e9) if last > first and n > 1 else None,
                end_to_end_output_tokens_per_second=n/duration)
            for metric, value in measured.items():
                close(request[metric], value); values[metric].append(value)
            if request['source_seal'] != row['source_seal'] or request['profile_sha256'] != row['profile_sha256']:
                raise ValueError('request provenance differs from repetition')
            starts.append(start); ends.append(end); tokens += n
        duration = (max(ends) - min(starts)) / 1e9
        close(row['burst_makespan_seconds'], duration)
        close(row['burst_output_tokens_per_second'], tokens / duration)
        close(row['dispatch_span_seconds'], (max(starts)-min(starts))/1e9)
        if (int(row['requests']) != len(population) or int(row['valid_requests']) != len(population) or
                int(row['failed_or_invalid_requests']) != 0 or int(row['valid_output_tokens']) != tokens or
                int(row['partial_output_tokens']) != 0 or row['warmup_excluded'] != 'True'):
            raise ValueError('request counts or excluded warmup differ')
        for metric, sample in values.items():
            for stat, value in quantiles(sample).items():
                close(row[metric+'_'+stat], value)
        hits = float(row['cache_hits']); queries = float(row['cache_queries'])
        if not 0 <= hits <= queries:
            raise ValueError('invalid cache counters')
        close(row['achieved_cache_hit_fraction'], hits / queries if queries else None)
        observed.setdefault(key[0], []).append(key[1]); burst_rates.setdefault(key[0], []).append(tokens/duration)
    if populations:
        raise ValueError('requests lack a declared repetition')
    cold = warm = 0; cohorts = {}
    for ident, cell in cell_map.items():
        shape(cell); present = sorted(observed.get(ident, [])); missing = [n for n in range(3) if n not in present]
        if (cell['accepted'] != str(not missing) or int(cell['repetitions_observed']) != len(present) or
                cell['missing_repetitions'] != ';'.join(map(str, missing))):
            raise ValueError('cell coverage contradicts requests')
        if not missing:
            if cell['cache'] == 'cold': cold += 1
            else: warm += 1
            q = quantiles(burst_rates[ident])
            for stat in ('p50', 'min', 'max'): close(cell['burst_rate_'+stat], q[stat])
            signature = tuple(contracts[(ident,i)] for i in range(3))
            shape_key = tuple(cell[k] for k in ('suite','input_tokens','concurrency','cache','output_tokens'))
            cohorts.setdefault((shape_key,signature), []).append(ident)
        elif any(cell['burst_rate_'+stat] for stat in ('p50','min','max')):
            raise ValueError('unaccepted cell has a complete-cell rate')
        status = 'descriptive_measurement_accepted' if not missing else 'partial_repetitions' if present else 'not_measured'
        if cell['status'] != status:
            raise ValueError('cell status differs from measured coverage')
    matched = {ident for group in cohorts.values() if len(group) == 4 for ident in group}
    if any(cell['all_four_matched'] != str(ident in matched) for ident,cell in cell_map.items()):
        raise ValueError('matched comparison flag lacks all four identical task contracts')
    if summary['all_four_matched_cells'] != len(matched) or summary['raw_source_seals'] != sorted(seals):
        raise ValueError('matched coverage or source seals differ')
    if (summary['accepted_cells'], summary['accepted_cold_cells'], summary['accepted_warm_cells'],
            summary['measured_repetitions'], summary['measured_requests'], summary['complete']) != (
            cold+warm, cold, warm, sum(map(len, observed.values())), len(requests), cold == 288 and warm == 48):
        raise ValueError('summary coverage contradicts retained table rows')
    return dict(status='sanitized_timing_usage_metrics_recomputed', accepted_cells=cold+warm,
        required_cells=336, observed_repetitions=sum(map(len, observed.values())), requests=len(requests),
        complete=summary['complete'], scope='Timing/usage arithmetic and coverage only; raw streams, profile qualification, placement and telemetry require the original capture verifiers.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    print(json.dumps(verify(parser.parse_args().directory)))
