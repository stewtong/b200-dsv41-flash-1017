"""Verify complete four-arm aggregate candidates without raw replay."""
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from render_sustained_speculation import verify_figures


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(left, right, label):
    require(math.isfinite(float(left)) and math.isfinite(float(right)) and
            math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-9), label)


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


SECURITY_KEYS = ('scheduled_turns', 'measured_turns', 'successful_turns', 'failed_or_unsent_turns',
                 'native_exporter_count', 'native_rank_count', 'native_series_count')
METRIC_KEYS = ('rate', 'acceptance_length', 'window_output_tokens_per_second_per_gpu', 'cohort_output_tokens_per_second',
               'cohort_elapsed_seconds', 'request_completion_p50_seconds', 'request_completion_p95_seconds',
               'request_completion_p99_seconds', 'drain_seconds', 'first_output_p50_seconds',
               'successful_turn_decode_tokens_per_second_p50', 'latest_engine_running', 'latest_engine_waiting')


def verify(root):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text())
    audit = json.loads((root / 'audit-summary.json').read_text())
    coverage = json.loads((root / 'coverage.json').read_text())
    config = json.loads((root / 'configuration.json').read_text())
    require(summary['sustainable_rate_by_completion_tier'] ==
            {a: {'30_seconds': None, '60_seconds': None} for a in 'ADBC'}, 'unqualified tier rate was invented')
    require(summary['schema'] == 'sustained-speculation-public/v2' and summary['execution_status'] == 'complete' and
            summary['domain_outcome'] == 'achieved' and summary['highest_sustainable_rate'] is None and
            summary['core_windows'] == 39 and summary['diagnostic_windows'] == 1 and
            summary['failed_core_windows'] == 4 and summary['inconclusive_native_windows'] == 1, 'final scope differs')
    accepted = audit['new_canonical']
    require(accepted['status'] == 'passed' and accepted['integrity_and_wire_checks'] == 'passed' and
            accepted['manifest_files'] == 88431 and accepted['canonical_indexed_rows'] == 73792 and
            accepted['raw_streams_reconstructed'] == 66240 and accepted['successful_streams_usage_timing_verified'] == 65483 and
            accepted['source_review_required'] == accepted['matched_input_review_required'] == 0 and
            accepted['protected_old_prepared_files_verified_through_shared_new_inodes'] == 2641 and
            len(accepted['cells']) == 35 and not accepted['excluded_partial_files'] and
            audit['receipts']['exit_code'] == 0 and not audit['raw_audit_repeated'], 'accepted audit differs')
    expected = {'failed_drain': 4, 'cache_confounded_diagnostic': 1, 'failure_stopped_unmeasured': 1,
                'optional_drift_omitted_temporal_stability_unresolved': 1, 'invalid_deployment_attempt': 1,
                'invalid_deployment_startup_timeout_zero_scored_windows': 1, 'excluded_diagnostic': 2,
                'inconclusive_native_telemetry': 1}
    entries = coverage['entries']
    require(coverage['all_scientific_inputs_closed'] and len(entries) == len({x['id'] for x in entries}) == 46 and
            all(sum(row['disposition'] == key for row in entries) == value for key, value in expected.items()),
            'coverage dispositions differ')
    require(config['arms'] == {'A': '2xtp4, DSpark K5', 'D': '2xtp4, NoSpec', 'B': 'dep8, NoSpec', 'C': 'dep8, DSpark K5'} and
            config['measurement'] == {'warmup_seconds': 60, 'completion_window_seconds': 360, 'drain_limit_seconds': 120,
                                     'forced_output_tokens': 400, 'gpu_count': 8} and
            config['external_context']['url'] == 'https://docs.nvidia.com/dynamo/dev/recipes/deepseek-v4-1-flash',
            'configuration differs')
    table, aggregates, provenance = rows(root / 'tables/cells.csv'), rows(root / 'tables/native-aggregates.csv'), rows(root / 'tables/capture-provenance.csv')
    require(len(table) == len(aggregates) == len(provenance) == 40 and len({row['capture_id'] for row in table}) == 40,
            'capture table identities differ')
    source = {row['capture_id']: row for row in summary['cells'] + [summary['diagnostic']]}
    require(len(source) == 40 and len(summary['cells']) == 39, 'duplicate or missing core captures')
    coverage_by_id = {row['id']: row for row in entries}
    coverage_table = rows(root / 'tables/coverage.csv')
    require(len(coverage_table) == 46 and len({row['id'] for row in coverage_table}) == 46,
            'coverage CSV identities differ')
    for published in coverage_table:
        require(published['id'] in coverage_by_id, 'unknown coverage CSV identity')
        require(published == {k: '' if v is None else str(v) for k, v in coverage_by_id[published['id']].items()},
                'coverage CSV differs from ledger')
    for identity, row in source.items():
        require(identity in coverage_by_id and all(coverage_by_id[identity][k] == row[k]
                for k in ['arm', 'workload', 'rate', 'repeat', 'cell', 'disposition']),
                'capture is not bound to coverage disposition')
    extras = {'D/W1/rate-4.050000000-r0': 'failure_stopped_unmeasured',
              'A/optional-final-temporal-drift': 'optional_drift_omitted_temporal_stability_unresolved',
              'B/initial-invalid-deployment': 'invalid_deployment_attempt',
              'C/initial-invalid-deployment-startup-timeout': 'invalid_deployment_startup_timeout_zero_scored_windows',
              'A-W1-0.150000000-r0-first-shape-attempt': 'excluded_diagnostic',
              'A-W1-0.050000000-r0-first-shape-diagnostic': 'excluded_diagnostic'}
    require({k: v['disposition'] for k, v in coverage_by_id.items() if k not in source} == extras,
            'unmeasured or excluded coverage changed')
    details = json.loads((root / 'measurement-details.json').read_text())['captures']
    details_by_id = {row['capture_id']: row for row in details}
    require(len(details) == len(details_by_id) == 40 and set(details_by_id) == set(source),
            'measurement detail coverage differs')
    old_audit = {row['cell']: row for row in audit['old_A']['cells']}
    canon = {(row['arm'], row['workload'], row['rate'], row['repeat'], Path(row['cell']).name): row for row in accepted['cells']}
    for published in table:
        identity = published['capture_id']; row = source[identity]
        require((published['arm'], published['workload']) == (row['arm'], row['workload']) and
                published['disposition'] == row['disposition'], 'capture disposition differs')
        for key in SECURITY_KEYS:
            require(int(published[key]) == row[key], 'population/native count differs: ' + key)
        for key in METRIC_KEYS:
            if row[key] is None:
                require(published[key] == '', 'null metric was invented: ' + key)
            else:
                near(published[key], row[key], 'published metric differs: ' + key)
        require(int(published['repeat']) == row['repeat'] and json.loads(published['counter_bracket_seconds']) ==
                row['counter_bracket_seconds'], 'repeat or bracket differs')
        require(row['request_completion_p95_seconds'] is not None or row['successful_turns'] < 20, 'sparse p95 changed')
        require(row['request_completion_p99_seconds'] is not None or row['successful_turns'] < 100, 'sparse p99 changed')
        native = next(row for row in aggregates if row['capture_id'] == identity)
        values = {key: float(native[key]) for key in ('draft_step_deltas', 'accepted_token_deltas', 'proposed_draft_token_deltas',
                                                       'prefix_cache_hits', 'prefix_cache_queries')}
        for key, value in values.items():
            near(row[key], value, 'native aggregate differs: ' + key)
        if values['draft_step_deltas'] > 0:
            near(1 + values['accepted_token_deltas'] / values['draft_step_deltas'], row['acceptance_length'], 'acceptance denominator')
        else:
            require(values['accepted_token_deltas'] == values['proposed_draft_token_deltas'] == 0 and
                    row['acceptance_length'] is None, 'NoSpec acceptance was invented')
        if values['prefix_cache_queries'] > 0:
            near(values['prefix_cache_hits'] / values['prefix_cache_queries'], row['prefix_cache_hit_fraction'], 'prefix denominator')
        capture = next(row for row in provenance if row['capture_id'] == identity)
        require(capture['config_sha256'] == row['config_sha256'] and capture['summary_sha256'] == row['summary_sha256'] and
                json.loads(capture['source_sha256']) == row['sources'], 'capture provenance differs')
        if row['arm'] == 'A' and row['repeat'] == 0 and row['cell'] in old_audit:
            near(row['acceptance_length'], old_audit[row['cell']]['acceptance_length'], 'old A audit join')
            near(row['window_output_tokens_per_second_per_gpu'],
                 old_audit[row['cell']]['window_tokens_per_second_per_gpu'], 'old A audit throughput')
            for percentile in ['p50', 'p95', 'p99']:
                key = 'request_completion_' + percentile + '_seconds'
                near(row[key], old_audit[row['cell']][key], 'old A audited latency')
            require(row['measured_turns'] == old_audit[row['cell']]['cohort_requests'] and
                    row['successful_turns'] == old_audit[row['cell']]['successful_cohort_requests'], 'old A population differs')
        else:
            audited = canon[(row['arm'], row['workload'], row['rate'], row['repeat'], row['cell'])]
            require(audited['cohort_turns'] == row['measured_turns'] and audited['successful_cohort_turns'] == row['successful_turns'] and
                    audited['source_hashes_without_blob'] == {} and not audited['planned_input_hashes_missing'],
                    'new canonical audit join differs')
            near(audited['window_output_tokens_per_second_per_gpu'], row['window_output_tokens_per_second_per_gpu'], 'audit throughput')
        detail = details_by_id[identity]
        require(detail['summary_sha256'] == row['summary_sha256'] and detail['config_sha256'] == row['config_sha256'],
                'closed summary/config hash join differs')
        measure, throughput, latency = detail['measurement'], detail['throughput'], detail['latency_seconds']
        require(measure['measured_cohort_requests'] == row['measured_turns'] and
                measure['successful_measured_cohort_requests'] == row['successful_turns'] and
                measure['failed_or_unsent_measured_cohort_requests'] == row['failed_or_unsent_turns'],
                'detail populations differ')
        near(throughput['successful_window_output_tokens_per_second_per_gpu'], row['window_output_tokens_per_second_per_gpu'],
             'detail window throughput differs')
        near(throughput['measured_cohort_output_tokens'] / throughput['cohort_first_scheduled_to_last_terminal_seconds'],
             row['cohort_output_tokens_per_second'], 'cohort elapsed denominator differs')
        near(throughput['cohort_first_scheduled_to_last_terminal_seconds'], row['cohort_elapsed_seconds'], 'cohort elapsed differs')
        for percentile in ['p50', 'p95', 'p99']:
            value = latency['request_completion'][percentile]
            published_value = row['request_completion_' + percentile + '_seconds']
            require(value == published_value, 'closed successful-turn latency differs')
        require(bool(detail['queue_backlog_trace']) and detail['client_safety_ceiling'] in [128, 256],
                'missing queue trace or changed client limit')
    failed = [row for row in summary['cells'] if row['disposition'] == 'failed_drain']
    require(sorted((row['arm'], row['rate']) for row in failed) == [('A', 4.05), ('B', 4.05), ('C', 4.05), ('D', 1.35)] and
            all(row['failed_or_unsent_turns'] > 0 and row['drain_seconds'] > 120 for row in failed), 'failed rates differ')
    dep8_native = [row for row in table if row['arm'] in {'B', 'C'}]
    require(all(int(row['native_exporter_count']) == 1 and int(row['native_rank_count']) == 8 and int(row['native_series_count']) == 88
                for row in dep8_native), 'DP8 full-label aggregation differs')
    control = json.loads((root / 'diagnostic-client-control.json').read_text())
    require(control['status'] == 'INCONCLUSIVE_CACHE_STATE_CONFOUNDED' and control['root_cause'] == 'unresolved' and
            control['combined_split_rows'] == 4568 and control['disjoint_split_request_identities'] and
            control['rows'][0]['native_prefix_hit_fraction'] == 0.8559437096809002 and
            control['rows'][1]['native_prefix_hit_fraction'] == 0.9989958895790336, 'split disposition changed')
    near(control['rows'][0]['successful_turn_p95_seconds'], 12.778185803, 'single-client diagnostic p95')
    near(control['rows'][1]['successful_turn_p95_seconds'], 2.963014064, 'split diagnostic p95')
    retention = json.loads((root / 'source-retention.json').read_text())
    require(retention['status'] == 'exact_sources_retained' and len(retention['producer_variants']) == 16 and
            not retention['raw_audit_repeated'], 'source-retention ledger differs')
    variants = retention['producer_variants']
    require(len({item['sha256'] for item in variants}) == len({item['path'] for item in variants}) == 16,
            'duplicate retained source variant')
    retained = {(item['basename'], item['sha256']) for item in variants}
    for row in source.values():
        require(set(details_by_id[row['capture_id']]['sources']) == {'replay.py', 'stream.py', 'workload.py'},
                'missing capture core source declarations')
        for name, digest in details_by_id[row['capture_id']]['sources'].items():
            require(row['sources'].get(name) == digest and (name, digest) in retained,
                    'capture source is not bound to retained bytes')
        for name in ['split_client.py', 'derive_cell.py']:
            if name in row['sources']:
                require((name, row['sources'][name]) in retained, 'capture helper source missing')
    for item in variants:
        require(hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['sha256'], 'source bytes differ')
    portable = root / 'portable-candidate'
    replay = (portable / 'replay.py').read_text()
    require('if legacy_profile.is_file():' in replay and 'result["complete"] and summary["finish_reason"] == "stop"' in replay and
            'if args.base_tokens is not None and abs(counts[0]["count"] - args.base_tokens)' in replay,
            'portable guards missing')
    originals = {'stream.py': 'stream-dab42bc10954.py', 'workload.py': 'workload-69764090584a.py',
                 'split_client.py': 'split_client-3f687aa3a1ed.py', 'derive_cell.py': 'derive_cell-6c02dbf197df.py'}
    for name, original in originals.items():
        require((portable / name).read_bytes() == (root / 'capture-source/producer-history' / original).read_bytes(),
                'unrelated portable bytes changed')
    fixtures = json.loads((root / 'portable-fixtures.json').read_text()); cases = fixtures['cases']
    require(fixtures['status'] == 'passed' and not fixtures['network_or_inference'] and
            fixtures['replay_sha256'] == hashlib.sha256((portable / 'replay.py').read_bytes()).hexdigest() and
            cases['eos_complete_stop']['final']['success'] and cases['eos_truncated']['final']['rejected'] and
            cases['eos_length']['final']['rejected'] and cases['forced_smoke_none']['previous_guarded']['error'].startswith('TypeError:') and
            cases['forced_smoke_none']['final']['error'] == '' and
            cases['forced_smoke_bad_receipt']['final']['error'] == 'RuntimeError: forced smoke qualification failed',
            'offline helper evidence differs')
    for name in ('acceptance.svg', 'completion-p95.svg'):
        svg = ET.parse(root / 'figures' / name).getroot()
        text = ' '.join(''.join(node.itertext()) for node in svg if node.tag == '{http://www.w3.org/2000/svg}text')
        require(svg.tag == '{http://www.w3.org/2000/svg}svg' and '39 core windows plus split diagnostic' in text and
                'Red X=failed drain' in text and 'categorical, not a fitted trend' in text, 'chart scope differs')
    verify_figures(root)
    turns = {'W1': 8, 'W2': 4}
    measured_rows = rows(root / 'tables/measured-arrivals.csv')
    measured = {row['capture_id']: row for row in measured_rows}
    require(len(measured_rows) == len(measured) == len(table) and set(measured) == {row['capture_id'] for row in table},
            'measured-arrival identities differ')
    for row in table:
        arrival = measured[row['capture_id']]
        conversations = int(row['measured_turns']) / turns[row['workload']]
        require(conversations.is_integer() and int(arrival['measured_conversations']) == conversations and
                arrival['workload'] == row['workload'] and arrival['arm'] == row['arm'] and arrival['rate_label'] == row['rate'] and
                arrival['repeat'] == row['repeat'], 'measured arrivals differ from measured turns')
        near(arrival['measured_arrivals_per_second'], round(conversations / 360, 4), 'measured arrival rate differs')
    report = (root / 'report.md').read_text()
    require('measured-arrivals.csv' in report and 'maximum sustainable rates remain null' in report and 'INCONCLUSIVE_CACHE_STATE_CONFOUNDED' in report and
            'turns within conversations are correlated' in report and 'No direct speedup or capacity ratio follows' in report and
            'not a claim of fresh-node or full-benchmark reproduction' in report and '\u2014' not in report,
            'final report qualifications differ')
    return {'status': 'passed', 'core_windows': 39, 'diagnostic_windows': 1, 'coverage_rows': 46,
            'accepted_new_cells': 35, 'old_accepted_cells': 5, 'source_variants': 16, 'portable_fixtures': 5,
            'maximum_sustainable_rate': None,
            'scope': 'Aggregate/source/coverage joins and stored offline regression receipts; private raw and corpus bytes are not re-audited.'}
