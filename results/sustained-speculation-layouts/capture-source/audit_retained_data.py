#!/usr/bin/env python3
"""Read-only coordinator audit of retained A evidence; no serving requests."""
import base64
import codecs
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import sys
ROOT = Path(sys.argv[1]).resolve()
MANIFEST_SHA = '4dd7c7f18b8258ffc321560233ff637583f8aa0ddcca4a51a45c801ac2f0f059'
SELECTED = ['rate-0.050000000-r0-v2', 'rate-0.150000000-r0-per-rate-v2',
            'rate-0.450000000-r0', 'rate-1.350000000-r0', 'rate-4.050000000-r0']
NAMES = {'vllm:spec_decode_num_drafts_total', 'vllm:spec_decode_num_accepted_tokens_total',
         'vllm:spec_decode_num_draft_tokens_total', 'vllm:prefix_cache_hits_total',
         'vllm:prefix_cache_queries_total', 'vllm:num_preemptions_total', 'vllm:request_success_total',
         'vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc'}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def near(a, b, label):
    check(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), label)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_wire(raw):
    decoder = codecs.getincrementaldecoder('utf-8')()
    pending, data, events, done = '', [], [], False
    h = hashlib.sha256()
    for chunk in raw['wire_chunks']:
        b = base64.b64decode(chunk['base64'], validate=True)
        h.update(b)
        pending += decoder.decode(b, final=bool(chunk.get('eof')))
        while '\n' in pending:
            line, pending = pending.split('\n', 1)
            line = line.rstrip('\r')
            if not line and data:
                payload = '\n'.join(data)
                data = []
                check(not done, 'event after DONE')
                if payload == '[DONE]':
                    kind, value, done = 'done', None, True
                else:
                    kind, value = 'event', json.loads(payload)
                events.append({'offset_ns': chunk['offset_ns'], 'kind': kind, 'data': value})
            elif line.startswith('data:'):
                value = line[5:]
                data.append(value[1:] if value.startswith(' ') else value)
    check(h.hexdigest() == raw['wire_sha256'], 'wire hash')
    check(events == raw['events'], 'reconstructed SSE events differ')
    if raw['complete']:
        check(done and not pending and not data, 'completed stream lacks final DONE')
    return events


def native(text):
    result = {}
    for line in text.splitlines():
        m = re.fullmatch(r'([\w:]+)(?:\{(.*)\})?\s+([-+0-9.eE]+)(?:\s+\d+)?', line)
        if not m or m[1] not in NAMES:
            continue
        labels = tuple(sorted(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m[2] or '')))
        key = (m[1], labels)
        check(key not in result, 'duplicate native series')
        result[key] = float(m[3])
    return result


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    mp = ROOT / 'output/manifests/A.json'
    check(digest(mp) == MANIFEST_SHA, 'raw manifest identity')
    manifest = json.loads(mp.read_text())
    files = {x['path']: x for x in manifest['files']}
    check(len(files) == len(manifest['files']), 'duplicate manifest path')
    actual, partials = set(), []
    for base in [ROOT / 'output/A', ROOT / 'prepared']:
        for p in base.rglob('*'):
            check(not p.is_symlink(), 'symlink in evidence')
            if p.is_file():
                rel = p.relative_to(ROOT).as_posix()
                if p.suffix == '.partial':
                    partials.append(rel)
                else:
                    actual.add(rel)
    check(actual == set(files), 'manifest file-set mismatch')
    hashed, source_digests = {}, set()
    for rel, entry in files.items():
        p = ROOT / rel
        check('..' not in Path(rel).parts and not Path(rel).is_absolute(), 'unsafe manifest path')
        st = p.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in hashed:
            hashed[key] = digest(p)
        check(st.st_size == entry['bytes'] and hashed[key] == entry['sha256'], 'file integrity: ' + rel)
    for base in [ROOT / 'code', ROOT / 'code-history', Path(__file__).resolve().parent]:
        for p in base.rglob('*'):
            if p.is_file() and not p.is_symlink():
                source_digests.add(digest(p))
    cells, streams, successes, indexed, source_missing = [], 0, 0, 0, []
    for cell in sorted((ROOT / 'output/A/W1').iterdir()):
        if not (cell / 'requests.jsonl').exists():
            continue
        rows = jsonl(cell / 'requests.jsonl')
        indexed += len(rows)
        check(len({x['request_id'] for x in rows}) == len(rows), 'duplicate request IDs')
        for row in rows:
            if 'raw_path' not in row:
                check(not row.get('success') and row.get('status') == 'unsent_drain_deadline', 'missing raw capture')
                continue
            p = cell / 'raw' / row['raw_path']
            check(files[p.relative_to(ROOT).as_posix()]['sha256'] == row['raw_sha256'], 'index/capture hash')
            raw = json.loads(gzip.decompress(p.read_bytes()))
            check(raw['request_id'] == row['request_id'] and raw['session_id'] == row['session_id'], 'capture identity')
            check(raw['request_headers']['X-Session-Id'] == row['session_id'], 'session affinity header')
            if row.get('actual_request_body_sha256'):
                check(row['actual_request_body_sha256'] == raw['request_body_sha256'], 'actual sent body hash')
            events = parse_wire(raw)
            streams += 1
            if row.get('success'):
                successes += 1
                values = [x['data'] for x in events if x['kind'] == 'event']
                usage = [x['usage'] for x in values if x.get('usage') is not None]
                finishes = [c['finish_reason'] for x in values for c in x.get('choices', []) if c.get('finish_reason') is not None]
                check(raw['complete'] and raw['http_status'] == 200 and len(usage) == 1 and finishes == ['length'], 'successful stream contract')
                check(usage[0] == row['usage'] and usage[0]['completion_tokens'] == 400, 'usage provenance')
                check(usage[0]['prompt_tokens'] == row['prepared_tokenize']['count'], 'exact input count')
                output = [x['offset_ns'] for x in events if x['kind'] == 'event' and any(
                    (c.get('delta') or {}).get(k) for c in x['data'].get('choices', [])
                    for k in ['content', 'reasoning_content', 'reasoning', 'tool_calls'])]
                near((raw['terminal_ns'] - raw['start_ns']) / 1e9, row['request_completion_seconds'], 'completion timing')
                near(min(output) / 1e9, row['first_output_seconds'], 'first-output timing')
                if max(output) > min(output):
                    near(399 / ((max(output) - min(output)) / 1e9), row['stream_decode_tokens_per_second'], 'chunk-clock rate')
        if cell.name not in SELECTED:
            continue
        config = json.loads((cell / 'config.json').read_text())
        summary = json.loads((cell / 'summary.json').read_text())
        start, end = config['measurement_start_wall_ns'], config['measurement_end_wall_ns']
        cohort = [x for x in rows if start <= config['start_wall_ns'] + x['conversation']['scheduled_offset_seconds'] * 1e9 < end]
        good = [x for x in cohort if x.get('success')]
        window = [x for x in rows if x.get('success') and start <= x['terminal_wall_ns'] < end]
        check(len(rows) == config['scheduled_conversations_total'] * 8, 'scheduled population')
        check(len(cohort) == summary['measurement']['measured_cohort_requests'] and len(good) == summary['measurement']['successful_measured_cohort_requests'], 'cohort populations')
        rate = sum(x['usage']['completion_tokens'] for x in window) / ((end - start) / 1e9) / 8
        near(rate, summary['throughput']['successful_window_output_tokens_per_second_per_gpu'], 'window throughput')
        durations = sorted(x['request_completion_seconds'] for x in good)
        for fraction, key in [(0.5, 'p50'), (0.95, 'p95'), (0.99, 'p99')]:
            near(durations[math.ceil(fraction * len(durations)) - 1], summary['latency_seconds']['request_completion'][key], 'latency quantile')
        telemetry = jsonl(cell / 'telemetry.jsonl')
        parsed = {}
        for sample in telemetry:
            rank = int(sample['rank'])
            check(sample['port'] == 31000 + rank and rank in [0, 1] and not sample.get('error'), 'true replica coverage')
            p = cell / 'raw/metrics' / f"{sample['sequence']:06d}-rank-{rank}.txt.gz"
            text = gzip.decompress(p.read_bytes()).decode()
            check(digest_bytes(text.encode()) == sample['raw_sha256'], 'native scrape hash')
            parsed[(rank, sample['sequence'])] = native(text)
        totals = {}
        for rank in [0, 1]:
            samples = [s for s in telemetry if int(s['rank']) == rank]
            before = max((s for s in samples if s['observed_wall_ns'] <= start), key=lambda s: s['observed_wall_ns'])
            after = min((s for s in samples if s['observed_wall_ns'] >= end), key=lambda s: s['observed_wall_ns'])
            a, b = parsed[(rank, before['sequence'])], parsed[(rank, after['sequence'])]
            for key in a.keys() & b.keys():
                if key[0].endswith('_total'):
                    delta = b[key] - a[key]
                    check(delta >= 0, 'native counter reset')
                    totals[key[0]] = totals.get(key[0], 0) + delta
        acceptance = 1 + totals['vllm:spec_decode_num_accepted_tokens_total'] / totals['vllm:spec_decode_num_drafts_total']
        near(acceptance, summary['engine']['acceptance_length'], 'weighted acceptance across distinct exporters')
        for name, value in totals.items():
            near(value, sum(x['delta'] for x in summary['engine']['counter_deltas'].values() if x['name'] == name), 'counter sum ' + name)
        missing = {k: v for k, v in config.get('source_sha256', {}).items() if v not in source_digests}
        if missing:
            source_missing.append({'cell': cell.name, 'hashes_without_code_blob_in_current_or_history': missing})
        cells.append({'cell': cell.name, 'cohort_requests': len(cohort), 'successful_cohort_requests': len(good), 'window_tokens_per_second_per_gpu': rate, 'acceptance_length': acceptance, 'counter_exporters': [31000, 31001]})
    check(indexed == manifest['requested_index_rows'], 'manifest index total')
    print(json.dumps({'schema': 'coordinator-independent-raw-audit/v1', 'started_utc': started,
                      'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'status': 'needs_source_review' if source_missing or partials else 'passed',
                      'metric_and_integrity_checks': 'passed',
                      'manifest_files': len(files), 'logical_bytes': sum(x['bytes'] for x in files.values()),
                      'indexed_records': indexed, 'raw_streams_reconstructed': streams, 'successful_streams_usage_timing_verified': successes,
                      'cells': cells, 'excluded_partial_files': partials, 'source_blob_coverage_findings': source_missing,
                      'limits': 'Source-blob findings are reported separately. This audit reads retained evidence only; no new inference and no maximum-sustainable-rate claim.'}))


if __name__ == '__main__':
    main()
