"""Persist a burst of explicitly supplied requests; never retries automatically."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import derive
import runtime
import stream

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute(url, requests, attempt_dir, timeout, clock_id):
    """Record one attempt, including transport failures, before deriving a summary.

    Task scoring here is an exact visible-text fixture, not a general benchmark
    correctness scorer. Real GPU cells require the campaign safety supervisor.
    """
    root = Path(attempt_dir)
    if not requests or len({r['id'] for r in requests}) != len(requests):
        raise ValueError('missing or duplicate requests')
    for request in requests:
        if not re.fullmatch(r'[A-Za-z0-9]+', request['id']): raise ValueError('request ID must be ASCII alphanumeric')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'summary.json').exists() or any((root/(r['id']+'.json')).exists() for r in requests):
        raise ValueError('attempt evidence already exists')
    def one(request):
        raw = stream.capture(url, request['body'], request['id'], timeout)
        text = ''.join((choice.get('delta') or {}).get('content') or ''
                       for event in raw['events'] if event['kind'] == 'event'
                       for choice in event['data'].get('choices', []))
        tokens, valid = None, False
        try:
            summary = stream.summarize_stream(raw)
            tokens = summary['usage']['completion_tokens']
            valid = text == request['expected'] and summary['finish_reason'] == 'stop'
        except (ValueError, KeyError, TypeError):
            pass
        record = {'request_id': request['id'], 'start_ns': raw['start_ns'],
                  'end_ns': raw['terminal_ns'], 'clock_id': clock_id,
                  'completion_tokens': tokens, 'valid': valid, 'raw': raw}
        path = root / (request['id'] + '.json')
        runtime.atomic_json(path, record)
        return path
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        paths = list(pool.map(one, requests))
    rows = [json.loads(p.read_text()) for p in paths]
    summary = derive.burst(rows)
    manifest = {'files': {p.name: sha(p) for p in paths}, 'summary': summary,
                'created_ns': time.monotonic_ns(), 'scope': 'fixture-only',
                'expected_request_ids': sorted(r['id'] for r in requests),
                'has_failed_request': any(not r['valid'] for r in rows)}
    runtime.atomic_json(root / 'summary.json', manifest)
    return manifest
