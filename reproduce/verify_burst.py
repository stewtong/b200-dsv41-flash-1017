"""Recompute fixture burst summary without importing its derivation code."""
import hashlib
import json
from pathlib import Path
import sys

def verify(root, contracts):
    root = Path(root)
    m = json.loads((root / 'summary.json').read_text())
    if m.get('scope') != 'fixture-only' or not m.get('files'): raise ValueError('missing fixture files')
    rows = []
    for name, expected in m['files'].items():
        if Path(name).name != name: raise ValueError('invalid evidence filename')
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected: raise ValueError('checksum mismatch')
        rows.append(json.loads(data))
    if len({r['request_id'] for r in rows}) != len(rows): raise ValueError('duplicate request')
    if sorted(r['request_id'] for r in rows) != sorted(contracts):
        raise ValueError('missing or unexpected request')
    for row in rows:
        raw = row['raw']; usage = []; finishes = []; text = ''
        if raw['request_id'] != row['request_id'] or raw['start_ns'] != row['start_ns'] or raw['terminal_ns'] != row['end_ns']:
            raise ValueError('raw request identity or clock mismatch')
        for entry in raw['events']:
            if entry['kind'] != 'event': continue
            event = entry['data']
            if event.get('usage') is not None: usage.append(event['usage'])
            for choice in event.get('choices', []):
                text += (choice.get('delta') or {}).get('content') or ''
                if choice.get('finish_reason') is not None: finishes.append(choice['finish_reason'])
        complete = raw['complete'] and len(usage) == 1 and len(finishes) == 1 and bool(text)
        tokens = usage[0].get('completion_tokens') if complete else None
        valid = complete and finishes[0] == 'stop' and text == contracts[row['request_id']]
        if row['completion_tokens'] != tokens or row['valid'] != valid:
            raise ValueError('raw response scoring mismatch')
    if len({r['clock_id'] for r in rows}) != 1: raise ValueError('mixed clocks')
    seconds = (max(r['end_ns'] for r in rows) - min(r['start_ns'] for r in rows)) / 1e9
    if seconds <= 0 or any(r['end_ns'] < r['start_ns'] for r in rows): raise ValueError('invalid clock')
    tokens = sum(r['completion_tokens'] for r in rows if r['valid'])
    invalid = [r['completion_tokens'] for r in rows if not r['valid']]
    expected = {'duration_seconds': seconds, 'valid_tokens': tokens,
                'invalid_partial_tokens': None if any(x is None for x in invalid) else sum(invalid),
                'valid_tokens_per_second': tokens / seconds,
                'valid_requests': sum(r['valid'] for r in rows), 'requests': len(rows)}
    if m['summary'] != expected or m['has_failed_request'] != any(not r['valid'] for r in rows):
        raise ValueError('summary mismatch')
    return True

if __name__ == '__main__':
    verify(sys.argv[1], {'one':'abc', 'two':'wrong'})
    print('fixture burst verified')
