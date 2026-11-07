#!/usr/bin/env python3
"""Independent fixed-oracle verifier for synthetic fixture evidence."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
state = json.loads((root / 'execution-state.json').read_text())
assert state['completed'] is True
assert len([a for a in state['attempts'] if a['status'] == 'passed']) == 1
for a in state['attempts']:
    path = root / a['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == a['sha256']
    raw = json.loads(path.read_text())
    assert raw['clock'] == {'start': 10, 'first': 12, 'last': 14, 'end': 15}
    assert raw['completion_tokens'] == 3 and raw['valid'] is True
    assert 'data: [DONE]' in raw['body']
    if a['status'] == 'passed':
        summary = json.loads((root / (path.stem + '-summary.json')).read_text())
        assert summary == {'ttft_seconds': 2, 'burst_valid_tokens_per_second': 0.6, 'stream_seconds_per_token': 1.0}
print('fixture evidence verified')
