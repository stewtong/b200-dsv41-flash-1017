"""Loopback health and known-answer readiness checks."""
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import stream
import workload

def health(url, timeout=5):
    p = urlsplit(url)
    if p.scheme != 'http' or p.hostname != '127.0.0.1' or not p.port:
        raise ValueError('health check requires loopback')
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status == 200

def known_answer(url, model, request_id, timeout=60):
    body = workload.request_body(model, [{'role':'user','content':'Reply with exactly the word READY and nothing else.'}],
                                 32, 32, 7, thinking=False, effort=50, fixed_output=False)
    raw = stream.capture(url, body, request_id, timeout)
    try:
        summary = stream.summarize_stream(raw)
    except (ValueError, KeyError, TypeError) as exc:
        return {'raw':raw, 'summary':None, 'passed':False, 'reason':type(exc).__name__}
    visible = ''.join((c.get('delta') or {}).get('content') or ''
                      for e in raw['events'] if e['kind'] == 'event'
                      for c in e['data'].get('choices', []))
    passed = visible.strip() == 'READY' and summary['finish_reason'] == 'stop'
    return {'raw':raw, 'summary':summary, 'passed':passed,
            'reason':None if passed else 'known-answer response failed'}
