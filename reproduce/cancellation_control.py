"""Exclusive unscored disconnect controls with retained all-rank work evidence."""
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

import readiness
import request_absence
import runtime
import stream
import verify_wire


def fetch(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port:
        raise ValueError('cancellation observations require explicit loopback')
    with urlopen(url, timeout=1.5) as response:
        data = response.read(2 * 1024 * 1024 + 1)
        if response.status != 200 or len(data) > 2 * 1024 * 1024:
            raise ValueError('cancellation observation status or size differs')
        return data.decode('utf-8', errors='strict')


def snapshot(expected, clock_id):
    if not expected or len(expected) > 8 or any(not url.endswith('/metrics') for url in expected):
        raise ValueError('invalid cancellation endpoint population')
    urls = list(expected) + [url.removesuffix('/metrics') + '/load' for url in expected]
    started = time.monotonic_ns(); pool = ThreadPoolExecutor(max_workers=len(urls))
    pending = {pool.submit(fetch, url): url for url in urls}
    try:
        done, incomplete = wait(pending, timeout=2.5)
        if incomplete: raise TimeoutError('all-rank cancellation observation exceeded collection bound')
        bodies = {pending[future]: future.result() for future in done}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    result = {'clock_id': clock_id, 'started_ns': started, 'monotonic_ns': time.monotonic_ns(),
              'metrics': {url: bodies[url] for url in expected},
              'loads': {url: json.loads(bodies[url]) for url in urls if url not in expected},
              'raw_load_bodies': {url: bodies[url] for url in urls if url not in expected}}
    request_absence.population(result, expected)
    return result


def body_for(task, run_id, mode):
    body = json.loads(json.dumps(task['body']))
    # Distinct early prefixes keep these recovery requests out of the next
    # pilot cell's cold-cache history. Their tokens are never scored.
    body['messages'].insert(0, {'role': 'system', 'content':
        'Cancellation control ' + run_id + ' ' + mode + '. This isolated request is outside measured tasks.'})
    body.update(max_tokens=8192, min_tokens=8192, ignore_eos=True)
    return body


def run(session, url, task, expected, root, mode, clock_id, *, timeout=90, absence_timeout=30):
    if mode not in ('prefill', 'decode') or not 0 < timeout <= 120 or not 0 < absence_timeout <= 60:
        raise ValueError('invalid cancellation control limits')
    if time.monotonic_ns() + int((timeout + absence_timeout + 60 + 120) * 1e9) > session.deadline_ns:
        raise runtime.RuntimeErrorClosed('cancellation control and owned cleanup do not fit')
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    request_id = session.run_id + '-cancel-' + mode
    body = body_for(task, session.run_id, mode)
    contract = {'schema': 'exclusive-cancellation-control/v1', 'run_id': session.run_id,
                'request_id': request_id, 'session_id': request_id, 'mode': mode,
                'clock_id': clock_id, 'url': url, 'body': body, 'expected_ranks': expected,
                'source_task_sha256': hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
                'timeout_seconds': timeout, 'absence_timeout_seconds': absence_timeout,
                'scope': 'unscored recovery control; no performance or task-correctness claim'}
    runtime.atomic_json(root / 'control-input.json', contract)
    runtime.atomic_json(root / 'source-task.json', task)
    cancellation = stream.Cancellation(); observed = []; output = threading.Event(); result = {}
    cancelled_ns = None; worker = None; accepted = False
    def event(row):
        if row['kind'] == 'event' and any(any((choice.get('delta') or {}).get(key)
            for key in ('content', 'reasoning', 'reasoning_content', 'tool_calls'))
            for choice in row['data'].get('choices', [])):
            if not output.is_set(): observed.append(time.monotonic_ns()); output.set()
    def capture():
        try:
            raw = stream.capture(url, body, request_id, timeout, cancellation, session_id=request_id, on_event=event)
            result['raw'] = raw
        except BaseException as exc:
            result['error_type'] = type(exc).__name__
    def observe(name):
        session.ensure_healthy(); value = snapshot(expected, clock_id)
        runtime.atomic_json(root / name, value); return value
    try:
        before = observe('before.json')
        if request_absence.population(before, expected) != {'running': 0, 'waiting': 0, 'frontend': 0}:
            raise ValueError('cancellation control does not own an empty engine population')
        worker = threading.Thread(target=capture, daemon=True); worker.start()
        limit = time.monotonic() + timeout; index = 0; busy = None
        while time.monotonic() < limit:
            current = observe('transition-%03d.json' % index); index += 1
            values = request_absence.population(current, expected)
            if values == {'running': 1, 'waiting': 0, 'frontend': 1} and (mode == 'prefill' or output.is_set()):
                busy = current; break
            if not worker.is_alive(): raise ValueError('request terminated before cancellation stage was established')
            if mode == 'prefill' and output.is_set(): raise ValueError('prefill stage ended before live-work observation')
            time.sleep(.05)
        if busy is None: raise TimeoutError('required live cancellation stage was not observed')
        if mode == 'prefill' and output.is_set(): raise ValueError('prefill cancellation would occur after output')
        runtime.atomic_json(root / 'busy.json', busy)
        cancelled_ns = time.monotonic_ns(); cancellation.cancel()
        runtime.atomic_json(root / 'cancellation.json', {'monotonic_ns': cancelled_ns, 'first_output_observed_ns': observed[0] if observed else None})
        worker.join(5)
        if worker.is_alive() or 'raw' not in result: raise RuntimeError('owned cancellation client has not terminated')
        raw = result['raw']; runtime.atomic_json(root / 'cancelled-request-raw.json', raw)
        verify_wire.retained_bytes(raw)
        if raw['complete'] or not raw['cancel_requested']:
            raise ValueError('request naturally completed instead of proving cancellation')
        if any(row['kind']=='done' or (row['kind']=='event' and any(c.get('finish_reason') is not None
               for c in row['data'].get('choices',[]))) for row in raw['events']):
            raise ValueError('terminal stream cannot qualify cancellation')
        if mode=='prefill' and observed and observed[0]<=cancelled_ns:
            raise ValueError('client output preceded the intended pre-output cancellation')
        after = []; limit = time.monotonic() + absence_timeout; index = 0
        while time.monotonic() < limit:
            session.ensure_healthy()
            target = (raw['terminal_ns'] + 1_500_000_000 if not after else after[-1]['monotonic_ns'] + 1_000_000_000)
            if time.monotonic_ns() < target:
                time.sleep(min(.05, (target - time.monotonic_ns()) / 1e9)); continue
            current = observe('after-attempt-%03d.json' % index); index += 1
            if request_absence.population(current, expected) == {'running': 0, 'waiting': 0, 'frontend': 0}:
                after.append(current)
                if len(after) == 2: break
            else: after = []
            time.sleep(.05)
        runtime.atomic_json(root / 'after.json', after)
        checked = request_absence.verify(before, busy, after, expected, client_terminal_ns=raw['terminal_ns'],
                                        cancellation_ns=cancelled_ns, clock_id=clock_id)
        health = {}
        for metric in expected:
            endpoint = metric.removesuffix('/metrics') + '/health'
            started=time.monotonic_ns();text=fetch(endpoint)
            health[endpoint] = {'status':200,'body':text,'started_ns':started,'monotonic_ns':time.monotonic_ns()}
        runtime.atomic_json(root / 'post-cancellation-health.json', health)
        session.ensure_healthy()
        semantic=readiness.known_answer(url,body['model'],request_id+'-semantic',timeout=60)
        runtime.atomic_json(root / 'post-cancellation-semantic.json',semantic)
        if not semantic['passed']:raise ValueError('semantic generation failed after cancellation')
        session.ensure_healthy()
        runtime.atomic_json(root / 'control-result.json', {'status': 'passed', 'work_absence': checked,
                            'client_terminal_ns': raw['terminal_ns'], 'cancellation_ns': cancelled_ns,
                            'scope': contract['scope']})
        import verify_cancellation
        verified=verify_cancellation.verify(root);accepted=True
        return verified
    except BaseException as exc:
        runtime.atomic_json(root / 'control-failure.json', {'type': type(exc).__name__,
                            'message': str(exc), 'scope': 'control failed; exact owned engine recovery required'})
        raise
    finally:
        cancellation.cancel()
        if worker is not None: worker.join(5)
        if 'raw' in result and not (root / 'cancelled-request-raw.json').exists():
            runtime.atomic_json(root / 'cancelled-request-raw.json',result['raw'])
        if not accepted:
            session.abort('cancellation control did not establish fresh all-rank work absence')
