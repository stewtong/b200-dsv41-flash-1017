"""Run one prepared pilot burst; abort owned engines on any failed request."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import re
import time
import derive
import runtime
import stream

def run(session, url, tasks, output_dir, timeout, clock_id, oracle=None,request_ids=None,session_ids=None):
    if (not tasks or len({t['id'] for t in tasks}) != len(tasks) or
        any(not re.fullmatch(r'[A-Za-z0-9]+', t['id']) for t in tasks)):
        raise ValueError('pilot task identities must be unique')
    custom_ids=request_ids is not None or session_ids is not None
    logical={t['id'] for t in tasks}
    request_ids={ident:ident for ident in logical} if request_ids is None else request_ids
    session_ids={ident:ident for ident in logical} if session_ids is None else session_ids
    if (set(request_ids)!=logical or set(session_ids)!=logical or len(set(request_ids.values()))!=len(tasks) or
        any(not isinstance(value,str) or not value or not value.isascii() or any(ord(c)<33 or ord(c)>126 for c in value)
            for mapping in (request_ids,session_ids) for value in mapping.values())):
        raise ValueError('request/session header identity coverage invalid')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    cancel = stream.Cancellation()
    pool = ThreadPoolExecutor(max_workers=len(tasks))
    def one(task):
        raw = stream.capture(url, task['body'], request_ids[task['id']], timeout, cancel, session_id=session_ids[task['id']])
        usage, finish, visible, summary = None, None, '', None
        try:
            summary = stream.summarize_stream(raw)
            usage, finish = summary['usage'], summary['finish_reason']
            visible = ''.join((c.get('delta') or {}).get('content') or ''
                              for e in raw['events'] if e['kind'] == 'event'
                              for c in e['data'].get('choices', []))
        except (ValueError, KeyError, TypeError):
            pass
        expected_input = task['input_tokens']
        prompt_tokens = usage.get('prompt_tokens') if isinstance(usage,dict) else None
        tokens = usage.get('completion_tokens') if isinstance(usage,dict) else None
        input_ok = type(prompt_tokens) is int and prompt_tokens >= 0 and abs(prompt_tokens-expected_input) <= expected_input * 0.0025
        if type(tokens) is not int or tokens < 0: tokens = None
        fixed = task.get('fixed_output', False)
        clean = ('\ufffd' not in visible and
                 re.search(r'[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]',visible) is None and
                 not any(ord(c) < 32 and c not in '\n\r\t' for c in visible))
        task_ok = (tokens is not None and tokens > 0 and bool(visible) and clean and finish in ('stop','length') and
                   (tokens == task['output_tokens'] and isinstance(task.get('expected_answer'),str) and
                    bool(task['expected_answer']) and task['expected_answer'] in visible
                    if fixed else finish == 'stop' and visible.strip() == task['expected_answer']))
        quality_score=None;oracle_error=False
        if oracle is not None:
            try:
                quality_score=oracle(task,raw)
                if not isinstance(quality_score,dict) or type(quality_score.get('correct')) is not bool:
                    raise ValueError('quality oracle must return an explicit correctness outcome')
                if 'fatal_content' in quality_score and type(quality_score['fatal_content']) is not bool:
                    raise ValueError('oracle fatal-content disposition must be boolean')
            except Exception as exc:
                oracle_error=True
                quality_score={'correct':False,'reason':'oracle_failure','error_type':type(exc).__name__}
            task_ok=quality_score['correct']
        valid=bool(input_ok and task_ok)
        # An explicitly typed oracle owns its content policy. The generic
        # pilot language screen remains the fallback for callers without one.
        content_fatal=(quality_score['fatal_content'] if quality_score is not None and
                       'fatal_content' in quality_score else not clean)
        fatal=(oracle_error or raw.get('complete') is not True or raw.get('http_status')!=200 or
               summary is None or not input_ok or content_fatal)
        timing=None
        if valid and summary is not None:
            duration=(raw['terminal_ns']-raw['start_ns'])/1e9
            if duration <= 0: valid=False
            else:
                timing={'first_output_seconds':summary['first_output_ns']/1e9,
                        'first_visible_seconds':summary['first_visible_ns']/1e9 if summary['first_visible_ns'] is not None else None,
                        'stream_window_seconds_per_token':derive.stream_window(summary['first_output_ns'],summary['last_output_ns'],tokens),
                        'request_output_tokens_per_second':tokens/duration,
                        'scope':'stream-derived; multi-token chunks do not reveal exact token times'}
        record = {'request_id':request_ids[task['id']], 'start_ns':raw['start_ns'], 'end_ns':raw['terminal_ns'],
                  'clock_id':clock_id, 'completion_tokens':tokens, 'valid':valid,
                  'prompt_tokens_match':input_ok, 'timing':timing, 'raw':raw}
        if custom_ids:record['task_id']=task['id']
        if oracle is not None:record.update(quality_score=quality_score,fatal_transport_or_content=fatal)
        runtime.atomic_json(root/(task['id']+'.json'), record)
        return record
    futures = {pool.submit(one, task):task['id'] for task in tasks}
    pending = set(futures); failed = False; abort_started = False; records = []
    try:
        while pending:
            session.ensure_healthy()
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result(); records.append(record)
                if not record['valid'] and (oracle is None or record['fatal_transport_or_content']) and not failed:
                    failed = True
                    cleanup_deadline = time.monotonic() + 120
                    cancel.cancel()
                    abort_started = True
                    session.abort('pilot request failed or timed out')
            if failed and pending:
                done, pending = wait(pending, timeout=max(0,cleanup_deadline-time.monotonic()))
                for future in done: records.append(future.result())
                if pending: raise runtime.RuntimeErrorClosed('clients did not drain after cleanup')
        summary = derive.burst(records)
        runtime.atomic_json(root/'summary.json', {'status':'failed' if failed else 'captured_with_invalid_requests_cleanup_pending' if summary['valid_requests']!=summary['requests'] else 'requests_passed_cleanup_pending', 'summary':summary})
        return summary
    except BaseException:
        cancel.cancel()
        if not abort_started:
            abort_started = True
            session.abort('pilot driver or safety failure')
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
