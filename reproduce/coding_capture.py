"""Four-turn natural-stop coding/tool capture through the retained-wire client."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import coding_panel
import coding_patch
import runtime
import stream
import verify_wire
import workload


def tool_schema(name, arguments):
    def schema(value):
        if isinstance(value, dict):
            return {'type': 'object', 'properties': {k: schema(v) for k, v in value.items()},
                    'required': list(value), 'additionalProperties': False}
        if isinstance(value, list):
            return {'type': 'array', 'items': {'type': 'string'}, 'minItems': len(value), 'maxItems': len(value)}
        return {'type': 'integer' if type(value) is int else 'string'}
    return {'type': 'function', 'function': {'name': name, 'description': 'Read deterministic synthetic project data. No side effects.',
                                            'parameters': schema(arguments)}}


def decode(raw):
    verify_wire.verify_complete(raw)
    summary = stream.summarize_stream(raw)
    usage = summary['usage']
    if (any(type(usage.get(k)) is not int or usage[k] <= 0 for k in ('prompt_tokens', 'completion_tokens')) or
        summary['finish_reason'] not in ('stop', 'tool_calls', 'length')):
        raise runtime.RuntimeErrorClosed('coding stream usage or finish is malformed')
    content = ''; calls = {}
    for event in raw['events']:
        if event['kind'] != 'event': continue
        for choice in event['data'].get('choices', []):
            if choice.get('index', 0) != 0: raise ValueError('multiple coding choices')
            delta = choice.get('delta') or {}
            if (delta.get('role', 'assistant') != 'assistant' or
                any(delta.get(key) not in (None, '') for key in ('reasoning', 'reasoning_content'))):
                raise ValueError('coding assistant role or thinking-off contract differs')
            piece = delta.get('content')
            if piece is not None:
                if not isinstance(piece, str): raise ValueError('malformed content delta')
                content += piece
            for fragment in delta.get('tool_calls') or []:
                index = fragment.get('index')
                if type(index) is not int or not 0 <= index < 8: raise ValueError('invalid tool fragment index')
                call = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if fragment.get('type', 'function') != 'function': raise ValueError('invalid tool fragment type')
                if 'id' in fragment: call['id'] += fragment['id']
                for key in ('name', 'arguments'):
                    if key in fragment.get('function', {}): call['function'][key] += fragment['function'][key]
    if calls and sorted(calls) != list(range(len(calls))): raise ValueError('tool fragment gap')
    return {'role': 'assistant', 'content': content or None,
            **({'tool_calls': [calls[i] for i in sorted(calls)]} if calls else {})}, summary


def conversation(session, url, model, task, root, timeout, cancellation, clock_id):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    runtime.atomic_json(root / 'task.json', task)
    messages = [{'role': 'system', 'content': 'You are repairing a small synthetic Python project. Use the requested inert tools with exact arguments. Keep identifiers case-sensitive. Never use external resources.'}]
    started = time.monotonic_ns(); tokens = 0; outcomes = []; patch = None
    result = {'task_id': task['id'], 'task_sha256': task['sha256'], 'clock_id': clock_id,
              'start_ns': started, 'tool_valid': False, 'conversation_valid': False,
              'executable_score': 'pending_separate_isolated_scoring', 'turns': outcomes}
    for turn in range(4):
        session.ensure_healthy()
        required = coding_panel.expected_calls(task, turn)
        if turn < 2:
            messages.append({'role': 'user', 'content': 'Call every listed tool once with these exact arguments: ' + coding_panel.encoded(required)})
        elif turn == 2:
            messages.extend([{'role': 'system', 'content': 'Mid-conversation requirement: preserve this exact marker: ' + task['exact_identifier'] + '. The patch may change only solution.py and must implement solve(value).'},
                             {'role': 'user', 'content': 'Before writing the patch, return only JSON with keys marker, path, function. Use the exact marker, solution.py, and solve.'}])
        else:
            messages.append({'role': 'user', 'content': task['description'] + '\n' + task['patch_contract'] + '\nReturn only the unified diff, with normal hunk line counts. No explanation.'})
        body = workload.request_body(model, messages, 1, task['initial_output_cap'],
                                     workload.seed_for('coding_panel', 0, 0, turn, task['id']), fixed_output=False)
        if required:
            body['tools'] = [tool_schema(name, args) for name, args in required]
            body['tool_choice'] = 'required'; body['parallel_tool_calls'] = True
        request_id = task['id'] + '-turn-' + str(turn)
        runtime.atomic_json(root / f'turn-{turn}-request.json', {'request_id': request_id, 'session_id': task['id'], 'body': body})
        raw = stream.capture(url, body, request_id, timeout, cancellation, session_id=task['id'])
        runtime.atomic_json(root / f'turn-{turn}-raw.json', raw)
        # Unsafe transport aborts the shared session. A semantic/tool miss stays
        # in the fixed 100-task denominator and never causes repeat-to-pass.
        assistant, summary = decode(raw)
        tokens += summary['usage']['completion_tokens']
        outcomes.append({'turn': turn, 'finish_reason': summary['finish_reason'], 'usage': summary['usage']})
        try:
            if turn < 2:
                if summary['finish_reason'] != 'tool_calls': raise ValueError('tool turn did not finish with tools')
                replies = coding_panel.tool_results(task, turn, assistant.get('tool_calls', []))
                messages.append(assistant); messages.extend(replies)
                if turn == 1: result['tool_valid'] = True
            elif turn == 2:
                if summary['finish_reason'] != 'stop' or assistant.get('tool_calls'): raise ValueError('metadata turn did not stop naturally')
                if coding_panel.strict_json(assistant['content']) != {'marker': task['exact_identifier'], 'path': 'solution.py', 'function': 'solve'}:
                    raise ValueError('mid-conversation exact identifier mismatch')
                messages.append(assistant)
            else:
                if summary['finish_reason'] != 'stop' or assistant.get('tool_calls'): raise ValueError('patch did not stop naturally')
                patch = assistant['content']; coding_patch.apply(task['source'], patch)
                runtime.atomic_bytes(root / 'candidate.diff', patch.encode())
                result['conversation_valid'] = True
        except (ValueError, TypeError, KeyError) as exc:
            result['failure_type'] = type(exc).__name__; result['failure_reason'] = str(exc)
            break
    ended = time.monotonic_ns()
    result.update(end_ns=ended, completion_tokens=tokens,
                  task_seconds=(ended - started) / 1e9)
    runtime.atomic_json(root / 'conversation.json', result)
    return result


def run(session, url, model, output, concurrency, timeout, clock_id):
    if concurrency not in (1, 8, 32) or not 0 < timeout <= 1800: raise ValueError('invalid coding panel limits')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    panel = coding_panel.tasks(); runtime.atomic_json(root / 'panel.json', panel)
    runtime.atomic_json(root / 'panel-contract.json', coding_panel.validate_panel(panel))
    cancellation = stream.Cancellation(); pool = ThreadPoolExecutor(max_workers=concurrency)
    pending = set(); rows = []; index = 0
    try:
        while index < len(panel) or pending:
            session.ensure_healthy()
            while index < len(panel) and len(pending) < concurrency:
                task = panel[index]; index += 1
                if time.monotonic_ns() + int((timeout * 4 + 120) * 1e9) > session.deadline_ns:
                    raise runtime.RuntimeErrorClosed('coding task and cleanup do not fit operational window')
                pending.add(pool.submit(conversation, session, url, model, task, root / task['id'], timeout, cancellation, clock_id))
            done, pending = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
            for future in done: rows.append(future.result())
        result = {'tasks': 100, 'captured_tasks': len(rows), 'concurrency_ceiling': concurrency,
                  'conversation_valid': sum(row['conversation_valid'] for row in rows),
                  'tool_valid': sum(row['tool_valid'] for row in rows),
                  'executable_score': 'pending_separate_isolated_scoring',
                  'scope': 'natural-stop four-turn task capture; no correctness or cap qualification claim'}
        runtime.atomic_json(root / 'capture-summary.json', result)
        return result
    except BaseException:
        cancellation.cancel(); session.abort('coding panel transport or ownership failed')
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
