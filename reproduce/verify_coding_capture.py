"""Recompute coding conversation outcomes from wire bytes, not capture verdicts."""
import hashlib
import json
from pathlib import Path

import coding_panel
import coding_patch
import verify_wire


def read(path): return json.loads(Path(path).read_text())


def response(raw):
    verify_wire.verify_complete(raw)
    content = ''; calls = {}; usage = []; finishes = []
    for event in raw['events']:
        if event['kind'] != 'event': continue
        data = event['data']
        if data.get('usage') is not None: usage.append(data['usage'])
        for choice in data.get('choices', []):
            if choice.get('index', 0) != 0: raise ValueError('coding choice population differs')
            if choice.get('finish_reason') is not None: finishes.append(choice['finish_reason'])
            delta = choice.get('delta') or {}
            if (delta.get('role', 'assistant') != 'assistant' or
                any(delta.get(key) not in (None, '') for key in ('reasoning', 'reasoning_content'))):
                raise ValueError('coding assistant role or thinking-off contract differs')
            content += delta.get('content') or ''
            for part in delta.get('tool_calls') or []:
                i = part.get('index')
                if type(i) is not int or not 0 <= i < 8: raise ValueError('coding tool index differs')
                row = calls.setdefault(i, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if part.get('type', 'function') != 'function': raise ValueError('coding tool type differs')
                row['id'] += part.get('id', '')
                row['function']['name'] += part.get('function', {}).get('name', '')
                row['function']['arguments'] += part.get('function', {}).get('arguments', '')
    if len(usage) != 1 or len(finishes) != 1 or sorted(calls) != list(range(len(calls))):
        raise ValueError('coding terminal or tool population differs')
    if any(type(usage[0].get(k)) is not int or usage[0][k] <= 0 for k in ('prompt_tokens', 'completion_tokens')):
        raise ValueError('coding usage differs')
    assistant = {'role': 'assistant', 'content': content or None}
    if calls: assistant['tool_calls'] = [calls[i] for i in sorted(calls)]
    return assistant, finishes[0], usage[0]


def required_calls(task, turn):
    project = {'task_id': task['id'], 'marker': task['exact_identifier']}
    if turn == 0: return [('read_project', {'project': project, 'paths': ['solution.py', 'contract.json']})]
    if turn == 1:
        return [('read_cases', {'project': project, 'selection': {'offset': 0, 'limit': 4}}),
                ('read_policy', {'project': project, 'policy': {'name': 'patch', 'version': 1}})]
    return []


def argument_schema(value):
    if isinstance(value, dict):
        return {'type': 'object', 'properties': {k: argument_schema(v) for k, v in value.items()},
                'required': list(value), 'additionalProperties': False}
    if isinstance(value, list): return {'type': 'array', 'items': {'type': 'string'}, 'minItems': len(value), 'maxItems': len(value)}
    return {'type': 'integer' if type(value) is int else 'string'}


def conversation(root):
    root = Path(root); task = read(root / 'task.json'); claimed = read(root / 'conversation.json')
    canonical = {row['id']: row for row in coding_panel.tasks()}
    if task != canonical.get(task.get('id')): raise ValueError('coding task differs from fixed panel')
    expected_history = [{'role': 'system', 'content': 'You are repairing a small synthetic Python project. Use the requested inert tools with exact arguments. Keep identifiers case-sensitive. Never use external resources.'}]
    turns = []; tokens = 0; valid = False; tools_valid = False; previous = None
    for turn in range(4):
        path = root / f'turn-{turn}-raw.json'
        if not path.exists(): raise ValueError('coding raw turn missing before terminal disposition')
        raw = read(path); request = read(root / f'turn-{turn}-request.json'); body = request['body']
        calls = required_calls(task, turn)
        if turn < 2:
            expected_history.append({'role': 'user', 'content': 'Call every listed tool once with these exact arguments: ' + coding_panel.encoded(calls)})
        elif turn == 2:
            expected_history.extend([{'role': 'system', 'content': 'Mid-conversation requirement: preserve this exact marker: ' + task['exact_identifier'] + '. The patch may change only solution.py and must implement solve(value).'},
                                     {'role': 'user', 'content': 'Before writing the patch, return only JSON with keys marker, path, function. Use the exact marker, solution.py, and solve.'}])
        else:
            expected_history.append({'role': 'user', 'content': task['description'] + '\n' + task['patch_contract'] + '\nReturn only the unified diff, with normal hunk line counts. No explanation.'})
        seed = int.from_bytes(hashlib.sha256(json.dumps(['coding_panel', 0, 0, turn, task['id']], separators=(',', ':')).encode()).digest()[:8], 'big') & 0x7fffffff
        if (request.get('request_id') != task['id'] + '-turn-' + str(turn) or
            request.get('session_id') != task['id'] or raw.get('request_id') != request['request_id'] or
            raw.get('session_id') != task['id'] or body.get('messages') != expected_history or
            body.get('temperature') != 1.0 or body.get('top_p') != 1.0 or body.get('seed') != seed or
            body.get('max_tokens') != 2048 or body.get('stream') is not True or
            body.get('stream_options') != {'include_usage': True} or
            body.get('chat_template_kwargs') != {'thinking': False, 'reasoning_effort': 50} or
            'min_tokens' in body or 'ignore_eos' in body or
            (previous is not None and raw['start_ns'] < previous)):
            raise ValueError('coding request, history, clocks or natural-stop contract differs')
        assistant, finish, usage = response(raw); tokens += usage['completion_tokens']; previous = raw['terminal_ns']
        if calls:
            expected_tools = [{'type': 'function', 'function': {'name': name, 'description': 'Read deterministic synthetic project data. No side effects.', 'parameters': argument_schema(args)}} for name, args in calls]
            if body.get('tools') != expected_tools or body.get('tool_choice') != 'required' or body.get('parallel_tool_calls') is not True:
                raise ValueError('coding tool schema differs')
        elif any(key in body for key in ('tools', 'tool_choice', 'parallel_tool_calls')):
            raise ValueError('coding text turn unexpectedly declares tools')
        turns.append({'turn': turn, 'finish_reason': finish, 'usage': usage})
        try:
            if turn < 2:
                actual = assistant.get('tool_calls', [])
                if finish != 'tool_calls' or len(actual) != len(calls) or len({c['id'] for c in actual}) != len(actual) or any(not c['id'] for c in actual):
                    raise ValueError('coding tool population invalid')
                parsed = [(c['function']['name'], coding_panel.strict_json(c['function']['arguments'])) for c in actual]
                if sorted(coding_panel.encoded(p) for p in parsed) != sorted(coding_panel.encoded(p) for p in calls):
                    raise ValueError('coding exact tool arguments invalid')
                expected_history.append(assistant)
                payloads = {'read_project': {'solution.py': task['source'], 'contract.json': task['description']},
                            'read_cases': task['cases'][:4],
                            'read_policy': {'path': task['path'], 'contract': task['patch_contract'], 'exact_identifier': task['exact_identifier']}}
                expected_history.extend({'role': 'tool', 'tool_call_id': c['id'], 'content': coding_panel.encoded(payloads[c['function']['name']])} for c in actual)
                if turn == 1: tools_valid = True
            elif turn == 2:
                if finish != 'stop' or assistant.get('tool_calls') or coding_panel.strict_json(assistant['content']) != {'marker': task['exact_identifier'], 'path': 'solution.py', 'function': 'solve'}:
                    raise ValueError('coding exact identifier or system-history check invalid')
                expected_history.append(assistant)
            else:
                if finish != 'stop' or assistant.get('tool_calls'): raise ValueError('coding patch was not natural-stop')
                coding_patch.apply(task['source'], assistant['content'])
                if (root / 'candidate.diff').read_text() != assistant['content']: raise ValueError('saved patch differs from raw response')
                valid = True
        except (ValueError, KeyError, TypeError):
            break
    expected_files = {f'turn-{turn}-{kind}.json' for turn in range(len(turns)) for kind in ('raw', 'request')}
    actual_files = {path.name for path in root.glob('turn-*.json')}
    if (actual_files != expected_files or claimed.get('task_id') != task['id'] or claimed.get('task_sha256') != task['sha256'] or
        claimed.get('turns') != turns or claimed.get('tool_valid') != tools_valid or claimed.get('conversation_valid') != valid or
        claimed.get('completion_tokens') != tokens or claimed.get('start_ns', 0) > read(root / 'turn-0-raw.json')['start_ns'] or
        claimed.get('end_ns', 0) < previous or claimed.get('end_ns', 0) <= claimed.get('start_ns', 0) or
        claimed.get('task_seconds') != (claimed['end_ns'] - claimed['start_ns']) / 1e9):
        raise ValueError('coding capture verdict or task arithmetic differs')
    return {'task_id': task['id'], 'conversation_valid': valid, 'tool_valid': tools_valid,
            'completion_tokens': tokens, 'task_seconds': claimed['task_seconds'],
            'scope': 'wire-recomputed conversation only; executable patch score remains separate'}


def verify(root):
    root = Path(root); panel = read(root / 'panel.json'); coding_panel.validate_panel(panel)
    if {path.name for path in root.iterdir() if path.is_dir()} != {task['id'] for task in panel}:
        raise ValueError('coding fixed population missing or duplicated')
    rows = [conversation(root / task['id']) for task in panel]
    result = {'tasks': 100, 'captured_tasks': len(rows), 'conversation_valid': sum(row['conversation_valid'] for row in rows),
              'tool_valid': sum(row['tool_valid'] for row in rows)}
    claimed = read(root / 'capture-summary.json')
    if any(claimed.get(key) != value for key, value in result.items()): raise ValueError('coding summary differs from raw conversations')
    result['scope'] = 'wire-recomputed conversations; no executable score, cap or model qualification claim'
    return result
