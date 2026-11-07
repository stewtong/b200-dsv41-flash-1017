"""Fixed synthetic coding tasks and inert, exact-argument tool conversations.

This module never executes a returned patch. Execution belongs to the separate
container scorer after serving ownership has been released.
"""
import hashlib
import json
import random

SCHEMA = 'coding-tool-panel/v1'
SOURCE = 'def solve(value):\n    raise NotImplementedError("TODO")\n'
FAMILIES = ('rotate', 'runs', 'intervals', 'counts', 'join', 'histogram',
            'pages', 'invoice', 'reachable', 'paths')
DESCRIPTIONS = {
    'rotate': 'Return items rotated left by offset modulo their length; an empty list stays empty. offset may be negative.',
    'runs': 'Return consecutive equal items as [item, count] pairs, preserving order. An empty list produces [].',
    'intervals': 'Sort closed integer intervals by start then end, and merge overlapping or touching endpoints. Return lists of [start, end]. Inputs always have start <= end.',
    'counts': 'Count exact case-sensitive string keys, retaining first appearance order. Return [[key, count], ...]. Do not normalize identifiers.',
    'join': 'For each left record in order, emit {id, name, total}; total is the sum of all matching right amounts or zero. Preserve duplicate left records and exact IDs.',
    'histogram': 'For each integer x, bucket it at floor(x / width) * width. Return [bucket_start, count] pairs sorted by bucket_start. width is a positive integer.',
    'pages': 'Split items into pages of size positive integer size, retaining a final partial page. Empty items produce [].',
    'invoice': 'For each row, add quantity * cents; subtract discount_cents once from the total, then clamp at zero. Values are integers; do not use floating point.',
    'reachable': 'Return all distinct nodes reachable from start through directed edges, including start, sorted lexicographically. Cycles and duplicate edges are permitted.',
    'paths': 'Normalize each absolute slash-separated path: ignore empty and dot components; dot-dot removes one component when available and otherwise stays at root. Return absolute paths in input order.',
}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def strict_json(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def answer(family, value):
    """Independent expected values are frozen in the panel, never model code."""
    if family == 'rotate':
        items = value['items']; k = value['offset'] % len(items) if items else 0
        return items[k:] + items[:k]
    if family == 'runs':
        result = []
        for item in value:
            if result and result[-1][0] == item: result[-1][1] += 1
            else: result.append([item, 1])
        return result
    if family == 'intervals':
        result = []
        for a, b in sorted(value):
            if result and a <= result[-1][1]: result[-1][1] = max(result[-1][1], b)
            else: result.append([a, b])
        return result
    if family == 'counts':
        result = {}
        for item in value: result[item] = result.get(item, 0) + 1
        return [[key, count] for key, count in result.items()]
    if family == 'join':
        totals = {}
        for row in value['right']: totals[row['id']] = totals.get(row['id'], 0) + row['amount']
        return [dict(id=row['id'], name=row['name'], total=totals.get(row['id'], 0)) for row in value['left']]
    if family == 'histogram':
        totals = {}
        for x in value['items']:
            key = x // value['width'] * value['width']; totals[key] = totals.get(key, 0) + 1
        return [[key, totals[key]] for key in sorted(totals)]
    if family == 'pages':
        return [value['items'][i:i + value['size']] for i in range(0, len(value['items']), value['size'])]
    if family == 'invoice':
        return max(0, sum(row['quantity'] * row['cents'] for row in value['rows']) - value['discount_cents'])
    if family == 'reachable':
        visited = {value['start']}
        while True:
            expanded = visited | {b for a, b in value['edges'] if a in visited}
            if expanded == visited: return sorted(visited)
            visited = expanded
    if family == 'paths':
        result = []
        for path in value:
            stack = []
            for part in path.split('/'):
                if part == '..':
                    if stack: stack.pop()
                elif part and part != '.': stack.append(part)
            result.append('/' + '/'.join(stack))
        return result
    raise ValueError('unknown coding family')


def examples(family, variant):
    rng = random.Random(830100 + FAMILIES.index(family) * 100 + variant)
    ids = [f'item_{variant}_{i}_Aa' for i in range(5)]
    values = []
    for count in (0, 1, 2, 3, 4, 7, 9, 12, 15, 20, 31, 40):
        nums = [rng.randint(-20, 20) for _ in range(count)]
        keys = [rng.choice(ids) for _ in range(count)]
        if family == 'rotate': value = {'items': keys, 'offset': rng.randint(-80, 80)}
        elif family in ('runs', 'counts'): value = keys
        elif family == 'intervals': value = [[n, n + rng.randint(0, 8)] for n in nums]
        elif family == 'join':
            value = {'left': [{'id': key, 'name': f'name_{i}'} for i, key in enumerate(keys)],
                     'right': [{'id': rng.choice(ids), 'amount': n} for n in nums]}
        elif family == 'histogram': value = {'items': nums, 'width': rng.randint(1, 7)}
        elif family == 'pages': value = {'items': keys, 'size': rng.randint(1, 8)}
        elif family == 'invoice':
            value = {'rows': [{'quantity': abs(n), 'cents': rng.randint(0, 100000)} for n in nums],
                     'discount_cents': rng.randint(0, 100000)}
        elif family == 'reachable':
            value = {'start': ids[variant % len(ids)], 'edges': [[key, rng.choice(ids)] for key in keys]}
        else:
            value = ['/' + '/'.join(rng.choice(ids + ['', '.', '..']) for _ in range(rng.randint(0, 12))) for _ in nums]
        values.append({'input': value, 'expected': answer(family, value)})
    return values


def tasks():
    result = []
    for family in FAMILIES:
        for variant in range(10):
            ident = f'coding-{family}-{variant:02d}'
            row = {'schema': SCHEMA, 'id': ident, 'family': family,
                   'description': DESCRIPTIONS[family], 'path': 'solution.py', 'source': SOURCE,
                   'exact_identifier': 'marker_' + hashlib.sha256(ident.encode()).hexdigest()[:20],
                   'cases': examples(family, variant), 'visible_case_count': 4,
                   'patch_contract': 'Unified diff for solution.py only; implement solve(value) and helper functions. Use builtins or collections, math, functools, itertools, heapq, bisect, re. No classes, global/nonlocal declarations, async, input/output, private attribute access or external resources.',
                   'initial_output_cap': 2048, 'cap_status': 'unqualified_pending_c1_calibration'}
            row['sha256'] = hashlib.sha256(encoded(row).encode()).hexdigest()
            result.append(row)
    return result


def expected_calls(task, turn):
    """Every tool call has nested arguments and exact case-sensitive values."""
    project = {'task_id': task['id'], 'marker': task['exact_identifier']}
    if turn == 0:
        return [('read_project', {'project': project, 'paths': ['solution.py', 'contract.json']})]
    if turn == 1:
        return [('read_cases', {'project': project, 'selection': {'offset': 0, 'limit': 4}}),
                ('read_policy', {'project': project, 'policy': {'name': 'patch', 'version': 1}})]
    return []


def tool_results(task, turn, calls):
    """Validate an exact set of inert calls before returning synthetic data.

    No paths, commands, Python or user-supplied expressions are evaluated.
    """
    required = expected_calls(task, turn)
    if turn not in (0, 1) or len(calls) != len(required): raise ValueError('wrong tool-call population')
    parsed = []; seen = set()
    for call in calls:
        if (set(call) != {'id', 'type', 'function'} or call['type'] != 'function' or
            not isinstance(call['id'], str) or not call['id'] or call['id'] in seen):
            raise ValueError('invalid or repeated tool-call identity')
        seen.add(call['id']); function = call['function']
        if set(function) != {'name', 'arguments'} or not isinstance(function['arguments'], str):
            raise ValueError('malformed tool function')
        # JSON bool must not compare equal to an integer argument.
        pair = (function['name'], strict_json(function['arguments']))
        if not any(pair[0] == name and encoded(pair[1]) == encoded(args) for name, args in required):
            raise ValueError('tool argument mismatch')
        if any(pair[0] == p[0] for p in parsed): raise ValueError('duplicate tool function')
        parsed.append(pair)
    payloads = {'read_project': {'solution.py': task['source'], 'contract.json': task['description']},
                'read_cases': task['cases'][:task['visible_case_count']],
                'read_policy': {'path': task['path'], 'contract': task['patch_contract'],
                                'exact_identifier': task['exact_identifier']}}
    return [{'role': 'tool', 'tool_call_id': call['id'], 'content': encoded(payloads[call['function']['name']])} for call in calls]


def validate_panel(panel):
    expected = tasks()
    if encoded(panel) != encoded(expected): raise ValueError('coding panel differs from fixed version')
    return {'schema': SCHEMA, 'tasks': 100, 'families': 10,
            'panel_sha256': hashlib.sha256(encoded(panel).encode()).hexdigest(),
            'scope': 'fixed synthetic tasks; no model, executable-score or cap qualification'}


if __name__ == '__main__':
    print(json.dumps({'panel': tasks(), 'validation': validate_panel(tasks())}, sort_keys=True))
