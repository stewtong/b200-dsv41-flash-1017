"""Apply one bounded unified diff as data; never execute a candidate on the host."""
import re


def apply(source, patch):
    if not isinstance(patch, str) or len(patch.encode()) > 65536 or '\x00' in patch:
        raise ValueError('invalid or oversized patch')
    if patch.startswith('```diff\n') and patch.endswith('\n```'):
        patch = patch[8:-4]
    lines = patch.splitlines(keepends=True)
    if len(lines) < 3 or lines[0].rstrip('\n') not in ('--- a/solution.py', '--- solution.py') or lines[1].rstrip('\n') not in ('+++ b/solution.py', '+++ solution.py'):
        raise ValueError('patch must modify only solution.py')
    original = source.splitlines(keepends=True); result = []; cursor = 0; index = 2; hunks = 0
    while index < len(lines):
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n?', lines[index])
        if not match: raise ValueError('malformed patch hunk')
        old_start, old_count, new_start, new_count = (int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1))
        target = old_start - 1 if old_count else old_start
        if target < cursor or target > len(original): raise ValueError('overlapping or out-of-range patch')
        result.extend(original[cursor:target]); cursor = target; index += 1
        if new_start != len(result) + (1 if new_count else 0): raise ValueError('new patch position mismatch')
        removed = added = 0
        while index < len(lines) and not lines[index].startswith('@@ '):
            line = lines[index]; index += 1
            if not line or line[0] not in ' +-': raise ValueError('unsupported patch line')
            kind, value = line[0], line[1:]
            if not value.endswith('\n'): value += '\n'
            if kind in ' -':
                if cursor >= len(original) or original[cursor] != value: raise ValueError('patch context differs from source')
                cursor += 1; removed += 1
            if kind in ' +': result.append(value); added += 1
        if removed != old_count or added != new_count: raise ValueError('patch hunk count mismatch')
        hunks += 1
    if not hunks: raise ValueError('empty patch')
    result.extend(original[cursor:])
    candidate = ''.join(result)
    if candidate == source or len(candidate.encode()) > 65536: raise ValueError('unchanged or oversized candidate')
    return candidate
