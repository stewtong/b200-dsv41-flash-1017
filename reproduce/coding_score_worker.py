"""Container-only executable scorer. The host must not import or run candidates.

Launch with isolated Python in a CPU-only, networkless, unprivileged, read-only
container. The explicit environment marker is a misuse guard, not isolation.
"""
import ast
import copy
import hashlib
import json
import os
import resource
import sys


def main():
    if os.environ.get('DSV41_ISOLATED_CODING_SCORER') != '1':
        raise SystemExit('container scorer environment required')
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    raw = sys.stdin.buffer.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024: raise ValueError('scorer input too large')
    request = json.loads(raw)
    if set(request) != {'source', 'cases'} or not isinstance(request['source'], str) or not isinstance(request['cases'], list) or not 1 <= len(request['cases']) <= 100:
        raise ValueError('invalid scorer input')
    source = request['source']; tree = ast.parse(source)
    # Narrow pure-Python task language. Container isolation remains mandatory.
    permitted_modules = {'collections', 'math', 'functools', 'itertools', 'heapq', 'bisect', 're'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            raise ValueError('candidate uses a disallowed construct')
        if isinstance(node, ast.Import) and any(alias.name not in permitted_modules for alias in node.names):
            raise ValueError('candidate import is not allowed')
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in permitted_modules or
                                                any(alias.name.startswith('_') or alias.name == '*' for alias in node.names)):
            raise ValueError('candidate import is not allowed')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('private attribute access forbidden')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('private name forbidden')
    import builtins
    def permitted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in permitted_modules or level or any(str(item).startswith('_') or item == '*' for item in fromlist):
            raise ValueError('candidate import is not allowed')
        return builtins.__import__(name, globals, locals, fromlist, level)
    names = ('abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float', 'int', 'isinstance',
             'len', 'list', 'map', 'max', 'min', 'range', 'reversed', 'round', 'set', 'sorted', 'str',
             'sum', 'tuple', 'zip', 'ValueError', 'TypeError')
    namespace = {'__builtins__': {name: getattr(builtins, name) for name in names}}
    namespace['__builtins__']['__import__'] = permitted_import
    exec(compile(tree, 'solution.py', 'exec'), namespace)
    solve = namespace.get('solve')
    if not callable(solve): raise ValueError('solve function missing')
    outcomes = []
    for index, case in enumerate(request['cases']):
        try:
            actual = solve(copy.deepcopy(case['input']))
            # Preserve JSON types: True must not satisfy expected 1.
            actual_json = json.dumps(actual, sort_keys=True, allow_nan=False)
            passed = actual_json == json.dumps(case['expected'], sort_keys=True, allow_nan=False)
            outcomes.append({'index': index, 'passed': passed, 'actual': json.loads(actual_json)})
        except Exception as exc:
            outcomes.append({'index': index, 'passed': False, 'error_type': type(exc).__name__})
    print(json.dumps({'schema': 'coding-executable-score/v1', 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'tests': outcomes, 'passed': all(row['passed'] for row in outcomes)}, sort_keys=True))


if __name__ == '__main__':
    main()
