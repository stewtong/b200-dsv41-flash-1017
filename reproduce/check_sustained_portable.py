"""Run five offline portable-helper regressions from this installed checkout."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / 'results/sustained-speculation-layouts/portable-candidate'


def require(condition, label):
    if not condition:
        raise ValueError('offline regression failed: ' + label)


def main():
    # This explicit executable tests this checkout's code, never code from a
    # caller-supplied data directory accepted by verify_result_package.py.
    sys.path.insert(0, str(PORTABLE))
    spec = importlib.util.spec_from_file_location('sustained_portable_fixture', PORTABLE / 'replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed = []
    with tempfile.TemporaryDirectory(prefix='sustained-portable-') as temp:
        base = Path(temp)
        (base / 'code.txt').write_text('alpha beta\n')
        (base / 'prose.txt').write_text('public text\n')
        for name, complete, reason, should_pass in [('complete_stop', True, 'stop', True),
                ('truncated', False, 'stop', False), ('length', True, 'length', False)]:
            args = SimpleNamespace(code_corpus=str(base / 'code.txt'), prose_corpus=str(base / 'prose.txt'),
                                   arm='fixture', router_port=0, output_dir=str(base / name))
            result = {'request_id': name, 'complete': complete, 'events': [{'kind': 'event',
                      'data': {'choices': [{'delta': {'content': 'public natural stop check passed'}}]}}]}
            rejected = False
            with patch.object(module.stream, 'capture', return_value=result), patch.object(module.stream,
                    'summarize_stream', return_value={'finish_reason': reason, 'usage': {}}):
                try:
                    module.natural_stop(args)
                except RuntimeError:
                    rejected = True
            receipt = json.loads((Path(args.output_dir) / 'natural-stop.json').read_text())
            require(receipt['success'] == should_pass and rejected != should_pass, name)
            passed.append(name)
        for name, valid in [('forced_smoke_none', True), ('forced_smoke_bad_receipt', False)]:
            args = SimpleNamespace(code_corpus=str(base / 'code.txt'), prose_corpus=str(base / 'prose.txt'),
                    workload='W1', rate=0.05, repeat=0, output_tokens=400, tokenize_url='', router_port=0,
                    arm='fixture', output_dir=str(base / name), base_tokens=None)
            summary = {'finish_reason': 'length', 'usage': {'prompt_tokens': 128,
                       'completion_tokens': 400 if valid else 399}}
            error = None
            with patch.object(module, 'calibrate_profile', side_effect=lambda corpus, *a, **kw:
                    (module.character_profile(corpus, 'W1'), [])), patch.object(module, 'tokenize',
                    return_value={'count': 128, 'response_keys': ['count'], 'request_sha256': 'fixture'}), \
                    patch.object(module.stream, 'capture', return_value={'request_id': name, 'complete': True, 'events': []}), \
                    patch.object(module.stream, 'summarize_stream', return_value=summary):
                try:
                    module.forced_smoke(args)
                except Exception as exc:
                    error = exc
            receipt = json.loads((Path(args.output_dir) / 'forced-smoke.json').read_text())
            require(receipt['success'] == valid, name)
            require(error is None if valid else isinstance(error, RuntimeError), name)
            passed.append(name)
    print(json.dumps({'status': 'passed', 'executed_offline_cases': passed, 'network_or_inference': False}))


if __name__ == '__main__':
    main()
