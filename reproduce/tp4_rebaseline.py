"""The owner's two exact, one-use TP4 control exceptions; no general replay API."""
import hashlib
import json
from pathlib import Path
import re
import runtime

OWNER_APPROVAL_SHA256 = '85ec4f81af2e061e5e146ecc345213c1b6ee303588861a193c40b61202c645d5'
PROFILE_SHA256 = 'd020daff21fc85db044138972ca3d4b0d61eddeb6bcbc81ca41b22ec726fe8ed'
SPECS = {
    'fixedqual-357ad7e778ea9e9cf88250b860ebd27da0d25bc46c792eb47cf2b413a5bd12e5': {
        'decision_id': 'tp4-v1-control-rebaseline-20261103',
        'cell_id': 'tuning_trial-2xtp4-control_before-1-8000-dspark_static', 'repetition': 0,
        'prior': {
            'attempt-0d14422a67b340e2aab9c3407c848b6e': ('interrupted', '1fbfe15e70f5744259f32b7759a21b2b20908b7f5389294e0c6d39c21d9e1912'),
            'attempt-ada26f5b0c734100b4aed330bdc1336d': ('passed', 'ff604fb4f68361f2ce4802c3e738241904c69bed4f9aadfef8f89f826a0ae92a')},
        'maximum_total_consumed_attempts': 3},
    'fixedqual-5ed8ed1f3c0b097cb18902603ab45955f4004e639c9e82cbb0833d8655341a6a': {
        'decision_id': 'tp4-c8r2-matched-control-rebaseline-20261103',
        'cell_id': 'tuning_trial-2xtp4-control_before-8-8000-dspark_static', 'repetition': 2,
        'prior': {'attempt-716924d5b6e64f6387bfcb65ff2a7a03':
                  ('passed', '5c8e668d36b8d1613b93c9d5beb91f378d5091b4d77ecef7655e51cf2fbb4313')},
        'maximum_total_consumed_attempts': 2}}


def arm_identities():
    return {'fixedqual-' + hashlib.sha256(json.dumps(
        [f'tuning_trial-2xtp4-control_before-{c}-8000-dspark_static', PROFILE_SHA256, r],
        separators=(',', ':')).encode()).hexdigest()
        for c in (1, 8, 32) for r in range(3)}


def reservation_context(cells, group_id, values):
    if (set(values) != set(SPECS) or {c['cell_id'] for c in cells} != arm_identities() or
            not re.fullmatch(r'tuning-static-2xtp4-before-i8-[0-9]{8}t[0-9]{4}z', group_id)):
        raise runtime.RuntimeErrorClosed('named rebaselines require the full registered shared opening reservation')


def declaration(cell_id, value):
    spec = SPECS.get(cell_id)
    if (spec is None or not isinstance(value, dict) or
            set(value) != {'decision_id', 'owner_approval_sha256', 'reason'} or
            value['decision_id'] != spec['decision_id'] or
            value['owner_approval_sha256'] != OWNER_APPROVAL_SHA256 or
            not isinstance(value['reason'], str) or not 12 <= len(value['reason'].strip()) <= 500 or
            '\n' in value['reason'] or '\r' in value['reason']):
        raise runtime.RuntimeErrorClosed('exact named owner rebaseline declaration required')
    return spec


def context(config, rows):
    """Both extra measurements belong to the already required nine-cell arm."""
    values = config.get('rebaseline_cells')
    if values is None:
        return
    if (not isinstance(values, dict) or set(values) != set(SPECS) or
            config.get('execution_phase') != 'qualification' or config.get('layout') != '2xtp4' or
            config.get('isolated_measurement_sampler') is not True or len(rows) != 9 or
            not set(values).issubset({r['identity'] for r in rows}) or
            config.get('retry', False) is not False or config.get('retry_reason') is not None):
        raise runtime.RuntimeErrorClosed('rebaseline requires the complete shared TP4 opening arm')
    expected = {(f'tuning_trial-2xtp4-control_before-{c}-8000-dspark_static', r)
                for c in (1, 8, 32) for r in range(3)}
    if {(r['config']['cell_id'], r['config']['repetition']) for r in rows} != expected:
        raise runtime.RuntimeErrorClosed('rebaseline shared arm population differs')
    for row in rows:
        prepared = row['prepared']
        if (prepared['binding']['profile_sha256'] != PROFILE_SHA256 or
                prepared['binding']['delta'] != {} or
                any(t.get('prompt_version') != 'fixed-performance-v1' for t in prepared['tasks'])):
            raise runtime.RuntimeErrorClosed('rebaseline requires the pinned NoSpec V1 inputs')
        if row['identity'] in values:
            spec = declaration(row['identity'], values[row['identity']])
            if (row['config']['cell_id'], row['config']['repetition']) != (spec['cell_id'], spec['repetition']):
                raise runtime.RuntimeErrorClosed('rebaseline cell identity differs')


def validate_history(root, cell_id, value, old, method_sha256):
    """Check the exact retained history before any block reservation is written."""
    spec = declaration(cell_id, value)
    if (len(old) != len(spec['prior']) or {r['directory'] for r in old} != set(spec['prior']) or
            not isinstance(method_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', method_sha256)):
        raise runtime.RuntimeErrorClosed('named rebaseline already consumed or prior history differs')
    prior_hashes = {}
    for row in old:
        status, method = spec['prior'][row['directory']]
        path = Path(root) / row['directory']
        metadata = path / 'attempt.json'
        if (path.is_symlink() or metadata.is_symlink() or row['status'] != status or
                row.get('method_sha256') != method or method_sha256 == method or
                row.get('rebaseline_authorization') is not None):
            raise runtime.RuntimeErrorClosed('named rebaseline prior status or changed method differs')
        raw = metadata.read_bytes()
        if json.loads(raw) != {k: v for k, v in row.items() if k != 'directory'}:
            raise runtime.RuntimeErrorClosed('named rebaseline prior metadata changed')
        evidence = row.get('evidence_sha256', {})
        if status == 'passed' and not evidence:
            raise runtime.RuntimeErrorClosed('named rebaseline prior pass evidence missing')
        for name, digest in evidence.items():
            file = path / name
            if (Path(name).name != name or file.is_symlink() or not file.is_file() or
                    hashlib.sha256(file.read_bytes()).hexdigest() != digest):
                raise runtime.RuntimeErrorClosed('named rebaseline prior evidence changed')
        prior_hashes[row['directory']] = hashlib.sha256(raw).hexdigest()
    return {'decision_id': spec['decision_id'], 'owner_approval_sha256': OWNER_APPROVAL_SHA256,
            'reason': value['reason'], 'prior_attempt_sha256': prior_hashes,
            'maximum_total_consumed_attempts': spec['maximum_total_consumed_attempts'],
            'scope': 'one additional named unscored TP4 opening control only; old records unchanged; no further retry'}
