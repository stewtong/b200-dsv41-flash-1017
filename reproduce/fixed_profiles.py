"""Freeze the four owner-selected NoSpec configurations without tuning claims."""
import copy
import hashlib
import json
import re
import comparison_scope
import launcher
import profile_registry


def digest(record):
    return hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != 'registration_sha256'},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def register(layout, manifest, evidence):
    runtime = manifest.get('runtime', {})
    if (manifest.get('scope_id') != comparison_scope.SCOPE_ID or layout not in comparison_scope.LAYOUTS or
            any(not re.fullmatch(r'[a-f0-9]{64}', str(manifest.get(k))) for k in ('plan_sha256', 'matrix_sha256')) or
            not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[a-f0-9]{64}', str(runtime.get('image_digest'))) or
            runtime.get('model_revision') != '1bae0e1088993b9b82de08f59de5be5a34e42170' or
            runtime.get('tokenizer_revision') != runtime.get('model_revision')):
        raise ValueError('fixed profile requires the bound336-cell plan, source and selected layout')
    if not isinstance(evidence, dict) or set(evidence) != {'known_answer', 'performance_path'}:
        raise ValueError('fixed profile requires retained basic qualification references')
    for item in evidence.values():
        if (not isinstance(item, dict) or set(item) != {'source_seal', 'evidence_sha256', 'verifier_sha256'} or
                any(not re.fullmatch(r'[a-f0-9]{64}', str(v)) for v in item.values())):
            raise ValueError('fixed profile qualification reference is malformed')
    cells = sorted(c['id'] for c in manifest['cells'] if c['layout'] == layout)
    if len(cells) != 84:
        raise ValueError('fixed profile does not cover all72cold and12warm cells')
    template = profile_registry.template(layout, {})
    profile = hashlib.sha256(json.dumps(template, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if profile != launcher.profile_sha256(layout):
        raise ValueError('fixed configuration differs from the qualified NoSpec starting profile')
    record = {'schema': 'frozen-fixed-profile/v1', 'id': 'fixed-nospec-' + layout, 'layout': layout,
              'profile_delta': {}, 'profile_sha256': profile, 'template': template,
              'fixed_cell_ids': cells, 'plan_sha256': manifest['plan_sha256'],
              'matrix_sha256': manifest['matrix_sha256'], 'image_digest': runtime['image_digest'],
              'model_revision': runtime['model_revision'], 'tokenizer_revision': runtime['tokenizer_revision'],
              'qualification_evidence': copy.deepcopy(evidence), 'status': 'frozen_fixed_configuration',
              'scope': 'owner-selected qualified NoSpec configuration; descriptive performance comparison, no optimization or quality-equivalence claim'}
    record['registration_sha256'] = digest(record)
    return record


def validate(record, manifest):
    if not isinstance(record, dict) or record.get('schema') != 'frozen-fixed-profile/v1':
        raise ValueError('frozen fixed configuration record missing')
    if record != register(record['layout'], manifest, record['qualification_evidence']):
        raise ValueError('frozen fixed configuration or qualification reference changed')
    return record


def resolve(config, manifest):
    layout = config['layout']
    record = validate(manifest['runtime'].get('fixed_profiles', {}).get(layout), manifest)
    if (config.get('profile_id') != record['id'] or config.get('trial_profile_id') is not None or
            'profile_overrides' in config or
            manifest['runtime'].get('profile_hashes', {}).get(layout) != record['profile_sha256']):
        raise ValueError('fixed execution does not select its frozen NoSpec configuration')
    return {'phase': 'fixed-performance', 'delta': {}, 'profile_sha256': record['profile_sha256'], 'record': record}
