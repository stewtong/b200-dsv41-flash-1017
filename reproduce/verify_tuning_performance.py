"""Pair verified tuning arms; throughput evidence alone never freezes a profile."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import campaign
import profile_registry
import runtime
import verify_fixed_block
import verify_fixed_qualification

ARMS = ('control_before', 'challenger', 'control_after')
METHOD = 'equal-anchor-paired-log-ratios/repetition-block-bootstrap-27/v1'


def geometric(values):
    values = list(values)
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('finite positive throughput observations required')
    return math.exp(sum(math.log(v) for v in values) / len(values))


def compare(observations, anchors):
    """Arithmetic only; verify() establishes raw provenance before calling this."""
    if not isinstance(anchors, list) or not anchors or len(set(anchors)) != len(anchors):
        raise ValueError('distinct preregistered anchors required')
    expected = {(shape, repetition, arm) for shape in anchors for repetition in range(3) for arm in ARMS}
    by_key = {}
    for row in observations:
        if type(row.get('repetition')) is not int or not 0 <= row['repetition'] < 3:
            raise ValueError('exact integer tuning repetition required')
        key = (row['anchor'], row['repetition'], row['arm'])
        if key not in expected or key in by_key:
            raise ValueError('duplicate or undeclared tuning observation')
        geometric([row['rate']])
        if row.get('all_requests_valid') is not True:
            raise ValueError('failed or partial trial cannot disappear into a performance comparison')
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError('missing tuning arms or repetitions; retain incomplete/failed trials without adoption')
    paired = []
    for shape in anchors:
        for repetition in range(3):
            arms = [by_key[(shape, repetition, arm)] for arm in ARMS]
            if (len({r['tasks_sha256'] for r in arms}) != 1 or
                    len({r['requests'] for r in arms}) != 1 or
                    any(type(r['requests']) is not int or r['requests'] < 1 for r in arms)):
                raise ValueError('paired tuning workload bytes or request counts differ')
            if (len({r['boot_id'] for r in arms}) != 1 or
                    any(type(r[k]) is not int or not 0 <= r['start_ns'] < r['end_ns']
                        for r in arms for k in ('start_ns', 'end_ns')) or
                    not arms[0]['end_ns'] <= arms[1]['start_ns'] or
                    not arms[1]['end_ns'] <= arms[2]['start_ns']):
                raise ValueError('opening, challenger and closing windows are not ordered on one boot')
            engines = [set(r['engine_ids']) for r in arms]
            if (any(not ids for ids in engines) or
                    any(engines[i] & engines[j] for i in range(3) for j in range(i + 1, 3))):
                raise ValueError('tuning arms require distinct fresh engine identities')
            a, b, z = [r['rate'] for r in arms]
            drift = z / a - 1
            paired.append({'anchor': shape, 'repetition': repetition, 'requests_per_arm': arms[0]['requests'],
                           'tasks_sha256': arms[0]['tasks_sha256'],
                           'control_before_rate': a, 'challenger_rate': b, 'control_after_rate': z,
                           'paired_ratio': b / math.sqrt(a * z),
                           'opening_to_closing_control_change': drift,
                           'within_five_percent_control_drift': abs(drift) <= .05})
    # Each anchor has one vote. Requests and tokens are not independent
    # replication units. Resample the three paired repetition blocks jointly
    # across anchors, retaining possible shared host/clock effects.
    per_repetition = [geometric(r['paired_ratio'] for r in paired if r['repetition'] == i) for i in range(3)]
    samples = sorted(geometric(per_repetition[i] for i in draw)
                     for draw in itertools.product(range(3), repeat=3))
    low, high = samples[math.ceil(.025 * len(samples)) - 1], samples[math.ceil(.975 * len(samples)) - 1]
    score = geometric(geometric(r['paired_ratio'] for r in paired if r['anchor'] == shape) for shape in anchors)
    stable = all(r['within_five_percent_control_drift'] for r in paired)
    gain = score - 1
    gate = stable and gain >= profile_registry.ACCEPTANCE['minimum_gain'] and low - 1 >= profile_registry.ACCEPTANCE['minimum_gain']
    return {'method': METHOD, 'anchor_count': len(anchors), 'paired_repetitions': len(paired),
            'source_repetitions': len(observations), 'requests_all_arms': sum(r['requests'] for r in observations),
            'equal_anchor_geomean_paired_ratio': score, 'observed_gain': gain,
            'repetition_block_ratios': per_repetition,
            'repetition_ratio_range': [min(per_repetition), max(per_repetition)],
            'paired_ratio_95pct_percentile_interval': [low, high],
            'uncertainty_scope': 'exhaustive 27 resamples of three paired repetition blocks; coarse small-sample percentile interval, not an independent-request or population guarantee',
            'controls_within_five_percent': stable,
            'performance_gate_passed': gate, 'performance_adoption_eligible': False,
            'pairs': paired,
            'remaining_adoption': ['declared integrity and task-quality controls',
                                   'paired GSM8K noninferiority and identifier/tool/stream checks',
                                   'protected p95 and required 100-request tail-validation evidence',
                                   'complete lever decisions and qualified profile freeze'],
            'scope': 'matched tuning performance evidence only; no frozen profile, scored coverage or automatic adoption'}


def verify(manifest_path, profile_id, catalog_path):
    manifest_path, catalog_path = Path(manifest_path), Path(catalog_path)
    deriver_sha = campaign.digest(Path(__file__))
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('matrix_sha256') != campaign.digest(campaign.HERE / 'matrix.json'):
        raise ValueError('tuning matrix differs from current source')
    profiles = manifest['runtime']['qualification_profiles']
    candidate = profile_registry.validate_record(profiles[profile_id], manifest)
    canonical = {r['id']: r for r in campaign.ledger(campaign.HERE / 'matrix.json')['cells']}
    declared = {r['id']: r for r in manifest['cells']}
    if len(declared) != len(manifest['cells']):
        raise ValueError('duplicate manifest cells')
    selected = [canonical.get(ident) for ident in candidate['deciding_cell_ids']]
    if any(r is None or r['suite'] not in ('tuning_trial', 'tuning_extra') for r in selected):
        raise ValueError('registered deciding cells are not tuning trials')
    groups = {(r['suite'], r['layout'], r['parameters']['lever']) for r in selected}
    if len(groups) != 1:
        raise ValueError('tuning comparison mixes suites, layouts or levers')
    suite, layout, lever = groups.pop()
    expected_cells = {ident for ident, r in canonical.items() if r['suite'] == suite and r['layout'] == layout and r['parameters']['lever'] == lever}
    if set(candidate['deciding_cell_ids']) != expected_cells or len(expected_cells) != 36:
        raise ValueError('retain all twelve planned anchors and all three tuning arms')
    if any({k: v for k, v in declared.get(ident, {}).items() if k != 'status'} !=
           {k: v for k, v in canonical[ident].items() if k != 'status'} for ident in expected_cells):
        raise ValueError('declared tuning cell differs from canonical ledger')
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    if not isinstance(catalog, list) or not catalog:
        raise ValueError('explicit retained tuning source catalog required')
    observations, evidence, seen_roots = [], [], set()
    for entry in catalog:
        if not isinstance(entry, dict) or set(entry) != {'root', 'seal'}:
            raise ValueError('catalog requires exact root and final seal')
        root = Path(entry['root']).resolve(strict=True)
        if root in seen_roots:
            raise ValueError('duplicate tuning source root')
        seen_roots.add(root)
        runtime.verify_evidence_manifest(root)
        seal = campaign.digest(root / 'SHA256SUMS')
        if seal != entry['seal']:
            raise ValueError('tuning source seal differs')
        if (json.loads((root / 'manifest.json').read_text()) != manifest or
                json.loads((root / 'cleanup-result.json').read_text()).get('success') is not True or
                json.loads((root / 'reconciliation.json').read_text()).get('status') != 'complete'):
            raise ValueError('tuning source dependencies, cleanup or reconciliation differ')
        config = json.loads((root / 'run-config.json').read_text())
        state = json.loads((root / 'execution-state.json').read_text())
        if isinstance(config.get('cells'), list):
            checked = verify_fixed_block.verify(root)
            roots = [root / 'cells' / row['cell_id'] for row in checked['cells']]
        else:
            checked = verify_fixed_qualification.verify(root)
            roots = [root]
        for child in roots:
            cfg = json.loads((child / 'run-config.json').read_text())
            if cfg['cell_id'] not in expected_cells or cfg.get('trial_profile_id') != profile_id:
                raise ValueError('source trial differs from selected candidate')
            cell = canonical[cfg['cell_id']]
            params = cell['parameters']
            inputs = child / 'fixed-input-source/fixed-tasks.json'
            tasks = json.loads(inputs.read_text())
            if any(t.get('prompt_version') != 'fixed-performance-v1' for t in tasks):
                raise ValueError('new tuning comparison cannot relabel a legacy workload')
            report = json.loads((child / 'fixed-attribution/requests/measured/fixed-verification.json').read_text())
            raw_rows = [json.loads((child / 'fixed-attribution/requests/measured' / (t['id'] + '.json')).read_text()) for t in tasks]
            observations.append({'anchor': (params['input'], params['concurrency']),
                                 'repetition': cfg['repetition'], 'arm': params['arm'],
                                 'rate': report['valid_tokens_per_second'], 'requests': report['requests'],
                                 'tasks_sha256': campaign.digest(inputs),
                                 'boot_id': state['driver_identity']['boot_id'],
                                 'engine_ids': state['container_ids'],
                                 'start_ns': min(r['start_ns'] for r in raw_rows),
                                 'end_ns': max(r['end_ns'] for r in raw_rows),
                                 'all_requests_valid': report['valid_requests'] == report['requests']})
        runtime.verify_evidence_manifest(root)
        if campaign.digest(root / 'SHA256SUMS') != seal:
            raise ValueError('tuning source changed during verification')
        evidence.append({'root': str(root), 'seal': seal, 'repetitions': len(roots)})
    anchors = sorted({(r['parameters']['input'], r['parameters']['concurrency']) for r in selected})
    result = compare(observations, anchors)
    if (manifest_path.read_bytes() != manifest_bytes or catalog_path.read_bytes() != catalog_bytes or
            campaign.digest(Path(__file__)) != deriver_sha):
        raise ValueError('tuning derivation inputs changed during verification')
    return {**result, 'at': datetime.now(timezone.utc).isoformat(), 'profile_id': profile_id,
            'profile_sha256': candidate['profile_sha256'], 'control_profile_sha256': candidate['control_profile_sha256'],
            'registration_sha256': candidate['registration_sha256'], 'layout': layout, 'lever': lever,
            'deriver_source_sha256': deriver_sha,
            'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(), 'evidence': evidence}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--profile-id', required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest, args.profile_id, args.catalog)
    campaign.atomic(args.output, result)
    print(json.dumps({k: result[k] for k in ('performance_gate_passed', 'performance_adoption_eligible', 'observed_gain', 'controls_within_five_percent')}))
