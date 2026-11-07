"""Synthetic registry/context checks for two exact owner-approved exceptions."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import attempts
import runtime
import tp4_rebaseline as policy


class TP4RebaselineTests(unittest.TestCase):
    def fixture(self, root):
        owner = {'boot_id': 'synthetic-boot', 'pid': os.getpid(), 'start_ticks': 1}
        declarations = {ident: {'decision_id': spec['decision_id'],
                               'owner_approval_sha256': policy.OWNER_APPROVAL_SHA256,
                               'reason': 'Use registered V1 inputs and isolated sampler in the shared opening arm.'}
                        for ident, spec in policy.SPECS.items()}
        for ident, spec in policy.SPECS.items():
            for name, (status, method) in spec['prior'].items():
                path = root / name
                path.mkdir()
                (path / 'raw.json').write_text('{"synthetic":true}\n')
                runtime.atomic_json(path / 'attempt.json', {
                    'attempt_id': name, 'cell_id': ident, 'status': status, 'method_sha256': method,
                    'evidence_sha256': {'raw.json': hashlib.sha256((path / 'raw.json').read_bytes()).hexdigest()}})
        retry_ids = sorted(policy.arm_identities() - set(policy.SPECS))[:3]
        for ident in retry_ids:
            path = attempts.begin(root, ident, method_sha256='a' * 64)
            attempts.mark_interrupted(path, 'Synthetic independently reconciled interruption.')
        specs = [{'cell_id': ident, 'method_sha256': 'f' * 64} for ident in sorted(policy.arm_identities())]
        retries = {ident: 'Changed V1 generator and isolated sampler after retained legacy interruption.' for ident in retry_ids}
        return owner, declarations, specs, retries

    def reserve(self, root, owner, declarations, specs, retries, suffix='20261103t0418z'):
        with patch.object(runtime, 'process_identity', return_value=owner):
            return attempts.begin_block(root, specs, 'tuning-static-2xtp4-before-i8-' + suffix, owner,
                                        retry_cells=retries, rebaseline_cells=declarations)

    def test_exact_mixed_shared_arm_preserves_history_and_exhausts_both_allowances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner, declarations, specs, retries = self.fixture(root)
            before = {p.name: (p / 'attempt.json').read_bytes() for p in root.glob('attempt-*')}
            with patch.object(runtime, 'process_identity', return_value=owner):
                with self.assertRaises(runtime.RuntimeErrorClosed):
                    attempts.begin_block(root, specs, 'ordinary', owner, retry_cells=retries)
            paths = self.reserve(root, owner, declarations, specs, retries)
            self.assertEqual(len(paths), 9)
            for name, raw in before.items():
                self.assertEqual((root / name / 'attempt.json').read_bytes(), raw)
            receipts = [json.loads((path / 'attempt.json').read_text()) for path in paths]
            self.assertEqual(sum('rebaseline_authorization' in r for r in receipts), 2)
            for row in receipts:
                if row['cell_id'] in policy.SPECS:
                    proof = row['rebaseline_authorization']
                    self.assertEqual(proof['maximum_total_consumed_attempts'], policy.SPECS[row['cell_id']]['maximum_total_consumed_attempts'])
                    self.assertEqual(set(proof['prior_attempt_sha256']), set(policy.SPECS[row['cell_id']]['prior']))
            for path in paths:
                (path / 'new-raw.json').write_text('{"synthetic_new":true}\n')
                attempts.finish(path, 'failed', ['new-raw.json'])
            count = len(attempts.load_attempts(root))
            with self.assertRaises(runtime.RuntimeErrorClosed):
                self.reserve(root, owner, declarations, specs, retries, '20261103t0420z')
            self.assertEqual(len(attempts.load_attempts(root)), count)
            for ident in policy.SPECS:
                with self.assertRaises(runtime.RuntimeErrorClosed):
                    attempts.begin(root, ident, retry=True, retry_reason='No further named control replay is permitted.', method_sha256='e' * 64)

    def test_wrong_authority_partial_arm_or_changed_history_reserves_nothing(self):
        for mutation in ('approval', 'unknown_cell', 'partial_arm', 'prior_status', 'prior_method', 'raw_changed', 'same_method'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                owner, declarations, specs, retries = self.fixture(root)
                ident = next(iter(policy.SPECS))
                old_name = next(n for n, (status, _) in policy.SPECS[ident]['prior'].items() if status == 'passed')
                old = root / old_name
                if mutation == 'approval': declarations[ident]['owner_approval_sha256'] = '0' * 64
                elif mutation == 'unknown_cell': declarations['unapproved'] = declarations.pop(ident)
                elif mutation == 'partial_arm': specs = [s for s in specs if s['cell_id'] == ident]; retries = {}
                elif mutation in ('prior_status', 'prior_method'):
                    row = json.loads((old / 'attempt.json').read_text())
                    row['status' if mutation == 'prior_status' else 'method_sha256'] = 'invalid' if mutation == 'prior_status' else 'b' * 64
                    runtime.atomic_json(old / 'attempt.json', row)
                elif mutation == 'raw_changed': (old / 'raw.json').write_text('{"tampered":true}\n')
                else:
                    next(s for s in specs if s['cell_id'] == ident)['method_sha256'] = policy.SPECS[ident]['prior'][old_name][1]
                before = {p.name: (p / 'attempt.json').read_bytes() for p in root.glob('attempt-*')}
                with self.assertRaises((ValueError, runtime.RuntimeErrorClosed)):
                    self.reserve(root, owner, declarations, specs, retries)
                self.assertEqual({p.name: (p / 'attempt.json').read_bytes() for p in root.glob('attempt-*')}, before)

    def test_request_context_requires_both_exact_v1_controls_in_shared_nospec_arm(self):
        declarations = {ident: {'decision_id': s['decision_id'], 'owner_approval_sha256': policy.OWNER_APPROVAL_SHA256,
                               'reason': 'Use the exact approved shared V1 opening control.'} for ident, s in policy.SPECS.items()}
        config = {'execution_phase': 'qualification', 'layout': '2xtp4',
                  'isolated_measurement_sampler': True, 'rebaseline_cells': declarations}
        rows = []
        for c in (1, 8, 32):
            for r in range(3):
                cell = f'tuning_trial-2xtp4-control_before-{c}-8000-dspark_static'
                ident = 'fixedqual-' + hashlib.sha256(json.dumps([cell, policy.PROFILE_SHA256, r], separators=(',', ':')).encode()).hexdigest()
                rows.append({'identity': ident, 'config': {'cell_id': cell, 'repetition': r},
                             'prepared': {'binding': {'profile_sha256': policy.PROFILE_SHA256, 'delta': {}},
                                          'tasks': [{'prompt_version': 'fixed-performance-v1'}]}})
        policy.context(config, rows)
        for mutate in (lambda c, r: c.update(isolated_measurement_sampler=False),
                       lambda c, r: r.pop(),
                       lambda c, r: r[0]['prepared']['tasks'][0].update(prompt_version='fixed-cache-v2'),
                       lambda c, r: r[0]['prepared']['binding'].update(delta={'unapproved': True})):
            altered, population = copy.deepcopy(config), copy.deepcopy(rows)
            mutate(altered, population)
            with self.assertRaises(runtime.RuntimeErrorClosed): policy.context(altered, population)


if __name__ == '__main__':
    unittest.main()
