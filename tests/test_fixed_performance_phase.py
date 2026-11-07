"""Selected fixed cells can proceed while later supplemental inputs stay open."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'reproduce'
sys.path.insert(0, str(SOURCE))
import campaign
import sampling_source


class FixedPerformancePhaseTests(unittest.TestCase):
    def setUp(self):
        self.manifest = campaign.ledger(SOURCE / 'matrix.json')
        self.manifest['plan_sha256'] = 'd' * 64
        proof = sampling_source.verify(Path(__file__).with_name('fixtures') / 'pinned-model-card.md')
        self.manifest['runtime'] = {
            'image_digest': 'example/image@sha256:' + 'a' * 64,
            'profile_hashes': dict.fromkeys(('2xtp4', '4xdep2', '2xdep4', 'dep8'), 'b' * 64),
            'operational_deadline': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            'model_revision': proof['revision'], 'tokenizer_revision': proof['revision'],
            'router_pin': 'c' * 64, 'retirement_gate': 'passed',
            'safety_controls': dict.fromkeys(('ownership', 'watchdog', 'cancellation', 'cleanup', 'persistence'), True),
            'capture_schema': 'http-body-chunks/v1', 'temperature': 1.0, 'top_p': 1.0,
            'sampling_source': proof,
        }
        self.cell = next(r['id'] for r in self.manifest['cells'] if r['suite'] == 'core')

    def test_partial_manifest_keeps_global_gate_and_all_planned_cells(self):
        before = copy.deepcopy(self.manifest)
        self.assertFalse(self.manifest['complete_universe'])
        self.assertTrue(campaign.validate_phase(self.manifest, 'fixed-performance', cell_ids=[self.cell]))
        for suite in ('output_sensitivity', 'historical_bridge_extra'):
            output = next(r['id'] for r in self.manifest['cells'] if r['suite'] == suite)
            self.assertTrue(campaign.validate_phase(self.manifest, 'fixed-performance', cell_ids=[output]))
        self.assertEqual(self.manifest, before)
        with self.assertRaisesRegex(ValueError, 'complete_universe'):
            campaign.validate_phase(self.manifest, 'scored')

    def test_selection_cannot_smuggle_unresolved_or_changed_workloads(self):
        quality = next(r['id'] for r in self.manifest['cells'] if r['suite'] == 'quality')
        for ids in (None, [], [self.cell, self.cell], ['unknown'], [quality]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                campaign.validate_phase(self.manifest, 'fixed-performance', cell_ids=ids)
        for mutation in ('contract', 'matrix', 'duplicate', 'missing_base'):
            m = copy.deepcopy(self.manifest)
            if mutation == 'contract':
                next(r for r in m['cells'] if r['id'] == self.cell)['request_contract']['automatic_retry'] = True
            elif mutation == 'matrix':
                m['matrix_sha256'] = '0' * 64
            elif mutation == 'duplicate':
                m['cells'].append(copy.deepcopy(m['cells'][0]))
            else:
                m['cells'] = [r for r in m['cells'] if r['id'] != quality]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                campaign.validate_phase(m, 'fixed-performance', cell_ids=[self.cell])

    def test_runtime_requirements_still_apply_to_selected_cells(self):
        for field in ('profile_hashes', 'sampling_source', 'capture_schema', 'router_pin'):
            m = copy.deepcopy(self.manifest)
            m['runtime'].pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                campaign.validate_phase(m, 'fixed-performance', cell_ids=[self.cell])
        m = copy.deepcopy(self.manifest)
        m['runtime']['safety_controls']['cleanup'] = False
        with self.assertRaises(ValueError):
            campaign.validate_phase(m, 'fixed-performance', cell_ids=[self.cell])

    def test_actual_cli_requires_exact_plan_and_selection_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.md'
            plan.write_text('fixture plan')
            manifest = root / 'manifest.json'
            self.manifest['plan_sha256'] = campaign.digest(plan)
            manifest.write_text(json.dumps(self.manifest))
            before = manifest.read_bytes()
            argv = [sys.executable, '-B', str(SOURCE / 'campaign.py'), 'validate-runtime',
                    '--manifest', str(manifest), '--plan', str(plan), '--phase', 'fixed-performance']
            passed = subprocess.run(argv + ['--cell', self.cell], capture_output=True, text=True, timeout=20)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn('does not qualify their evidence or start a run', passed.stdout)
            missing = subprocess.run(argv, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(missing.returncode, 0)
            plan.write_text('changed plan')
            mismatch = subprocess.run(argv + ['--cell', self.cell], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn('plan content hash mismatch', mismatch.stderr)
            self.assertEqual(manifest.read_bytes(), before)
