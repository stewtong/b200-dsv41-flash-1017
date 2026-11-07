"""Matched-control arithmetic and fail-closed tuning report boundary controls."""
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'reproduce'
sys.path.insert(0, str(SOURCE))
import campaign
import profile_registry
import verify_tuning_performance as tuning


def observations(anchors, ratios=(1.3, 1.3, 1.3)):
    rows = []
    for shape in anchors:
        for repetition in range(3):
            # Large seed-to-seed differences are present in BOTH controls.
            base = (100 + shape[0] / 1000) * (.8, 1.1, 1.0)[repetition]
            for index, arm in enumerate(tuning.ARMS):
                start = index * 100000 + repetition * 1000
                rows.append({'anchor': shape, 'repetition': repetition, 'arm': arm,
                             'rate': base * (ratios[repetition] if arm == 'challenger' else 1),
                             'requests': shape[1], 'tasks_sha256': str(shape) + str(repetition),
                             'all_requests_valid': True, 'boot_id': 'fixture-boot',
                             'engine_ids': ['fixture-engine-' + arm],
                             'start_ns': start, 'end_ns': start + 100})
    return rows


class TuningPerformanceTests(unittest.TestCase):
    def test_matched_bookends_are_distinct_from_different_seed_spread(self):
        anchors = [(n, c) for n in (8000, 128000, 512000, 1000000) for c in (1, 8, 32)]
        result = tuning.compare(observations(anchors), anchors)
        self.assertEqual(result['paired_repetitions'], 36)
        self.assertTrue(result['controls_within_five_percent'])
        self.assertTrue(result['performance_gate_passed'])
        self.assertAlmostEqual(result['equal_anchor_geomean_paired_ratio'], 1.3)
        self.assertFalse(result['performance_adoption_eligible'])
        self.assertTrue(result['remaining_adoption'])

    def test_equal_anchor_weights_prevent_large_bursts_hiding_regressions(self):
        anchors = [(8000, 1), (1000000, 32)]
        rows = observations(anchors)
        for row in rows:
            if row['arm'] == 'challenger':
                row['rate'] *= (.5 if row['anchor'][1] == 1 else 2) / 1.3
        result = tuning.compare(rows, anchors)
        self.assertAlmostEqual(result['equal_anchor_geomean_paired_ratio'], 1)
        self.assertFalse(result['performance_gate_passed'])

    def test_uncertainty_or_one_unstable_control_blocks_performance_gate(self):
        anchors = [(8000, 1)]
        result = tuning.compare(observations(anchors, (1.02, 1.4, 1.4)), anchors)
        self.assertGreater(result['observed_gain'], .10)
        self.assertAlmostEqual(result['paired_ratio_95pct_percentile_interval'][0], 1.02)
        self.assertFalse(result['performance_gate_passed'])
        rows = observations(anchors)
        next(r for r in rows if r['arm'] == 'control_after' and r['repetition'] == 1)['rate'] *= 1.06
        result = tuning.compare(rows, anchors)
        self.assertFalse(result['controls_within_five_percent'])
        self.assertFalse(result['performance_gate_passed'])
        self.assertEqual(len(result['pairs']), 3)

    def test_missing_failed_duplicate_or_unmatched_trials_cannot_disappear(self):
        anchors = [(8000, 1), (128000, 8)]
        original = observations(anchors)
        cases = [original[:-1], original + [copy.deepcopy(original[0])]]
        for field, value in (('all_requests_valid', False), ('tasks_sha256', 'different'),
                             ('requests', 2), ('repetition', True), ('rate', math.nan)):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            cases.append(rows)
        for rows in cases:
            with self.subTest(row=rows[0]), self.assertRaises(ValueError):
                tuning.compare(rows, anchors)

    def test_temporal_order_same_boot_and_fresh_engines_are_required(self):
        anchors = [(8000, 1)]
        for change in ('order', 'boot', 'engine'):
            rows = observations(anchors)
            candidate = next(r for r in rows if r['arm'] == 'challenger' and r['repetition'] == 0)
            if change == 'order':
                candidate.update(start_ns=0, end_ns=100)
            elif change == 'boot':
                candidate['boot_id'] = 'other-boot'
            else:
                candidate['engine_ids'] = ['fixture-engine-control_before']
            with self.subTest(change=change), self.assertRaises(ValueError):
                tuning.compare(rows, anchors)

    def test_actual_cli_rejects_missing_evidence_without_replacing_report(self):
        manifest = campaign.ledger(SOURCE / 'matrix.json')
        manifest['plan_sha256'] = 'd' * 64
        manifest['runtime'] = {
            'image_digest': 'example/image@sha256:' + 'a' * 64,
            'model_revision': '1bae0e1088993b9b82de08f59de5be5a34e42170',
            'tokenizer_revision': '1bae0e1088993b9b82de08f59de5be5a34e42170',
            'operational_deadline': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        cells = [r['id'] for r in manifest['cells'] if r['suite'] == 'tuning_trial' and
                 r['layout'] == '2xtp4' and r['parameters']['lever'] == 'prefill_batch']
        record = profile_registry.register('fixture-candidate', '2xtp4', {'max_num_batched_tokens': 8192},
            manifest, hypothesis='Synthetic fixture for fail-closed evidence selection only.',
            deciding_cells=cells, protected_tail='p95_request_seconds')
        manifest['runtime']['qualification_profiles'] = {record['id']: record}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m, catalog, output = root / 'manifest.json', root / 'catalog.json', root / 'report.json'
            m.write_text(json.dumps(manifest))
            catalog.write_text('[]')
            output.write_text('prior protected report\n')
            before = {f.name: f.read_bytes() for f in root.iterdir()}
            for prefix in ([sys.executable, '-B', str(SOURCE / 'verify_tuning_performance.py')],
                           [sys.executable, '-B', str(SOURCE / 'campaign.py'), 'verify-tuning-performance']):
                result = subprocess.run(prefix + [
                    '--manifest', str(m), '--profile-id', record['id'], '--catalog', str(catalog),
                    '--output', str(output)], capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('explicit retained tuning source catalog', result.stderr)
                self.assertEqual({f.name: f.read_bytes() for f in root.iterdir()}, before)
