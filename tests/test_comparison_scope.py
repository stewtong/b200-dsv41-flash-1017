"""Actual CLI and independent request checks for the approved336-cell design."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'reproduce'
sys.path.insert(0, str(SOURCE))
import campaign
import comparison_scope
import fixed_workload
import verify_fixed_task
from test_vllm_prompt import FakeTokenizer
import test_fixed_performance_phase as phase_fixture


class ComparisonScopeTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(SOURCE / 'campaign.py'), *args],
                              capture_output=True, text=True, timeout=20)

    def test_default_cli_has_exact_cold_warm_counts_and_context_output_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.md'
            plan.write_text('Synthetic scope fixture, not operational authorization.')
            output = root / 'ledger.json'
            result = self.cli('plan', '--plan', str(plan), '--output', str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text())
            self.assertEqual(manifest['scope_id'], 'context-concurrency-336-v1')
            cells = manifest['cells']
            self.assertEqual(len(cells), 336)
            self.assertEqual(len({c['id'] for c in cells}), 336)
            cold = [c for c in cells if c['suite'] == 'core']
            warm = [c for c in cells if c['suite'] == 'warm_prefix_panel']
            self.assertEqual((len(cold), len(warm)), (288, 48))
            self.assertEqual((sum(c['repetitions'] for c in cold), sum(c['repetitions'] for c in warm)), (864, 144))
            self.assertEqual((sum(c['repetitions'] * c['concurrency'] for c in cold),
                              sum(c['repetitions'] * c['concurrency'] for c in warm)), (27540, 1968))
            self.assertEqual({c['cache'] for c in cold}, {'cold'})
            self.assertEqual({c['cache'] for c in warm}, {'warm_prefix'})
            self.assertTrue(all(c['output_tokens'] == (512 if c['input_tokens'] == 1000000 else 2048) for c in cells))
            self.assertTrue(all(c['request_contract']['stopping'] == 'fixed' for c in cells))
            self.assertTrue(all(c['status'] == 'planned' for c in cells))
            self.assertEqual(json.loads(result.stdout)['measured_requests'], 29508)
            self.assertEqual(manifest['historical_scope']['core_cells'], 704)

    def test_bad_schema_axes_caps_and_counts_cannot_replace_prior_output(self):
        matrix = json.loads((SOURCE / 'comparison-matrix.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, source = root / 'ledger.json', root / 'matrix.json'
            output.write_text('retained prior output\n')
            for change in ('schema', 'output', 'counts', 'warm_axes'):
                altered = copy.deepcopy(matrix)
                if change == 'schema': altered['schema'] = 'dsv41-b200-campaign/v4'
                elif change == 'output': altered['output_tokens_by_input']['1000000'] = 2048
                elif change == 'counts': altered['total_fixed_output_cells'] = 337
                else: altered['warm_prefix_panel']['node_concurrency'].append(128)
                source.write_text(json.dumps(altered))
                result = self.cli('plan', '--matrix', str(source), '--output', str(output))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output.read_text(), 'retained prior output\n')

    def test_actual_migration_keeps_every_legacy_row_without_transferring_results(self):
        old = campaign.ledger(SOURCE / 'matrix.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, output = root / 'plan.md', root / 'migration.json'
            plan.write_text('Synthetic migration fixture.')
            result = self.cli('scope-migration', '--plan', str(plan), '--output', str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads(output.read_text())
            self.assertEqual({r['old_cell_id'] for r in saved['rows']}, {c['id'] for c in old['cells']})
            self.assertEqual(saved['new_required_cells'], 336)
            self.assertEqual(saved['historical_denominators'], {'core': 704, 'fixed_output': 1036, 'quality': 704})
            self.assertTrue(all(r['measurement_credit_transferred'] is False for r in saved['rows']))
            rows = {r['old_cell_id']: r for r in saved['rows']}
            self.assertEqual(rows['core-2xtp4-i8000-c1-cold-o2048']['disposition'], 'required')
            self.assertEqual(rows['core-2xtp4-i1000000-c1-cold-o2048']['disposition'], 'superseded_owner_scope')
            quality = next(c['id'] for c in old['cells'] if c['suite'] == 'quality')
            self.assertEqual(rows[quality]['disposition'], 'deferred_owner_scope')
            self.assertEqual(saved['status'], 'planning_migration_history_binding_pending')

    def test_new_task_shapes_keep_tokenizer_and_independent_input_oracle(self):
        tokenizer = FakeTokenizer()
        cold = fixed_workload.build(tokenizer, '/model', 1000000, 512, 'core', 1, 0, 0,
                                    prompt_version='fixed-performance-v1')
        warm = fixed_workload.build(tokenizer, '/model', 128000, 2048, 'warm_prefix_panel', 1, 0, 0,
                                    'warm_prefix', 128, prompt_version='fixed-performance-v1')
        for task in (cold, warm):
            self.assertEqual(verify_fixed_task.verify(task, tokenizer)['status'], 'tokenizer_input_verified')
            self.assertEqual(task['body']['max_tokens'], task['output_tokens'])
        with self.assertRaises(ValueError):
            fixed_workload.build(tokenizer, '/model', 8000, 512, 'core', 1, 0, 0,
                                 prompt_version='fixed-performance-v1')
        wrong = copy.deepcopy(warm)
        wrong['cache_designation'] = 'cold'
        with self.assertRaises(ValueError): verify_fixed_task.verify(wrong)

    def test_selected_v3_phase_keeps_exact336_rows_and_runtime_gates(self):
        fixture = phase_fixture.FixedPerformancePhaseTests()
        fixture.setUp()
        manifest = campaign.ledger(SOURCE / 'comparison-matrix.json')
        manifest.update(plan_sha256='d' * 64, runtime=copy.deepcopy(fixture.manifest['runtime']))
        chosen = [next(c['id'] for c in manifest['cells'] if c['suite'] == family)
                  for family in ('core', 'warm_prefix_panel')]
        self.assertTrue(campaign.validate_phase(manifest, 'fixed-performance', cell_ids=chosen))
        wrong = copy.deepcopy(manifest)
        wrong['cells'].append(copy.deepcopy(fixture.manifest['cells'][-1]))
        with self.assertRaises(ValueError): campaign.validate_phase(wrong, 'fixed-performance', cell_ids=chosen)
        wrong = copy.deepcopy(manifest)
        wrong['runtime']['safety_controls']['cleanup'] = False
        with self.assertRaises(ValueError): campaign.validate_phase(wrong, 'fixed-performance', cell_ids=chosen)


if __name__ == '__main__':
    unittest.main()
