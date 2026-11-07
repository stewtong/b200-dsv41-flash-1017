"""Frozen configuration and actual preparation controls; synthetic inputs only."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'reproduce'
sys.path.insert(0, str(SOURCE))
import campaign, comparison_scope, fixed_profiles, fixed_workload, launcher
import node_fixed_block, node_fixed_qualification, node_smoke, profile_registry, runtime
import test_fixed_performance_phase as phase_fixture
from test_vllm_prompt import FakeTokenizer


def manifest_fixture():
    prior = phase_fixture.FixedPerformancePhaseTests()
    prior.setUp()
    manifest = campaign.ledger(SOURCE / 'comparison-matrix.json')
    manifest.update(plan_sha256='d' * 64, runtime=copy.deepcopy(prior.manifest['runtime']))
    refs = {name: dict(source_seal='1' * 64, evidence_sha256='2' * 64, verifier_sha256='3' * 64)
            for name in ('known_answer', 'performance_path')}
    records = {layout: fixed_profiles.register(layout, manifest, refs) for layout in comparison_scope.LAYOUTS}
    manifest['runtime'].update(fixed_profiles=records,
        profile_hashes={layout: row['profile_sha256'] for layout, row in records.items()})
    return manifest


class FixedProfileTests(unittest.TestCase):
    def test_all_four_profiles_bind_exact_nospec_commands_and_full_cell_sets(self):
        manifest = manifest_fixture()
        for layout in comparison_scope.LAYOUTS:
            record = manifest['runtime']['fixed_profiles'][layout]
            self.assertEqual(record['profile_sha256'], launcher.profile_sha256(layout))
            self.assertEqual(len(record['fixed_cell_ids']), 84)
            config = dict(layout=layout, execution_phase='fixed-performance', profile_id=record['id'])
            binding = profile_registry.resolve(config, manifest)
            self.assertEqual(binding['phase'], 'fixed-performance')
            self.assertEqual(binding['delta'], {})
            self.assertEqual(profile_registry.bound_hash(config, manifest), record['profile_sha256'])
            for fields in ({'trial_profile_id': 'control'}, {'profile_overrides': {}}, {'profile_id': 'other'}):
                with self.subTest(layout=layout, fields=fields), self.assertRaises(ValueError):
                    profile_registry.resolve(dict(config, **fields), manifest)
        for field in ('template', 'fixed_cell_ids', 'qualification_evidence', 'plan_sha256'):
            changed = copy.deepcopy(manifest['runtime']['fixed_profiles']['2xtp4'])
            if field == 'template': changed[field]['commands'][0]['argv'].append('--changed')
            elif field == 'fixed_cell_ids': changed[field].pop()
            elif field == 'qualification_evidence': changed[field]['known_answer']['source_seal'] = 'bad'
            else: changed[field] = '0' * 64
            changed['registration_sha256'] = fixed_profiles.digest(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                fixed_profiles.validate(changed, manifest)

    def test_actual_preparation_accepts_cold_warm_and_1m_without_tuning_relabelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = manifest_fixture()
            layout = '2xdep4'
            config = dict(execution_phase='fixed-performance', profile_id='fixed-nospec-' + layout,
                layout=layout, image=manifest['runtime']['image_digest'], matrix=str(SOURCE / 'comparison-matrix.json'),
                run_id='fixed-scored-fixture', run_dir=str(root / 'dsv41-fixed-scored-fixture'),
                expected_ranks={'http://127.0.0.1:31000/metrics': [0, 1, 2, 3],
                                'http://127.0.0.1:31001/metrics': [0, 1, 2, 3]}, repetition=0,
                request_timeout_seconds=7200, deadline_utc='2026-11-04T00:00:00Z')
            for key, name, value in (('manifest', 'manifest.json', manifest),
                    ('checkpoint_receipt', 'checkpoint-receipt.json', {'synthetic': True}),
                    ('image_qualification', 'image-qualification.json', {'synthetic': True})):
                path = root / name
                path.write_text(json.dumps(value))
                config[key] = str(path)
            entries = []
            for index, (suite, n, cache) in enumerate((('core', 8000, 'cold'),
                    ('core', 1000000, 'cold'), ('warm_prefix_panel', 128000, 'warm_prefix'))):
                output = 512 if n == 1000000 else 2048
                cell = f'{suite}-{layout}-i{n}-c1-{cache}-o{output}'
                source = root / f'dsv41-fixed-input-fixture-{index}'
                source.mkdir()
                for key in ('manifest', 'checkpoint_receipt', 'image_qualification'):
                    path = Path(config[key]); (source / path.name).write_bytes(path.read_bytes())
                (source / 'run-config.json').write_text(json.dumps(dict(cell_id=cell, layout=layout,
                    image=config['image'], task_family='fixed', repetition=0, fixed_prompt_version='fixed-performance-v1')))
                (source / 'prompt-result.json').write_text(json.dumps(dict(status='passed', cleanup='passed',
                    task_family='fixed', cell_id=cell, repetition=0, tasks=1)))
                task = fixed_workload.build(FakeTokenizer(), '/model', n, output, suite, 1, 0, 0,
                    cache, 128, prompt_version='fixed-performance-v1')
                path = source / 'fixed-tasks.json'; path.write_text(json.dumps([task])); runtime.seal_evidence(source)
                entry = dict(cell_id=cell, tasks_file=str(path), repetition=0, request_timeout_seconds=7200)
                prepared = node_fixed_qualification.prepare(dict(config, **entry), root)
                self.assertTrue(prepared['scored_performance'])
                self.assertEqual(prepared['attempt_cell_id'], node_fixed_qualification.identity(
                    cell, launcher.profile_sha256(layout), 0))
                self.assertIn('three-repetition', node_fixed_qualification.result_scope(prepared))
                entries.append(entry)
            for key in ('repetition', 'request_timeout_seconds'): config.pop(key)
            rows, budget = node_fixed_block.prepare(dict(config, cells=entries), root)
            self.assertEqual(len(rows), 3)
            self.assertEqual(budget, 4 * (7200 + 20))
            with self.assertRaises(ValueError):
                node_fixed_block.prepare(dict(config, cells=entries + [entries[0]]), root)

    def test_extended_budget_requires_exact_new_phase_and_stays_bounded(self):
        now = datetime(2026, 11, 3, tzinfo=timezone.utc)
        manifest = manifest_fixture()
        manifest['runtime']['operational_deadline'] = (now + timedelta(days=2)).isoformat()
        cfg = dict(execution_phase='fixed-performance', deadline_utc=(now + timedelta(hours=30)).isoformat(),
            cells=[dict(cell_id='core-2xtp4-i1000000-c1-cold-o512', tasks_file='/synthetic',
                        request_timeout_seconds=7200, repetition=0)])
        self.assertEqual(node_smoke.session_seconds(cfg, manifest, budget='fixed-performance-block', now=now), 108000)
        for fields in ({'execution_phase': 'qualification'}, {'cells': []},
                {'deadline_utc': (now + timedelta(hours=53)).isoformat()},
                {'deadline_utc': (now + timedelta(days=3)).isoformat()}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                node_smoke.session_seconds(dict(cfg, **fields), manifest, budget='fixed-performance-block', now=now)
        with self.assertRaises(ValueError): node_smoke.session_seconds(cfg, manifest, now=now)


if __name__ == '__main__': unittest.main()
