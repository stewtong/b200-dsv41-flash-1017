from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import operational_authority as authority


class OperationalAuthorityTests(unittest.TestCase):
    """Synthetic plans test invariants without distributing private authority files."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.original = Path(temporary.name)
        before = 'Matched inputs and balanced execution order remain required.'
        after = ('Matched inputs and balanced placement remain required. For unstarted work, Section 6 adopts a common '
                 'within-configuration order and records the increased temporal-drift exposure from running configurations sequentially.')
        old = '# Synthetic authority fixture\n' + ''.join(
            f'## {n}. Section\n' + (before if n == 3 else f'Fixed protection {n}.') + '\n'
            for n in range(2, 9))
        new = old.replace(before, after)
        prior = hashlib.sha256(old.encode()).hexdigest()
        current = hashlib.sha256(new.encode()).hexdigest()
        files = {
            'benchmark-plan.md': new,
            'benchmark-matrix.json': '{"fixture": true}\n',
            authority.DECISION+'previous-benchmark-plan.md': old,
            authority.DECISION+'source-proposal.md': 'Synthetic operational proposal.\n',
        }
        matrix = hashlib.sha256(files['benchmark-matrix.json'].encode()).hexdigest()
        amendment = dict(schema='dsv41-operational-amendment/v1', scope_id='context-concurrency-336-v1',
            prior_plan_sha256=prior, current_plan_sha256=current,
            unchanged_matrix_sha256=matrix, ownership_transferred=False)
        files[authority.DECISION+'amendment.json'] = json.dumps(amendment)
        hashes = {}
        for name, text in files.items():
            path = self.original/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        # Scoped patches restore every module value after each test.
        patches = patch.multiple(authority, PRIOR=prior, CURRENT=current, MATRIX=matrix,
            AMENDMENT=hashes[authority.DECISION+'amendment.json'],
            PROPOSAL=hashes[authority.DECISION+'source-proposal.md'], FILES=hashes)
        patches.start()
        self.addCleanup(patches.stop)

    def source(self):
        return self.original

    def fixture(self, root):
        for name in authority.FILES:
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source()/name, target)

    def test_synthetic_amendment_preserves_scientific_pin(self):
        result = authority.verify(self.source())
        self.assertEqual(result['scientific_plan_sha256'], authority.PRIOR)
        self.assertEqual(result['operational_plan_sha256'], authority.CURRENT)
        self.assertFalse(result['historical_manifests_rewritten'])
        self.assertEqual(Path(result['scientific_plan']).read_bytes(),
                         (self.source()/(authority.DECISION+'previous-benchmark-plan.md')).read_bytes())

    def test_every_named_dependency_tamper_is_rejected(self):
        for name in authority.FILES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                with (root/name).open('ab') as stream:
                    stream.write(b'\nchanged')
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    authority.verify(root)

    def test_same_bytes_from_external_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'authority'
            root.mkdir()
            self.fixture(root)
            target = root/'benchmark-plan.md'
            raw = target.read_bytes()
            target.unlink()
            external = Path(d)/'outside.md'
            external.write_bytes(raw)
            target.symlink_to(external)
            with self.assertRaisesRegex(ValueError, 'contained'):
                authority.verify(root)

    def test_changed_scientific_section_is_rejected_after_rehash(self):
        root = self.original
        plan = root/'benchmark-plan.md'
        plan.write_text(plan.read_text().replace('Fixed protection 2.', 'Changed protection 2.'))
        current = hashlib.sha256(plan.read_bytes()).hexdigest()
        amendment = root/(authority.DECISION+'amendment.json')
        data = json.loads(amendment.read_text())
        data['current_plan_sha256'] = current
        amendment.write_text(json.dumps(data))
        hashes = dict(authority.FILES)
        hashes['benchmark-plan.md'] = current
        hashes[authority.DECISION+'amendment.json'] = hashlib.sha256(amendment.read_bytes()).hexdigest()
        with patch.multiple(authority, CURRENT=current, FILES=hashes):
            with self.assertRaisesRegex(ValueError, 'unchanged scientific'):
                authority.verify(root)


if __name__ == '__main__':
    unittest.main()
