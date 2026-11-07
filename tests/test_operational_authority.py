from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import operational_authority as authority


class OperationalAuthorityTests(unittest.TestCase):
    def source(self):
        packaged = Path(__file__).resolve().parents[1]/'authority'
        if packaged.is_dir():
            return packaged
        return Path(__file__).resolve().parents[3]/'nebius_cloud/projects/deepseek-v41-flash-1017'

    def fixture(self, root):
        for name in authority.FILES:
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source()/name, target)

    def test_actual_authority_preserves_scientific_pin(self):
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


if __name__ == '__main__':
    unittest.main()
