from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import runtime
import verify_pilot


class EvidenceManifestTests(unittest.TestCase):
    def test_nested_manifest_is_covered_and_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            nested = root / 'nested'
            nested.mkdir()
            (nested / 'SHA256SUMS').write_text('original\n')
            runtime.seal_evidence(root)
            self.assertIn('  nested/SHA256SUMS\n', (root / 'SHA256SUMS').read_text())
            (nested / 'SHA256SUMS').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_pilot.verify(root)
