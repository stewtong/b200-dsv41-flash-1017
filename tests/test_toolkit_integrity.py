import hashlib,json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import toolkit_integrity


class ToolkitIntegrityTests(unittest.TestCase):
    def test_changed_or_unlisted_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'reproduce').mkdir();source=root/'reproduce/runner.py';source.write_text('x=1\n')
            manifest=root/'MANIFEST.json';manifest.write_text(json.dumps({'reproduce/runner.py':hashlib.sha256(source.read_bytes()).hexdigest()}))
            self.assertEqual(toolkit_integrity.verify(manifest),hashlib.sha256(manifest.read_bytes()).hexdigest())
            for name in ('extra.txt','extra.so','extra'):
                unknown=root/'reproduce'/name;unknown.write_text('unlisted')
                with self.assertRaisesRegex(ValueError,'unlisted'):toolkit_integrity.verify(manifest)
                unknown.unlink()
            extra=root/'reproduce/MANIFEST.json';extra.write_text('{}')
            with self.assertRaisesRegex(ValueError,'unlisted'):toolkit_integrity.verify(manifest)
            extra.unlink();source.write_text('x=2\n')
            with self.assertRaisesRegex(ValueError,'source hash mismatch'):toolkit_integrity.verify(manifest)
