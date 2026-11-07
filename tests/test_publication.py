import sys
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import verify_publication


class PublicationTest(unittest.TestCase):
    def inventory(self, paths, images=None):
        return verify_publication.read_inventory(json.dumps(dict(
            schema='publication-result-inventory/v1', paths=sorted(paths),
            reviewed_png_sha256=images or {})))

    def chunk(self, kind, data):
        return (struct.pack('>I', len(data)) + kind + data +
                struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))

    def png(self):
        return (b'\x89PNG\r\n\x1a\n' +
                self.chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)) +
                self.chunk(b'IDAT', zlib.compress(b'\0\0\0\0')) + self.chunk(b'IEND', b''))

    def test_reject_runtime_evidence_and_unapproved_results(self):
        for name in ['LESSONS.md', 'evidence/failure.json', 'reproduce/raw/response.json',
                     'results/partial-run/summary.json', 'results/fixed-comparison/server.log',
                     'configs/.env', '../README.md']:
            self.assertTrue(verify_publication.check(name, b'{}'), name)

    def test_secret_detection_does_not_echo_value(self):
        secret = b'gh' + b'p_' + b'a' * 36
        issues = verify_publication.check('configs/example.json', secret)
        self.assertTrue(issues)
        self.assertNotIn(secret.decode(), str(issues))

    def test_private_connection_details(self):
        for value in [b'/home/' + b'stew/private', b'204.' + b'12.161.24']:
            self.assertTrue(verify_publication.check('README.md', value))

    def test_portable_source_and_sanitized_results(self):
        for name in ['reproduce/attempts.py', 'tests/test_faults.py',
                     'results/fixed-comparison/tables/requests.csv']:
            self.assertEqual(verify_publication.check(name, b'http://127.0.0.1:8000'), [])

    def test_failure_json_and_other_unreviewed_result_paths_rejected(self):
        inventory = self.inventory(['results/fixed-comparison/summary.json'])
        for name in ['results/fixed-comparison/failure.json',
                     'results/fixed-comparison/report/incident.json',
                     'results/tuning-adaptation/captured-response.json']:
            issues = verify_publication.check(name, b'{"status":"failed"}', inventory)
            self.assertIn('result path is absent from reviewed publication inventory', issues)

    def test_png_all_bytes_including_compressed_metadata_are_bound(self):
        name = 'results/fixed-comparison/report/chart.png'
        original = self.png()
        inventory = self.inventory([name], {name: hashlib.sha256(original).hexdigest()})
        self.assertEqual(verify_publication.check(name, original, inventory), [])
        secret = b'gh' + b'p_' + b'a'*36
        for kind, payload in [(b'tEXt', b'Comment\0' + secret),
                              (b'zTXt', b'Comment\0\0' + zlib.compress(secret)),
                              (b'iTXt', b'Comment\0\1\0\0\0' + zlib.compress(secret))]:
            changed = original[:-12] + self.chunk(kind, payload) + original[-12:]
            issues = verify_publication.check(name, changed, inventory)
            self.assertIn('PNG bytes differ from reviewed publication inventory', issues)
            self.assertNotIn(secret.decode(), str(issues))
        self.assertTrue(verify_publication.check(name, original + b'extra', inventory))

    def test_inventory_rejects_duplicate_paths_or_missing_image_hash(self):
        for data in [dict(schema='publication-result-inventory/v1',
                          paths=['results/fixed-comparison/x.json']*2, reviewed_png_sha256={}),
                     dict(schema='publication-result-inventory/v1',
                          paths=['results/fixed-comparison/x.png'], reviewed_png_sha256={})]:
            with self.assertRaises(ValueError):
                verify_publication.read_inventory(json.dumps(data))

    def test_cli_uses_index_inventory_and_checks_unchanged_indexed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True)
            git('init', '-q')
            config = root/'configs/publication-results.json'
            config.parent.mkdir()
            approved = 'results/fixed-comparison/summary.json'
            bad = 'results/fixed-comparison/failure.json'
            inventory = self.inventory([approved])
            config.write_text(json.dumps(inventory))
            for name in (approved, bad):
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{}')
            git('add', '--', 'configs', 'results')
            # This unstaged approval must not authorize the staged failure file.
            config.write_text(json.dumps(self.inventory([approved, bad])))
            result = subprocess.run([sys.executable, '-B', verify_publication.__file__, '--staged'],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn(bad, result.stdout)
            # Treat the current index as HEAD without writing a commit or config.
            tree = git('write-tree').stdout.decode().strip()
            git('read-tree', tree)
            # The scanner includes indexed result blobs independently of diff output.
            from unittest.mock import patch
            original = verify_publication.subprocess.check_output
            def output(command):
                if command[:3] == ['git', 'diff', '--cached']:
                    return b''
                return original(['git', '-C', str(root)] + command[1:])
            with patch.object(verify_publication.subprocess, 'check_output', side_effect=output):
                self.assertIn(bad, dict(verify_publication.staged_blobs()))


if __name__ == '__main__':
    unittest.main()
