import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import verify_publication


class PublicationTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
