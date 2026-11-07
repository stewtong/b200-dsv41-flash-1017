import importlib.util
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reproduce'))
spec = importlib.util.spec_from_file_location('campaign', ROOT / 'reproduce/campaign.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class LocalTests(unittest.TestCase):
    def test_planning_counts_and_unresolved(self):
        x = c.ledger(ROOT / 'reproduce/matrix.json')
        self.assertGreater(len(x['cells']), 1928)
        self.assertFalse(x['complete_universe'])
        self.assertIsNone(x['runtime']['image_digest'])
        pilots = [a for a in x['cells'] if a['suite'].startswith('pilot_')]
        self.assertEqual(len(pilots), 38)
        self.assertEqual(sum(a['concurrency'] for a in pilots), 342)
        self.assertEqual(sum(a['suite'] == 'growing' for a in x['cells']), 64)
        self.assertEqual(sum(a['suite'] == 'cache_control' for a in x['cells']), 128)
        self.assertEqual(sum(a['suite'] == 'arrival_ladder' for a in x['cells']), 84)
        self.assertEqual(sum(a['suite'] == 'admission' for a in x['cells']), 352)
        self.assertEqual(sum(a['suite'] == 'api_protocol' for a in x['cells']), 264)
        self.assertEqual(sum(a['suite'] == 'tuning_trial' for a in x['cells']), 864)

    def test_phase_validation_does_not_require_future_profiles(self):
        layouts=('2xtp4','4xdep2','2xdep4','dep8','tp8','4xtp2')
        rev='1bae0e1088993b9b82de08f59de5be5a34e42170'
        m = {'complete_universe':False, 'plan_sha256':'d'*64, 'runtime':{'image_digest':'example/image@sha256:'+'a'*64,
             'starting_profiles':dict.fromkeys(layouts,'b'*64),'operational_deadline':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
             'model_revision':rev,'tokenizer_revision':rev,'router_pin':'c'*64,
             'safety_controls':dict.fromkeys(('ownership','watchdog','cancellation','cleanup','persistence'),True)}}
        m['runtime']['retirement_gate'] = 'passed'
        self.assertTrue(c.validate_phase(m, 'pilot'))
        m['plan_sha256'] = None
        with self.assertRaises(ValueError): c.validate_phase(m, 'pilot')
        m['plan_sha256'] = 'd'*64
        with self.assertRaises(ValueError): c.validate_phase(m, 'scored')
        m['runtime']['router_pin'] = None
        with self.assertRaises(ValueError): c.validate_phase(m, 'pilot')

    def test_sse(self):
        self.assertEqual(c.parse_sse('data: {"a":1}\n\ndata: {"b":2}\n\ndata: [DONE]\n\n'), [{'a':1},{'b':2}])
        with self.assertRaises(ValueError): c.parse_sse('data: {"a":1}\n\n')
        with self.assertRaises(ValueError): c.parse_sse('data: {bad}\n\ndata: [DONE]\n\n')

    def test_resume_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError): c.fixture_run(d, True)
            self.assertEqual(c.fixture_run(d), 'completed')
            self.assertEqual(c.fixture_run(d), 'already completed')
            s = json.loads((Path(d)/'execution-state.json').read_text())
            self.assertEqual([a['status'] for a in s['attempts']], ['incomplete','passed'])
            subprocess.run([sys.executable, str(ROOT/'reproduce/verify_fixture.py'), d], check=True, capture_output=True)
            (Path(d)/'attempt-0002-summary.json').write_text('{}')
            self.assertNotEqual(subprocess.run([sys.executable, str(ROOT/'reproduce/verify_fixture.py'), d], capture_output=True).returncode, 0)

    def test_orphan_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d)/'attempt-0001.json'; f.write_text('{"orphan": true}')
            c.fixture_run(d)
            self.assertEqual(json.loads(f.read_text()), {'orphan': True})
            self.assertEqual(len(json.loads((Path(d)/'execution-state.json').read_text())['attempts']), 2)

if __name__ == '__main__': unittest.main()
