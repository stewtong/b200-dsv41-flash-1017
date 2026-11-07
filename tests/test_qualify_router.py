import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import qualify_router as q

class RouterQualificationTests(unittest.TestCase):
    def test_requires_exact_dependency_versions_and_flags(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); venv=root/'env'; (venv/'bin').mkdir(parents=True)
            (venv/'bin/vllm-router').write_text('fixture')
            lock=root/'lock'; lock.write_text('vllm-router==0.1.14\nrequests==2.0\n')
            pin=hashlib.sha256(lock.read_bytes()).hexdigest()
            def fake(argv, timeout=30):
                if 'check' in argv: return 'No broken requirements found.'
                if 'freeze' in argv: return 'vllm-router==0.1.14\nrequests==2.0\npip==1.0\n'
                return '\n'.join(q.FLAGS)
            result=q.qualify(venv,lock,pin,root/'out',fake)
            self.assertEqual(result['installed_packages']['vllm-router'], '0.1.14')
            with self.assertRaises(ValueError): q.qualify(venv,lock,'0'*64,root/'bad',fake)
