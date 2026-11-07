import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import check_fix
import qualify_image as q

IMAGE='example/image@sha256:'+'a'*64
REV='b'*40

class QualificationTests(unittest.TestCase):
    def test_digest_revision_and_fix_checks(self):
        class Session:
            run_id = 'fixture-1'
            preflight_done = True
            def create_start(self, argv):
                self.argv = argv
                return 'd'*64
            def ensure_healthy(self): pass
        session = Session()
        def fake(argv, timeout=30):
            if 'inspect' in argv and 'image' not in argv:
                return json.dumps([{'Id':'d'*64, 'Image':'sha256:'+'c'*64, 'Config':{'Labels':{'campaign':'dsv41-1017','run_id':'fixture-1'}}, 'State':{'Running':True}}])
            if 'inspect' in argv:
                return json.dumps([{'Id':'sha256:'+'c'*64,'RepoDigests':[IMAGE], 'Config':{'Entrypoint':['vllm','serve'],'Labels':{'org.opencontainers.image.revision':REV}}}])
            if 'nvidia-smi' in argv: return '\n'.join(['580.1']*8)
            if 'python3' in argv:
                return json.dumps({'invariants':{n:'invariants-present' for names in check_fix.REQUIRED.values() for n in names}, 'cuda_version':'fixture','vllm_version':'fixture','torch_version':'fixture','nccl_version':[1,2,3]})
            return '\n'.join(q.FLAGS)
        with tempfile.TemporaryDirectory() as d:
            result=q.qualify(IMAGE, REV, 'fixture-1', Path(d)/'qualification', session, fake)
            self.assertEqual(result['image'], IMAGE)
            self.assertIn('unqualified', result['scope'])
            self.assertEqual(session.argv[:3], ['docker','run','-d'])
            self.assertNotIn('--rm', session.argv)
            session.preflight_done = False
            with self.assertRaises(q.runtime.RuntimeErrorClosed): q.qualify(IMAGE, REV, 'fixture-1', Path(d)/'no-preflight', session, fake)
            with self.assertRaises(ValueError): q.qualify(IMAGE, 'c'*40, 'fixture-1', Path(d)/'bad', session, fake)
