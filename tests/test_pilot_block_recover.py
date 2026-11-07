import json,os
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import attempts,pilot_block_recover,runtime


class BlockRecoveryTests(unittest.TestCase):
    def reserve(self,base):
        owner={'boot_id':'fixture','pid':os.getpid(),'start_ticks':1}
        with patch.object(runtime,'process_identity',return_value=owner):
            paths=attempts.begin_block(base/'registry',[{'cell_id':c,'method_sha256':'a'*64} for c in ('one','two')],'fixture',owner)
        return paths

    def recover(self,base):
        return pilot_block_recover.recover(base/'dsv41-fixture',attempts_root=base/'registry',
            node_lock=base/'node.lock',allowed_root=base,alive=lambda identity:False,preflight=lambda *_args:None)

    def test_unlaunched_reservations_release_without_consuming_retry(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);paths=self.reserve(base)
            self.assertEqual(self.recover(base)['status'],'interrupted_block_reconciled')
            self.assertEqual({r['status'] for r in attempts.load_attempts(base/'registry')},{'not_dispatched'})
            fresh=attempts.begin(base/'registry','two',method_sha256='a'*64)
            self.assertNotIn(fresh,paths)
            self.assertTrue(all(p.exists() for p in paths))

    def test_dispatch_intent_is_interrupted_and_never_released_as_unlaunched(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);paths=self.reserve(base)
            (paths[0]/'dispatch-intent.json').write_text('{}')
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'may have dispatched'):
                attempts.release_not_dispatched(paths[0],'fixture recovery')
            self.recover(base)
            rows={r['cell_id']:r['status'] for r in attempts.load_attempts(base/'registry')}
            self.assertEqual(rows,{'one':'interrupted','two':'not_dispatched'})
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'already terminal'):
                attempts.begin(base/'registry','one',method_sha256='a'*64)
