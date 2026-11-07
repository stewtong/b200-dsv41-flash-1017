"""Block reservations retain per-cell attempts and fail closed on interruptions."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import attempts
import runtime

class BlockReservationTests(unittest.TestCase):
    def setUp(self):
        # Linux /proc incarnation is supplied explicitly on macOS fixtures.
        owner={'boot_id':'fixture','pid':os.getpid(),'start_ticks':100}
        mock=patch.object(runtime,'process_identity',return_value=owner)
        mock.start();self.addCleanup(mock.stop)

    def cells(self):
        return [{'cell_id':c,'method_sha256':'a'*64} for c in ('one','two')]

    def test_capture_remains_unresolved_and_blocks_other_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);owner=runtime.process_identity(os.getpid())
            paths=attempts.begin_block(root,self.cells(),'group1',owner)
            self.assertEqual(len(paths),2)
            (paths[0]/'raw.json').write_text('{}')
            attempts.capture_pending(paths[0],['raw.json'],'group1',owner)
            self.assertEqual({r['status'] for r in attempts.load_attempts(root)},{'running'})
            receipt=json.loads((paths[0]/'capture-pending.json').read_text())
            self.assertEqual(receipt['status'],'captured_cleanup_pending')
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin(root,'third')
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin_block(root,[self.cells()[0]],'group2',owner)
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.capture_pending(paths[0],['raw.json'],'group1',owner)
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.capture_pending(paths[1],['raw.json'],'other',owner)

    def test_terminal_cells_cannot_be_replayed_by_a_block(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=attempts.begin(root,'one',method_sha256='a'*64)
            (old/'raw.json').write_text('{}');attempts.finish(old,'failed',['raw.json'])
            before=sorted(p.name for p in root.iterdir())
            with self.assertRaises(runtime.RuntimeErrorClosed):
                attempts.begin_block(root,self.cells(),'group',runtime.process_identity(os.getpid()))
            self.assertEqual(before,sorted(p.name for p in root.iterdir()))

    def test_bad_owner_or_duplicate_cell_creates_no_reservations(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);owner=runtime.process_identity(os.getpid())
            for cells,identity in [(self.cells(),dict(owner,pid=owner['pid']+1)),([self.cells()[0]]*2,owner)]:
                with self.assertRaises((ValueError,runtime.RuntimeErrorClosed)):
                    attempts.begin_block(root,cells,'group',identity)
                self.assertFalse(list(root.glob('attempt-*')))
