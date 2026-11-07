from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import node_smoke
import reconcile
import runtime
import stage_checkpoint


class NodePathsTests(unittest.TestCase):
    def test_campaign_components_share_lock_in_writable_parent(self):
        self.assertEqual(node_smoke.NODE_LOCK, runtime.DEFAULT_NODE_LOCK)
        self.assertEqual(reconcile.reconcile.__defaults__[0], runtime.DEFAULT_NODE_LOCK)
        self.assertEqual(Path(runtime.DEFAULT_NODE_LOCK).parent, stage_checkpoint.OWNER.parent)
        self.assertEqual(Path(runtime.DEFAULT_NODE_LOCK).parent, Path('/data/hf-cache'))
