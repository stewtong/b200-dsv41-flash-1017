import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import node_qualify_image
import runtime


class NodeQualificationTests(unittest.TestCase):
    def test_refuses_without_tmux_before_starting_any_work(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(node_qualify_image.supervisor, 'Supervisor') as supervisor:
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed, 'requires tmux'):
                node_qualify_image.run({})
            supervisor.assert_not_called()

    def test_refuses_invalid_run_identity_before_starting_any_work(self):
        config = {'run_id':'bad/name', 'run_dir':'/data/dsv41-bad/name'}
        with patch.dict(os.environ, {'TMUX':'fixture'}, clear=True), patch.object(node_qualify_image.supervisor, 'Supervisor') as supervisor:
            with self.assertRaisesRegex(ValueError, 'invalid run'):
                node_qualify_image.run(config)
            supervisor.assert_not_called()
