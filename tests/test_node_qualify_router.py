import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_qualify_router
import runtime


class RouterQualificationTests(unittest.TestCase):
    def test_refuses_without_tmux(self):
        with patch.dict(os.environ,{},clear=True), patch.object(node_qualify_router.supervisor,'Supervisor') as supervisor:
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'requires tmux'):node_qualify_router.run({})
            supervisor.assert_not_called()
