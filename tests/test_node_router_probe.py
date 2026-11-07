import os
from pathlib import Path
import sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_router_probe
import runtime


class NodeRouterProbeTests(unittest.TestCase):
    def test_refuses_without_tmux_before_starting_work(self):
        with patch.dict(os.environ,{},clear=True), patch.object(node_router_probe.supervisor,'Supervisor') as supervisor:
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'requires tmux'):node_router_probe.run({})
            supervisor.assert_not_called()
