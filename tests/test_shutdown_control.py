from pathlib import Path
import json,sys,tempfile,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_gpu_probe

class ShutdownControlTests(unittest.TestCase):
    def test_four_disjoint_exact_owned_cpu_controls_use_launcher_gpu_encoding(self):
        commands=[];checks=[]
        class Session:
            deadline_ns=time.monotonic_ns()+600_000_000_000
            def ensure_healthy(self):pass
            def create_start(self,argv):commands.append(argv);return format(len(commands),'064x')
        with tempfile.TemporaryDirectory() as d,patch.object(node_gpu_probe.runtime,'command',return_value='SHUTDOWN_CONTROL_READY'):
            root=Path(d);node_gpu_probe.shutdown_control(Session(),root,'fixture','fixture-image',lambda:checks.append(True))
            self.assertEqual(len(commands),4);self.assertTrue(checks)
            for i,argv in enumerate(commands):
                self.assertEqual(argv[argv.index('--gpus')+1],f'"device={2*i},{2*i+1}"')
                self.assertIn('run_id=fixture',argv);self.assertIn('signal.SIG_IGN',argv[-1])
                self.assertEqual(argv[argv.index('--name')+1],f'dsv41-fixture-{i}')
            receipt=json.loads((root/'probe-observation.json').read_text())
            self.assertEqual(len(set(receipt['container_ids'])),4)
            self.assertTrue(receipt['term_ignored'])
