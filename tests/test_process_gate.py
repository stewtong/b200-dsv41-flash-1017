from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

GATE = Path(__file__).resolve().parents[1]/'reproduce/process_gate.py'

class ProcessGateTests(unittest.TestCase):
    def test_child_waits_until_gate_and_fails_on_watchdog_signal(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d)/'executed'
            code = 'from pathlib import Path; Path(sys.argv[1]).write_text("yes")'
            code = 'import sys; ' + code
            child = subprocess.Popen([sys.executable, str(GATE), d, sys.executable, '-c', code, str(marker)])
            time.sleep(0.15)
            self.assertFalse(marker.exists())
            (Path(d)/'watchdog-failure.json').write_text('{}')
            self.assertNotEqual(child.wait(timeout=2), 0)
            self.assertFalse(marker.exists())

    def test_gate_releases_child(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d)/'executed'
            (Path(d)/'router-start.ok').write_text('{}')
            code = 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text("yes")'
            child = subprocess.Popen([sys.executable, str(GATE), d, sys.executable, '-c', code, str(marker)])
            self.assertEqual(child.wait(timeout=2), 0)
            self.assertEqual(marker.read_text(), 'yes')
