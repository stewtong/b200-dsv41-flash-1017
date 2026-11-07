from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import telemetry

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'# TYPE active gauge\nactive{rank="0"} 3\n')
    def log_message(self, *args): pass

class TelemetryTests(unittest.TestCase):
    def test_missing_metric_fails_closed(self):
        server=HTTPServer(('127.0.0.1',0), Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        url=f'http://127.0.0.1:{server.server_port}/metrics'
        try:
            self.assertIn('active', telemetry.scrape(url, ['active']))
            captured=[]
            with self.assertRaises(ValueError): telemetry.scrape(url, ['queued'],capture=captured.append)
            self.assertIn('active',captured[0])
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_ecc_and_xid_fail_closed(self):
        gpu = '\n'.join(f'{i}, 0, 0, 0, 1000, 100, 0' for i in range(8))
        def fake(argv, timeout=30): return gpu if argv[0] == 'nvidia-smi' else ''
        with patch.object(telemetry.runtime, 'safety_sample', return_value={'monotonic_ns':1}):
            result=telemetry.safety_sample('/', fake)
            self.assertEqual(result['uncorrected_ecc'], {i:0 for i in range(8)})
            def xid(argv, timeout=30): return gpu if argv[0] == 'nvidia-smi' else 'NVRM: Xid 31'
            with self.assertRaises(ValueError): telemetry.safety_sample('/', xid)
            with self.assertRaises(ValueError): telemetry.safety_sample('/', fake, {i:-1 for i in range(8)})

    def test_startup_gpu_timeout_has_matching_journal_window(self):
        gpu = '\n'.join(f'{i}, 0, 0, 0, 1000, 100, 0' for i in range(8)); calls=[]
        def fake(argv,timeout=30):
            calls.append((argv,timeout));return gpu if argv[0]=='nvidia-smi' else ''
        with patch.object(telemetry.runtime,'safety_sample',return_value={'monotonic_ns':1}):
            result=telemetry.safety_sample('/',fake,gpu_timeout=8,journal_window=14)
        self.assertEqual(calls[0][1],8)
        self.assertIn('14 seconds ago',calls[1][0])
        self.assertEqual(result['journal_window_seconds'],14)
        with patch.object(telemetry.runtime,'safety_sample',return_value={'monotonic_ns':1}):
            with self.assertRaisesRegex(telemetry.runtime.RuntimeErrorClosed,'deadline'):
                telemetry.safety_sample('/',fake,gpu_timeout=8,journal_window=14,deadline=0)

    def test_nvml_safety_keeps_ecc_and_kernel_checks(self):
        calls=[]; data={'devices':[{'index':i,'volatile_uncorrected_ecc':0} for i in range(8)]}
        def fake(argv,timeout=30):calls.append(argv);return ''
        with patch.object(telemetry.runtime,'safety_sample',return_value={'monotonic_ns':1}):
            result=telemetry.safety_sample('/',fake,nvml_data=data)
        self.assertEqual(result['gpu_source'],'nvml')
        self.assertEqual(result['uncorrected_ecc'],{i:0 for i in range(8)})
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][2],'journalctl')

    def test_rank_coverage_requires_every_series(self):
        metrics={'u':'active{engine="0"} 1\nactive{engine="1"} 2\nqueued{engine="0"} 0\nqueued{engine="1"} 0\n'}
        telemetry.check_rank_coverage(metrics,['active','queued'],{'u':[0,1]})
        with self.assertRaises(ValueError): telemetry.check_rank_coverage(metrics,['active','queued'],{'u':[0,1,2]})
