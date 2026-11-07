from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import threading
import unittest

REPRO = Path(__file__).resolve().parents[1] / 'reproduce'
sys.path.insert(0, str(REPRO))
import burst
import verify_burst

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200); self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n')
        self.wfile.write(b'data: [DONE]\n\n')
    def log_message(self, *args): pass

class BurstTests(unittest.TestCase):
    def test_real_loopback_failed_request_and_independent_verifier(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)/'attempt1'
                result = burst.execute(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    [{'id':'a','body':{},'expected':'ok'}, {'id':'b','body':{},'expected':'other'}], root, 5, 'fixture')
                self.assertEqual(result['summary']['valid_requests'], 1)
                self.assertTrue(result['has_failed_request'])
                self.assertTrue(verify_burst.verify(root, {'a':'ok','b':'other'}))
                original = json.loads((root/'summary.json').read_text())
                missing = json.loads(json.dumps(original)); missing['files'].pop('b.json')
                (root/'summary.json').write_text(json.dumps(missing))
                with self.assertRaises(ValueError): verify_burst.verify(root, {'a':'ok','b':'other'})
                (root/'summary.json').write_text(json.dumps(original))
                data = json.loads((root/'summary.json').read_text()); data['summary']['valid_tokens'] = 99
                (root/'summary.json').write_text(json.dumps(data))
                with self.assertRaises(ValueError): verify_burst.verify(root, {'a':'ok','b':'other'})
        finally:
            server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
