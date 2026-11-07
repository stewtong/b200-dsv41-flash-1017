"""An observed first output can trigger cancellation without erasing raw data."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import stream
import verify_wire


class ProgressTests(unittest.TestCase):
    def test_first_output_cancels_client_but_does_not_prove_server_work_absent(self):
        working = threading.Event(); release = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length'])); working.set()
                self.send_response(200); self.end_headers()
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':'first'}}]})+'\n\n').encode()); self.wfile.flush()
                release.wait(3); working.clear()
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        cancellation = stream.Cancellation(); observed = []
        def first(row):
            observed.append(row)
            if row['kind'] == 'event': cancellation.cancel()
        try:
            raw = stream.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', {}, 'progress', 2,
                                 cancellation=cancellation, on_event=first)
            self.assertFalse(raw['complete']); self.assertTrue(raw['cancel_requested'])
            self.assertEqual(raw['events'], observed)
            self.assertIn(b'first', verify_wire.retained_bytes(raw))
            self.assertTrue(working.is_set(), 'client cancellation alone must not establish server cancellation')
        finally:
            release.set(); server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': unittest.main()
