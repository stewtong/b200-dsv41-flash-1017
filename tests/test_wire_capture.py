import base64
import copy
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import stream,verify_wire


class WireCaptureTests(unittest.TestCase):
    def capture(self,payload,status=200,limit=64*1024*1024,extra_length=0):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(status);self.send_header('Content-Length',str(len(payload)+extra_length));self.end_headers()
                self.wfile.write(payload);self.wfile.flush()
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:return stream.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',{},'wire',5,max_wire_bytes=limit)
        finally:server.shutdown();server.server_close();thread.join()

    def test_complete_wire_replays_independently_and_tampering_fails(self):
        payload=b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
        raw=self.capture(payload);self.assertTrue(raw['complete'],raw['error'])
        self.assertEqual(verify_wire.retained_bytes(raw),payload)
        self.assertEqual(verify_wire.verify_complete(raw)['events'],3)
        truncated=self.capture(payload,extra_length=20)
        self.assertFalse(truncated['complete']);self.assertEqual(verify_wire.retained_bytes(truncated),payload)
        for mutate in (lambda r:r['events'][0]['data']['choices'][0]['delta'].update(content='wrong'),
                       lambda r:r['wire_chunks'][0].update(base64=base64.b64encode(b'changed').decode()),
                       lambda r:r['wire_chunks'].pop()):
            altered=copy.deepcopy(raw);mutate(altered)
            with self.assertRaises(ValueError):verify_wire.verify_complete(altered)

    def test_malformed_utf8_or_json_retains_original_body(self):
        for payload in (b'data: {bad}\n\n',b'data: \xff\n\n'):
            raw=self.capture(payload);self.assertFalse(raw['complete'])
            self.assertEqual(verify_wire.retained_bytes(raw),payload)
            with self.assertRaises(ValueError):verify_wire.verify_complete(raw)

    def test_http_error_and_oversized_response_retain_bounded_evidence(self):
        payload=b'{"error":"fixture rejection"}'
        raw=self.capture(payload,503);self.assertFalse(raw['complete']);self.assertEqual(raw['http_status'],503)
        self.assertEqual(verify_wire.retained_bytes(raw),payload)
        raw=self.capture(b'x'*100,limit=32)
        self.assertFalse(raw['complete']);self.assertTrue(raw['wire_truncated'])
        self.assertEqual(raw['wire_byte_count'],32);self.assertEqual(verify_wire.retained_bytes(raw),b'x'*32)
