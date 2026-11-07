import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('stream', Path(__file__).resolve().parents[1] / 'reproduce/stream.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

class StreamTests(unittest.TestCase):
    def test_fragmentation_utf8_and_multiple_events(self):
        d = s.SSEDecoder()
        raw = 'data: {"choices":[{"delta":{"content":"é"}}]}\n\ndata: {"choices":[]}\n\ndata: [DONE]\n\n'.encode()
        events = []
        for byte in raw: events += d.feed(bytes([byte]))
        events += d.feed(b'', final=True)
        self.assertEqual([x[0] for x in events], ['event', 'event', 'done'])
        self.assertEqual(events[0][1]['choices'][0]['delta']['content'], 'é')

    def test_truncated_and_malformed_stream_fail(self):
        with self.assertRaises(ValueError): s.SSEDecoder().feed(b'data: {"x":1}\n\n', final=True)
        with self.assertRaises(ValueError): s.SSEDecoder().feed(b'data: {bad}\n\n')
        d = s.SSEDecoder(); d.feed(b'data: [DONE]\n\n')
        with self.assertRaises(ValueError): d.feed(b'data: {}\n\n')

    def test_reasoning_precedes_visible_and_usage_is_unique(self):
        result = {'complete': True, 'events': [
            {'kind':'event', 'offset_ns':2, 'data':{'choices':[{'delta':{'reasoning_content':'r'}}]}},
            {'kind':'event', 'offset_ns':4, 'data':{'choices':[{'delta':{'content':'a'}, 'finish_reason':'stop'}]}},
            {'kind':'event', 'offset_ns':5, 'data':{'choices':[], 'usage':{'completion_tokens':2}}}]}
        summary = s.summarize_stream(result)
        self.assertEqual((summary['first_output_ns'], summary['first_visible_ns'], summary['last_output_ns']), (2, 4, 4))
        self.assertEqual(summary['task_validity'], 'unscored')
        result['events'].append(result['events'][-1])
        with self.assertRaises(ValueError): s.summarize_stream(result)

    def test_reject_remote_and_invalid_deadline(self):
        with self.assertRaises(ValueError): s.capture('http://example.org:80/v1/chat/completions', {}, 'a', 1)
        with self.assertRaises(ValueError): s.capture('http://127.0.0.1:1/v1/chat/completions', {}, 'a', 0)

    def test_loopback_capture_and_valid_terminal(self):
        sessions = [];payloads=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                sessions.append(self.headers.get('X-Session-Id'))
                payloads.append(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n')
                self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n')
                self.wfile.write(b'data: [DONE]\n\n')
            def log_message(self, *args): pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            result = s.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', {}, 'test-1', 5, session_id='session-a')
        finally:
            server.shutdown(); server.server_close(); thread.join()
        self.assertTrue(result['complete'], result['error'])
        self.assertEqual(sessions, ['session-a'])
        import hashlib
        self.assertEqual(result['request_body_sha256'],hashlib.sha256(payloads[0]).hexdigest())
        self.assertEqual(result['request_payload_bytes'],len(payloads[0]))
        self.assertEqual(result['request_headers']['X-Request-Id'],'test-1')
        self.assertEqual(result['request_headers']['X-Session-Id'],'session-a')
        self.assertEqual(s.summarize_stream(result)['usage']['completion_tokens'], 1)

    def test_header_wait_uses_remaining_deadline(self):
        class Sock:
            timeout=None
            def settimeout(self,value): self.timeout=value
        class Conn:
            def __init__(self,*args,**kwargs): self.sock=Sock(); self.seen=None
            def request(self,*args,**kwargs): time.sleep(.05)
            def getresponse(self): self.seen=self.sock.timeout; raise TimeoutError('fixture')
            def close(self): pass
        conn=Conn()
        with patch.object(s.http.client,'HTTPConnection',return_value=conn), patch.object(s,'_connect_socket',return_value=conn.sock):
            result=s.capture('http://127.0.0.1:8000/v1/chat/completions',{},'test',.2)
        self.assertFalse(result['complete'])
        self.assertGreater(conn.seen,0)
        self.assertLess(conn.seen,.17)

    def test_connect_can_be_cancelled_before_completion(self):
        class Sock:
            closed=False
            def setblocking(self,value):pass
            def connect_ex(self,address):return s.errno.EINPROGRESS
            def shutdown(self,*args):pass
            def close(self):self.closed=True
        sock=Sock();cancel=s.Cancellation()
        def poll(*args):cancel.cancel();return [],[],[]
        with patch.object(s.socket,'socket',return_value=sock),patch.object(s.select,'select',side_effect=poll):
            with self.assertRaises(InterruptedError):s._connect_socket('127.0.0.1',8000,time.monotonic_ns(),100,cancel)
        self.assertTrue(sock.closed);self.assertFalse(cancel.sockets)

    def test_timeout_does_not_prove_server_cancellation(self):
        started = threading.Event(); finished = threading.Event()
        class Slow(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200); self.end_headers(); started.set()
                time.sleep(0.3); finished.set()
            def log_message(self, *args): pass
        server = HTTPServer(('127.0.0.1', 0), Slow)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            result = s.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions', {}, 'slow', 0.05)
            self.assertTrue(started.is_set())
            self.assertFalse(result['complete'])
            self.assertFalse(finished.is_set())
            self.assertTrue(finished.wait(1))
        finally:
            server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
