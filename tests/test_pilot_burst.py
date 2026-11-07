from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import pilot_burst

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200); self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":1}}\n\n')
        self.wfile.write(b'data: [DONE]\n\n')
    def log_message(self,*args): pass

class FakeSession:
    def __init__(self): self.aborts=[]
    def ensure_healthy(self): pass
    def abort(self, reason): self.aborts.append(reason)

class PilotTests(unittest.TestCase):
    def test_fixed_length_report_with_identifier_and_cjk_fails_and_retains_raw(self):
        raw={'start_ns':1,'terminal_ns':2,'complete':True,'http_status':200,
             'events':[{'kind':'event','data':{'choices':[{'delta':{'content':'synthetic-code English report \\u5bb9'}}]}}]}
        # Construct the actual decoded wire character, not a literal escape.
        raw['events'][0]['data']['choices'][0]['delta']['content']='synthetic-code English report '+chr(0x5bb9)
        summary={'usage':{'prompt_tokens':128000,'completion_tokens':512},'finish_reason':'length'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'attempt';session=FakeSession()
            task={'id':'s0','body':{},'input_tokens':128000,'output_tokens':512,'expected_answer':'synthetic-code','fixed_output':True}
            with patch.object(pilot_burst.stream,'capture',return_value=raw),patch.object(pilot_burst.stream,'summarize_stream',return_value=summary):
                result=pilot_burst.run(session,'http://127.0.0.1:1',[task],root,3,'fixture')
            self.assertEqual(result['valid_requests'],0);self.assertEqual(len(session.aborts),1)
            record=json.loads((root/'s0.json').read_text());self.assertFalse(record['valid']);self.assertEqual(record['raw'],raw)

    def test_fixed_length_filler_without_the_requested_identifier_fails(self):
        raw={'start_ns':1,'terminal_ns':2,'complete':True,'http_status':200,
             'events':[{'kind':'event','data':{'choices':[{'delta':{'content':'the '*512}}]}}]}
        summary={'usage':{'prompt_tokens':128000,'completion_tokens':512},'finish_reason':'length'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'attempt';session=FakeSession()
            task={'id':'s0','body':{},'input_tokens':128000,'output_tokens':512,'expected_answer':'synthetic-code','fixed_output':True}
            with patch.object(pilot_burst.stream,'capture',return_value=raw),patch.object(pilot_burst.stream,'summarize_stream',return_value=summary):
                result=pilot_burst.run(session,'http://127.0.0.1:1',[task],root,3,'fixture')
            self.assertEqual(result['valid_requests'],0);self.assertEqual(len(session.aborts),1)
            record=json.loads((root/'s0.json').read_text());self.assertFalse(record['valid']);self.assertEqual(record['raw'],raw)

    def test_boolean_usage_fails_and_aborts(self):
        raw={'start_ns':1,'terminal_ns':2,'events':[{'kind':'event','data':{'choices':[{'delta':{'content':'ok'}}]}}]}
        summary={'usage':{'prompt_tokens':100,'completion_tokens':True},'finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'attempt';session=FakeSession()
            task={'id':'a','body':{},'input_tokens':100,'output_tokens':10,'expected_answer':'ok'}
            with patch.object(pilot_burst.stream,'capture',return_value=raw), \
                 patch.object(pilot_burst.stream,'summarize_stream',return_value=summary):
                result=pilot_burst.run(session,'http://127.0.0.1:1',[task],root,3,'fixture')
            self.assertEqual(result['valid_requests'],0)
            self.assertEqual(len(session.aborts),1)
            self.assertFalse(json.loads((root/'a.json').read_text())['valid'])

    def test_failed_response_aborts_and_preserves_attempt(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'attempt'; session=FakeSession()
                task={'id':'a','body':{},'input_tokens':100,'output_tokens':10,'expected_answer':'wrong'}
                result=pilot_burst.run(session, f'http://127.0.0.1:{server.server_port}/v1/chat/completions', [task], root, 3, 'fixture')
                self.assertEqual(result['valid_requests'], 0)
                self.assertEqual(len(session.aborts), 1)
                self.assertEqual(json.loads((root/'summary.json').read_text())['status'], 'failed')
                self.assertFalse(json.loads((root/'a.json').read_text())['valid'])
        finally:
            server.shutdown(); server.server_close(); thread.join()
