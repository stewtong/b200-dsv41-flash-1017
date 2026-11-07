from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import threading
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import readiness

class Handler(BaseHTTPRequestHandler):
    answer = 'READY'
    def do_GET(self): self.send_response(200); self.end_headers()
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200); self.end_headers()
        self.wfile.write(('data: {"choices":[{"delta":{"content":"'+self.answer+'"},"finish_reason":"stop"}]}\n\n').encode())
        self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n')
        self.wfile.write(b'data: [DONE]\n\n')
    def log_message(self, *args): pass

class ReadinessTests(unittest.TestCase):
    def test_health_alone_does_not_prove_semantic_readiness(self):
        server=HTTPServer(('127.0.0.1', 0), Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            self.assertTrue(readiness.health(base+'/health'))
            self.assertEqual(readiness.known_answer(base+'/v1/chat/completions','model','ready')['summary']['finish_reason'], 'stop')
            Handler.answer='WRONG'
            failed = readiness.known_answer(base+'/v1/chat/completions','model','wrong')
            self.assertFalse(failed['passed'])
            self.assertTrue(failed['raw']['complete'])
        finally:
            Handler.answer='READY'; server.shutdown(); server.server_close(); thread.join()
