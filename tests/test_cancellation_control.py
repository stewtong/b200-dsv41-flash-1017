from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import cancellation_control
import verify_cancellation
import campaign


class CancellationControlTests(unittest.TestCase):
    def test_actual_disconnect_requires_live_busy_to_zero_transition_and_semantic_health(self):
        working=threading.Event();stop=threading.Event();ignore=[False]
        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.0'
            def send(self,text):
                self.send_response(200);self.send_header('Content-Length',str(len(text)));self.end_headers();self.wfile.write(text)
            def do_GET(self):
                if self.path=='/metrics':
                    text=''.join('# TYPE vllm:num_requests_'+name+' gauge\nvllm:num_requests_'+name+'{engine="0"} '+str(int(working.is_set()) if name=='running' else 0)+'\n' for name in ('running','waiting'))
                elif self.path=='/load':text=json.dumps({'server_load':int(working.is_set())})
                else:text=''
                self.send(text.encode())
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                if self.headers['X-Request-Id'].endswith('-semantic'):
                    events=[{'choices':[{'delta':{'content':'READY'}}]},
                            {'choices':[{'delta':{},'finish_reason':'stop'}]},
                            {'choices':[],'usage':{'prompt_tokens':12,'completion_tokens':1}}]
                    self.send((''.join('data: '+json.dumps(row)+'\n\n' for row in events)+'data: [DONE]\n\n').encode());return
                working.set();self.send_response(200);self.end_headers();time.sleep(.35)
                try:
                    while not stop.is_set():
                        try:
                            self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n');self.wfile.flush()
                        except (BrokenPipeError,ConnectionResetError):
                            if not ignore[0]:break
                        time.sleep(.02)
                finally:working.clear()
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        class Session:
            run_id='fixture-cancel';deadline_ns=time.monotonic_ns()+600_000_000_000
            def __init__(self):self.aborted=False
            def ensure_healthy(self):pass
            def abort(self,reason):self.aborted=True;stop.set()
        base='http://127.0.0.1:'+str(server.server_port)
        task={'body':{'model':'fixture','messages':[{'role':'user','content':'synthetic'}]}}
        try:
            with tempfile.TemporaryDirectory() as d:
                for mode in ('prefill','decode'):
                    root=Path(d)/mode;session=Session()
                    result=cancellation_control.run(session,base+'/v1/chat/completions',task,{base+'/metrics':[0]},root,mode,'fixture-clock',timeout=5,absence_timeout=4)
                    self.assertEqual(result['status'],'passed');self.assertFalse(session.aborted)
                    self.assertEqual(verify_cancellation.verify(root),result)
                    output=io.StringIO()
                    with patch.object(sys,'argv',['campaign.py','verify-cancellation','--run-dir',str(root)]),redirect_stdout(output):campaign.main()
                    self.assertEqual(json.loads(output.getvalue()),result)
                    after=root/'after.json';old=after.read_text();changed=json.loads(old);changed[0]['loads'][base+'/load']['server_load']=1;after.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):verify_cancellation.verify(root)
                    after.write_text(old)
                    raw=root/'cancelled-request-raw.json';old=raw.read_text();changed=json.loads(old);changed['complete']=True;raw.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,'transport'):verify_cancellation.verify(root)
                    raw.write_text(old)
                ignore[0]=True;root=Path(d)/'ignored';session=Session()
                with self.assertRaisesRegex(ValueError,'observations'):
                    cancellation_control.run(session,base+'/v1/chat/completions',task,{base+'/metrics':[0]},root,'decode','fixture-clock',timeout=5,absence_timeout=2)
                self.assertTrue(session.aborted)
                self.assertTrue((root/'control-failure.json').exists());self.assertTrue((root/'cancelled-request-raw.json').exists())
                self.assertFalse((root/'control-result.json').exists())
        finally:
            stop.set();server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
