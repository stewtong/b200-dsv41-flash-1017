"""Loopback-only prerecorded SSE payload for the actual client instrument."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import base64,hashlib,json,os,threading,time


class Backend:
    def __init__(self,port,events,piece_bytes):
        self.records=[];self.lock=threading.Lock();parent=self
        frames=[{'choices':[{'index':0,'delta':{'content':'x'*piece_bytes}}]} for _ in range(events)]
        frames += [{'choices':[{'index':0,'delta':{},'finish_reason':'length'}]},
            {'choices':[],'usage':{'prompt_tokens':1,'completion_tokens':events,'total_tokens':events+1}}]
        base={'id':'fixture','object':'chat.completion.chunk','created':0,'model':'/model'}
        frames=[{**base,**frame} for frame in frames]
        payload=(''.join('data: '+json.dumps(f)+'\n\n' for f in frames)+'data: [DONE]\n\n').encode()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ('/health','/health_generate','/v1/models'):self.send_error(404);return
                data=b'{"data":[{"id":"/model"}]}' if self.path=='/v1/models' else b'{}'
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                length=self.headers.get('Content-Length','')
                if self.path!='/v1/chat/completions' or not length.isdigit() or not 1<=int(length)<=8_001_024:
                    self.send_error(400);return
                raw=self.rfile.read(int(length));body=json.loads(raw)
                if body.get('stream') is not True or body.get('max_tokens')!=events:self.send_error(400);return
                with parent.lock:
                    parent.records.append({'request_id':self.headers.get('X-Request-ID'),'session_id':self.headers.get('X-Session-ID'),
                        'rank_headers':self.headers.get_all('X-data-parallel-rank',[]),'backend_port':parent.port,
                        'body_sha256':hashlib.sha256(raw).hexdigest(),'body_bytes':len(raw),'observed_ns':time.monotonic_ns(),
                        'body_base64':base64.b64encode(raw).decode('ascii'),
                        'server_cpus':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None})
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            def log_message(self,*args):pass
        class Server(ThreadingHTTPServer):request_queue_size=256
        self.server=Server(('127.0.0.1',port),Handler);self.port=self.server.server_address[1]
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=10)
        if self.thread.is_alive():raise RuntimeError('client fixture backend did not stop')
