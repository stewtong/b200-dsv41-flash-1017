"""Exercise the installed router against loopback-only synthetic backends."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import time
import urllib.error
import urllib.request


class Backend:
    def __init__(self,port):
        self.port=port;self.records=[];self.lock=threading.Lock();parent=self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ('/health','/health_generate','/v1/models'):
                    self.send_error(404);return
                data=b'{"data":[{"id":"/model"}]}' if self.path=='/v1/models' else b'{}'
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                if self.path!='/v1/chat/completions':self.send_error(404);return
                length=self.headers.get('Content-Length')
                if not length or not length.isdigit() or int(length)>1_000_000:self.send_error(400);return
                self.rfile.read(int(length))
                row={'request_id':self.headers.get('X-Request-ID'),'session_id':self.headers.get('X-Session-ID'),
                     'rank_headers':self.headers.get_all('X-data-parallel-rank',[]),'backend_port':parent.port}
                with parent.lock:parent.records.append(row)
                if row['session_id']=='no-retry':
                    self.send_response(503);self.send_header('Content-Length','0');self.end_headers();return
                data=json.dumps({'id':'probe','object':'chat.completion','choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}]}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
        self.port=self.server.server_address[1]
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=10)
        if self.thread.is_alive():raise RuntimeError('synthetic backend did not stop')


def request(port,session,request_id):
    data=json.dumps({'model':'/model','messages':[{'role':'user','content':'probe'}],'max_tokens':1}).encode()
    req=urllib.request.Request('http://127.0.0.1:'+str(port)+'/v1/chat/completions',data=data,
                               headers={'Content-Type':'application/json','X-Session-ID':session,'X-Request-ID':request_id},method='POST')
    with urllib.request.urlopen(req,timeout=10) as response:
        payload=json.load(response)
        if response.status!=200 or payload.get('id')!='probe':raise ValueError('router probe response invalid')


def verify_no_retry(port, records):
    try:request(port,'no-retry','no-retry-r0')
    except urllib.error.HTTPError as exc:
        try: status=exc.code
        finally: exc.close()
        if not 400<=status<=599:raise ValueError('unexpected rejection status')
    else:raise ValueError('synthetic backend failure was hidden')
    time.sleep(1)  # Retain a bounded window for a delayed duplicate dispatch.
    matches=[r for r in records() if r['session_id']=='no-retry' and r['request_id']=='no-retry-r0']
    if len(matches)!=1:raise ValueError('router retried or lost the failed request')
    return {'observed_backend_attempts':1,'status':'passed'}


def verify(records, sessions, attempts, dp, backend_ports):
    expected={(session,f'{session}-r{index}') for session in sessions for index in range(attempts)}
    if len(records)!=len(expected):raise ValueError('router probe request count mismatch')
    observed=set(); mapping={}
    for row in records:
        key=(row['session_id'],row['request_id']);headers=row['rank_headers'];port=row['backend_port']
        if key not in expected or key in observed or port not in backend_ports:raise ValueError('router probe identity mismatch')
        observed.add(key)
        if dp==1:
            if len(headers)>1 or (headers and headers[0]!='0'):raise ValueError('unexpected rank header')
            rank=0
        else:
            if len(headers)!=1 or not headers[0].isdigit() or not 0<=int(headers[0])<dp:raise ValueError('missing or invalid rank header')
            rank=int(headers[0])
        placement=(port,rank)
        if row['session_id'] in mapping and mapping[row['session_id']]!=placement:raise ValueError('session affinity changed')
        mapping[row['session_id']]=placement
    if observed!=expected:raise ValueError('missing probe request')
    return {'sessions':{k:{'backend_port':v[0],'rank':v[1]} for k,v in mapping.items()},
            'requests':len(records),'scope':'synthetic backend routing only; real GPU rank placement unverified'}
