"""Capture streamed Chat Completions without retries or fabricated token clocks."""
import codecs
import base64
import hashlib
import errno
import select
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit

CAPTURE_SCHEMA='http-body-chunks/v1'

class SSEDecoder:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.buffer = ''
        self.data = []
        self.done = False

    def feed(self, chunk, final=False):
        self.buffer += self.decoder.decode(chunk, final=final)
        frames = []
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line = line.rstrip('\r')
            if not line:
                if self.data:
                    payload = '\n'.join(self.data)
                    self.data = []
                    if self.done:
                        raise ValueError('event after DONE')
                    if payload == '[DONE]':
                        self.done = True
                        frames.append(('done', None))
                    else:
                        frames.append(('event', json.loads(payload)))
            elif line.startswith('data:'):
                value = line[5:]
                self.data.append(value[1:] if value.startswith(' ') else value)
        if final and (self.buffer or self.data or not self.done):
            raise ValueError('truncated stream or missing DONE')
        return frames

class Cancellation:
    def __init__(self):
        self.lock = threading.Lock()
        self.sockets = set()
        self.cancelled = False

    def register(self, sock):
        if sock is None: return
        with self.lock:
            if self.cancelled:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                sock.close()
            else:
                self.sockets.add(sock)

    def unregister(self,sock):
        with self.lock:self.sockets.discard(sock)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            for sock in self.sockets:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                sock.close()
            self.sockets.clear()

def _connect_socket(host,port,start_ns,timeout,cancellation):
    """Register the socket before connecting, with bounded cancellation polls."""
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        sock.setblocking(False)
        if cancellation:cancellation.register(sock)
        if cancellation and cancellation.cancelled:raise InterruptedError('request cancelled before connect')
        code=sock.connect_ex(('127.0.0.1' if host=='localhost' else host,port))
        pending={errno.EINPROGRESS,errno.EWOULDBLOCK,errno.EALREADY}
        while code not in (0,errno.EISCONN):
            if code not in pending:raise OSError(code,errno.errorcode.get(code,'connect failed'))
            if cancellation and cancellation.cancelled:raise InterruptedError('request cancelled during connect')
            remaining=timeout-(time.monotonic_ns()-start_ns)/1e9
            if remaining<=0:raise TimeoutError('request connect deadline expired')
            _,writable,exceptional=select.select([],[sock],[sock],min(.1,remaining))
            if writable or exceptional:code=sock.getsockopt(socket.SOL_SOCKET,socket.SO_ERROR)
        remaining=timeout-(time.monotonic_ns()-start_ns)/1e9
        if remaining<=0:raise TimeoutError('request connect deadline expired')
        sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);sock.setblocking(True);sock.settimeout(remaining)
        return sock
    except BaseException:
        if cancellation:cancellation.unregister(sock)
        sock.close()
        raise


def capture(url, body, request_id, timeout, cancellation=None, session_id=None, max_wire_bytes=64*1024*1024, on_event=None):
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or not parsed.port:
        raise ValueError('only an explicit loopback HTTP endpoint is permitted')
    if not isinstance(request_id, str) or not request_id or '\n' in request_id or '\r' in request_id:
        raise ValueError('invalid request id')
    if session_id is not None and (not isinstance(session_id, str) or not session_id or '\n' in session_id or '\r' in session_id):
        raise ValueError('invalid session id')
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    if type(max_wire_bytes) is not int or not 1<=max_wire_bytes<=64*1024*1024:
        raise ValueError('invalid bounded wire evidence limit')
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    start = time.monotonic_ns()
    result = {'request_id': request_id, 'session_id': session_id, 'start_ns': start, 'events': [], 'http_status': None,
              'terminal_ns': None, 'error': None, 'complete': False, 'wire_chunks':[],
              'wire_byte_count':0,'wire_bytes_observed':0,'wire_truncated':False,
              'wire_limit_bytes':max_wire_bytes,'wire_response_kind':None}
    transport_sock=None;phase='prepare'
    decoder = SSEDecoder()
    wire_digest=hashlib.sha256()
    def save_wire(chunk,now):
        result['wire_bytes_observed']+=len(chunk)
        room=result['wire_limit_bytes']-result['wire_byte_count']
        kept=chunk[:room];wire_digest.update(kept);result['wire_byte_count']+=len(kept)
        result['wire_chunks'].append({'offset_ns':now-start,'base64':base64.b64encode(kept).decode('ascii'),'eof':not chunk})
        if len(kept)!=len(chunk):
            result['wire_truncated']=True
            raise ValueError('bounded wire evidence limit exceeded')
    try:
        headers = {'Content-Type': 'application/json', 'X-Request-Id': request_id}
        if session_id is not None: headers['X-Session-Id'] = session_id
        payload = json.dumps(body,allow_nan=False,sort_keys=True,separators=(',',':'))
        result.update(request_serialization='canonical-json-ensure-ascii/v1',
                      request_body_sha256=hashlib.sha256(payload.encode('utf-8')).hexdigest(),
                      request_payload_bytes=len(payload.encode('utf-8')),
                      request_headers={'Content-Type':headers['Content-Type'],'X-Request-Id':request_id,
                                       **({'X-Session-Id':session_id} if session_id is not None else {})})
        remaining = timeout - (time.monotonic_ns() - start) / 1e9
        if remaining <= 0: raise TimeoutError('request deadline expired before transmission')
        phase='connect'
        transport_sock=_connect_socket(parsed.hostname,parsed.port,start,timeout,cancellation)
        conn.sock=transport_sock
        phase='request_send'
        conn.timeout=remaining;conn.sock.settimeout(max(.000001,timeout-(time.monotonic_ns()-start)/1e9))
        conn.request('POST', parsed.path or '/v1/chat/completions', body=payload, headers=headers)
        if cancellation: cancellation.register(conn.sock)
        remaining = timeout - (time.monotonic_ns() - start) / 1e9
        if remaining <= 0: raise TimeoutError('request deadline expired before response headers')
        if conn.sock is None: raise RuntimeError('cannot enforce response-header deadline')
        conn.sock.settimeout(remaining)
        phase='response_headers'
        response = conn.getresponse()
        result['http_status'] = response.status
        sock = conn.sock or getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
        if sock is None:
            raise RuntimeError('cannot enforce stream deadline on response socket')
        if cancellation: cancellation.register(sock)
        result['response_headers']={k.lower():v for k,v in response.getheaders() if k.lower() in ('content-type','content-length','x-request-id')}
        phase='response_body'
        result['wire_response_kind']='sse' if response.status==200 else 'http_error_body'
        length=response.getheader('Content-Length')
        if not response.chunked and length is not None and not length.isascii():raise ValueError('invalid HTTP Content-Length')
        if not response.chunked and length is not None and not length.isdigit():raise ValueError('invalid HTTP Content-Length')
        result['wire_expected_body_bytes']=int(length) if not response.chunked and length is not None else None
        def read_chunk(message):
            remaining=timeout-(time.monotonic_ns()-start)/1e9
            if remaining<=0:raise TimeoutError(message)
            if response.isclosed():return b''
            sock.settimeout(remaining)
            return response.read1(4096)
        if response.status!=200:
            result['wire_limit_bytes']=min(max_wire_bytes,65536)
            while True:
                chunk=read_chunk('error-response body deadline expired');save_wire(chunk,time.monotonic_ns())
                if not chunk:break
            raise ValueError('non-200 HTTP response')
        while True:
            chunk=read_chunk('request deadline expired')
            now = time.monotonic_ns()
            save_wire(chunk,now)
            for kind, event in decoder.feed(chunk, final=not chunk):
                row={'offset_ns': now - start, 'kind': kind, 'data': event}
                result['events'].append(row)
                if on_event is not None: on_event(row)
            if not chunk:
                break
        expected=result['wire_expected_body_bytes']
        if expected is not None and result['wire_byte_count']!=expected:raise ValueError('incomplete HTTP Content-Length body')
        if time.monotonic_ns()-start>timeout*1e9:raise TimeoutError('request deadline expired during response processing')
        result['complete'] = True
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        result['error_phase']=phase
    finally:
        result['wire_sha256']=wire_digest.hexdigest()
        result['terminal_ns'] = time.monotonic_ns()
        result['cancel_requested']=bool(cancellation and cancellation.cancelled)
        try:conn.close()
        finally:
            if cancellation and transport_sock is not None:cancellation.unregister(transport_sock)
    return result

def summarize_stream(result):
    """Expose clocks and usage; validity requires independent task scoring."""
    usage, finishes, first_output, first_visible, last_output = [], [], None, None, None
    for row in result['events']:
        if row['kind'] != 'event': continue
        event = row['data']
        if event.get('usage') is not None: usage.append(event['usage'])
        for choice in event.get('choices', []):
            if choice.get('finish_reason') is not None: finishes.append(choice['finish_reason'])
            delta = choice.get('delta') or {}
            visible = delta.get('content') or ''
            reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
            tools = delta.get('tool_calls') or []
            if visible or reasoning or tools:
                if first_output is None: first_output = row['offset_ns']
                last_output = row['offset_ns']
            if visible and first_visible is None: first_visible = row['offset_ns']
    if not result['complete'] or len(usage) != 1 or len(finishes) != 1 or first_output is None:
        raise ValueError('incomplete stream, missing output, finish or unique usage')
    return {'first_output_ns': first_output, 'first_visible_ns': first_visible, 'last_output_ns': last_output,
            'finish_reason': finishes[0], 'usage': usage[0], 'task_validity': 'unscored'}
