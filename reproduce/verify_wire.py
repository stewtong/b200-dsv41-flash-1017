"""Replay retained HTTP body chunks independently of the serving SSE parser."""
import base64
import codecs
import hashlib
import json


def retained_bytes(raw):
    if 'wire_chunks' not in raw:raise ValueError('raw wire evidence unavailable')
    chunks=[];previous=-1
    for row in raw['wire_chunks']:
        stamp=row['offset_ns']
        if type(stamp) is not int or stamp<previous or stamp<0 or stamp>raw['terminal_ns']-raw['start_ns']:
            raise ValueError('invalid wire clock')
        previous=stamp
        try:chunk=base64.b64decode(row['base64'],validate=True)
        except (ValueError,TypeError):raise ValueError('malformed wire bytes')
        if type(row.get('eof')) is not bool or (row['eof'] and chunk):raise ValueError('invalid wire EOF')
        chunks.append(chunk)
    data=b''.join(chunks)
    if (len(data)!=raw['wire_byte_count'] or len(data)>raw['wire_limit_bytes'] or
        hashlib.sha256(data).hexdigest()!=raw['wire_sha256'] or raw['wire_bytes_observed']<len(data)):
        raise ValueError('wire evidence integrity mismatch')
    return data


def verify_complete(raw):
    data=retained_bytes(raw)
    if (raw.get('complete') is not True or raw.get('http_status')!=200 or raw.get('error') is not None or
        raw.get('wire_truncated') is not False or raw.get('wire_response_kind')!='sse' or
        raw['wire_byte_count']!=raw['wire_bytes_observed']):raise ValueError('wire response is incomplete')
    expected=raw.get('wire_expected_body_bytes')
    if expected is not None and (type(expected) is not int or expected!=len(data)):raise ValueError('HTTP Content-Length evidence differs')
    decoder=codecs.getincrementaldecoder('utf-8')('strict');buffer='';pending=[];done=False;events=[];eof=False
    for row in raw['wire_chunks']:
        if eof:raise ValueError('wire data after EOF')
        chunk=base64.b64decode(row['base64'],validate=True);eof=row['eof']
        if not chunk and not eof:raise ValueError('empty nonterminal wire chunk')
        buffer+=decoder.decode(chunk,final=eof)
        while '\n' in buffer:
            line,buffer=buffer.split('\n',1);line=line.rstrip('\r')
            if not line:
                if not pending:continue
                payload='\n'.join(pending);pending=[]
                if done:raise ValueError('SSE event after DONE')
                if payload=='[DONE]':kind='done';value=None;done=True
                else:kind='event';value=json.loads(payload)
                events.append({'offset_ns':row['offset_ns'],'kind':kind,'data':value})
            elif line.startswith('data:'):
                value=line[5:];pending.append(value[1:] if value.startswith(' ') else value)
    if not eof or buffer or pending or not done:raise ValueError('wire SSE terminal evidence missing')
    if events!=raw['events']:raise ValueError('decoded events differ from retained wire chunks')
    return {'status':'wire_events_verified','wire_bytes':len(data),'events':len(events)}
