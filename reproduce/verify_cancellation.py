"""Recompute disconnect stage and work-absence proof from raw control records."""
import base64
import codecs
import hashlib
import json
import math
from pathlib import Path
import re

import verify_wire


def read(path): return json.loads(Path(path).read_text())


def totals(row, expected):
    """Independent parser: do not import the control or its absence validator."""
    loads={url[:-8]+'/load' for url in expected}
    if set(row['metrics'])!=set(expected) or set(row['loads'])!=loads or set(row['raw_load_bodies'])!=loads:
        raise ValueError('cancellation endpoint coverage differs')
    start=row['started_ns'];end=row['monotonic_ns']
    if type(start) is not int or type(end) is not int or not 0<=start<=end or end-start>2_500_000_000:
        raise ValueError('cancellation collection clocks differ')
    result=[0,0,0]
    for url,ranks in expected.items():
        types={};values={}
        for line in row['metrics'][url].splitlines():
            parts=line.split()
            if len(parts)==4 and parts[:2]==['#','TYPE'] and parts[2] in ('vllm:num_requests_running','vllm:num_requests_waiting'):
                if parts[2] in types:raise ValueError('duplicate cancellation metric type')
                types[parts[2]]=parts[3]
            if not re.match(r'^vllm:num_requests_(?:running|waiting)(?:\s|\{)',line):continue
            match=re.fullmatch(r'(vllm:num_requests_(running|waiting))\{([^}]*)\}\s+(\S+)\s*',line)
            if match is None:raise ValueError('malformed cancellation work series')
            engine=re.search(r'(?:^|,)\s*engine="([^"]+)"(?:,|$)',match[3])
            if engine is None:raise ValueError('cancellation engine absent')
            key=(match[1],engine[1]);value=float(match[4])
            if key in values or not math.isfinite(value) or value<0 or not value.is_integer():
                raise ValueError('cancellation work value invalid')
            values[key]=int(value)
        wanted={(name,str(rank)) for name in ('vllm:num_requests_running','vllm:num_requests_waiting') for rank in ranks}
        if not ranks or len(set(map(str,ranks)))!=len(ranks) or set(values)!=wanted or types!=dict.fromkeys(('vllm:num_requests_running','vllm:num_requests_waiting'),'gauge'):
            raise ValueError('cancellation rank or gauge population differs')
        result[0]+=sum(v for (name,_),v in values.items() if name.endswith('_running'))
        result[1]+=sum(v for (name,_),v in values.items() if name.endswith('_waiting'))
    for url,value in row['loads'].items():
        if json.loads(row['raw_load_bodies'][url])!=value or set(value)!={'server_load'} or type(value['server_load']) is not int or value['server_load']<0:
            raise ValueError('cancellation frontend raw load differs')
        result[2]+=value['server_load']
    return result


def first_output(raw):
    verify_wire.retained_bytes(raw)
    # Cancellation may cut the last event or UTF-8 codepoint. Only complete
    # retained frames establish output; any complete terminal frame rejects it.
    decoder=codecs.getincrementaldecoder('utf-8')('strict');buffer='';pending=[];first=None
    for chunk in raw['wire_chunks']:
        buffer+=decoder.decode(base64.b64decode(chunk['base64'],validate=True),final=False)
        while '\n' in buffer:
            line,buffer=buffer.split('\n',1);line=line.rstrip('\r')
            if line.startswith('data:'):pending.append(line[5:].lstrip(' '))
            elif not line and pending:
                value='\n'.join(pending);pending=[]
                if value=='[DONE]':raise ValueError('terminal stream cannot qualify cancellation')
                event=json.loads(value)
                for choice in event.get('choices',[]):
                    if choice.get('finish_reason') is not None:raise ValueError('terminal stream cannot qualify cancellation')
                    delta=choice.get('delta') or {}
                    if first is None and any(delta.get(k) for k in ('content','reasoning','reasoning_content','tool_calls')):
                        first=raw['start_ns']+chunk['offset_ns']
    return first


def verify(root):
    root=Path(root);contract=read(root/'control-input.json');task=read(root/'source-task.json')
    raw=read(root/'cancelled-request-raw.json');cancel=read(root/'cancellation.json');result=read(root/'control-result.json')
    mode=contract['mode'];run_id=contract['run_id'];expected=contract['expected_ranks']
    body=json.loads(json.dumps(task['body']));body['messages'].insert(0,{'role':'system','content':
        'Cancellation control '+run_id+' '+mode+'. This isolated request is outside measured tasks.'})
    body.update(max_tokens=8192,min_tokens=8192,ignore_eos=True)
    if (contract['schema']!='exclusive-cancellation-control/v1' or mode not in ('prefill','decode') or
        contract['body']!=body or contract['source_task_sha256']!=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest() or
        contract['request_id']!=run_id+'-cancel-'+mode or contract['session_id']!=contract['request_id'] or
        raw['request_id']!=contract['request_id'] or raw['session_id']!=contract['session_id'] or
        raw['complete'] is not False or raw['cancel_requested'] is not True or raw['wire_truncated'] is not False or
        raw['http_status'] not in (None,200)):
        raise ValueError('cancellation request, provenance or partial transport differs')
    cancelled=cancel['monotonic_ns'];terminal=raw['terminal_ns'];output=first_output(raw)
    if (type(cancelled) is not int or type(terminal) is not int or not raw['start_ns']<=cancelled<=terminal or
        (mode=='prefill' and output is not None and output<=cancelled) or
        (mode=='decode' and (output is None or output>cancelled))):
        raise ValueError('cancellation stage or client clocks differ')
    before=read(root/'before.json');busy=read(root/'busy.json');after=read(root/'after.json')
    rows=[before,busy]+after
    if (len(after)!=2 or any(row['clock_id']!=contract['clock_id'] for row in rows) or
        totals(before,expected)!=[0,0,0] or totals(busy,expected)!=[1,0,1] or
        any(totals(row,expected)!=[0,0,0] for row in after) or
        before['monotonic_ns']>busy['started_ns'] or busy['monotonic_ns']>cancelled or
        after[0]['started_ns']<terminal+1_500_000_000 or after[1]['started_ns']<after[0]['monotonic_ns']+1_000_000_000):
        raise ValueError('exclusive live-work transition or separated zero proof differs')
    health=read(root/'post-cancellation-health.json')
    if set(health)!={url[:-8]+'/health' for url in expected} or any(v['status']!=200 or
        not after[-1]['monotonic_ns']<=v['started_ns']<=v['monotonic_ns'] for v in health.values()):
        raise ValueError('post-cancellation backend health differs')
    semantic=read(root/'post-cancellation-semantic.json');semantic_raw=semantic['raw']
    verify_wire.verify_complete(semantic_raw)
    text=''.join((c.get('delta') or {}).get('content') or '' for e in semantic_raw['events'] if e['kind']=='event' for c in e['data'].get('choices',[]))
    finishes=[c['finish_reason'] for e in semantic_raw['events'] if e['kind']=='event' for c in e['data'].get('choices',[]) if c.get('finish_reason') is not None]
    usage=[e['data']['usage'] for e in semantic_raw['events'] if e['kind']=='event' and e['data'].get('usage') is not None]
    if (semantic_raw['request_id']!=contract['request_id']+'-semantic' or semantic_raw['start_ns']<after[-1]['monotonic_ns'] or
        text.strip()!='READY' or finishes!=['stop'] or len(usage)!=1 or any(type(usage[0].get(k)) is not int or usage[0][k]<=0 for k in ('prompt_tokens','completion_tokens'))):
        raise ValueError('post-cancellation semantic raw response differs')
    if result['status']!='passed' or result['client_terminal_ns']!=terminal or result['cancellation_ns']!=cancelled or (root/'control-failure.json').exists():
        raise ValueError('cancellation result differs from raw proof')
    return {'status':'passed','mode':mode,'expected_ranks':sum(map(len,expected.values())),
            'scope':'exclusive sampled work absence and client-observed cancellation stage; no per-engine prefill-stage or scored-overhead claim'}
