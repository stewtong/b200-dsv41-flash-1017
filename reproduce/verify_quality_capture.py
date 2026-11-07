"""Recompute quality-capture populations without importing the capture runner."""
import json
import hashlib
from pathlib import Path
import re
import verify_quality
import verify_wire


def verify_conditioning(root,tasks,contract):
    """Recompute the unscored prefix pass separately from quality task accuracy."""
    expected={t['id']+'.json' for t in tasks}|{'summary.json'}
    if ({p.name for p in root.iterdir()}!=expected or any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError('quality conditioning file coverage differs')
    rows=[];tokens=0
    for task in tasks:
        row=json.loads((root/(task['id']+'.json')).read_text());raw=row['raw']
        verify_wire.verify_complete(raw)
        ident=contract['request_namespace']+'c'+str(task['session'])
        verify_quality.verify_request_binding(raw,task['conditioning_body'],ident,contract['session_ids'][task['id']])
        if (row.get('request_id')!=ident or row.get('task_id')!=task['id'] or
            row.get('clock_id')!=contract['clock_id'] or raw.get('complete') is not True or raw.get('http_status')!=200 or
            row.get('valid') is not True or row.get('prompt_tokens_match') is not True or
            row.get('start_ns')!=raw.get('start_ns') or row.get('end_ns')!=raw.get('terminal_ns') or
            any(type(row.get(k)) is not int for k in ('start_ns','end_ns')) or row['end_ns']<=row['start_ns']):
            raise ValueError('quality conditioning identity, transport or clocks differ')
        usage=[];finishes=[];visible=''
        for event in raw['events']:
            if event['kind']!='event':continue
            data=event['data']
            if data.get('usage') is not None:usage.append(data['usage'])
            for choice in data.get('choices',[]):
                visible+=(choice.get('delta') or {}).get('content') or ''
                if choice.get('finish_reason') is not None:finishes.append(choice['finish_reason'])
        if (len(usage)!=1 or finishes!=['stop'] or visible.strip()!=task['expected_conditioning_answer'] or
            type(usage[0].get('prompt_tokens')) is not int or
            abs(usage[0]['prompt_tokens']-task['prefix_tokens'])>task['prefix_tokens']*.0025 or
            type(usage[0].get('completion_tokens')) is not int or not 0<usage[0]['completion_tokens']<=128 or
            row.get('completion_tokens')!=usage[0]['completion_tokens']):
            raise ValueError('quality conditioning answer or usage differs')
        tokens+=usage[0]['completion_tokens'];rows.append(row)
    seconds=(max(r['end_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
    summary={'duration_seconds':seconds,'valid_tokens':tokens,'invalid_partial_tokens':0,
             'valid_tokens_per_second':tokens/seconds,'valid_requests':len(rows),'requests':len(rows)}
    if json.loads((root/'summary.json').read_text())!={'status':'requests_passed_cleanup_pending','summary':summary}:
        raise ValueError('quality conditioning arithmetic differs')
    return rows


def verify(root, tasks, repetition, target):
    root=Path(root)
    ids=[t['id'] for t in tasks]
    if (not ids or len(set(ids))!=len(ids) or
        any(not isinstance(i,str) or not re.fullmatch(r'[A-Za-z0-9]+',i) for i in ids)):
        raise ValueError('invalid quality population')
    caches={t.get('cache_designation','cold') for t in tasks}
    if len(caches)!=1:raise ValueError('quality population mixes cache states')
    contract=json.loads((root/'quality-capture-contract.json').read_text())
    namespace=contract.get('request_namespace');affinity=contract.get('session_ids')
    if (contract.get('schema')!='quality-capture/v2' or not isinstance(namespace,str) or
        not re.fullmatch(r'ql[a-f0-9]{24}',namespace) or not isinstance(affinity,dict) or set(affinity)!=set(ids) or
        any(not isinstance(s,str) or not s or not s.isascii() or any(not 33<=ord(c)<=126 for c in s) for s in affinity.values()) or
        len(set(affinity.values()))!=len(ids) or not isinstance(contract.get('clock_id'),str) or not contract['clock_id'] or
        contract.get('cache')!=next(iter(caches)) or contract.get('repetition')!=repetition or contract.get('target')!=target or
        contract.get('scheduled_measured_requests')!=len(tasks) or
        contract.get('tasks_sha256')!=hashlib.sha256(json.dumps(tasks,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()):
        raise ValueError('quality capture contract differs')
    expected={'quality-capture-contract.json','measured'}|({'conditioning'} if caches=={'warm_prefix'} else set())
    if ({p.name for p in root.iterdir()}!=expected or any(p.is_symlink() for p in root.iterdir()) or
        not (root/'quality-capture-contract.json').is_file() or not (root/'measured').is_dir()):
        raise ValueError('quality capsule file coverage differs')
    conditions=[]
    if caches=={'warm_prefix'}:
        if not (root/'conditioning').is_dir():
            raise ValueError('warm quality directories differ')
        conditions=verify_conditioning(root/'conditioning',tasks,contract)
    root=root/'measured'
    expected={i+'.json' for i in ids}|{'summary.json'}
    if ({p.name for p in root.iterdir()}!=expected or
        any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError('quality request file coverage differs')
    rows=[];fatal=False
    for task in tasks:
        row=json.loads((root/(task['id']+'.json')).read_text());raw=row['raw']
        ident=namespace+'m'+str(task['session'])
        if (row.get('request_id')!=ident or raw.get('request_id')!=ident or row.get('task_id')!=task['id'] or
            raw.get('session_id')!=affinity[task['id']] or row.get('clock_id')!=contract['clock_id'] or row.get('start_ns')!=raw.get('start_ns') or
            row.get('end_ns')!=raw.get('terminal_ns') or
            any(type(row.get(k)) is not int or row[k]<0 for k in ('start_ns','end_ns')) or
            row['end_ns']<row['start_ns']):raise ValueError('quality identity or clocks differ')
        score=verify_quality.evaluate(task,raw,repetition,len(tasks),target,require_wire=True,
            request_id=ident,session_id=affinity[task['id']])
        if row.get('quality_score')!=score:raise ValueError('quality score differs from raw response')
        usage=[];finishes=[];visible='';first=None;first_visible=None;last=None;previous=-1
        for event in raw['events']:
            offset=event['offset_ns']
            if type(offset) is not int or not previous<=offset<=row['end_ns']-row['start_ns']:
                raise ValueError('quality event clocks differ')
            previous=offset
            if event['kind']!='event':continue
            value=event['data']
            if value.get('usage') is not None:usage.append(value['usage'])
            for choice in value.get('choices',[]):
                delta=choice.get('delta') or {};text=delta.get('content') or '';visible+=text
                output=text or delta.get('reasoning_content') or delta.get('reasoning') or delta.get('tool_calls')
                if output:
                    if first is None:first=offset
                    last=offset
                if text.strip() and first_visible is None:first_visible=offset
                if choice.get('finish_reason') is not None:finishes.append(choice['finish_reason'])
        parsed=raw.get('complete') is True and len(usage)==len(finishes)==1 and first is not None
        tokens=usage[0].get('completion_tokens') if parsed else None
        if type(tokens) is not int or tokens<0:tokens=None
        prompt=usage[0].get('prompt_tokens') if parsed else None
        input_ok=type(prompt) is int and prompt>=0 and abs(prompt-task['input_tokens'])<=task['input_tokens']*.0025
        clean='\ufffd' not in visible and not any(ord(c)<32 and c not in '\n\r\t' for c in visible)
        bad=raw.get('complete') is not True or raw.get('http_status')!=200 or not parsed or not input_ok or not clean
        valid=input_ok and score['correct'];fatal|=bad
        if (row.get('valid') is not valid or row.get('completion_tokens')!=tokens or
            row.get('prompt_tokens_match') is not input_ok or row.get('fatal_transport_or_content') is not bad):
            raise ValueError('quality validity or token population differs')
        duration=(row['end_ns']-row['start_ns'])/1e9
        if duration<=0:raise ValueError('invalid quality request duration')
        timing=None
        if valid:
            timing={'first_output_seconds':first/1e9,'first_visible_seconds':first_visible/1e9 if first_visible is not None else None,
                'stream_window_seconds_per_token':(last-first)/1e9/(tokens-1) if tokens>1 else None,
                'request_output_tokens_per_second':tokens/duration,
                'scope':'stream-derived; multi-token chunks do not reveal exact token times'}
        if row.get('timing')!=timing:raise ValueError('quality timing arithmetic differs')
        rows.append(row)
    if len({r.get('clock_id') for r in rows})!=1 or not rows[0].get('clock_id'):
        raise ValueError('mixed quality clock domains')
    if conditions and (any(r.get('clock_id')!=rows[0]['clock_id'] for r in conditions) or
                       max(r['end_ns'] for r in conditions)>min(r['start_ns'] for r in rows)):
        raise ValueError('quality conditioning overlaps measurement or uses a different clock')
    seconds=(max(r['end_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
    tokens=sum(r['completion_tokens'] for r in rows if r['valid']);invalid=[r['completion_tokens'] for r in rows if not r['valid']]
    count=sum(r['valid'] for r in rows)
    summary={'duration_seconds':seconds,'valid_tokens':tokens,'invalid_partial_tokens':None if None in invalid else sum(invalid),
             'valid_tokens_per_second':tokens/seconds,'valid_requests':count,'requests':len(rows)}
    status='failed' if fatal else 'captured_with_invalid_requests_cleanup_pending' if count<len(rows) else 'requests_passed_cleanup_pending'
    if json.loads((root/'summary.json').read_text())!={'status':status,'summary':summary}:
        raise ValueError('quality capture summary differs')
    return {'status':status,'correct':count,'requests':len(rows),
            'scope':'raw transport, declared task contracts and population arithmetic; parent cleanup, tokenizer provenance and scored eligibility require separate verification'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('tasks');p.add_argument('--repetition',type=int,required=True);p.add_argument('--target',type=int,required=True)
    args=p.parse_args()
    print(json.dumps(verify(args.root,json.loads(Path(args.tasks).read_text()),args.repetition,args.target),sort_keys=True))
