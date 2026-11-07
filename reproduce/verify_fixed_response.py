"""Score retained fixed-output HTTP evidence without importing its capturer."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import verify_wire
import verify_fixed_task


def obvious_periodic_filler(text,identifier):
    """Detect dominant short loops, not general task quality or prose style.

    Ignore the required code. At least64 words, four complete motif lengths
    and95% exact positional agreement are required. Ordinary repeated headings
    or vocabulary alone do not trigger this conservative corruption check.
    """
    body=text.lstrip()
    if body.startswith(identifier):body=body[len(identifier):]
    words=re.findall(r'[A-Za-z0-9]+',body.lower())
    if len(words)<64:return False
    for period in range(1,min(128,len(words)//4)+1):
        if sum(word==words[i%period] for i,word in enumerate(words))/len(words)>=.95:
            return True
    return False


def evaluate(task,raw,request_id=None,session_id=None):
    verify_fixed_task.verify(task)
    ident=task['id'] if request_id is None else request_id
    affinity=ident if session_id is None else session_id
    if raw.get('request_id')!=ident or raw.get('session_id')!=affinity:raise ValueError('fixed response identity differs')
    start=raw.get('start_ns');end=raw.get('terminal_ns')
    if type(start) is not int or type(end) is not int or not 0<=start<end:raise ValueError('fixed response clocks invalid')
    payload=json.dumps(task['body'],allow_nan=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    expected={'Content-Type':'application/json','X-Request-Id':ident,'X-Session-Id':affinity}
    binding=(raw.get('request_serialization')=='canonical-json-ensure-ascii/v1' and
        raw.get('request_body_sha256')==hashlib.sha256(payload).hexdigest() and raw.get('request_payload_bytes')==len(payload) and
        raw.get('request_headers')==expected)
    if raw.get('complete') is True and not binding:raise ValueError('fixed response lacks exact transmitted request binding')
    if raw.get('complete') is True:verify_wire.verify_complete(raw)
    else:verify_wire.retained_bytes(raw)
    usage=[];finishes=[];visible=[];first=None;first_visible=None;last=None;unexpected=False
    for row in raw.get('events',[]):
        if row.get('kind')!='event':continue
        event=row['data'];offset=row['offset_ns']
        if type(offset) is not int or not 0<=offset<=end-start:raise ValueError('fixed output event clock invalid')
        if event.get('usage') is not None:usage.append(event['usage'])
        for choice in event.get('choices',[]):
            delta=choice.get('delta') or {}
            if choice.get('index',0)!=0 or delta.get('role') not in (None,'assistant') or any(delta.get(k) for k in ('tool_calls','reasoning','reasoning_content')):
                unexpected=True
            content=delta.get('content')
            if content is not None and not isinstance(content,str):raise ValueError('fixed visible delta is not text')
            if content:
                visible.append(content);first=offset if first is None else first;last=offset
                if content.strip() and first_visible is None:first_visible=offset
            if choice.get('finish_reason') is not None:finishes.append(choice['finish_reason'])
    text=''.join(visible);u=usage[0] if len(usage)==1 and isinstance(usage[0],dict) else {}
    prompt=u.get('prompt_tokens');tokens=u.get('completion_tokens');cap=task['output_tokens']
    prompt_ok=type(prompt) is int and abs(prompt-task['input_tokens'])<=task['input_tokens']*.0025
    output_ok=type(tokens) is int and tokens==cap and len(finishes)==1 and finishes[0] in ('stop','length')
    performance=task['prompt_version']=='fixed-performance-v1'
    # The wire verifier checks strict UTF-8. Performance validity does not
    # imply task correctness, a language restriction or non-repetitive prose.
    clean=bool(text) if performance else bool(text) and text.isascii() and all(ord(c)>=32 or c in '\n\r\t' for c in text)
    identifier=text.lstrip().startswith(task['expected_answer'])
    nonperiodic=not obvious_periodic_filler(text,task['expected_answer'])
    transport=raw.get('complete') is True and raw.get('http_status')==200 and raw.get('cancel_requested') is not True
    correct=bool(binding and transport and prompt_ok and output_ok and clean and (performance or (identifier and nonperiodic)) and not unexpected and first is not None)
    reasons=[]
    for ok,label in ((binding,'request_binding'),(transport,'transport'),(prompt_ok,'input_usage'),(output_ok,'fixed_output_usage'),
                     (clean,'nonempty_content' if performance else 'ascii_content'),(performance or identifier,'leading_identifier'),(performance or nonperiodic,'periodic_filler'),(not unexpected,'unexpected_output_channel')):
        if not ok:reasons.append(label)
    timing=None
    if correct:
        seconds=(end-start)/1e9
        timing={'request_seconds':seconds,'first_output_seconds':first/1e9,'first_visible_seconds':first_visible/1e9 if first_visible is not None else None,
            'stream_window_seconds_per_token':(last-first)/1e9/(tokens-1) if tokens>1 else None,
            'request_output_tokens_per_second':tokens/seconds}
    scope='performance transport/UTF-8/nonempty/usage/channel validity; token throughput is not correct-task goodput' if performance else 'fixed transport/length/ASCII/identifier/obvious-periodic-filler check; no general task-quality or exact event-token timing claim'
    return {**({'performance_valid':correct,'content_anomalies':(['periodic_filler'] if not nonperiodic else [])} if performance else {}),'correct':correct,'reasons':reasons,'fatal_content':bool(transport and (not clean or (not performance and (not identifier or not nonperiodic)) or unexpected or not output_ok)),
        'completion_tokens':tokens if raw.get('complete') is True and type(tokens) is int and tokens>=0 else None,'prompt_tokens':prompt if type(prompt) is int else None,
        'timing':timing,'visible_bytes':len(text.encode('utf-8')),'visible_token_count':None,'reasoning_token_count':None,
        'scope':scope}


def population(tasks,records,request_ids,session_ids,clock_id,*,progress=None):
    if progress is not None and not callable(progress):raise ValueError('verification progress check must be callable')
    if (not tasks or len({t['id'] for t in tasks})!=len(tasks) or set(records)!={t['id'] for t in tasks} or
        set(request_ids)!=set(records) or set(session_ids)!=set(records) or len(set(request_ids.values()))!=len(tasks) or
        not isinstance(clock_id,str) or not clock_id):raise ValueError('fixed population coverage differs')
    rows=[]
    for task in tasks:
        if progress is not None:progress()
        raw=records[task['id']];score=evaluate(task,raw,request_ids[task['id']],session_ids[task['id']])
        rows.append({'task_id':task['id'],'request_id':request_ids[task['id']],'clock_id':clock_id,
            'start_ns':raw['start_ns'],'end_ns':raw['terminal_ns'],**score})
    duration=(max(r['end_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
    if not math.isfinite(duration) or duration<=0:raise ValueError('fixed population makespan invalid')
    tokens=sum(r['completion_tokens'] for r in rows if r['correct']);invalid=[r['completion_tokens'] for r in rows if not r['correct']]
    return {'requests':len(rows),'valid_requests':sum(r['correct'] for r in rows),'valid_tokens':tokens,'duration_seconds':duration,
        'valid_tokens_per_second':tokens/duration,'invalid_partial_tokens':sum(invalid) if all(x is not None for x in invalid) else None,
        'rows':rows,'scope':'all scheduled fixed requests retained; attribution and shared cleanup verification remain separate'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task',type=Path);p.add_argument('raw',type=Path)
    p.add_argument('--request-id');p.add_argument('--session-id');a=p.parse_args()
    result=evaluate(json.loads(a.task.read_text()),json.loads(a.raw.read_text()),a.request_id,a.session_id)
    print(json.dumps(result,sort_keys=True));raise SystemExit(0 if result['correct'] else 1)


if __name__=='__main__':main()
