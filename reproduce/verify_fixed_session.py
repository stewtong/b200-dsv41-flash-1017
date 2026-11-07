"""Independently audit a fixed request capsule and its warm conditioning."""
import hashlib,json,re
from pathlib import Path
import runtime,verify_fixed_capture,verify_wire


def conditioning_score(task,raw,request_id,session_id):
    begin=raw.get('start_ns');end=raw.get('terminal_ns')
    if (raw.get('request_id')!=request_id or raw.get('session_id')!=session_id or
        type(begin) is not int or type(end) is not int or not 0<=begin<end):raise ValueError('conditioning identity or clocks differ')
    data=json.dumps(task['conditioning_body'],allow_nan=False,sort_keys=True,separators=(',',':')).encode()
    binding=(raw.get('request_serialization')=='canonical-json-ensure-ascii/v1' and
        raw.get('request_body_sha256')==hashlib.sha256(data).hexdigest() and raw.get('request_payload_bytes')==len(data) and
        raw.get('request_headers')=={'Content-Type':'application/json','X-Request-Id':request_id,'X-Session-Id':session_id})
    if raw.get('complete'):verify_wire.verify_complete(raw)
    else:verify_wire.retained_bytes(raw)
    text='';usage=[];finishes=[];unexpected=False
    for event in raw.get('events',[]):
        if event.get('kind')!='event':continue
        value=event['data']
        if value.get('usage') is not None:usage.append(value['usage'])
        for choice in value.get('choices',[]):
            delta=choice.get('delta') or {}
            if choice.get('index',0)!=0 or delta.get('role') not in (None,'assistant') or any(delta.get(k) for k in ('tool_calls','reasoning','reasoning_content')):unexpected=True
            piece=delta.get('content')
            if piece is not None and not isinstance(piece,str):raise ValueError('conditioning content is not text')
            text+=piece or ''
            if choice.get('finish_reason') is not None:finishes.append(choice['finish_reason'])
    u=usage[0] if len(usage)==1 and isinstance(usage[0],dict) else {};prompt=u.get('prompt_tokens');tokens=u.get('completion_tokens')
    performance=task.get('prompt_version')=='fixed-performance-v1'
    # Only the request's encoded prefix is guaranteed reusable. The synthetic
    # assistant turn is fixed input, not a correctness claim about this reply.
    reply_ok=(bool(text) and len(finishes)==1 and finishes[0] in ('stop','length')) if performance else (finishes==['stop'] and text.strip()==task['expected_conditioning_answer'])
    correct=bool(binding and raw.get('complete') is True and raw.get('http_status')==200 and
        raw.get('cancel_requested') is not True and type(prompt) is int and abs(prompt-task['prefix_tokens'])<=task['prefix_tokens']*.0025 and
        type(tokens) is int and 0<tokens<=128 and reply_ok and not unexpected)
    return {'correct':correct,'fatal_content':not correct,'completion_tokens':tokens if type(tokens) is int else None,
            'scope':('unscored performance prefix conditioning transport/usage; reply semantics excluded, exact prefix and achieved cache reuse checked separately' if performance else 'unscored conditioning acknowledgement and exact transmitted request; cache hit remains unverified')}


def verify(directory,*,progress=None):
    if progress is not None and not callable(progress):raise ValueError('verification progress check must be callable')
    if progress is not None:progress()
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda name:json.loads((root/name).read_text())
    contract=read('fixed-session-contract.json');result=read('fixed-session-result.json')
    inputs=json.loads((root/'measured/fixed-inputs.json').read_text());tasks=inputs['tasks'];namespace=contract.get('request_namespace')
    cache={t.get('cache_designation') for t in tasks}
    if (contract.get('schema')!='fixed-session-capture/v1' or cache!={contract.get('cache')} or
        not isinstance(namespace,str) or not re.fullmatch(r'fx[a-f0-9]{24}',namespace) or
        contract.get('scheduled_measured_requests')!=len(tasks) or
        inputs['request_namespace']!=namespace or inputs['session_ids']!=contract.get('session_ids') or
        inputs['clock_id']!=contract.get('clock_id')):raise ValueError('fixed session contract differs from measured inputs')
    measured=verify_fixed_capture.verify(root/'measured',progress=progress)
    process_contract=contract.get('client_processes')
    clients=root/'measured/clients'
    if process_contract is not None:
        if (not isinstance(process_contract,dict) or set(process_contract)!={'workers','cpus'} or
                type(process_contract['workers']) is not int or process_contract['workers'] not in (1,2) or
                not isinstance(process_contract['cpus'],list) or len(set(process_contract['cpus']))!=len(process_contract['cpus']) or
                len(process_contract['cpus'])<process_contract['workers'] or
                any(type(i) is not int or i<0 for i in process_contract['cpus']) or not clients.is_dir()):
            raise ValueError('fixed session subprocess contract differs')
        observed=json.loads((clients/'model-client-contract.json').read_text())
        if any(observed[k]!=process_contract[k] for k in ('workers','cpus')):
            raise ValueError('fixed session subprocess mode or CPU partition differs')
    elif clients.exists():raise ValueError('undeclared fixed session subprocess mode')
    expected={'fixed-session-contract.json','fixed-session-result.json','SHA256SUMS','measured'}
    if cache=={'warm_prefix'}:expected.add('conditioning')
    if {p.name for p in root.iterdir()}!=expected:raise ValueError('fixed session file coverage differs')
    measured_rows=[json.loads((root/'measured'/(t['id']+'.json')).read_text()) for t in tasks]
    condition_tokens=0;condition_seconds=0
    if cache=={'warm_prefix'}:
        cond=root/'conditioning';runtime.verify_evidence_manifest(cond)
        if {p.name for p in cond.iterdir()}!={t['id']+'.json' for t in tasks}|{'summary.json','SHA256SUMS'}:raise ValueError('conditioning population differs')
        rows=[]
        for task in tasks:
            if progress is not None:progress()
            row=json.loads((cond/(task['id']+'.json')).read_text());raw=row['raw'];ident=namespace+'c'+str(task['session'])
            score=conditioning_score(task,raw,ident,contract['session_ids'][task['id']])
            if (not score['correct'] or row.get('task_id')!=task['id'] or row.get('request_id')!=ident or
                row.get('clock_id')!=contract['clock_id'] or row.get('quality_score')!=score or row.get('valid') is not True or
                row.get('start_ns')!=raw['start_ns'] or row.get('end_ns')!=raw['terminal_ns'] or row.get('completion_tokens')!=score['completion_tokens']):
                raise ValueError('conditioning raw outcome differs')
            condition_tokens+=score['completion_tokens'];rows.append(row)
        condition_seconds=(max(r['end_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
        expected_summary={'duration_seconds':condition_seconds,'valid_tokens':condition_tokens,'invalid_partial_tokens':0,
            'valid_tokens_per_second':condition_tokens/condition_seconds,'valid_requests':len(tasks),'requests':len(tasks)}
        if json.loads((cond/'summary.json').read_text())!={'status':'requests_passed_cleanup_pending','summary':expected_summary}:raise ValueError('conditioning denominator differs')
        if max(r['end_ns'] for r in rows)>min(r['start_ns'] for r in measured_rows):raise ValueError('measurement overlaps conditioning')
    elif cache!={'cold'}:raise ValueError('unknown fixed cache designation')
    if result!={'status':'captured_cleanup_pending','measured_requests':len(tasks),'conditioning_requests':len(tasks) if cache=={'warm_prefix'} else 0,
        'scope':'conditioning excluded from measured makespan and token-rate denominator; achieved cache hits and routing remain separate'}:
        raise ValueError('fixed session result differs')
    runtime.verify_evidence_manifest(root)
    return {'status':'raw_fixed_session_verified','requests':measured['requests'],'valid_requests':measured['valid_requests'],
        'conditioning_requests':len(tasks) if cache=={'warm_prefix'} else 0,'conditioning_tokens':condition_tokens,'conditioning_seconds':condition_seconds,
        'scope':'raw request sequencing and distinct wire IDs; excludes tokenizer provenance, achieved cache/rank evidence, frozen profile, telemetry and shared cleanup acceptance'}
