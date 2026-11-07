"""Independently score one natural-stop synthetic quality response."""
import hashlib
import json
import math
import re
import verify_wire


def verify_request_binding(raw, body, request_id, session_id):
    data=json.dumps(body,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
    if (raw.get('request_id')!=request_id or raw.get('session_id')!=session_id or
        raw.get('request_serialization')!='canonical-json-ensure-ascii/v1' or
        raw.get('request_body_sha256')!=hashlib.sha256(data).hexdigest() or
        raw.get('request_payload_bytes')!=len(data) or
        raw.get('request_headers')!={'Content-Type':'application/json','X-Request-Id':request_id,'X-Session-Id':session_id}):
        raise ValueError('quality transmitted request binding differs')


def evaluate(task, raw, repetition, concurrency, target, tokenizer=None, require_wire=False,
             request_id=None, session_id=None):
    if require_wire or 'wire_chunks' in raw:
        if raw.get('complete'):verify_wire.verify_complete(raw)
        else:verify_wire.retained_bytes(raw)
    session=task['session']
    if (type(session) is not int or type(concurrency) is not int or not 0<=session<concurrency or
        type(repetition) is not int or repetition<0 or type(target) is not int or target<=0):
        raise ValueError('quality task dimensions invalid')
    seed=int.from_bytes(hashlib.sha256(json.dumps(['quality',target,concurrency,repetition,session],separators=(',',':')).encode()).digest()[:8],'big') & 0x7fffffff
    if task.get('suite')!='quality' or task.get('seed')!=seed or task.get('id')!='s'+str(session):raise ValueError('quality task identity mismatch')
    kind=('identifier','multi_key','arithmetic')[(session+repetition)%3]
    if task.get('task_kind')!=kind:raise ValueError('quality task kind mismatch')
    def digest(label):return hashlib.sha256((str(seed)+':'+label).encode()).hexdigest()
    if kind=='identifier':
        expected='value_'+digest('value')[:8]; keys=['key_'+digest('key')[:8]]; clues=[keys[0]+'='+expected+';']; question='Return only the value for '+keys[0]+'.'
    elif kind=='multi_key':
        keys=['key_'+digest('key'+str(i))[:8] for i in range(3)]
        expected=['value_'+digest('value'+str(i))[:8] for i in range(3)]
        clues=[k+'='+v+';' for k,v in zip(keys,expected)]
        question='Return only a JSON array of values in this key order: '+', '.join(keys)+'.'
    else:
        keys=['calc_'+digest('calc')[:8]]
        a=int(digest('a')[:6],16)%900+100; b=int(digest('b')[:6],16)%900+100
        expected=str(a+b); clues=[keys[0]+'='+str(a)+'+'+str(b)+';'];question='Return only the integer result for '+keys[0]+'.'
    version=task.get('prompt_version');cache=task.get('cache_designation','cold');warm=cache=='warm_prefix'
    if version not in (None,'quality-cache-v2') or cache not in ('cold','warm_prefix') or (warm and version is None):
        raise ValueError('quality prompt version or cache state invalid')
    body=task['body']; messages=body.get('messages',[])
    if (task.get('expected_answer')!=expected or task.get('fixed_output') is not False or body.get('seed')!=seed or
        body.get('model')!='/model' or body.get('stream') is not True or body.get('stream_options')!={'include_usage':True} or
        body.get('temperature')!=1.0 or body.get('top_p')!=1.0 or body.get('chat_template_kwargs')!={'thinking':False,'reasoning_effort':50} or
        'min_tokens' in body or body.get('ignore_eos') or body.get('max_tokens')!=task.get('output_tokens') or
        type(task.get('input_tokens')) is not int or abs(task['input_tokens']-target)>target*.0025 or
        type(task.get('output_tokens')) is not int or task['output_tokens']<=0 or len(messages)!=(4 if warm else 2) or
        [m.get('role') for m in messages]!=(['system','user','assistant','user'] if warm else ['system','user'])):
        raise ValueError('quality request contract mismatch')
    user=messages[1].get('content','')
    tag='session_'+digest('session:'+cache+':quality-v2' if version else 'session')[:16]+';'
    acknowledgement='ready_'+digest('conditioning')[:16]
    combined='\n'.join(m.get('content','') for m in messages)
    if (not isinstance(user,str) or not user.startswith(tag) or task.get('session_prefix')!=tag or
        messages[0].get('content')!='Follow the user instructions precisely.' or not messages[-1].get('content','').endswith('\n'+question) or
        any(combined.count(key)!=2 for key in keys) or any(combined.count(clue)!=1 for clue in clues)):
        raise ValueError('quality target occurrence mismatch')
    if warm:
        condition=task.get('conditioning_body',{});block=task.get('cache_block_size');prefix=task.get('prefix_tokens')
        if (type(block) is not int or block<=0 or type(prefix) is not int or prefix<=0 or
            prefix!=((target-256)//block)*block or task.get('target_prefix_tokens')!=prefix or
            not re.fullmatch(r'[0-9a-f]{64}',str(task.get('prefix_token_sha256'))) or
            task.get('expected_conditioning_answer')!=acknowledgement or messages[2].get('content')!=acknowledgement or
            not user.endswith('\nReply only with '+acknowledgement+'.') or condition.get('messages')!=messages[:2] or
            condition.get('model')!='/model' or condition.get('seed')!=seed or condition.get('max_tokens')!=128 or
            condition.get('stream') is not True or condition.get('stream_options')!={'include_usage':True} or
            condition.get('temperature')!=1.0 or condition.get('top_p')!=1.0 or
            condition.get('chat_template_kwargs')!={'thinking':False,'reasoning_effort':50} or
            'min_tokens' in condition or condition.get('ignore_eos') or
            set(condition)-{'model','messages','stream','stream_options','temperature','top_p','seed','max_tokens','chat_template_kwargs'}):
            raise ValueError('warm quality conditioning contract mismatch')
    targets=task.get('target_depths'); observed=task.get('observed_depths')
    if (not isinstance(targets,list) or not isinstance(observed,list) or len(targets)!=len(keys) or len(observed)!=len(keys) or
        any(type(x) not in (int,float) or type(y) not in (int,float) or not math.isfinite(x) or not math.isfinite(y) or not 0<=y<=1 or abs(x-y)>.005 for x,y in zip(targets,observed))):
        raise ValueError('quality depth evidence invalid')
    expected_depths=[.05,.5,.95] if kind=='multi_key' else [(.05,.5,.95)[(session//3+repetition)%3]]
    if targets!=expected_depths:raise ValueError('quality target depth contract mismatch')
    depth_verification='declared_only'
    if tokenizer is not None:
        rendered=tokenizer.apply_chat_template(messages,thinking=False,reasoning_effort=50,tokenize=False)
        tokens=tokenizer.encode(rendered,add_special_tokens=False)
        if len(tokens)!=task['input_tokens'] or hashlib.sha256(rendered.encode()).hexdigest()!=task.get('prompt_sha256'):
            raise ValueError('quality rendered prompt mismatch')
        if warm:
            conditioned=tokenizer.encode(tokenizer.apply_chat_template(messages[:2],thinking=False,reasoning_effort=50,tokenize=False),add_special_tokens=False)
            if (len(conditioned)!=prefix or tokens[:prefix]!=conditioned or
                hashlib.sha256(json.dumps(conditioned,separators=(',',':')).encode()).hexdigest()!=task['prefix_token_sha256']):
                raise ValueError('warm quality rendered token prefix mismatch')
        context=user.rsplit('\n',1)[0];context_tokens=len(tokenizer.encode(context,add_special_tokens=False))
        if context_tokens<=0:raise ValueError('empty quality context')
        recalculated=[len(tokenizer.encode(context[:context.find(clue)],add_special_tokens=False))/context_tokens for clue in clues]
        if any(abs(a-b)>1e-12 for a,b in zip(recalculated,observed)):
            raise ValueError('quality observed depth mismatch')
        depth_verification='provided_tokenizer_recomputed'
    request_id=task['id'] if request_id is None else request_id
    session_id=task['id'] if session_id is None else session_id
    if raw.get('request_id')!=request_id:raise ValueError('quality response identity mismatch')
    if require_wire:verify_request_binding(raw,body,request_id,session_id)
    outcome={'correct':False,'extraction_failed':False,'reason':None,'depth_verification':depth_verification}
    if not raw.get('complete') or raw.get('http_status')!=200:
        outcome['reason']='request_failed';return outcome
    events=raw.get('events',[])
    if not events or events[-1].get('kind')!='done' or sum(item.get('kind')=='done' for item in events)!=1:
        outcome['reason']='missing_terminal_stream';return outcome
    usages=[];finishes=[];visible=''
    for item in raw.get('events',[]):
        if item.get('kind')!='event':continue
        event=item['data']
        if event.get('usage') is not None:usages.append(event['usage'])
        for choice in event.get('choices',[]):
            visible+=(choice.get('delta') or {}).get('content') or ''
            if choice.get('finish_reason') is not None:finishes.append(choice['finish_reason'])
    if len(usages)!=1 or len(finishes)!=1:
        outcome['reason']='missing_terminal_usage';return outcome
    prompt=usages[0].get('prompt_tokens');completion=usages[0].get('completion_tokens')
    if (type(prompt) is not int or type(completion) is not int or completion<=0 or completion>task['output_tokens'] or
        abs(prompt-task['input_tokens'])>task['input_tokens']*.0025):
        outcome['reason']='usage_mismatch';return outcome
    if finishes[0]!='stop':outcome['reason']='budget_exhausted' if finishes[0]=='length' else 'invalid_finish';return outcome
    text=visible.strip()
    if not text:outcome.update(extraction_failed=True,reason='missing_visible_answer');return outcome
    if kind=='multi_key':
        try:value=json.loads(text)
        except (json.JSONDecodeError,TypeError):outcome.update(extraction_failed=True,reason='invalid_json');return outcome
        if not isinstance(value,list) or len(value)!=3 or any(not isinstance(x,str) for x in value):
            outcome.update(extraction_failed=True,reason='invalid_answer_shape');return outcome
    elif kind=='identifier':
        if not re.fullmatch(r'value_[0-9a-f]{8}',text):outcome.update(extraction_failed=True,reason='invalid_answer_shape');return outcome
        value=text
    else:
        if not text.isascii() or not text.isdigit():outcome.update(extraction_failed=True,reason='invalid_answer_shape');return outcome
        value=text
    outcome['correct']=value==expected
    if not outcome['correct']:outcome['reason']='wrong_answer'
    return outcome
