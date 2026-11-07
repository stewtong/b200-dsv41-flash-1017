"""Independently verify fixed-grid request, namespace and tokenizer proofs."""
import hashlib
import json
import re


def verify(task,tokenizer=None):
    suite=task.get('suite');outputs={'core':{512,2048},'warm_prefix_panel':{512,2048},'output_sensitivity':{512,8192},'historical_bridge_extra':{512},'anchor':{2048},'anchor_greedy':{2048}}
    n=task.get('target_input_tokens');c=task.get('concurrency');r=task.get('repetition');sid=task.get('session');cache=task.get('cache_designation')
    # Concurrency is supplied by the containing cell, not inferred from a
    # session number. Producers must retain it in each standalone task.
    if (suite not in outputs or task.get('output_tokens') not in outputs[suite] or type(n) is not int or n<=0 or
        type(c) is not int or c not in (1,2,4,8,16,32,64,128) or type(r) is not int or not 0<=r<3 or
        type(sid) is not int or not 0<=sid<c or cache not in ('cold','warm_prefix') or task.get('prompt_version') not in ('fixed-performance-v1','fixed-cache-v1','fixed-cache-v2','fixed-cache-v2-ascii-instruction-v1','fixed-cache-v2-ascii-instruction-v2')):
        raise ValueError('fixed task dimensions or version differ')
    if ((suite=='core' and task['output_tokens']==512 and
         (n!=1000000 or task['prompt_version']!='fixed-performance-v1')) or
        (suite=='warm_prefix_panel' and (cache!='warm_prefix' or n not in (128000,256000,512000,1000000) or
         task['output_tokens']!=(512 if n==1000000 else 2048) or task['prompt_version']!='fixed-performance-v1'))):
        raise ValueError('scoped fixed-output or warm-prefix contract differs')
    matched='anchor' if suite=='anchor_greedy' and task['prompt_version']!='fixed-cache-v1' else suite
    seed=int.from_bytes(hashlib.sha256(json.dumps([matched,n,c,r,sid],separators=(',',':')).encode()).digest()[:8],'big') & 0x7fffffff
    marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
    identity=[task['prompt_version'],matched,n,c,r,sid,cache]
    if task['prompt_version']!='fixed-cache-v1':identity.append(task['output_tokens'])
    if matched!=suite and (task.get('matched_sampling_control')!='anchor' or task.get('reuse_requires')!='fresh engine or independently verified prefix-cache reset'):
        raise ValueError('greedy sampling control lineage missing')
    namespace='session_'+hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()[:20]
    body=task['body'];messages=body.get('messages',[]);warm=cache=='warm_prefix';cap=task['output_tokens'];temperature=0.0 if suite=='anchor_greedy' else 1.0
    allowed={'model','messages','stream','stream_options','temperature','top_p','seed','max_tokens','min_tokens','ignore_eos','chat_template_kwargs'}
    options=body.get('stream_options');kwargs=body.get('chat_template_kwargs')
    if (set(body)!=allowed or body.get('model')!='/model' or task.get('id')!='s'+str(sid) or type(task.get('seed')) is not int or task.get('seed')!=seed or type(body.get('seed')) is not int or body.get('seed')!=seed or
        task.get('expected_answer')!=marker or task.get('session_prefix')!=namespace or body.get('stream') is not True or
        not isinstance(options,dict) or set(options)!={'include_usage'} or options['include_usage'] is not True or
        type(body.get('temperature')) not in (int,float) or body['temperature']!=temperature or type(body.get('top_p')) not in (int,float) or body['top_p']!=1.0 or
        not isinstance(kwargs,dict) or set(kwargs)!={'thinking','reasoning_effort'} or kwargs['thinking'] is not False or type(kwargs['reasoning_effort']) is not int or kwargs['reasoning_effort']!=50 or task.get('fixed_output') is not True or
        body.get('max_tokens')!=cap or body.get('min_tokens')!=cap or body.get('ignore_eos') is not True or
        len(messages)!=(4 if warm else 2) or [m.get('role') for m in messages]!=(['system','user','assistant','user'] if warm else ['system','user']) or
        any(set(m)!={'role','content'} or not isinstance(m['content'],str) for m in messages) or
        messages[0].get('content')!=('Follow the user instructions precisely. ' if task['prompt_version']=='fixed-performance-v1' else 'Follow the user instructions precisely. Reply in English using only ASCII characters. ')+namespace+'.'):
        raise ValueError('fixed task request contract differs')
    instruction='Read the following material. The verification code is '+marker+'. When asked for the code, copy it exactly.\n'
    ending='Begin with the verification code, then write a detailed technical report with explanations, concrete examples and several sections. Continue the report in full prose; do not end after the code.'
    if task['prompt_version']=='fixed-cache-v2-ascii-instruction-v1':
        ending='Begin with the verification code. Write a detailed English report with sections, explanations and examples. Use ASCII only: straight quotes and ordinary hyphens. Keep writing.'
    if task['prompt_version']=='fixed-cache-v2-ascii-instruction-v2':
        ending='Begin with the code exactly, with no label or introduction. Write a detailed English report in sections with examples. ASCII only: straight quotes and hyphens. Keep writing.'
    if task['prompt_version']=='fixed-performance-v1':
        ending='Write a detailed technical report with explanations, concrete examples and several sections. Continue developing the report in full prose.'
    user=messages[1].get('content','');suffix='\n'+('Return only the verification code.' if warm else ending)
    pattern=r'(?: the)* {0,16}' if warm else r'(?:A clear report contains facts and examples\. )*'
    if not user.startswith(instruction) or not user.endswith(suffix) or not re.fullmatch(pattern,user[len(instruction):-len(suffix)]):
        raise ValueError('fixed task material or instruction differs')
    if (type(task.get('input_tokens')) is not int or task['input_tokens']<=0 or abs(task['input_tokens']-n)>n*.0025 or
        task['input_tokens']+cap>1048576 or not re.fullmatch(r'[a-f0-9]{64}',str(task.get('prompt_sha256')))):
        raise ValueError('fixed declared input count or prompt hash invalid')
    if warm:
        condition=task.get('conditioning_body',{});block=task.get('cache_block_size');prefix=task.get('prefix_tokens')
        if (type(block) is not int or block<=0 or type(prefix) is not int or prefix<=0 or prefix!=((n-256)//block)*block or task.get('target_prefix_tokens')!=prefix or
            not re.fullmatch(r'[a-f0-9]{64}',str(task.get('prefix_token_sha256'))) or
            condition.get('messages')!=messages[:2] or task.get('expected_conditioning_answer')!=marker or messages[2].get('content')!=marker or
            not re.fullmatch(r'Additional material:(?: the)*\n'+re.escape(ending),messages[3].get('content','')) or
            set(condition)!=allowed-{'min_tokens','ignore_eos'} or condition.get('max_tokens')!=128 or
            condition.get('stream_options',{}).get('include_usage') is not True or
            condition.get('chat_template_kwargs',{}).get('thinking') is not False or
            type(condition.get('chat_template_kwargs',{}).get('reasoning_effort')) is not int or
            type(condition.get('temperature')) not in (int,float) or type(condition.get('top_p')) not in (int,float) or
            type(condition.get('seed')) is not int or
            any(condition.get(k)!=body[k] for k in ('model','stream','stream_options','temperature','top_p','seed','chat_template_kwargs'))):
            raise ValueError('fixed conditioning contract differs')
    elif any(k in task for k in ('conditioning_body','prefix_tokens','prefix_token_sha256')):
        raise ValueError('cold fixed task contains warm conditioning')
    if tokenizer is None:
        return {'status':'input_contract_only','scope':'tokenizer length, render hash and prefix identity are unverified without the pinned encoder'}
    rendered=tokenizer.apply_chat_template(messages,thinking=False,reasoning_effort=50,tokenize=False)
    ids=tokenizer.encode(rendered,add_special_tokens=False)
    if len(ids)!=task['input_tokens'] or hashlib.sha256(rendered.encode()).hexdigest()!=task['prompt_sha256']:
        raise ValueError('fixed task tokenizer count or prompt hash differs')
    if warm:
        initial=tokenizer.encode(tokenizer.apply_chat_template(messages[:2],thinking=False,reasoning_effort=50,tokenize=False),add_special_tokens=False)
        if len(initial)!=prefix or ids[:prefix]!=initial or hashlib.sha256(json.dumps(initial,separators=(',',':')).encode()).hexdigest()!=task['prefix_token_sha256']:
            raise ValueError('fixed conditioning token-prefix proof differs')
    return {'status':'tokenizer_input_verified','scope':'input contract only; live usage, cache/routing and scored acceptance remain unverified'}
