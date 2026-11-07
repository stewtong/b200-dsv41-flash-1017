"""Deterministic synthetic quality tasks calibrated with the serving tokenizer."""
import hashlib
import json
import re
import workload

KINDS=('identifier','multi_key','arithmetic')
DEPTHS=(.05,.50,.95)
FILLER=' note'


def _digest(seed, label):
    return hashlib.sha256((str(seed)+':'+label).encode()).hexdigest()


def contract(seed,kind,depth):
    if kind=='identifier':
        key='key_'+_digest(seed,'key')[:8]; value='value_'+_digest(seed,'value')[:8]
        return [key+'='+value+';'], 'Return only the value for '+key+'.', value, [key], [depth]
    if kind=='multi_key':
        keys=['key_'+_digest(seed,'key'+str(i))[:8] for i in range(3)]
        vals=['value_'+_digest(seed,'value'+str(i))[:8] for i in range(3)]
        if len(set(keys))!=3 or len(set(vals))!=3: raise ValueError('synthetic key collision')
        clues=[k+'='+v+';' for k,v in zip(keys,vals)]
        question='Return only a JSON array of values in this key order: '+', '.join(keys)+'.'
        return clues,question,vals,keys,list(DEPTHS)
    if kind=='arithmetic':
        a=int(_digest(seed,'a')[:6],16)%900+100; b=int(_digest(seed,'b')[:6],16)%900+100
        key='calc_'+_digest(seed,'calc')[:8]
        return [key+'='+str(a)+'+'+str(b)+';'], 'Return only the integer result for '+key+'.',str(a+b),[key],[depth]
    raise ValueError('unsupported quality task kind')


def build(tokenizer,target,output_tokens,concurrency,repetition,session,cache_state='cold',block_size=None):
    if cache_state not in ('cold','warm_prefix'):raise ValueError('invalid quality cache state')
    warm=cache_state=='warm_prefix'
    if warm and (type(block_size) is not int or block_size<=0):raise ValueError('warm quality requires cache block size')
    seed=workload.seed_for('quality',target,concurrency,repetition,session)
    kind=KINDS[(session+repetition)%len(KINDS)];depth=DEPTHS[(session//len(KINDS)+repetition)%len(DEPTHS)]
    clues,question,expected,keys,depths=contract(seed,kind,depth)
    session_tag='session_'+_digest(seed,'session:'+cache_state+':quality-v2')[:16]+';'
    acknowledgement='ready_'+_digest(seed,'conditioning')[:16]
    prefix_target=((target-256)//block_size)*block_size if warm else target
    if prefix_target<=0:raise ValueError('quality prefix target is too short')
    padding=''
    def context(units,positions):
        result=[session_tag];last=0
        for pos,clue in zip(positions,clues):
            result.append(FILLER*(pos-last));result.append(clue);last=pos
        result.append(FILLER*(units-last))
        return ''.join(result)+padding
    def messages(units,positions=None):
        if positions is None: positions=[round(units*x) for x in depths]
        return [{'role':'system','content':'Follow the user instructions precisely.'},
                {'role':'user','content':context(units,positions)+'\n'+
                    ('Reply only with '+acknowledgement+'.' if warm else question)}]
    def render(units,positions=None):
        return tokenizer.apply_chat_template(messages(units,positions),thinking=False,reasoning_effort=50,tokenize=False)
    def encode(text):return tokenizer.encode(text,add_special_tokens=False)
    calibrated=workload.calibrate(lambda n:render(n),lambda text:len(encode(text)),prefix_target,128 if warm else output_tokens)
    units=calibrated['filler_units']; positions=[round(units*x) for x in depths]
    def measure(candidate,index):
        trial=positions[:];trial[index]=candidate; text=context(units,trial)
        offset=text.find(clues[index])
        if offset<0: raise ValueError('quality target missing')
        total=len(encode(text))
        return len(encode(text[:offset]))/total
    for _ in range(2):
        for index,desired in enumerate(depths):
            low=positions[index-1] if index else 0
            high=positions[index+1] if index+1<len(positions) else units
            while low<high:
                mid=(low+high)//2
                if measure(mid,index)<desired:low=mid+1
                else:high=mid
            options=[max(positions[index-1] if index else 0,low-1),low]
            positions[index]=min(options,key=lambda n:abs(measure(n,index)-desired))
    if warm and len(encode(render(units,positions)))!=prefix_target:
        # Normally one filler unit is one serving token. Also handle tokenizers
        # where a unit spans several tokens, without rounding the prefix target.
        exact=None
        for candidate in range(max(positions[-1],units-8),units+9):
            for spaces in range(17):
                padding=' '*spaces
                if len(encode(render(candidate,positions)))==prefix_target:
                    exact=(candidate,padding);break
            if exact:break
        if exact is None:raise ValueError('exact quality conditioning prefix is unreachable')
        units,padding=exact
    content=context(units,positions);base=messages(units,positions);extra={}
    if warm:
        conditioned=encode(tokenizer.apply_chat_template(base,thinking=False,reasoning_effort=50,tokenize=False))
        if len(conditioned)!=prefix_target:raise ValueError('quality conditioning prefix changed')
        def followup(n):
            return base+[{'role':'assistant','content':acknowledgement},
                         {'role':'user','content':'Additional material:'+FILLER*n+'\n'+question}]
        def full_render(n):return tokenizer.apply_chat_template(followup(n),thinking=False,reasoning_effort=50,tokenize=False)
        measured=workload.calibrate(full_render,lambda text:len(encode(text)),target,output_tokens)
        msgs=followup(measured['filler_units']);text=measured['prompt']
        if encode(text)[:len(conditioned)]!=conditioned:raise ValueError('quality conditioning is not an exact token prefix')
        extra={'conditioning_body':workload.request_body('/model',base,len(conditioned),128,seed,thinking=False,effort=50,fixed_output=False),
               'expected_conditioning_answer':acknowledgement,'prefix_tokens':len(conditioned),
               'target_prefix_tokens':prefix_target,'cache_block_size':block_size,
               'prefix_token_sha256':hashlib.sha256(json.dumps(conditioned,separators=(',',':')).encode()).hexdigest()}
    else:
        msgs=base;text=render(units,positions)
    actual=len(encode(text))
    if abs(actual-target)>target*.0025 or actual+output_tokens>workload.MODEL_LIMIT:
        raise ValueError('quality prompt changed outside token tolerance')
    if any(text.count(clue)!=1 for clue in clues) or any(text.count(k)!=2 for k in keys):
        raise ValueError('quality target key occurrence mismatch')
    context_tokens=len(encode(content))
    observed=[len(encode(content[:content.find(clue)]))/context_tokens for clue in clues]
    if any(abs(actual_depth-desired)>.005 for actual_depth,desired in zip(observed,depths)):
        raise ValueError('quality target depth outside tolerance')
    body=workload.request_body('/model',msgs,actual,output_tokens,seed,thinking=False,effort=50,fixed_output=False)
    return {'id':'s'+str(session),'suite':'quality','session':session,'seed':seed,'body':body,
            'input_tokens':actual,'output_tokens':output_tokens,'fixed_output':False,
            'task_kind':kind,'expected_answer':expected,'target_depths':depths,'observed_depths':observed,
            'session_prefix':session_tag,'cache_designation':cache_state,'prompt_version':'quality-cache-v2',
            **extra,
            'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'qualification':'tokenizer calibration; response usage unverified'}


def score(task,text):
    if not isinstance(text,str) or not text.strip(): return {'correct':False,'extraction_failed':True}
    if task['task_kind']=='multi_key':
        try:value=json.loads(text.strip())
        except (json.JSONDecodeError,TypeError):return {'correct':False,'extraction_failed':True}
        if not isinstance(value,list) or len(value)!=len(task['expected_answer']) or any(not isinstance(x,str) for x in value):
            return {'correct':False,'extraction_failed':True}
        return {'correct':value==task['expected_answer'],'extraction_failed':False}
    if task['task_kind'] in ('identifier','arithmetic'):
        value=text.strip()
        if task['task_kind']=='identifier' and not re.fullmatch(r'value_[0-9a-f]{8}',value):
            return {'correct':False,'extraction_failed':True}
        if task['task_kind']=='arithmetic' and (not value.isascii() or not value.isdigit()):
            return {'correct':False,'extraction_failed':True}
        return {'correct':value==task['expected_answer'],'extraction_failed':False}
    raise ValueError('unknown quality task kind')


def main():
    import argparse
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model');parser.add_argument('target',type=int);parser.add_argument('output',type=int)
    parser.add_argument('concurrency',type=int);parser.add_argument('repetition',type=int);parser.add_argument('session',type=int)
    parser.add_argument('--cache-state',choices=('cold','warm_prefix'),default='cold');parser.add_argument('--block-size',type=int)
    args=parser.parse_args()
    tokenizer=DeepseekV41Tokenizer.from_pretrained(args.model,local_files_only=True)
    task=build(tokenizer,args.target,args.output,args.concurrency,args.repetition,args.session,args.cache_state,args.block_size)
    import verify_quality
    verify_quality.evaluate(task,{'request_id':task['id'],'complete':False},args.repetition,args.concurrency,args.target,tokenizer)
    print(json.dumps(task))

if __name__=='__main__': main()
