"""Prepare fixed-output grid inputs with isolated cold and warm namespaces.

Tokenizer proofs qualify inputs only. Live usage, cache hits, placement and
profile qualification remain execution requirements.
"""
import argparse
import hashlib
import json
import workload

SUITES={'core':{512,2048},'warm_prefix_panel':{512,2048},'output_sensitivity':{512,8192},
        'historical_bridge_extra':{512},'anchor':{2048},'anchor_greedy':{2048}}
VERSION='fixed-cache-v2-ascii-instruction-v2'
PERFORMANCE_VERSION='fixed-performance-v1'
PERFORMANCE_ENDING='Write a detailed technical report with explanations, concrete examples and several sections. Continue developing the report in full prose.'
EXECUTION_VERSIONS={PERFORMANCE_VERSION,'fixed-cache-v2','fixed-cache-v2-ascii-instruction-v1',VERSION}
FILLER='A clear report contains facts and examples. '
ENDING='Begin with the verification code, then write a detailed technical report with explanations, concrete examples and several sections. Continue the report in full prose; do not end after the code.'
ASCII_ENDING_V1='Begin with the verification code. Write a detailed English report with sections, explanations and examples. Use ASCII only: straight quotes and ordinary hyphens. Keep writing.'
ASCII_ENDING='Begin with the code exactly, with no label or introduction. Write a detailed English report in sections with examples. ASCII only: straight quotes and hyphens. Keep writing.'


def build(tokenizer,model,target,output_tokens,suite,concurrency,repetition,session,cache_state='cold',block_size=None,*,prompt_version=VERSION):
    if (suite not in SUITES or output_tokens not in SUITES[suite] or type(target) is not int or target<=0 or
        type(concurrency) is not int or concurrency not in (1,2,4,8,16,32,64,128) or
        type(repetition) is not int or not 0<=repetition<3 or type(session) is not int or not 0<=session<concurrency or
        cache_state not in ('cold','warm_prefix') or prompt_version not in EXECUTION_VERSIONS):
        raise ValueError('invalid fixed-grid task dimensions')
    if ((suite=='core' and output_tokens==512 and
         (target!=1000000 or prompt_version!=PERFORMANCE_VERSION)) or
        (suite=='warm_prefix_panel' and (cache_state!='warm_prefix' or
         target not in (128000,256000,512000,1000000) or
         output_tokens!=(512 if target==1000000 else 2048) or prompt_version!=PERFORMANCE_VERSION))):
        raise ValueError('scoped fixed-output or warm-prefix dimensions differ')
    warm=cache_state=='warm_prefix'
    if warm and (type(block_size) is not int or block_size<=0):raise ValueError('warm task requires measured cache block size')
    sampling_control='anchor' if suite=='anchor_greedy' else suite
    seed=workload.seed_for(sampling_control,target,concurrency,repetition,session)
    marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
    namespace='session_'+hashlib.sha256(json.dumps([prompt_version,sampling_control,target,concurrency,repetition,session,cache_state,output_tokens],separators=(',',':')).encode()).hexdigest()[:20]
    ending=PERFORMANCE_ENDING if prompt_version==PERFORMANCE_VERSION else ASCII_ENDING if prompt_version==VERSION else ASCII_ENDING_V1 if prompt_version=='fixed-cache-v2-ascii-instruction-v1' else ENDING
    system=('Follow the user instructions precisely. ' if prompt_version==PERFORMANCE_VERSION else 'Follow the user instructions precisely. Reply in English using only ASCII characters. ')+namespace+'.'
    instruction='Read the following material. The verification code is '+marker+'. When asked for the code, copy it exactly.\n'
    def render(messages):return tokenizer.apply_chat_template(messages,thinking=False,reasoning_effort=50,tokenize=False)
    def encode(text):return tokenizer.encode(text,add_special_tokens=False)
    padding=''
    def initial(n):return [{'role':'system','content':system},{'role':'user','content':instruction+((' the'*n)+padding if warm else FILLER*n)+'\n'+('Return only the verification code.' if warm else ending)}]
    extra={'matched_sampling_control':'anchor','reuse_requires':'fresh engine or independently verified prefix-cache reset'} if suite=='anchor_greedy' else {}
    if warm:
        prefix_target=((target-256)//block_size)*block_size
        if prefix_target<=0:raise ValueError('warm prefix has no input room')
        calibrated=workload.calibrate(lambda n:render(initial(n)),lambda text:len(encode(text)),prefix_target,128)
        units=calibrated['filler_units']
        if calibrated['input_tokens']!=prefix_target:
            exact=None
            for candidate in range(max(0,units-8),units+9):
                for spaces in range(17):
                    padding=' '*spaces
                    if len(encode(render(initial(candidate))))==prefix_target:
                        exact=candidate;break
                if exact is not None:break
            if exact is None:raise ValueError('fixed conditioning prefix target is unreachable')
            units=exact
        base=initial(units);conditioned=encode(render(base))
        def messages(n):return base+[{'role':'assistant','content':marker},{'role':'user','content':'Additional material:'+(' the'*n)+'\n'+ending}]
        measured=workload.calibrate(lambda n:render(messages(n)),lambda text:len(encode(text)),target,output_tokens)
        actual_messages=messages(measured['filler_units']);tokens=encode(measured['prompt'])
        if tokens[:len(conditioned)]!=conditioned:raise ValueError('fixed conditioning is not an exact token prefix')
        conditioning=workload.request_body(model,base,len(conditioned),128,seed,thinking=False,effort=50,fixed_output=False)
        extra.update({'conditioning_body':conditioning,'expected_conditioning_answer':marker,'prefix_tokens':len(conditioned),
            'target_prefix_tokens':prefix_target,'cache_block_size':block_size,
            'prefix_token_sha256':hashlib.sha256(json.dumps(conditioned,separators=(',',':')).encode()).hexdigest()})
    else:
        measured=workload.calibrate(lambda n:render(initial(n)),lambda text:len(encode(text)),target,output_tokens)
        actual_messages=initial(measured['filler_units'])
    body=workload.request_body(model,actual_messages,measured['input_tokens'],output_tokens,seed,thinking=False,effort=50,fixed_output=True)
    if suite=='anchor_greedy':
        body['temperature']=0.0
        if warm:extra['conditioning_body']['temperature']=0.0
    return {'id':'s'+str(session),'suite':suite,'session':session,'concurrency':concurrency,'seed':seed,'repetition':repetition,
        'body':body,'input_tokens':measured['input_tokens'],'target_input_tokens':target,'output_tokens':output_tokens,
        'fixed_output':True,'expected_answer':marker,'prompt_sha256':measured['prompt_sha256'],
        'session_prefix':namespace,'cache_designation':cache_state,'prompt_version':prompt_version,**extra,
        'qualification':'tokenizer-prepared input only; live usage/cache/routing unverified'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('model');p.add_argument('target',type=int);p.add_argument('output',type=int)
    p.add_argument('suite',choices=sorted(SUITES));p.add_argument('concurrency',type=int);p.add_argument('repetition',type=int);p.add_argument('session',type=int)
    p.add_argument('--cache-state',choices=('cold','warm_prefix'),default='cold');p.add_argument('--block-size',type=int)
    p.add_argument('--prompt-version',choices=sorted(EXECUTION_VERSIONS),default=VERSION)
    p.add_argument('--batch',action='store_true',help='Build all sessions with one tokenizer; session must be zero, at most8million total input tokens.')
    a=p.parse_args()
    if a.batch and (a.session!=0 or not 1<=a.concurrency<=128 or not 0<a.target*a.concurrency<=8_000_000):
        p.error('batch requires session zero and at most8million total input tokens')
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    tokenizer=DeepseekV41Tokenizer.from_pretrained(a.model,local_files_only=True)
    import verify_fixed_task
    tasks=[]
    for index in range(a.concurrency) if a.batch else [a.session]:
        task=build(tokenizer,'/model',a.target,a.output,a.suite,a.concurrency,a.repetition,index,a.cache_state,a.block_size,prompt_version=a.prompt_version)
        verify_fixed_task.verify(task,tokenizer);tasks.append(task)
    print(json.dumps(tasks if a.batch else tasks[0],sort_keys=True))


if __name__=='__main__':main()
