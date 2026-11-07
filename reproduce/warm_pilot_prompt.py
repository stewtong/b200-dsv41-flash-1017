"""Versioned warm pilot input with an explicit long-form measured request."""
import argparse
import hashlib
import json
import workload

VERSION="pilot-warm-report-v2"
INSTRUCTION='Begin with the verification code exactly, without a label or introduction. Then write a detailed technical report in English about reliable distributed model serving, with explanations, concrete examples and several sections. Continue the report in full prose; do not end after the code.'

def build(tokenizer, model, target, output_tokens, concurrency, repetition, session, block_size, answer):
    """Prepare conditioning and follow-up with an exactly verified token prefix."""
    if block_size <= 0 or not answer:
        raise ValueError('block size and exact conditioning response required')
    desired = ((target - 256) // block_size) * block_size
    if desired <= 0: raise ValueError('warm prefix target is too short')
    seed = workload.seed_for('pilot_warm', target, concurrency, repetition, session)
    marker = hashlib.sha256(str(seed).encode()).hexdigest()[:16]
    if answer != marker: raise ValueError('conditioning response must exactly match the seeded answer')
    def initial(n):
        return [{'role':'system','content':'Follow the user instructions precisely.'},
                {'role':'user','content':'Remember this code: '+marker+'.' + ' the'*n + '\nReturn the code.'}]
    def render(messages):
        return tokenizer.apply_chat_template(messages, thinking=False, reasoning_effort=50, tokenize=False)
    def encode(text): return tokenizer.encode(text, add_special_tokens=False)
    prefix = workload.calibrate(lambda n:render(initial(n)), lambda t:len(encode(t)), desired, 1, tolerance=0)
    base = initial(prefix['filler_units'])
    conditioned = encode(render(base))
    def followup(n):
        return base + [{'role':'assistant','content':answer}, {'role':'user','content':'More context:'+' the'*n+'\n'+INSTRUCTION}]
    measured = workload.calibrate(lambda n:render(followup(n)), lambda t:len(encode(t)), target, output_tokens)
    measured_messages = followup(measured['filler_units'])
    measured_ids = encode(render(measured_messages))
    if measured_ids[:len(conditioned)] != conditioned:
        raise ValueError('conditioned prompt is not an exact measured token prefix')
    conditioning = workload.request_body(model, base, len(conditioned), 128, seed, thinking=False, effort=50, fixed_output=False)
    request = workload.request_body(model, measured_messages, measured['input_tokens'], output_tokens, seed,
                                    thinking=False, effort=50, fixed_output=True)
    return {'id':'s'+str(session), 'suite':'pilot_warm', 'prompt_version':VERSION, 'fixed_report_instruction':INSTRUCTION, 'fixed_output':True, 'output_tokens':output_tokens,
            'conditioning_body':conditioning, 'body':request, 'input_tokens':measured['input_tokens'],
            'prefix_tokens':len(conditioned), 'target_prefix_tokens':desired, 'cache_block_size':block_size,
            'prefix_token_sha256':hashlib.sha256(json.dumps(conditioned, separators=(',', ':')).encode()).hexdigest(),
            'prompt_sha256':measured['prompt_sha256'], 'expected_conditioning_answer':marker, 'expected_answer':marker,
            'seed':seed, 'session':session, 'cache_designation':'warm_prefix',
            'qualification':'token-prefix identity only; cache-hit and placement evidence required'}

def main():
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model');parser.add_argument('target',type=int);parser.add_argument('output',type=int)
    parser.add_argument('suite');parser.add_argument('concurrency',type=int);parser.add_argument('repetition',type=int);parser.add_argument('session',type=int)
    parser.add_argument('--block-size',type=int,required=True)
    a=parser.parse_args()
    if a.suite!='pilot_warm':raise ValueError('versioned generator is only for warm pilot inputs')
    tokenizer=DeepseekV41Tokenizer.from_pretrained(a.model,local_files_only=True)
    seed=workload.seed_for(a.suite,a.target,a.concurrency,a.repetition,a.session)
    answer=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
    print(json.dumps(build(tokenizer,a.model,a.target,a.output,a.concurrency,a.repetition,a.session,a.block_size,answer)))

if __name__=='__main__':main()
