"""Deterministic request identities and tokenizer-measured prompt calibration."""
import hashlib
import json

MODEL_LIMIT = 1_048_576

def seed_for(suite, context, concurrency, repetition, session):
    # Layout is intentionally absent so arms receive identical prompts and seeds.
    data = json.dumps([suite, context, concurrency, repetition, session], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'big') & 0x7fffffff

def request_body(model, messages, input_tokens, output_tokens, seed, thinking=False, effort=50, fixed_output=True):
    if not model or not messages or not isinstance(thinking, bool): raise ValueError('missing request fields')
    if not isinstance(effort, int) or isinstance(effort, bool) or not 1 <= effort <= 100:
        raise ValueError('effort must be an integer in 1..100')
    if not isinstance(seed, int) or seed < 0 or input_tokens <= 0 or output_tokens <= 0 or input_tokens + output_tokens > MODEL_LIMIT:
        raise ValueError('invalid seed or token budget')
    result = {'model':model, 'messages':messages, 'stream':True,
              'stream_options':{'include_usage':True}, 'temperature':1.0, 'top_p':1.0,
              'seed':seed, 'max_tokens':output_tokens,
              'chat_template_kwargs':{'thinking':thinking, 'reasoning_effort':effort}}
    if fixed_output:
        result['min_tokens'] = output_tokens
        result['ignore_eos'] = True
    return result

def calibrate(render, count_tokens, target, output_tokens, tolerance=0.0025):
    """Bracket filler units, measuring the fully rendered prompt each time.

    render(n) must generate a deterministic complete prompt including system and
    tool overhead; count_tokens must use the pinned tokenizer. Nonmonotonic or
    unreachable targets fail instead of silently changing the requested shape.
    """
    if target <= 0 or output_tokens <= 0 or target + output_tokens > MODEL_LIMIT:
        raise ValueError('invalid target or no output room')
    lower, upper = 0, 1
    observed = {}
    def measure(n):
        if n not in observed:
            observed[n] = count_tokens(render(n))
            if not isinstance(observed[n], int) or observed[n] < 0: raise ValueError('invalid token count')
            ordered = sorted(observed.items())
            if any(a[1] > b[1] for a,b in zip(ordered, ordered[1:])):
                raise ValueError('token count not monotonic')
        return observed[n]
    if measure(0) > target * (1 + tolerance): raise ValueError('prompt overhead exceeds target')
    while measure(upper) < target:
        lower, upper = upper, upper * 2
        if upper > target * 16 + 16384: raise ValueError('unreachable token target')
    # Repeated filler is nearly affine, so refine around an interpolated guess
    # inside the already measured safe bracket. Every guess is still tokenized.
    # Exponential expansion around that guess preserves a valid bracket even
    # for nonlinear counts, then ordinary bisection finds the same crossing.
    if upper-lower>2 and measure(lower)<target<measure(upper):
        estimate=lower+(target-measure(lower))*(upper-lower)//(measure(upper)-measure(lower))
        estimate=max(lower+1,min(upper-1,estimate));step=1
        if measure(estimate)<target:
            lower=estimate
            while True:
                candidate=min(upper,estimate+step)
                if measure(candidate)>=target:
                    upper=candidate;break
                lower=candidate;step*=2
        else:
            upper=estimate
            while True:
                candidate=max(lower,estimate-step)
                if measure(candidate)<target:
                    lower=candidate;break
                upper=candidate;step*=2
    while upper - lower > 1:
        mid = (lower + upper) // 2
        if measure(mid) < target: lower = mid
        else: upper = mid
    best = min((lower, upper), key=lambda n: (abs(measure(n) - target), n))
    length = measure(best)
    if abs(length - target) > target * tolerance or length + output_tokens > MODEL_LIMIT:
        raise ValueError('rendered prompt outside tolerance or sequence limit')
    prompt = render(best)
    return {'prompt':prompt, 'input_tokens':length, 'filler_units':best,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
