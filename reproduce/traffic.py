"""Seeded offered traffic and Section 5 SLO adjudication from retained requests.

This module does not dispatch inference. Missing dispatches remain offered work.
"""
import math
import random

CONTEXTS=(8000,128000,512000,1000000)


def offered_schedule(rate, seed, measured_seconds=300, ramp_seconds=60, gamma_shape=1.0):
    if (not math.isfinite(rate) or rate<=0 or not math.isfinite(gamma_shape) or gamma_shape<=0 or
        type(seed) is not int or measured_seconds<=0 or ramp_seconds<0):
        raise ValueError('invalid offered schedule')
    arrivals=random.Random(str(seed)+':arrivals');work=random.Random(str(seed)+':work')
    result=[];at=0.0
    while True:
        at+=arrivals.gammavariate(gamma_shape,1/(rate*gamma_shape))
        if at>=ramp_seconds+measured_seconds:break
        u=work.random();context=8000 if u<.5 else 128000 if u<.75 else 512000 if u<.9 else 1000000
        u=work.random();output=512 if u<.6 else 2048 if u<.9 else 8192
        i=len(result)
        result.append({'id':f'traffic-{seed}-{i}','scheduled_seconds':at,
                       'phase':'ramp' if at<ramp_seconds else 'measured',
                       'input_tokens':context,'output_tokens':output,'designated_cache':'cold',
                       'session':f'traffic-session-{seed}-{i}'})
    return result


def queue_evidence(samples, start, end):
    """One sample per second; deduplicate IDs across client/router/engine layers."""
    if end-start<120:raise ValueError('queue window must span at least120 seconds')
    selected=sorted((s for s in samples if start<=s['seconds']<end),key=lambda s:s['seconds'])
    expected=set(range(math.ceil(end-start)))
    covered=set();points=[]
    for s in selected:
        t=s['seconds']-start;slot=int(t)
        if slot in covered:raise ValueError('duplicate queue second')
        covered.add(slot)
        if set(s['layers'])!={'client','router','engine'}:raise ValueError('queue layer missing')
        ids=set()
        for values in s['layers'].values():
            if len(set(values))!=len(values):raise ValueError('duplicate queue identity in one layer')
            ids.update(values)
        points.append((t,len(ids)))
    if covered!=expected:return {'verdict':'inconclusive','reason':'missing 1 Hz queue coverage'}
    first=[n for t,n in points if t<60];last=[n for t,n in points if t>=end-start-60]
    mean_t=sum(t for t,n in points)/len(points);mean_n=sum(n for t,n in points)/len(points)
    denominator=sum((t-mean_t)**2 for t,n in points)
    increase=sum((t-mean_t)*(n-mean_n) for t,n in points)/denominator*(end-start)
    difference=sum(last)/len(last)-sum(first)/len(first)
    return {'verdict':'passed' if difference<=1 and increase<=1 else 'failed',
            'final_minus_initial_mean':difference,'fitted_increase':increase,'samples':len(points)}


def ttft_limit(context,cache):
    if cache=='cold':return 10 if context<=128000 else 30 if context<=512000 else 120
    if cache!='warm_prefix':raise ValueError('unknown designated cache state')
    if context<=128000:return 3
    try:return {256000:5,400000:8,512000:10,768000:15,1000000:20}[context]
    except KeyError:raise ValueError('unknown warm context')


def p95(values):
    if len(values)<20:raise ValueError('p95 requires at least20 independent requests')
    return sorted(values)[math.ceil(.95*len(values))-1]


def assess_rate(repetitions, required_buckets):
    """No pooling away failed repetitions; latency populations use valid cohort tasks.

    A repetition contains scheduled request rows (including missing dispatch), exact
    window and drain bounds, and queue samples. terminal_seconds includes ramp and
    drain; valid means the external semantic/transport oracle passed.
    """
    if not repetitions or not required_buckets:raise ValueError('empty SLO population')
    required=set(tuple(x) for x in required_buckets)
    pooled={b:[] for b in required};reports=[];failed=[];uncertain=[]
    for rep in repetitions:
        start,end,drain=rep['start'],rep['end'],rep['drain_end']
        if not 0<=start<end<=drain:raise ValueError('invalid measurement bounds')
        rows=rep['requests'];ids=[r['id'] for r in rows]
        if len(set(ids))!=len(ids):raise ValueError('duplicate offered identity')
        cohort=[r for r in rows if start<=r['scheduled_seconds']<end]
        def valid(r):
            return (r.get('valid') is True and isinstance(r.get('terminal_seconds'),(float,int)) and
                    math.isfinite(r['terminal_seconds']) and r['scheduled_seconds']<=r['terminal_seconds']<=min(r['deadline_seconds'],drain))
        complete=[r for r in cohort if valid(r)]
        terminal=[r for r in rows if valid(r) and start<=r['terminal_seconds']<end]
        fraction=len(complete)/len(cohort) if cohort else None
        ratio=len(terminal)/len(cohort) if cohort else None
        queue=queue_evidence(rep['queue_samples'],start,end)
        if fraction is None:uncertain.append('zero offered cohort')
        elif fraction<.99:failed.append('cohort valid completion below99%')
        if ratio is not None and ratio<.9:failed.append('in-window valid completion below90% offered rate')
        if queue['verdict']=='failed':failed.append('growing queue')
        elif queue['verdict']!='passed':uncertain.append('missing queue coverage')
        buckets={b:[] for b in required}
        for r in complete:
            bucket=(r['input_tokens'],r['designated_cache'])
            if bucket not in buckets:raise ValueError('offered bucket outside frozen workload')
            first=r.get('first_visible_seconds');decode=r.get('decode_seconds_per_token')
            if not isinstance(first,(int,float)) or not math.isfinite(first) or not r['scheduled_seconds']<=first<=r['terminal_seconds']:
                uncertain.append('missing visible latency');continue
            ttft=first-r['scheduled_seconds']
            if bucket[1]=='warm_prefix' and bucket[0]<=128000 and (not isinstance(decode,(int,float)) or not math.isfinite(decode) or decode<0):
                uncertain.append('missing decode latency');continue
            buckets[bucket].append((ttft,decode));pooled[bucket].append((ttft,decode))
        for b,values in buckets.items():
            if len(values)>=20 and (p95([v[0] for v in values])>=ttft_limit(*b) or
                (b[1]=='warm_prefix' and b[0]<=128000 and p95([v[1] for v in values])>=.033)):
                failed.append('repetition bucket latency exceeds SLO')
        reports.append({'offered':len(cohort),'valid_cohort':len(complete),'in_window_valid_terminals':len(terminal),
                        'cohort_valid_fraction':fraction,'in_window_rate_fraction':ratio,'queue':queue})
    populations=[]
    for b,values in sorted(pooled.items()):
        row={'input_tokens':b[0],'designated_cache':b[1],'observations':len(values)}
        if len(values)<20:uncertain.append('undersampled required bucket')
        else:
            row['p95_ttft_seconds']=p95([v[0] for v in values])
            if row['p95_ttft_seconds']>=ttft_limit(*b):failed.append('pooled bucket TTFT exceeds SLO')
            if b[1]=='warm_prefix' and b[0]<=128000:
                row['p95_decode_seconds_per_token']=p95([v[1] for v in values])
                if row['p95_decode_seconds_per_token']>=.033:failed.append('pooled bucket decode exceeds SLO')
        populations.append(row)
    return {'verdict':'failed' if failed else 'inconclusive' if uncertain else 'passed',
            'failures':sorted(set(failed)),'uncertainties':sorted(set(uncertain)),
            'repetitions':reports,'buckets':populations,'quantile':'nearest-rank p95'}
