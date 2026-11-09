"""Verify publication metrics from the included prompt-free CSV observations."""
import csv
import math
from pathlib import Path
import random
import statistics
import json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
END=60_000_000_000


def read(name):
    with (DATA/name).open() as stream:
        return list(csv.DictReader(stream))


def p95(values):
    return sorted(values)[math.ceil((len(values)-1)*.95)]


def indexed(name, fields):
    rows=read(name)
    values={tuple(r[k] for k in fields):r for r in rows}
    if len(values)!=len(rows):
        raise ValueError('duplicate table identity: '+name)
    return values


def population_checks():
    windows={'mixed-low-r1':90,'mixed-low-r2':90,'mixed-high-r1':360,
             'mixed-high-r2':360,'anchor-r1':5,'fast-only-high-r1':240}
    cohorts={(a,w,p) for a in ['nospec','k5'] for w in windows
             for p in (['all','fast','think'] if w.startswith('mixed-') else ['all','fast'])}
    assert set(indexed('load-windows.csv',['arm','window','profile']))==cohorts
    assert set(indexed('load-pairs.csv',['window','profile']))=={(w,p) for a,w,p in cohorts}
    assert set(indexed('load-deadlines.csv',['arm','window','profile','deadline_seconds']))=={
        (*key,str(d)) for key in cohorts for d in [1,5,15,60]}
    requests=indexed('load-requests.csv',['arm','window','request_index'])
    assert set(requests)=={(a,w,str(i)) for a in ['nospec','k5'] for w,n in windows.items() for i in range(n)}
    assert all(r['profile'] in {'fast','think'} and r['success'] in {'0','1'} for r in requests.values())
    assert set(indexed('load-tasks.csv',['arm','window','task_id']))=={
        (r['arm'],r['window'],r['task_id']) for r in requests.values()}
    regimes={'8000-c1','8000-c8','64000-c1','64000-c8'}
    assert set(indexed('natural-regimes.csv',['regime']))=={(r,) for r in regimes}
    assert set(indexed('natural-blocks.csv',['regime','block']))=={
        (r,f'context-block-{i:02}') for r in regimes for i in range(1,6)}
    assert len(indexed('natural-requests.csv',['arm','block','input_target','concurrency','request_index']))==196
    assert set(indexed('quality.csv',['profile']))=={('fast',),('think',)}
    outcomes=indexed('quality-outcomes.csv',['arm','profile','task_id','seed'])
    assert len(outcomes)==880 and all(r['success'] in {'0','1'} for r in outcomes.values())
    for profile,n in [('fast',200),('think',20)]:
        tasks={r['task_id'] for r in outcomes.values() if r['profile']==profile}
        assert len(tasks)==n
        expected={(a,profile,t,str(s)) for a in ['nospec','k5'] for t in tasks for s in [19,43]}
        assert {k for k in outcomes if k[1]==profile}==expected
        for task in tasks:
            assert len({r['family'] for r in outcomes.values() if r['profile']==profile and r['task_id']==task})==1
        counts={f:len({r['task_id'] for r in outcomes.values() if r['profile']==profile and r['family']==f})
                for f in ['coding','grounded','reasoning','tools']}
        assert set(counts.values())=={n//4}


def plotted_arithmetic_checks():
    requests=read('load-requests.csv')
    for c in read('load-windows.csv'):
        rr=[r for r in requests if r['arm']==c['arm'] and r['window']==c['window'] and
            (c['profile']=='all' or r['profile']==c['profile'])]
        disp=[r for r in rr if r['dispatched_ns'] and 0<=int(r['dispatched_ns'])<END]
        done=[r for r in disp if r['completed_ns'] and 0<=int(r['completed_ns'])<END]
        correct=[r for r in done if r['success']=='1']
        one=sum(int(r['completed_ns'])-int(r['dispatched_ns'])<=10**9 for r in correct)
        for field,value in dict(seconds=60,offered_per_second=len(rr)/60,
                                completed_per_second=len(done)/60,
                                successful_per_second=len(correct)/60,
                                completion_fraction=len(done)/len(disp),
                                success_given_completion=len(correct)/len(done),
                                successful_within_1s=one,successful_within_1s_per_second=one/60).items():
            assert math.isclose(value,float(c[field]),rel_tol=1e-10,abs_tol=1e-12),field
    for row in read('natural-regimes.csv'):
        ratios=[float(r['token_throughput_ratio']) for r in read('natural-blocks.csv') if r['regime']==row['regime']]
        rng=random.Random(20261108)
        draws=sorted(statistics.median([ratios[rng.randrange(5)] for _ in range(5)]) for _ in range(10000))
        assert float(row['prompt_blocks'])==5
        assert math.isclose(draws[math.floor(.025*9999)],float(row['interval_low']),rel_tol=1e-12)
        assert math.isclose(draws[math.ceil(.975*9999)],float(row['interval_high']),rel_tol=1e-12)
    for row in read('quality.csv'):
        assert int(row['seeds_per_task'])==2
        actual=100*(float(row['k5_success'])-float(row['nospec_success']))
        assert math.isclose(actual,float(row['difference_pp']),rel_tol=1e-10,abs_tol=1e-12)


def main():
    if not __debug__:
        raise ValueError('replay requires Python assertion checks; remove optimization flags')
    population_checks()
    requests=read('load-requests.csv')
    for row in requests:
        for key in ['scheduled_ns','dispatched_ns','first_visible_ns','completed_ns']:
            row[key]=int(row[key]) if row[key] else None
        row['success']=bool(int(row['success']))
    checks=0
    for claim in read('load-windows.csv'):
        selected=[r for r in requests if r['arm']==claim['arm'] and r['window']==claim['window'] and (claim['profile']=='all' or r['profile']==claim['profile'])]
        disp=[r for r in selected if r['dispatched_ns'] is not None and 0<=r['dispatched_ns']<END]
        done=[r for r in disp if r['completed_ns'] is not None and 0<=r['completed_ns']<END]
        success=[r for r in done if r['success']]
        assert len(selected)==int(claim['scheduled'])
        assert len(disp)==int(claim['dispatched'])
        assert len(done)==int(claim['completed_in_window'])
        assert len(success)==int(claim['successful_in_window'])
        allfv=p95([(r['first_visible_ns']-r['dispatched_ns'])/1e9 for r in disp if r['first_visible_ns'] is not None])
        visible=[(r['first_visible_ns']-r['dispatched_ns'])/1e9 for r in disp if r['first_visible_ns'] is not None]
        assert len(visible)==int(claim['all_recorded_first_visible_samples'])
        assert len(success)==int(claim['successful_in_window_first_visible_samples'])
        assert abs(allfv-float(claim['all_recorded_first_visible_p95']))<1e-9
        assert abs(p95([(r['first_visible_ns']-r['dispatched_ns'])/1e9 for r in success])-float(claim['successful_in_window_first_visible_p95']))<1e-9
        assert abs(p95([(r['completed_ns']-r['dispatched_ns'])/1e9 for r in success])-float(claim['successful_in_window_e2e_p95']))<1e-9
        assert abs(len(success)/60-float(claim['successful_per_second']))<1e-12
        for deadline in [1,5,15,60]:
            count=sum(r['completed_ns']-r['dispatched_ns']<=deadline*1e9 for r in success)
            c=next(x for x in read('load-deadlines.csv') if x['arm']==claim['arm'] and x['window']==claim['window'] and x['profile']==claim['profile'] and int(x['deadline_seconds'])==deadline)
            assert count==int(c['correct_within_deadline'])
            assert abs(count/60-float(c['correct_within_deadline_per_second']))<1e-12
            assert abs(count/60-float(c['correct_within_deadline_per_second']))<1e-12
        checks+=1
    windows=read('load-windows.csv')
    for pair in read('load-pairs.csv'):
        sides={a:next(x for x in windows if x['arm']==a and x['window']==pair['window'] and x['profile']==pair['profile']) for a in ['nospec','k5']}
        ns,k5=sides['nospec'],sides['k5']
        assert abs(float(k5['successful_per_second'])/float(ns['successful_per_second'])-float(pair['successful_rate_ratio']))<1e-12
        assert abs(float(ns['all_recorded_first_visible_p95'])/float(k5['all_recorded_first_visible_p95'])-float(pair['first_visible_p95_nospec_over_k5']))<1e-9
        if pair['e2e_1s_goodput_ratio']:
            assert abs(float(k5['successful_within_1s_per_second'])/float(ns['successful_within_1s_per_second'])-float(pair['e2e_1s_goodput_ratio']))<1e-12
    natural=read('natural-blocks.csv')
    natural_req=read('natural-requests.csv')
    for row in natural:
        for arm in ['nospec','k5']:
            rr=[r for r in natural_req if r['arm']==arm and r['block']==row['block'] and r['input_target']==row['input_target'] and r['concurrency']==row['concurrency']]
            assert sum(int(r['output_tokens']) for r in rr)==int(row[arm+'_output_tokens'])
            assert all(abs(float(r['group_seconds'])-float(row[arm+'_group_seconds']))<1e-12 for r in rr)
        ratio=(float(row['k5_output_tokens'])/float(row['k5_group_seconds']))/(float(row['nospec_output_tokens'])/float(row['nospec_group_seconds']))
        assert abs(ratio-float(row['token_throughput_ratio']))<1e-9
    for row in read('natural-regimes.csv'):
        ratios=[float(r['token_throughput_ratio']) for r in natural if r['regime']==row['regime']]
        assert len(ratios)==5 and abs(statistics.median(ratios)-float(row['median_ratio']))<1e-12
    outcomes=read('quality-outcomes.csv')
    quality_checks=[]
    for claim in read('quality.csv'):
        selected=[r for r in outcomes if r['profile']==claim['profile']]
        tasks=list(dict.fromkeys(r['task_id'] for r in selected))
        assert len(tasks)==int(claim['tasks'])
        for arm in ['nospec','k5']:
            rr=[r for r in selected if r['arm']==arm]
            assert len(rr)==2*len(tasks)
            assert abs(sum(int(r['success']) for r in rr)/len(rr)-float(claim[arm+'_success']))<1e-12
        differences=[]
        for task in tasks:
            sides={a:[int(r['success']) for r in selected if r['arm']==a and r['task_id']==task] for a in ['nospec','k5']}
            assert len(sides['nospec'])==len(sides['k5'])==2
            differences.append(statistics.mean(sides['k5'])-statistics.mean(sides['nospec']))
        rng=random.Random(20261108);n=len(differences)
        bootstrap=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
        lo,hi=100*bootstrap[math.floor(.025*9999)],100*bootstrap[math.ceil(.975*9999)]
        assert abs(lo-float(claim['interval_low_pp']))<1e-9
        assert abs(hi-float(claim['interval_high_pp']))<1e-9
        quality_checks.append(dict(profile=claim['profile'],task_clusters=n,interval_pp=[lo,hi]))
    plotted_arithmetic_checks()
    return dict(status='passed',load_requests=len(requests),profile_window_checks=checks,
        natural_block_pairs=len(natural),quality_outcomes=len(outcomes),quality=quality_checks,
        scope='Fixed study coverage and plotted metric arithmetic from included CSV observations; no prompt/oracle rescoring, GPU execution, fresh-node or independent-lab replication.')


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
