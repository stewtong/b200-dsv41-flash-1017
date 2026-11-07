"""Bind native router session selections to physical engines and cache counters."""
import re

ANSI=re.compile(r'\x1b\[[0-9;]*m')
ROUTE=re.compile(r"Consistent hash routing: key='header:x-session-id:([^']+)' -> worker='(http://127\.0\.0\.1:[0-9]+)(?:@([0-9]+))?' \(index=[0-9]+\)")


def verify(text,tasks,expected_ranks,deltas,conditioning=False):
    expected={task['id']:task for task in tasks}
    if len(expected)!=len(tasks) or not expected:raise ValueError('ambiguous routing tasks')
    observations={key:[] for key in expected}
    clean=ANSI.sub('',text)
    for line in clean.splitlines():
        if "Consistent hash routing: key='header:x-session-id:" not in line:continue
        match=ROUTE.search(line)
        if match is None:raise ValueError('malformed native router selection')
        session,worker,rank=match.groups()
        rank=rank or ''
        if session not in expected:raise ValueError('unregistered routed session')
        endpoint=worker+'/metrics'
        if endpoint not in expected_ranks:raise ValueError('router selected an unowned engine')
        if rank=='':
            if list(map(str,expected_ranks[endpoint]))!=['0']:raise ValueError('distributed rank absent from router selection')
            rank='0'
        if rank not in set(map(str,expected_ranks[endpoint])):raise ValueError('router selected an unqualified rank')
        observations[session].append((endpoint,rank))
    counts=2 if conditioning else 1
    if any(len(values)!=counts or len(set(values))!=1 for values in observations.values()):
        raise ValueError('missing, repeated or changed router session selection')
    expected_queries={(url,str(rank)):0 for url,ranks in expected_ranks.items() for rank in ranks}
    records=[]
    for session,values in sorted(observations.items()):
        endpoint,rank=values[0];expected_queries[(endpoint,rank)]+=expected[session]['input_tokens']
        records.append({'request_id':session,'session_id':session,'endpoint':endpoint,'rank':rank})
    actual={}
    for d in deltas:
        key=(d['endpoint'],str(d['rank']))
        if key in actual:raise ValueError('duplicate cache rank')
        actual[key]=d['queries']
    if set(actual)!=set(expected_queries):raise ValueError('cache/routing rank coverage differs')
    for key,tokens in expected_queries.items():
        if (tokens==0 and actual[key]!=0) or (tokens>0 and abs(actual[key]-tokens)>max(16*counts,tokens*.0025)):
            raise ValueError('router-selected work disagrees with observed per-rank query counters')
    return {'status':'router_selection_and_rank_counters_verified','requests':records,
            'scope':'per-session native router selection; per-rank aggregate query counters agree; TP requests span all physically mapped shards of the chosen engine'}
