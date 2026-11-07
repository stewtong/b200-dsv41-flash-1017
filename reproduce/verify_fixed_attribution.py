"""Recompute native fixed-session routes and cache work from retained records."""
import hashlib
import json
import math
from pathlib import Path
import re
import engine_counters,runtime,verify_fixed_session,speculation_evidence
import verify_worker_observations

ROUTE=re.compile(r"Consistent hash routing: key='header:x-session-id:([^']+)' -> worker='(http://127\.0\.0\.1:[0-9]+)(?:@([0-9]+))?' \(index=[0-9]+\)")


def verify_cache_profile(directory,binding):
    root=Path(directory);contract=json.loads((root/'attribution-contract.json').read_text())
    expected=(binding.get('delta') or {}).get('enable_prefix_caching',True)
    if type(expected) is not bool or contract.get('prefix_caching',True) is not expected:
        raise ValueError('fixed cache accounting differs from parent profile')
    if not expected and json.loads((root/'prefix-cache-profile-binding.json').read_text())!=binding:
        raise ValueError('fixed cache proof differs from parent profile binding')


def cache_values(snapshot,ranks):
    if set(snapshot)!=set(ranks):raise ValueError('cache endpoints differ')
    result={}
    for endpoint,text in snapshot.items():
        types={}
        for line in text.splitlines():
            m=re.fullmatch(r'# TYPE (vllm:prefix_cache_(?:queries|hits)) (\S+)',line)
            if m:
                if m[1] in types and types[m[1]]!=m[2]:raise ValueError('conflicting cache counter type')
                types[m[1]]=m[2]
            if not line.startswith(('vllm:prefix_cache_queries_total','vllm:prefix_cache_hits_total')):continue
            m=re.fullmatch(r'(vllm:prefix_cache_(?:queries|hits)_total)\{([^}]*)\}\s+(\S+)',line)
            if not m:raise ValueError('malformed cache counter')
            pairs=re.findall(r'(\w+)="([^"\\]*)"',m[2]);labels=dict(pairs)
            if (len(pairs)!=len(labels) or ','.join(k+'="'+v+'"' for k,v in pairs)!=m[2] or
                    set(labels)!={'engine','model_name'} or labels['model_name']!='/model'):
                raise ValueError('cache counter labels differ')
            key=(endpoint,labels['engine'],m[1]);value=float(m[3])
            if key in result or not math.isfinite(value) or value<0 or not value.is_integer():raise ValueError('invalid cache counter')
            if types.get(m[1].removesuffix('_total'))!='counter':raise ValueError('cache metric is not a counter')
            result[key]=int(value)
    expected={(url,str(rank),'vllm:prefix_cache_'+kind+'_total') for url,rs in ranks.items() for rank in rs for kind in ('queries','hits')}
    if set(result)!=expected:raise ValueError('cache rank population differs')
    return result


def verify(directory,*,progress=None):
    if progress is not None and not callable(progress):raise ValueError('verification progress check must be callable')
    if progress is not None:progress()
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda name:json.loads((root/name).read_text())
    contract=read('attribution-contract.json');ranks=contract['expected_ranks'];boundaries=read('boundaries.json')
    result=verify_fixed_session.verify(root/'requests',progress=progress);requests=root/'requests'
    inputs=json.loads((requests/'measured/fixed-inputs.json').read_text());tasks=inputs['tasks'];session=read('requests/fixed-session-contract.json')
    warm={t['cache_designation'] for t in tasks}=={'warm_prefix'}
    phases=['before_conditioning','after_conditioning','before_measured','after_measured'] if warm else ['before_measured','after_measured']
    if (contract.get('schema')!='fixed-attribution-capture/v1' or contract.get('session_ids')!=session['session_ids'] or
            contract.get('request_namespace')!=session['request_namespace'] or contract.get('clock_id')!=session['clock_id'] or
            [b.get('phase') for b in boundaries]!=phases or result['valid_requests']!=len(tasks)):
        raise ValueError('fixed attribution contract or boundary population differs')
    if warm and (type(contract.get('cache_block_size')) is not int or contract['cache_block_size']<=0 or
            any(t.get('cache_block_size')!=contract['cache_block_size'] for t in tasks)):
        raise ValueError('fixed attribution cache block differs')
    snapshots={};last=0;by_phase={};worker_snapshots={};worker_context=None
    if 'worker_observation_schema' in contract:
        if (contract['worker_observation_schema']!='worker-target-attribution/v1' or
                hashlib.sha256((root/'worker-context.json').read_bytes()).hexdigest()!=contract.get('worker_context_sha256')):
            raise ValueError('fixed worker attribution source contract differs')
        worker_context=read('worker-context.json')
    for boundary in boundaries:
        a=boundary.get('started_ns');z=boundary.get('completed_ns');limit=boundary.get('deadline_ns');name=boundary.get('snapshot_file','')
        if (type(a) is not int or type(z) is not int or not last<=a<z or
                type(limit) is not int or not z<limit<=a+10_000_000_000 or
                not re.fullmatch(boundary['phase']+r'-[0-9]{3}\.json',name) or
                hashlib.sha256((root/name).read_bytes()).hexdigest()!=boundary.get('snapshot_sha256')):
            raise ValueError('fixed boundary clocks or source differ')
        last=z;by_phase[boundary['phase']]=boundary;snapshots[boundary['phase']]=read(name)
        if worker_context is not None:
            worker_name=boundary.get('worker_snapshot_file','')
            if (not re.fullmatch(boundary['phase']+r'-[0-9]{3}-worker\.json',worker_name) or
                    hashlib.sha256((root/worker_name).read_bytes()).hexdigest()!=boundary.get('worker_snapshot_sha256')):
                raise ValueError('fixed worker boundary source differs')
            observation=read(worker_name)
            if not a<=observation.get('observed_at_ns',0)<=z:raise ValueError('fixed worker observation is outside its boundary')
            worker_snapshots[boundary['phase']]=verify_worker_observations.boundary_rows(observation,worker_context,z)
        elif any(k.startswith('worker_snapshot') for k in boundary):
            raise ValueError('undeclared fixed worker boundary')
    outcomes=[]
    for phase,folder in ([('conditioning','conditioning')] if warm else [])+[('measured','measured')]:
        before='before_'+phase;after='after_'+phase;rows=[json.loads((requests/folder/(t['id']+'.json')).read_text()) for t in tasks]
        if (any(r['clock_id']!=session['clock_id'] for r in rows) or
                min(r['start_ns'] for r in rows)<by_phase[before]['completed_ns'] or
                max(r['end_ns'] for r in rows)>by_phase[after]['started_ns']):
            raise ValueError('fixed requests lie outside their retained boundaries')
        old=cache_values(snapshots[before],ranks);new=cache_values(snapshots[after],ranks)
        delta={k:new[k]-old[k] for k in old}
        if any(v<0 for v in delta.values()):raise ValueError('fixed cache counter reset')
        accounting=engine_counters.differences(snapshots[before],snapshots[after],ranks)
        if sum(r['successfully_finished_requests'] for r in accounting['ranks'])!=len(tasks) or any(
                r['finished_by_reason'][reason] for r in accounting['ranks'] for reason in ('abort','error','repetition')):
            raise ValueError('fixed engine completion population differs')
        prompt_usage=sum(next(e['data']['usage']['prompt_tokens'] for e in r['raw']['events']
            if e['kind']=='event' and e['data'].get('usage') is not None) for r in rows)
        if sum(r['finished_request_prompt_tokens'] for r in accounting['ranks'])!=prompt_usage:
            raise ValueError('fixed completed engine input work differs from wire usage')
        if sum(r['finished_request_generation_tokens'] for r in accounting['ranks'])!=sum(r['completion_tokens'] for r in rows):
            raise ValueError('fixed completed engine output work differs from wire')
        outcomes.append({'phase':phase,'counter_delta':delta,'engine_work':accounting})
    raw=(root/'routing.log').read_bytes();span=read('routing-span.json')
    if (type(span.get('start_byte')) is not int or type(span.get('end_byte')) is not int or
            span['start_byte']!=contract.get('router_start_byte') or span['start_byte']<0 or
            span['end_byte']-span['start_byte']!=len(raw) or span.get('sha256')!=hashlib.sha256(raw).hexdigest()):
        raise ValueError('fixed router byte span differs')
    selected={value:[] for value in session['session_ids'].values()}
    if len(selected)!=len(tasks):raise ValueError('fixed session affinity is not unique')
    for line in re.sub(r'\x1b\[[0-9;]*m','',raw.decode()).splitlines():
        if "Consistent hash routing: key='header:x-session-id:" not in line:continue
        m=ROUTE.search(line)
        if not m or m[1] not in selected:raise ValueError('unknown or malformed fixed native route')
        endpoint=m[2]+'/metrics';rank=m[3]
        if endpoint not in ranks:raise ValueError('fixed native route selected an unowned endpoint')
        if rank is None:
            if list(map(str,ranks[endpoint]))!=['0']:raise ValueError('fixed native DP rank missing')
            rank='0'
        if rank not in set(map(str,ranks[endpoint])):raise ValueError('fixed native rank differs')
        selected[m[1]].append((endpoint,rank))
    if any(len(values)!=(2 if warm else 1) or len(set(values))!=1 for values in selected.values()):
        raise ValueError('fixed route missing, repeated or affinity changed')
    speculative=contract.get('speculative_contract')
    if 'speculative_contract' in contract:
        if (not isinstance(speculative,dict) or set(speculative)!={'max_spec_tokens','placement'} or
                type(speculative['max_spec_tokens']) is not int or not 1<=speculative['max_spec_tokens']<=64 or
                speculative['placement']!={t['id']:list(selected[session['session_ids'][t['id']]][0]) for t in tasks}):
            raise ValueError('speculative contract differs from actual native request routes')
    enabled=contract.get('prefix_caching',True)
    if type(enabled) is not bool or warm and not enabled:raise ValueError('fixed cache accounting mode differs')
    if not enabled:
        proof=(root/'prefix-cache-profile-binding.json').read_bytes();binding=json.loads(proof)
        if (hashlib.sha256(proof).hexdigest()!=contract.get('prefix_cache_profile_sha256') or
                binding.get('phase')!='qualification' or binding.get('delta',{}).get('enable_prefix_caching') is not False or
                binding.get('record',{}).get('profile_delta')!=binding.get('delta') or
                not binding.get('record',{}).get('template',{}).get('commands') or
                any(c['argv'].count('--disable-prefix-caching')!=1 or '--enable-prefix-caching' in c['argv']
                    for c in binding['record']['template']['commands'])):
            raise ValueError('cache-off attribution profile proof differs')
    elif (root/'prefix-cache-profile-binding.json').exists() or 'prefix_cache_profile_sha256' in contract:
        raise ValueError('undeclared cache accounting profile proof')
    populations=[]
    for outcome in outcomes:
        phase=outcome['phase'];delta=outcome['counter_delta'];expected={(url,str(rank)):0 for url,rs in ranks.items() for rank in rs}
        for task in tasks:
            key=selected[session['session_ids'][task['id']]][0]
            expected[key]+=task['prefix_tokens'] if phase=='conditioning' else task['input_tokens']
        per_rank=[]
        for key,tokens in expected.items():
            queries=delta[(*key,'vllm:prefix_cache_queries_total')];hits=delta[(*key,'vllm:prefix_cache_hits_total')]
            if (not enabled and (queries or hits) or enabled and
                    (hits>queries or (not tokens and queries) or tokens and abs(queries-tokens)>max(16,tokens*.0025))):
                raise ValueError('fixed route-selected work differs from cache counters')
            per_rank.append({'endpoint':key[0],'rank':key[1],'queries':queries,'hits':hits,'hit_fraction':hits/queries if queries else None})
        population={'phase':phase,'ranks':per_rank,'engine_work':outcome['engine_work']}
        if speculative is not None:
            rows={t['id']:read('requests/'+phase+'/'+t['id']+'.json') for t in tasks}
            population['speculative_accounting']=speculation_evidence.reconcile_requests(
                snapshots['before_'+phase],snapshots['after_'+phase],ranks,
                speculative['max_spec_tokens'],rows,speculative['placement'])
        if worker_context is not None:
            native=verify_worker_observations.native_target_population(worker_context,ranks,population.get('speculative_accounting'))
            population['target_observation_accounting']=verify_worker_observations.reconcile_target_positions(
                worker_snapshots['before_'+phase],worker_snapshots['after_'+phase],native,speculative=speculative is not None)
        populations.append(population)
    if read('attribution-result.json')!={'status':'captured_cleanup_pending','requests':len(tasks),
            'scope':'request and aggregate cache/work accounting only; full launcher acceptance remains required'}:
        raise ValueError('fixed attribution result differs')
    runtime.verify_evidence_manifest(root)
    return {'status':'raw_fixed_attribution_verified','requests':len(tasks),'conditioning_requests':len(tasks) if warm else 0,
        'session_routes':{k:list(v[0]) for k,v in selected.items()},'populations':populations,
        'scope':'raw native affinity, achieved cache counters and engine work; excludes frozen map/profile, physical workers, full telemetry, overhead and shared cleanup acceptance'}
