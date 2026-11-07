"""Recover request-draft acceptance distributions from pinned metric semantics.

The installed SpecDecodingStats.observe_draft increments every accepted prefix
position. Position counters therefore form a survival-count vector. This does
not expose a separate count of all target-model verified tokens.
"""
from fractions import Fraction
import math
import re
import verify_wire

SCALARS=('num_drafts','num_draft_tokens','num_accepted_tokens')
PREFIX='vllm:spec_decode_'
POSITION=PREFIX+'num_accepted_tokens_per_pos_total'


def snapshot(text, ranks, positions):
    if (type(positions) is not int or not 1<=positions<=64 or not ranks or
            len(set(map(str,ranks)))!=len(ranks)):
        raise ValueError('explicit speculative geometry required')
    types={};values={}
    required={PREFIX+s+'_total' for s in SCALARS}|{POSITION}
    for line in text.splitlines():
        match=re.fullmatch(r'# TYPE (\S+) (\S+)',line)
        if match:
            name,kind=match.groups()
            if name in types and types[name]!=kind:raise ValueError('conflicting speculative metric type')
            types[name]=kind
        match=re.fullmatch(r'(\S+?)\{([^}]*)\}\s+(\S+)',line)
        if not match or match[1] not in required:continue
        name,raw_labels,number=match.groups();pairs=re.findall(r'(\w+)="([^"\\]*)"',raw_labels)
        keys=[k for k,v in pairs]
        if len(set(keys))!=len(keys):raise ValueError('duplicate speculative label')
        labels=dict(pairs);rank=labels.get('engine');pos=labels.get('position')
        expected_labels={'model_name','engine'}|({'position'} if name==POSITION else set())
        if (set(labels)!=expected_labels or labels.get('model_name')!='/model' or
                ','.join(k+'="'+v+'"' for k,v in pairs)!=raw_labels):
            raise ValueError('speculative metric model or labels differ from pinned source')
        if rank not in set(map(str,ranks)):raise ValueError('unexpected speculative rank')
        if name==POSITION:
            if pos not in set(map(str,range(positions))):raise ValueError('unexpected draft position')
        elif pos is not None:raise ValueError('position on scalar speculative counter')
        key=(name,rank,pos);value=float(number)
        if key in values or not math.isfinite(value) or value<0 or not value.is_integer():
            raise ValueError('invalid speculative counter')
        values[key]=int(value)
    expected={(PREFIX+s+'_total',str(rank),None) for rank in ranks for s in SCALARS}
    expected|={(POSITION,str(rank),str(pos)) for rank in ranks for pos in range(positions)}
    if set(values)!=expected:raise ValueError('missing speculative metric rank or position')
    if any(types.get(name.removesuffix('_total'))!='counter' and types.get(name)!='counter' for name in required):
        raise ValueError('speculative counter type missing')
    return values


def verify_delta(before,after,ranks,max_spec_tokens):
    if type(max_spec_tokens) is not int or not 1<=max_spec_tokens<=64 or not ranks:
        raise ValueError('explicit speculative geometry required')
    if set(before)!=set(ranks) or set(after)!=set(ranks):raise ValueError('speculative endpoint coverage differs')
    reports=[];pooled=[0]*(max_spec_tokens+1)
    for endpoint,engines in sorted(ranks.items()):
        if not engines or len(set(map(str,engines)))!=len(engines):raise ValueError('ambiguous speculative rank contract')
        old=snapshot(before[endpoint],engines,max_spec_tokens);new=snapshot(after[endpoint],engines,max_spec_tokens)
        delta={k:new[k]-old[k] for k in old}
        if any(v<0 for v in delta.values()):raise ValueError('speculative counter reset')
        for rank in engines:
            rank=str(rank)
            drafts,drafted,accepted=[delta[(PREFIX+s+'_total',rank,None)] for s in SCALARS]
            vector=[delta[(POSITION,rank,str(i))] for i in range(max_spec_tokens)]
            if (any(a<b for a,b in zip([drafts]+vector,vector)) or sum(vector)!=accepted or
                not accepted<=drafted<=drafts*max_spec_tokens):raise ValueError('inconsistent speculative prefix counts')
            counts=[drafts-vector[0]]+[vector[i]-vector[i+1] for i in range(max_spec_tokens-1)]+[vector[-1]]
            if sum(counts)!=drafts or sum(i*c for i,c in enumerate(counts))!=accepted:
                raise ValueError('speculative distribution arithmetic differs')
            pooled=[a+b for a,b in zip(pooled,counts)]
            reports.append({'endpoint':endpoint,'rank':rank,'draft_observations':drafts,'draft_tokens':drafted,
                'accepted_tokens':accepted,'accepted_count_distribution':counts,
                'mean_acceptance_length_including_bonus':1+accepted/drafts if drafts else None})
    total=sum(pooled);accepted=sum(i*c for i,c in enumerate(pooled))
    variance=(sum(Fraction(i*i*c,total) for i,c in enumerate(pooled))-Fraction(accepted,total)**2) if total else None
    return {'status':'partial_missing_verified_token_counter','ranks':reports,'draft_observations':total,
        'accepted_count_distribution':pooled,'accepted_count_variance':float(variance) if variance is not None else None,
        'zero_variance_requires_investigation':variance==0 if variance is not None else None,
        'verified_token_count':None,'verified_token_count_source':'not exported by these installed counters',
        'scope':'per-request draft-observation distribution from cumulative position counters; not a kernel trace or complete speculative attribution qualification'}


def detailed_payload(payload,max_spec_tokens):
    """Validate the installed RequestSpecDecodeMetrics.to_dict response schema."""
    if type(max_spec_tokens) is not int or not 1<=max_spec_tokens<=64 or not isinstance(payload,dict):
        raise ValueError('invalid detailed speculative contract')
    fields=('num_spec_tokens','num_spec_steps','num_accepted_draft_tokens','num_draft_tokens')
    if any(type(payload.get(k)) is not int or payload[k]<0 for k in fields) or payload['num_spec_tokens']!=max_spec_tokens:
        raise ValueError('invalid detailed speculative counts')
    steps=payload['num_spec_steps'];hist=payload.get('acceptance_histogram')
    accepted=payload.get('per_step_accepted',[]);drafted=payload.get('per_step_drafted',[])
    if (not isinstance(hist,list) or len(hist)!=max_spec_tokens+1 or
        any(type(n) is not int or n<0 for n in hist) or not isinstance(accepted,list) or not isinstance(drafted,list) or
        len(accepted)!=steps or len(drafted)!=steps or
        any(type(a) is not int or type(d) is not int or not 0<=a<=d<=max_spec_tokens for a,d in zip(accepted,drafted))):
        raise ValueError('detailed per-step speculative arrays missing or invalid')
    recovered=[accepted.count(i) for i in range(max_spec_tokens+1)]
    count=sum(accepted);proposed=sum(drafted)
    mean=1+count/steps if steps else 1.0;rate=count/proposed if proposed else 0.0
    if (hist!=recovered or sum(hist)!=steps or payload['num_accepted_draft_tokens']!=count or
        payload['num_draft_tokens']!=proposed or
        any(type(payload.get(k)) not in (int,float) or not math.isfinite(payload[k]) for k in ('mean_acceptance_length','draft_acceptance_rate')) or
        payload['mean_acceptance_length']!=mean or payload['draft_acceptance_rate']!=rate):
        raise ValueError('detailed speculative summary differs from ordered steps')
    variance=sum(Fraction(a*a,steps) for a in accepted)-Fraction(count,steps)**2 if steps else None
    return {'status':'detailed_request_steps_verified','verification_steps':steps,
        'proposed_valid_draft_tokens':proposed,'accepted_draft_tokens':count,'acceptance_histogram':hist,
        'mean_acceptance_length_including_bonus':mean,'accepted_count_variance':float(variance) if variance is not None else None,
        'zero_variance_requires_investigation':variance==0 if variance is not None else None,
        'verified_token_count':None,
        'scope':'ordered per-request verification steps and grammar-adjusted proposed drafts; not a separate target-model verified-token counter'}


def from_stream(raw,max_spec_tokens):
    verify_wire.verify_complete(raw)
    observed=[]
    for row in raw['events']:
        if row['kind']!='event':continue
        data=row['data'];metrics=data.get('metrics') or {};value=metrics.get('speculative_decoding')
        if value is None:continue
        if not isinstance(data.get('usage'),dict) or data.get('choices')!=[]:
            raise ValueError('speculative metrics are not on the final usage chunk')
        observed.append(value)
    if len(observed)!=1:raise ValueError('unique detailed speculative response metrics missing')
    return detailed_payload(observed[0],max_spec_tokens)


class UnsettledSpeculationError(ValueError):
    """Valid counters do not yet describe the completed request population."""


def reconcile_requests(before,after,ranks,max_spec_tokens,rows,placement):
    """Bind cumulative draft/position deltas to ordered wire records per rank.

    Proposed drafts remain grammar-adjusted proposals, never a count of all
    target-model verified tokens. A valid but different population may settle
    within the caller's existing boundary deadline; malformed metrics fail.
    """
    if not rows or set(rows)!=set(placement):raise ValueError('speculative request placement population differs')
    expected={(endpoint,str(rank)):{'drafts':0,'drafted':0,'accepted':0,'hist':[0]*(max_spec_tokens+1)}
              for endpoint,rs in ranks.items() for rank in rs}
    for task,row in rows.items():
        if row.get('task_id')!=task:raise ValueError('speculative request identity differs')
        route=placement[task]
        if not isinstance(route,(list,tuple)) or len(route)!=2 or tuple(route) not in expected:
            raise ValueError('speculative request selected an unknown physical rank')
        record=from_stream(row['raw'],max_spec_tokens);total=expected[tuple(route)]
        total['drafts']+=record['verification_steps'];total['drafted']+=record['proposed_valid_draft_tokens']
        total['accepted']+=record['accepted_draft_tokens']
        total['hist']=[a+b for a,b in zip(total['hist'],record['acceptance_histogram'])]
    report=verify_delta(before,after,ranks,max_spec_tokens)
    for rank in report['ranks']:
        total=expected[(rank['endpoint'],rank['rank'])]
        if (rank['draft_observations']!=total['drafts'] or rank['draft_tokens']!=total['drafted'] or
                rank['accepted_tokens']!=total['accepted'] or rank['accepted_count_distribution']!=total['hist']):
            raise UnsettledSpeculationError('speculative counters differ from completed request steps per rank')
    return {**report,'request_population_status':'ordered_wire_steps_match_cumulative_counters_per_rank',
            'requests':len(rows)}
