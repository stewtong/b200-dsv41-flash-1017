"""Retain installed group-aware KV capacity and gate same-profile pool drift."""
import hashlib
import json
import math
from pathlib import Path
import re
import launcher
import runtime
import profile_registry

SOURCE_SHA256='195159dd716e8fa2e597d94625d79618e025218ca2baffa38cb2e5977516b9e3'
NUMBER=r'(?:[1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]*)'
LINE=re.compile(r'\(EngineCore(?:_DP([0-9]+))? pid=[0-9]+\).*?\b(GPU(?: \+ host)?) KV cache size: ('+NUMBER+
                r') tokens, Maximum concurrency for ('+NUMBER+r') tokens per request: ([0-9]+(?:\.[0-9]+)?)x')


def parse(text,expected_ranks):
    if (not isinstance(expected_ranks,list) or not expected_ranks or
        any(type(n) is not int or not 0<=n<8 for n in expected_ranks) or len(set(expected_ranks))!=len(expected_ranks)):
        raise ValueError('invalid KV engine coverage contract')
    rows=[]
    for line in text.splitlines():
        if ' KV cache size:' not in line:continue
        match=LINE.search(line)
        if not match:raise ValueError('unparseable installed KV capacity line')
        rank=int(match[1] or 0);tokens=int(match[3].replace(',',''));limit=int(match[4].replace(',',''));concurrency=float(match[5])
        if (not math.isfinite(concurrency) or concurrency<=0 or
            abs(tokens/limit-concurrency)>.00501+1/limit):
            raise ValueError('KV capacity and rounded concurrency disagree')
        rows.append({'logical_rank':rank,'pool':match[2],'equivalent_full_context_tokens':tokens,
                     'max_request_tokens':limit,'max_concurrency_at_limit':concurrency})
    if len(rows)!=len(expected_ranks) or {r['logical_rank'] for r in rows}!=set(expected_ranks):
        raise ValueError('KV engine capacity coverage incomplete or duplicated')
    return sorted(rows,key=lambda r:r['logical_rank'])


def collect(root,config):
    root=Path(root);state=json.loads((root/'execution-state.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text());plan=json.loads((root/'launch-plan.json').read_text())
    ids=state['container_ids'];commands=plan['commands'];tp,dp,_=launcher.LAYOUTS[config['layout']]
    if (state['run_id']!=config['run_id'] or len(ids)!=len(commands) or
        len(ids)!=8//(tp*dp) or len(set(ids))!=len(ids) or any(not runtime.re_full_id(c) for c in ids)):
        raise ValueError('KV snapshot ownership or geometry mismatch')
    replicas=[]
    for cid,command in zip(ids,commands):
        raw=(root/('rank-startup-'+cid+'.log')).read_bytes()
        engines=parse(raw.decode('utf-8',errors='replace'),list(range(dp)))
        if any(e['max_request_tokens']!=1_048_576 for e in engines):raise ValueError('KV sequence limit differs from campaign')
        replicas.append({'container_id':cid,'gpu_ids':command['gpu_ids'],
                         'startup_log_sha256':hashlib.sha256(raw).hexdigest(),'engines':engines})
    return {'schema':'group-aware-kv-capacity/v1','layout':config['layout'],'image':config['image'],
            'profile_sha256':profile_registry.bound_hash(config,manifest),
            'installed_source_sha256':SOURCE_SHA256,'replicas':replicas,
            'scope':'group-aware equivalent full-context capacity, limited by device and any host pool; not physical bytes or measured sustainable capacity'}


def compare(current,baseline):
    if any(current.get(k)!=baseline.get(k) for k in ('schema','layout','image','profile_sha256','installed_source_sha256')):
        raise ValueError('KV baseline differs from the exact starting profile or installed source')
    def indexed(record):
        rows={}
        for replica in record['replicas']:
            for engine in replica['engines']:
                key=(tuple(replica['gpu_ids']),engine['logical_rank'])
                if key in rows:raise ValueError('duplicate KV physical engine')
                rows[key]=engine
        return rows
    now= indexed(current);old=indexed(baseline)
    if not old or now.keys()!=old.keys():raise ValueError('KV baseline physical coverage differs')
    changes=[]
    for key in sorted(old):
        if any(now[key][k]!=old[key][k] for k in ('pool','max_request_tokens')):
            raise ValueError('KV pool or sequence contract changed')
        before=old[key]['equivalent_full_context_tokens'];after=now[key]['equivalent_full_context_tokens']
        change=abs(after-before)/before
        changes.append({'gpu_ids':list(key[0]),'logical_rank':key[1],'relative_change':change})
    if any(row['relative_change']>.02 for row in changes):
        raise runtime.RuntimeErrorClosed('KV pool deviated over two percent; investigate before dispatch')
    return {'status':'within_two_percent','changes':changes}


def validate_baseline(config,manifest):
    """Reject a cross-profile baseline before paying for engine startup."""
    baseline=config.get('kv_baseline_evidence')
    if baseline is None:return
    root=Path(baseline);runtime.verify_evidence_manifest(root)
    old=json.loads((root/'run-config.json').read_text())
    old_manifest=json.loads((root/'manifest.json').read_text())
    if (old.get('layout')!=config['layout'] or old.get('image')!=config['image'] or
        profile_registry.bound_hash(old,old_manifest)!=profile_registry.bound_hash(config,manifest)):
        raise ValueError('KV baseline differs from the exact starting profile before startup')


def record(root,config):
    current=collect(root,config)
    runtime.atomic_json(Path(root)/'kv-budget.json',current)
    baseline=config.get('kv_baseline_evidence')
    if baseline is not None:
        runtime.verify_evidence_manifest(baseline)
        baseline=Path(baseline);oldconfig=json.loads((baseline/'run-config.json').read_text())
        oldstate=json.loads((baseline/'execution-state.json').read_text())
        selected=['run-config.json','manifest.json','execution-state.json','launch-plan.json']
        selected+=['rank-startup-'+cid+'.log' for cid in oldstate['container_ids']]
        excerpt=Path(root)/'kv-baseline-source';excerpt.mkdir()
        for name in selected:runtime.atomic_bytes(excerpt/name,(baseline/name).read_bytes())
        runtime.atomic_bytes(excerpt/'SOURCE_SHA256SUMS',(baseline/'SHA256SUMS').read_bytes())
        result=compare(current,collect(excerpt,oldconfig))
        result.update(baseline_manifest_sha256=hashlib.sha256((baseline/'SHA256SUMS').read_bytes()).hexdigest(),
                      scope='selected raw baseline files copied from a full-set verified source; excerpt is not the complete prior run')
    else:result={'status':'initial_observation','scope':'no established same-profile baseline supplied'}
    runtime.atomic_json(Path(root)/'kv-budget-comparison.json',result)
    return current
