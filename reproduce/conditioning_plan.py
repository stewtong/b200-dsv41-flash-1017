"""Expand warm-grid requests and measured multi-turn dependencies separately."""
import hashlib
import json
import os
from pathlib import Path
import workload

SUITES={'core','quality','output_sensitivity','anchor','pilot_warm'}


def rows(ledger):
    for cell in ledger['cells']:
        if cell.get('cache')!='warm_prefix' or cell.get('suite') not in SUITES:
            continue
        for repetition in range(cell['repetitions']):
            for session in range(cell['concurrency']):
                yield {'kind':'conditioning_request','id':cell['id']+'-r'+str(repetition)+'-s'+str(session),
                       'parent_cell':cell['id'], 'suite':cell['suite'], 'layout':cell['layout'],
                       'input_tokens':cell['input_tokens'], 'concurrency':cell['concurrency'],
                       'repetition':repetition, 'session':session,
                       'seed':workload.seed_for(cell['suite'],cell['input_tokens'],cell['concurrency'],repetition,session),
                       'status':'planned'}

    for cell in ledger['cells']:
        if cell['suite'] not in ('growing','cache_control'):continue
        fields=cell['parameters']; n=fields['input'];c=fields['concurrency']
        arm=fields.get('control','affinity')
        for repetition in range(cell['repetitions']):
            for session in range(c):
                prefix=cell['id']+'-r'+str(repetition)+'-s'+str(session)
                for turn in range(4):
                    reuse=turn>0 and arm in ('affinity','no_affinity')
                    yield {'kind':'measured_turn_dependency','id':prefix+'-turn'+str(turn),
                           'parent_cell':cell['id'],'suite':cell['suite'],'layout':cell['layout'],
                           'repetition':repetition,'session':session,'turn':turn,
                           'designated_cache':'warm_prefix' if reuse else 'cold',
                           'prefix_source':prefix+'-turn'+str(turn-1) if reuse else None,
                           'separate_conditioning_request':False,
                           'initial_input_tokens':n,'output_tokens':512 if cell['suite']=='growing' else 256,
                           'append_user_tokens':8000 if cell['suite']=='growing' and turn>0 else None,
                           'seed':workload.seed_for(cell['suite'],n,c,repetition,session),
                           'input_count_source':'installed tokenizer over actual retained history; output and role overhead included',
                           'routing':'randomized session map per turn' if arm=='no_affinity' else 'stable affinity',
                           'cache_control':arm,'status':'planned'}


def write(ledger_path, output):
    source=Path(ledger_path); data=source.read_bytes(); ledger=json.loads(data)
    if not ledger.get('plan_sha256') or not ledger.get('matrix_sha256'):
        raise ValueError('plan and matrix provenance required')
    target=Path(output); temporary=target.with_name(target.name+'.tmp.'+str(os.getpid()))
    if target.exists() or target.is_symlink(): raise FileExistsError('conditioning ledger already exists')
    count=0
    with temporary.open('x') as f:
        header={'schema':'dsv41-conditioning-ledger/v1','source_ledger_sha256':hashlib.sha256(data).hexdigest(),
                'plan_sha256':ledger['plan_sha256'],'matrix_sha256':ledger['matrix_sha256'],
                'scope':sorted(SUITES|{'growing','cache_control'}),'complete_universe':False,
                'omissions':['tokenized actual multi-turn histories and any separately registered warm traffic variants'],
                'counting':'conditioning_request adds unmeasured work; measured_turn_dependency refers to measured turns, never extra conditioning'}
        f.write(json.dumps(header,sort_keys=True,separators=(',',':'))+'\n')
        for row in rows(ledger):
            f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n');count+=1
        f.flush();os.fsync(f.fileno())
    if target.exists() or target.is_symlink(): raise FileExistsError('conditioning ledger appeared during generation')
    os.link(temporary,target);temporary.unlink()
    return count
