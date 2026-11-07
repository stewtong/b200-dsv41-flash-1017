"""Run one pilot cell with a supervised node session."""
import hashlib
import json
from pathlib import Path
import re
import time
import campaign
import cache_evidence
import launcher
import node_smoke
import pilot_burst
import pilot_registry
import runtime
import routing_evidence
import telemetry
import workload
import verify_pilot

PILOT_METRICS = {'vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc',
                 'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total', 'vllm:num_preemptions_total'}

def clock_domain(run_id):
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + str(run_id)

def prepare(matrix, cell_id, tasks):
    ledger = campaign.ledger(matrix)
    cells = [x for x in ledger['cells'] if x['id'] == cell_id and x['suite'].startswith('pilot_')]
    if len(cells) != 1: raise ValueError('unknown pilot cell')
    cell = cells[0]
    if cell['cache'] not in ('cold','warm_prefix'): raise ValueError('unsupported pilot cache state')
    if len(tasks) != cell['concurrency']: raise ValueError('pilot task count mismatch')
    seeds, hashes, ids = set(), set(), set()
    for task in tasks:
        body = task['body']
        allowed = {'model','messages','stream','stream_options','temperature','top_p','seed','max_tokens',
                   'chat_template_kwargs','min_tokens','ignore_eos'}
        if set(body) - allowed: raise ValueError('unexpected request fields')
        if body.get('model') != '/model' or body.get('stream') is not True or body.get('temperature') != 1.0 or body.get('top_p') != 1.0:
            raise ValueError('request contract mismatch')
        if body.get('stream_options') != {'include_usage':True} or body.get('chat_template_kwargs') != {'thinking':False,'reasoning_effort':50}:
            raise ValueError('stream or reasoning contract mismatch')
        if body.get('max_tokens') != cell['output_tokens'] or task['output_tokens'] != cell['output_tokens']:
            raise ValueError('output token contract mismatch')
        if abs(task['input_tokens'] - cell['input_tokens']) > cell['input_tokens']*0.0025:
            raise ValueError('input token target mismatch')
        fixed = cell['suite'] != 'pilot_smoke'
        if bool(task.get('fixed_output')) != fixed:
            raise ValueError('pilot fixed-output contract mismatch')
        if fixed and (body.get('min_tokens') != cell['output_tokens'] or body.get('ignore_eos') is not True):
            raise ValueError('fixed output not enforced')
        if not fixed and ('min_tokens' in body or body.get('ignore_eos')):
            raise ValueError('known-answer smoke must allow natural stopping')
        expected_seed = workload.seed_for(cell['suite'], cell['input_tokens'], cell['concurrency'], 0, task['session'])
        if (not isinstance(task['session'], int) or not 0 <= task['session'] < cell['concurrency'] or
            not re.fullmatch(r'[0-9a-f]{64}', task['prompt_sha256']) or task.get('suite') != cell['suite'] or
            task['seed'] != expected_seed or body.get('seed') != expected_seed or task['id'] != 's'+str(task['session'])):
            raise ValueError('pilot seed or session identity mismatch')
        marker = hashlib.sha256(str(expected_seed).encode()).hexdigest()[:16]
        if task.get('expected_answer') != marker or not any(marker in str(m.get('content',''))[:512] for m in body.get('messages',[])):
            raise ValueError('pilot seeded answer missing from early prompt')
        if cell['cache'] == 'warm_prefix':
            import warm_pilot_prompt
            if (task.get('prompt_version')!=warm_pilot_prompt.VERSION or
                    task.get('fixed_report_instruction')!=warm_pilot_prompt.INSTRUCTION or
                    not isinstance(body.get('messages'),list) or not body['messages'] or
                    not body['messages'][-1].get('content','').endswith(warm_pilot_prompt.INSTRUCTION)):
                raise ValueError('warm pilot requires its versioned long-form report instruction')
            condition=task.get('conditioning_body', {})
            block=task.get('cache_block_size'); prefix=task.get('prefix_tokens')
            if (cell['suite'] != 'pilot_warm' or cell['concurrency'] != 1 or
                not isinstance(block,int) or block <= 0 or not isinstance(prefix,int) or
                prefix != ((cell['input_tokens']-256)//block)*block or task.get('target_prefix_tokens') != prefix or
                not re.fullmatch(r'[0-9a-f]{64}',str(task.get('prefix_token_sha256'))) or
                task.get('expected_conditioning_answer') != marker or not isinstance(condition.get('messages'),list) or
                not condition['messages'] or set(condition)-allowed or 'min_tokens' in condition or condition.get('ignore_eos') or
                not any(marker in str(m.get('content',''))[:512] for m in condition['messages']) or condition.get('model') != '/model' or
                condition.get('seed') != expected_seed or condition.get('max_tokens') != 128 or
                condition.get('stream') is not True or condition.get('stream_options') != {'include_usage':True} or
                condition.get('temperature') != 1.0 or condition.get('top_p') != 1.0 or
                condition.get('chat_template_kwargs') != {'thinking':False,'reasoning_effort':50} or
                condition.get('messages') != body.get('messages',[])[:len(condition.get('messages',[]))] or
                body.get('messages',[])[len(condition.get('messages',[])):len(condition.get('messages',[]))+1] != [{'role':'assistant','content':marker}]):
                raise ValueError('warm conditioning contract mismatch')
        if task['id'] in ids or body['seed'] in seeds or task['prompt_sha256'] in hashes:
            raise ValueError('duplicate session identity, seed or prompt')
        ids.add(task['id']); seeds.add(body['seed']); hashes.add(task['prompt_sha256'])
    return cell

def verified_tasks(config, allowed_root='/data'):
    path=Path(config['tasks_file'])
    if path.is_symlink() or path.name!='pilot-tasks.json' or not path.is_absolute():
        raise ValueError('pilot tasks must be a sealed prompt artifact')
    root=path.parent.resolve(strict=True)
    if not root.is_relative_to(Path(allowed_root).resolve()) or not re.fullmatch(r'dsv41-[a-z0-9-]+',root.name):
        raise ValueError('pilot tasks outside a campaign evidence directory')
    runtime.verify_evidence_manifest(root)
    result=node_smoke.read(root/'prompt-result.json'); source=node_smoke.read(root/'run-config.json')
    reused=None
    if config.get('prompt_reuse') is not None:
        import prompt_reuse
        reused=prompt_reuse.load(config,root)[0]
    expected_cell=reused['source_cell_id'] if reused else config['cell_id']
    expected_layout=config['prompt_reuse']['source_layout'] if reused else config['layout']
    if (result.get('status')!='passed' or result.get('cleanup')!='passed' or
        result.get('cell_id')!=expected_cell or source.get('cell_id')!=expected_cell or
        source.get('layout')!=expected_layout or source.get('image')!=config['image']):
        raise ValueError('prompt generation evidence does not match pilot')
    for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
        if reused and key=='manifest':continue
        if (root/name).read_bytes()!=Path(config[key]).read_bytes():
            raise ValueError('prompt generation dependency evidence changed')
    data=path.read_bytes(); tasks=json.loads(data)
    if result.get('tasks')!=len(tasks): raise ValueError('prompt generation task count mismatch')
    runtime.verify_evidence_manifest(root)
    return data

def copy_prompt_provenance(config, root, task_bytes):
    source=Path(config['tasks_file']).parent
    runtime.verify_evidence_manifest(source)
    dest=Path(root)/'prompt-provenance'; dest.mkdir()
    names=('SHA256SUMS','run-config.json','prompt-result.json','manifest.json','checkpoint-receipt.json','image-qualification.json')
    for name in names: runtime.atomic_bytes(dest/name,(source/name).read_bytes())
    if config.get('prompt_reuse') is not None:
        import prompt_reuse
        binding,old_bytes,new_bytes=prompt_reuse.load(config,source)
        runtime.atomic_bytes(dest/'source-toolkit-MANIFEST.json',old_bytes)
        runtime.atomic_bytes(dest/'target-toolkit-MANIFEST.json',new_bytes)
        runtime.atomic_json(dest/'prompt-reuse.json',binding)
    runtime.verify_evidence_manifest(source)
    if (source/'pilot-tasks.json').read_bytes()!=task_bytes:
        raise ValueError('prompt tasks changed during provenance capture')
    for name in names:
        if (dest/name).read_bytes()!=(source/name).read_bytes(): raise ValueError('prompt provenance changed during capture')

def capture_cell(session, root, endpoint, config, cell, tasks, task_bytes, matrix_bytes, attribution, router_log_path=None, cache_config_path=None):
    """Capture one cell inside the caller's still-owned supervised session.

    This writes request evidence only. Cleanup and independent acceptance remain
    the owning runner's responsibility before any attempt can pass.
    """
    router_source=Path(router_log_path) if router_log_path is not None else root/'router.log'
    route_start=router_source.stat().st_size if router_log_path is not None else 0
    task_sha=hashlib.sha256(task_bytes).hexdigest()
    timeout = config['request_timeout_seconds']
    if not isinstance(timeout, (int,float)) or timeout <= 0 or time.monotonic_ns() + int((timeout * (2 if cell['cache']=='warm_prefix' else 1) + 120)*1e9) > session.deadline_ns:
        raise runtime.RuntimeErrorClosed('cell deadline plus cleanup does not fit session deadline')
    runtime.atomic_bytes(root/'pilot-tasks.json',task_bytes)
    copy_prompt_provenance(config, root, task_bytes)
    runtime.atomic_bytes(root/'pilot-matrix.json',matrix_bytes)
    runtime.atomic_json(root/'pilot-cell.json', {'cell':cell, 'tasks_sha256':task_sha,
                                                 'matrix_sha256':hashlib.sha256(matrix_bytes).hexdigest(),
                                                 'cache_proof':'warm token prefix generated and rank/hit counters required' if cell['cache']=='warm_prefix' else 'cache-busted cold; flush and rank placement unverified'})
    metrics_urls = sorted(config['expected_ranks'])
    def snapshot():
        return {url:telemetry.scrape(url, list(PILOT_METRICS)) for url in metrics_urls}
    clock_id = clock_domain(session.run_id)
    condition_rank = None
    if cell['cache'] == 'warm_prefix':
        task=tasks[0]
        live=node_smoke.read(Path(cache_config_path) if cache_config_path is not None else root/'cache-config.json')
        if live['block_size'] != task['cache_block_size']:
            raise runtime.RuntimeErrorClosed('live cache block size differs from generated prefix')
        condition_before=snapshot(); runtime.atomic_json(root/'cache-conditioning-before.json',condition_before)
        condition_task={'id':task['id'],'body':task['conditioning_body'],'input_tokens':task['prefix_tokens'],
                        'output_tokens':128,'fixed_output':False,'expected_answer':task['expected_conditioning_answer']}
        condition_summary=pilot_burst.run(session, endpoint, [condition_task], root/'pilot-conditioning', timeout, clock_id)
        if condition_summary['valid_requests'] != 1: raise runtime.RuntimeErrorClosed('warm conditioning failed')
        limit=time.monotonic()+10; index=0
        while True:
            session.ensure_healthy(); condition_after=snapshot()
            runtime.atomic_json(root/('cache-conditioning-after-%02d.json'%index),condition_after)
            deltas=cache_evidence.differences(condition_before,condition_after,config['expected_ranks'])
            active=[x for x in deltas if x['queries']>0]
            if active: break
            if time.monotonic()>=limit: raise runtime.RuntimeErrorClosed('conditioning cache counters did not advance')
            index+=1; time.sleep(1)
        if len(active)!=1: raise runtime.RuntimeErrorClosed('conditioning cannot be attributed to one rank')
        condition_rank=(active[0]['endpoint'],active[0]['rank'])
        before=condition_after
    else:
        before=snapshot()
    runtime.atomic_json(root/'cache-before.json', before)
    output = root/'pilot-attempt'
    summary = pilot_burst.run(session, endpoint, tasks, output, timeout, clock_id)
    if summary['valid_requests'] != summary['requests']:
        raise runtime.RuntimeErrorClosed('pilot cell failed; retained raw attempt')
    observed_deadline = time.monotonic() + 10
    index = 0
    while True:
        session.ensure_healthy()
        after = snapshot()
        runtime.atomic_json(root/('cache-after-%02d.json' % index), after)
        delta = cache_evidence.differences(before, after, config['expected_ranks'])
        if sum(x['queries'] for x in delta) > 0: break
        if time.monotonic() >= observed_deadline: raise runtime.RuntimeErrorClosed('cache counters did not advance')
        index += 1; time.sleep(1)
    active = [x for x in delta if x['queries'] > 0]
    if cell['concurrency'] == 1 and len(active) != 1:
        raise runtime.RuntimeErrorClosed('single request cannot be attributed to exactly one rank')
    if condition_rank is not None and (len(active)!=1 or (active[0]['endpoint'],active[0]['rank'])!=condition_rank):
        raise runtime.RuntimeErrorClosed('warm follow-up did not use the conditioning rank')
    attribution.update({'rank_counter_deltas':delta, 'selected_rank':active[0] if len(active)==1 else None,
                        'per_request_rank':'observed by counter' if cell['concurrency']==1 else 'unverified',
                        'cache_label':'warm workload; observed hits recorded' if condition_rank else 'cache-busted cold; flush unverified',
                        'conditioning_rank':list(condition_rank) if condition_rank else None,
                        'hit_fraction':active[0]['hits']/active[0]['queries'] if condition_rank else None})
    route_bytes=router_source.read_bytes()
    route_end=len(route_bytes)
    if route_end<route_start:raise runtime.RuntimeErrorClosed('router log shrank during capture')
    raw_routes=route_bytes[route_start:route_end]
    route_text=raw_routes.decode('utf-8')
    runtime.atomic_bytes(root/'pilot-routing.log',raw_routes)
    if router_log_path is not None:
        runtime.atomic_json(root/'routing-log-span.json',{'start_byte':route_start,'end_byte':route_end,
            'sha256':hashlib.sha256(raw_routes).hexdigest(),
            'scope':'exact byte span within the retained shared router log'})
    route_proof=routing_evidence.verify(route_text,tasks,config['expected_ranks'],delta,condition_rank is not None)
    runtime.atomic_json(root/'routing-evidence.json',route_proof)
    if cell['concurrency']>1:
        attribution['per_request_rank']='native router-selected rank; aggregate counters agree'
    runtime.atomic_json(root/'pilot-attribution.json', attribution)

def validate_rank_contract(config,cell):
    tp,dp,_=launcher.LAYOUTS[cell['layout']]
    urls={'http://127.0.0.1:'+str(31000+i)+'/metrics' for i in range(8//(tp*dp))}
    ranks=config.get('expected_ranks',{})
    if (set(ranks)!=urls or any(len(values)!=dp or set(map(str,values))!=set(map(str,range(dp))) for values in ranks.values())):
        raise ValueError('pilot requires explicit per-engine rank coverage matching installed geometry')

def run(config):
    task_bytes = verified_tasks(config)
    tasks = json.loads(task_bytes)
    matrix_bytes = Path(config['matrix']).read_bytes()
    if matrix_bytes != (campaign.HERE/'matrix.json').read_bytes():
        raise ValueError('matrix differs from the campaign source')
    cell = prepare(config['matrix'], config['cell_id'], tasks)
    if Path(config['matrix']).read_bytes() != matrix_bytes:
        raise ValueError('matrix changed during expansion')
    if config['layout'] != cell['layout']: raise ValueError('layout mismatch')
    if not PILOT_METRICS.issubset(set(config.get('required_metric_names', []))):
        raise ValueError('required pilot metric contract incomplete')
    validate_rank_contract(config,cell)
    task_sha = hashlib.sha256(task_bytes).hexdigest()
    node_smoke.validate(config)
    attempt=pilot_registry.begin(config,task_sha)
    attribution = {}
    def execute(session, root, endpoint):
        capture_cell(session, root, endpoint, config, cell, tasks, task_bytes, matrix_bytes, attribution)
    try:
        node_smoke.run(config, execute)
        result_path=Path(config['run_dir'])/'pilot-result.json'
        evidence_root=Path(config['run_dir'])
        qualified = ((evidence_root/'rank-placement.json').exists() and
                     (evidence_root/'backend-selection.json').exists() and
                     node_smoke.read(evidence_root/'rank-placement.json').get('status')=='passed' and
                     node_smoke.read(evidence_root/'backend-selection.json').get('status')=='configured_backend_confirmed' and
                     node_smoke.read(evidence_root/'routing-evidence.json').get('status')=='router_selection_and_rank_counters_verified')
        runtime.atomic_json(result_path,
                            {'cell_id':cell['id'], 'status':'passed' if qualified else 'invalid',
                             'requests_status':'passed', 'tasks_sha256':task_sha, 'attribution':attribution,
                             'scope':'feasibility pilot observation; not scored performance or a frozen-profile qualification',
                             'reason':None if qualified else 'installed backend engagement and per-request rank evidence are not fully verified'})
        runtime.seal_evidence(config['run_dir'])
        verification=verify_pilot.verify(config['run_dir'])
        pilot_registry.finish(attempt,result_path,verification)
    except Exception as exc:
        pilot_registry.fail(attempt,exc)
        raise
