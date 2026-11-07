"""Independently verify a completed pilot against its saved task contract."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import verify_wire
import verify_telemetry

LINE = re.compile(r'^(vllm:prefix_cache_(?:queries|hits)_total)\{([^}]*)\}\s+([^\s]+)')
ENGINE = re.compile(r'(?:^|,)\s*engine="([^"]+)"(?:,|$)')

def read(path): return json.loads(Path(path).read_text())

def verify_block_provenance(root,config):
    path=root/'block-provenance.json'
    if not path.exists():
        if (root/'shared-run-config.json').exists():raise ValueError('shared session provenance missing')
        return
    proof=read(path);shared=read(root/'shared-run-config.json');state=read(root/'execution-state.json')
    files=proof.get('shared_file_sha256',{})
    required={'shared-run-config.json','execution-state.json','cleanup-result.json','gpu-idle-after-cleanup.json',
              'manifest.json','startup-timing.json','rank-placement.json','backend-selection.json','router.log'}
    if (proof.get('schema')!='shared-pilot-session/v1' or not required.issubset(files) or
        proof.get('run_id')!=config.get('run_id') or shared.get('run_id')!=config.get('run_id') or
        state.get('run_id')!=config.get('run_id') or proof.get('driver_identity')!=state.get('driver_identity')):
        raise ValueError('shared session ownership or file coverage differs')
    for name,digest in files.items():
        rel=Path(name)
        allowed=(len(rel.parts)==1 or len(rel.parts)==2 and rel.parts[0]=='kv-baseline-source')
        if (not allowed or rel.is_absolute() or '..' in rel.parts or rel.as_posix()!=name or
            not re.fullmatch(r'[0-9a-f]{64}',str(digest)) or
            hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest):
            raise ValueError('shared session raw record changed')
    entries=shared.get('cells',[])
    chosen=[e for e in entries if e.get('cell_id')==config.get('cell_id')]
    if (len({e.get('cell_id') for e in entries})!=len(entries) or len(chosen)!=1 or
        any(chosen[0].get(k)!=config.get(k) for k in ('tasks_file','request_timeout_seconds','prompt_reuse')) or
        any(shared.get(k)!=config.get(k) for k in ('layout','image','run_dir','manifest','model_dir'))):
        raise ValueError('cell configuration differs from registered shared session')

def verify_cleanup(root,config):
    state=read(root/'execution-state.json');receipt=read(root/'cleanup-result.json')
    ids=state.get('container_ids');elapsed=receipt.get('elapsed_seconds')
    if (not isinstance(ids,list) or not ids or len(set(ids))!=len(ids) or
        any(not re.fullmatch(r'[0-9a-f]{64}',str(cid)) for cid in ids) or
        state.get('run_id')!=config.get('run_id') or state.get('stopping') is not True or
        state.get('watchdog_stop') is not True or receipt.get('success') is not True or
        receipt.get('deadline_seconds')!=120 or type(elapsed) not in (int,float) or
        not math.isfinite(elapsed) or not 0<=elapsed<=120 or
        read(root/'gpu-idle-after-cleanup.json').get('idle') is not True):
        raise ValueError('pilot cleanup did not independently satisfy the shared deadline and idle contract')
    outcomes=receipt.get('outcomes',[])
    containers=[x for x in outcomes if 'container_id' in x]
    processes=[x for x in outcomes if 'pid' in x]
    if (not outcomes or any(x.get('status')!='stopped' for x in outcomes) or
        len(containers)!=len(ids) or {x['container_id'] for x in containers}!=set(ids) or
        any(x.get('log_status')!='captured' for x in containers) or
        len(processes)!=len(state.get('owned_processes',[])) or
        {x['pid'] for x in processes}!={x['pid'] for x in state.get('owned_processes',[])}):
        raise ValueError('pilot cleanup outcome coverage differs from exact owned identities')
    for name in ('watchdog-failure.json','driver-failure.json','smoke-failure.json','primary-failure.json'):
        if (root/name).exists():raise ValueError('pilot has retained failure evidence; recovery is not an original pass')
    idle=read(root/'gpu-idle-after-cleanup.json')
    manifest=read(root/'manifest.json') if (root/'manifest.json').exists() else {}
    shutdown_schema='owned-cleanup-gpu-proof/v1'
    if (state.get('safety_backend')=='nvml' and manifest.get('runtime',{}).get('shutdown_safety_schema')==shutdown_schema and
        state.get('shutdown_safety_schema')!=shutdown_schema):raise ValueError('shutdown safety phase binding missing')
    if state.get('shutdown_safety_schema')==shutdown_schema:
        initial=read(root/'nvml-initial-safety.json');baseline=state.get('nvml_ecc_baseline')
        if (not isinstance(baseline,dict) or set(baseline)!=set(map(str,range(8))) or
            any(type(v) is not int or v<0 for v in baseline.values()) or
            initial.get('pid')!=state.get('nvml_identity',{}).get('pid') or
            len(initial.get('devices',[]))!=8 or {d.get('index') for d in initial['devices']}!=set(range(8)) or
            {str(d['index']):d.get('volatile_uncorrected_ecc') for d in initial['devices']}!=baseline):
            raise ValueError('shutdown ECC activation baseline differs')
        found=False
        for path in sorted(root.glob('nvml-samples-*.jsonl')):
            with path.open() as file:
                for line in file:
                    if json.loads(line)==initial:found=True;break
            if found:break
        if not found:raise ValueError('shutdown ECC baseline lacks retained raw sample')
    budget_path=root/'gpu-drain-budget.json'
    needs_budget=(state.get('safety_backend')=='nvml' and manifest.get('runtime',{}).get('cleanup_drain_schema')=='remaining-shared-cleanup-deadline/v1')
    if needs_budget and not budget_path.exists():raise ValueError('pilot shared GPU drain budget missing')
    if budget_path.exists():
        budget=read(budget_path);started=budget.get('started_ns');drain_end=budget.get('drain_deadline_ns');cleanup_end=budget.get('cleanup_deadline_ns')
        if (budget.get('schema')!='remaining-shared-cleanup-deadline/v1' or budget.get('run_id')!=state.get('run_id') or
            any(type(v) is not int for v in (started,drain_end,cleanup_end)) or not started<drain_end<cleanup_end or
            cleanup_end!=state.get('cleanup_deadline_ns') or budget.get('collector_release_reserve_seconds')!=20 or
            abs(cleanup_end-drain_end-20_000_000_000)>1000 or receipt['monotonic_ns']>cleanup_end):
            raise ValueError('pilot GPU drain budget exceeds the shared cleanup contract')
    if idle.get('source')=='isolated-nvml-compute-and-memory':
        # Recompute the raw drain predicate without importing the collector or
        # cleanup implementation. A boolean cleanup receipt is not this proof.
        proof=read(root/'gpu-drain-proof.json');request=read(root/'nvml-drain-request.json')
        data=proof.get('sample',{});rows=data.get('devices',[])
        times=[request.get('requested_ns'),proof.get('after_stopped_ns'),
               data.get('sample_started_ns'),data.get('monotonic_ns'),
               proof.get('observed_ns'),idle.get('monotonic_ns')]
        if (any(type(t) is not int for t in times) or times!=sorted(times) or times[-1]-times[2]>10_000_000_000 or
            proof.get('run_id')!=state['run_id'] or request.get('run_id')!=state['run_id'] or
            proof.get('identity')!=state.get('nvml_identity') or request.get('identity')!=state.get('nvml_identity') or
            data.get('pid')!=state.get('nvml_identity',{}).get('pid') or
            proof.get('source')!=idle['source'] or idle.get('sample_ns')!=data.get('monotonic_ns') or
            len(rows)!=8 or {r.get('index') for r in rows}!=set(range(8)) or len({r.get('uuid') for r in rows})!=8 or
            any(not r.get('uuid') or type(r.get('compute_process_count')) is not int or r['compute_process_count']!=0 or
                type(r.get('used_bytes')) is not int or not 0<=r['used_bytes']<=2048*1024**2 or
                type(r.get('total_bytes')) is not int or r['total_bytes']<=0 or
                type(r.get('volatile_uncorrected_ecc')) is not int or r['volatile_uncorrected_ecc']<0 for r in rows)):
            raise ValueError('pilot raw NVML drain proof is incomplete, stale or not idle')
        if state.get('nvml_ecc_baseline') is not None and (
            {str(d['index']):d['volatile_uncorrected_ecc'] for d in rows}!=state['nvml_ecc_baseline'] or
            {d['uuid'] for d in rows}!={d['uuid'] for d in read(root/'nvml-initial-safety.json')['devices']}):
            raise ValueError('final drain ECC/device identity differs from activation baseline')
    elif state.get('safety_backend')=='nvml' and (root/'nvml-drain-request.json').exists():
        raise ValueError('pilot NVML drain request lacks matching raw idle proof')

def verify_kv_budget(root, config):
    """Check raw group-aware capacity without importing its producer."""
    def profile_hash(manifest,cfg):
        if cfg.get('execution_phase','pilot')=='pilot':return manifest['runtime']['starting_profiles'][cfg['layout']]
        phase=cfg.get('execution_phase')
        if phase=='fixed-performance':
            record=manifest['runtime'].get('fixed_profiles',{}).get(cfg['layout'],{})
            if (record.get('schema')!='frozen-fixed-profile/v1' or record.get('id')!=cfg.get('profile_id') or
                record.get('status')!='frozen_fixed_configuration'):
                raise ValueError('KV frozen profile binding differs')
        elif phase=='qualification':record=manifest['runtime'].get('qualification_profiles',{}).get(cfg.get('profile_id'),{})
        else:raise ValueError('unknown KV profile phase')
        if record.get('layout')!=cfg['layout'] or not re.fullmatch(r'[0-9a-f]{64}',str(record.get('profile_sha256'))):
            raise ValueError('KV candidate profile binding differs')
        return record['profile_sha256']
    def raw_record(directory,cfg):
        state=read(directory/'execution-state.json');plan=read(directory/'launch-plan.json');manifest=read(directory/'manifest.json')
        ids=state['container_ids'];layout=cfg['layout'];dp={'dep8':8,'4xdep2':2,'2xdep4':4}.get(layout,1)
        if (not ids or len(ids)!=len(set(ids)) or any(not re.fullmatch(r'[0-9a-f]{64}',str(cid)) for cid in ids) or
            len(ids)!=len(plan['commands']) or state['run_id']!=cfg['run_id']):raise ValueError('raw KV ownership differs')
        entries={};replicas=[]
        for cid,command in zip(ids,plan['commands']):
            data=(directory/('rank-startup-'+cid+'.log')).read_bytes();engines=[]
            for line in data.decode('utf-8',errors='replace').splitlines():
                if ' KV cache size:' not in line:continue
                match=re.search(r'\(EngineCore(?:_DP(\d+))? pid=\d+\).*?(GPU(?: \+ host)?) KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x',line)
                if not match:raise ValueError('raw KV capacity line malformed')
                if any(not re.fullmatch(r'(?:[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)',match[k]) for k in (3,4)):
                    raise ValueError('raw KV capacity integer malformed')
                rank=int(match[1] or 0);tokens=int(match[3].replace(',',''));limit=int(match[4].replace(',',''));maximum=float(match[5])
                if limit!=1048576 or tokens<=0 or not math.isfinite(maximum) or abs(tokens/limit-maximum)>.00501+1/limit:
                    raise ValueError('raw KV token capacity and concurrency disagree')
                engine={'logical_rank':rank,'pool':match[2],'equivalent_full_context_tokens':tokens,
                        'max_request_tokens':limit,'max_concurrency_at_limit':maximum}
                key=(tuple(command['gpu_ids']),rank)
                if key in entries:raise ValueError('raw KV physical engine duplicated')
                entries[key]=engine;engines.append(engine)
            if len(engines)!=dp or {e['logical_rank'] for e in engines}!=set(range(dp)):
                raise ValueError('raw KV logical coverage differs')
            replicas.append({'container_id':cid,'gpu_ids':command['gpu_ids'],
                'startup_log_sha256':hashlib.sha256(data).hexdigest(),'engines':sorted(engines,key=lambda e:e['logical_rank'])})
        return manifest,replicas,entries
    manifest,replicas,current=raw_record(root,config);budget=read(root/'kv-budget.json')
    if (budget.get('schema')!='group-aware-kv-capacity/v1' or budget.get('layout')!=config['layout'] or
        budget.get('image')!=config['image'] or budget.get('profile_sha256')!=profile_hash(manifest,config) or
        budget.get('installed_source_sha256')!='195159dd716e8fa2e597d94625d79618e025218ca2baffa38cb2e5977516b9e3' or
        budget.get('replicas')!=replicas):raise ValueError('KV capacity receipt differs from raw startup evidence')
    comparison=read(root/'kv-budget-comparison.json')
    if config.get('kv_baseline_evidence') is None:
        if comparison.get('status')!='initial_observation':raise ValueError('KV initial observation mislabeled')
        return
    source=root/'kv-baseline-source';sha=(source/'SOURCE_SHA256SUMS').read_bytes()
    listed={}
    for line in sha.decode().splitlines():
        digest,sep,name=line.partition('  ')
        if (not sep or not re.fullmatch(r'[0-9a-f]{64}',digest) or name in listed or
            Path(name).is_absolute() or '..' in Path(name).parts):raise ValueError('KV baseline source manifest malformed')
        listed[name]=digest
    oldconfig=read(source/'run-config.json');oldmanifest,_,old=raw_record(source,oldconfig)
    names={'run-config.json','manifest.json','execution-state.json','launch-plan.json'}
    names|={'rank-startup-'+c+'.log' for c in read(source/'execution-state.json')['container_ids']}
    if ({p.name for p in source.iterdir()}!=names|{'SOURCE_SHA256SUMS'} or
        any(p.is_symlink() or not p.is_file() for p in source.iterdir()) or
        any(listed.get(name)!=hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names) or
        hashlib.sha256(sha).hexdigest()!=comparison.get('baseline_manifest_sha256') or
        oldconfig['image']!=config['image'] or oldconfig['layout']!=config['layout'] or
        profile_hash(oldmanifest,oldconfig)!=budget['profile_sha256'] or old.keys()!=current.keys()):
        raise ValueError('KV baseline excerpt ownership, profile or integrity differs')
    changes=[]
    for key in sorted(old):
        if any(current[key][k]!=old[key][k] for k in ('pool','max_request_tokens')):raise ValueError('KV baseline pool changed')
        change=abs(current[key]['equivalent_full_context_tokens']-old[key]['equivalent_full_context_tokens'])/old[key]['equivalent_full_context_tokens']
        if change>.02:raise ValueError('KV baseline drift exceeds two percent')
        changes.append({'gpu_ids':list(key[0]),'logical_rank':key[1],'relative_change':change})
    if comparison.get('status')!='within_two_percent' or comparison.get('changes')!=changes:
        raise ValueError('KV baseline comparison differs from raw records')


def verify_single_request_placement(root, config, result, *, expected_profile=None, require_selected_rank=True):
    """Recompute the physical worker join from retained raw records, independently."""
    layouts={'dep8':(1,8,True,'FLASHMLA_MEGA_ATTN_DSV41'),
             '4xdep2':(1,2,True,'FLASHINFER_MLA_SPARSE_DSV41'),
             '2xdep4':(1,4,True,'FLASHINFER_MLA_SPARSE_DSV41'),
             '2xtp4':(4,1,False,'FLASHINFER_MLA_SPARSE_DSV41'),
             '4xtp2':(2,1,False,'FLASHINFER_MLA_SPARSE_DSV41'),
             'tp8':(8,1,False,'FLASHINFER_MLA_SPARSE_DSV41')}
    if config['layout'] not in layouts: raise ValueError('pilot physical placement is not implemented for this layout')
    tp,dp,expert,backend=layouts[config['layout']]
    dtype='fp8_ds_mla' if config['layout']=='dep8' else 'fp8';spec=None;moe='deep_gemm_mega_moe'
    if expected_profile is not None:
        if config.get('execution_phase') not in ('qualification','fixed-performance') or set(expected_profile)!={'backend','kv_dtype','speculative_config','moe_backend'}:
            raise ValueError('registered startup profile scope differs')
        backend=expected_profile['backend'];dtype=expected_profile['kv_dtype'];spec=expected_profile['speculative_config'];moe=expected_profile['moe_backend']
    width=tp*dp;axis='dp' if dp>1 else 'tp'
    state=read(root/'execution-state.json'); ids=state['container_ids']
    if len(ids)!=8//width or len(set(ids))!=len(ids) or any(not re.fullmatch(r'[0-9a-f]{64}',c) for c in ids):
        raise ValueError('pilot container coverage mismatch')
    snapshot=read(root/'rank-process-snapshot.json')
    devices=read(root/'nvml-live.json')['devices']
    device_map={d['uuid']:d['index'] for d in devices}
    if len(device_map)!=8 or set(device_map.values())!=set(range(8)):
        raise ValueError('pilot device coverage mismatch')
    placement=read(root/'rank-placement.json'); proof=read(root/'backend-selection.json')
    if (placement.get('status')!='passed' or proof.get('status')!='configured_backend_confirmed' or
        {x['container_id'] for x in placement['containers']}!=set(ids) or
        {x['container_id'] for x in proof['containers']}!=set(ids)):
        raise ValueError('pilot qualification receipts incomplete')
    for index,cid in enumerate(ids):
        text=(root/('rank-startup-'+cid+'.log')).read_text()
        configs=[json.JSONDecoder().raw_decode(l.split('--args-json ',1)[1])[0]
                 for l in text.splitlines() if 'Launching Rust frontend:' in l and '--args-json ' in l]
        if len(configs)!=1: raise ValueError('missing installed backend configuration')
        installed=configs[0]
        actual_spec=installed.get('speculative_config')
        if spec is None and actual_spec is not None:
            raise ValueError('pilot must use the no-speculation starting profile')
        if spec is not None and (not isinstance(actual_spec,dict) or actual_spec.get('method')!='dspark' or
            actual_spec.get('rejection_sample_method')!='block' or
            any(re.search(r'golden|synthetic|forced.accept|fake.accept',k,re.I) for k in actual_spec) or
            any(k not in actual_spec or type(actual_spec[k]) is not type(v) or actual_spec[k]!=v for k,v in spec.items())):
            raise ValueError('installed real-block candidate differs from registered profile')
        if (not str(installed.get('attention_config',{}).get('backend','')).startswith('<AttentionBackendEnum.'+backend+':') or
            installed.get('attention_config',{}).get('indexer_kv_dtype')!='mxfp4' or
            (expert and (installed.get('kernel_config',{}).get('moe_backend')!=moe or
                         installed.get('enable_expert_parallel') is not True)) or
            (not expert and installed.get('enable_expert_parallel',False) is not False) or
            any(k in installed and installed[k]!=v for k,v in
                {'data_parallel_size':dp,'data_parallel_size_local':dp,'tensor_parallel_size':tp}.items())):
            raise ValueError('installed backend configuration mismatch')
        if installed.get('kv_cache_dtype')!=dtype:
            raise ValueError('installed KV precision mismatch')
        if set(re.findall(r'world_size=(\d+)\s+rank=',text))!={str(width)}:
            raise ValueError('installed world size mismatch')
        packed=dtype
        if backend=='FLASHMLA_MEGA_ATTN_DSV41' and set(re.findall(r"Using DeepSeek's ([a-z0-9_]+) KV cache format",text))!={packed}:
            raise ValueError('pilot packed precision mismatch')
        workers={}
        pattern=(r'\(Worker_DP(\d+)_EP\d+\s+pid=(\d+)\)' if axis=='dp' else
                 r'\(Worker_TP(\d+)\s+pid=(\d+)\)')
        for rank,pid,size,selected in re.findall(pattern+r'.*?Setting kv cache block size to (\d+) for ([^/\s]+)',text):
            value=(int(pid),int(size),selected)
            if int(rank) in workers and workers[int(rank)]!=value: raise ValueError('worker incarnation changed')
            workers[int(rank)]=value
        if set(workers)!=set(range(width)) or any(v[1]<=0 or v[2]!=backend for v in workers.values()):
            raise ValueError('worker backend coverage mismatch')
        owned={int(x) for x in (root/('rank-top-'+cid+'.txt')).read_text().splitlines() if x.strip().isdigit()}
        observed=next(x['workers'] for x in placement['containers'] if x['container_id']==cid)
        if len(observed)!=width or {x[axis+'_rank'] for x in observed}!=set(range(width)):
            raise ValueError('physical worker receipt coverage mismatch')
        physical=[]
        for rank,(namespace,_,_) in workers.items():
            matches=[int(pid) for pid,row in snapshot['processes'].items() if int(pid) in owned and row.get('namespace_pid')==namespace]
            if len(matches)!=1: raise ValueError('ambiguous namespace ownership')
            host=matches[0]; gpu=[x['gpu_uuid'] for x in snapshot['gpu_processes'] if x['pid']==host]
            if len(gpu)!=1 or gpu[0] not in device_map: raise ValueError('ambiguous physical GPU assignment')
            row=next(x for x in observed if x[axis+'_rank']==rank)
            if row!={axis+'_rank':rank,'namespace_pid':namespace,'host_pid':host,
                     'start_ticks':snapshot['processes'][str(host)]['start_ticks'],
                     'gpu_uuid':gpu[0],'gpu_index':device_map[gpu[0]]}:
                raise ValueError('physical worker receipt does not match raw join')
            physical.append(device_map[gpu[0]])
        if set(physical)!=set(range(index*width,(index+1)*width)):
            raise ValueError('physical partition differs from candidate')
    chosen=result['attribution'].get('selected_rank')
    if require_selected_rank and config.get('concurrency',1)==1 and (not chosen or chosen['endpoint'] not in config['expected_ranks'] or str(chosen['rank']) not in {str(x) for x in config['expected_ranks'][chosen['endpoint']]}):
        raise ValueError('single request lacks a physically mapped counter rank')

def verify_routes(root,tasks,config,deltas,conditioning=False):
    """Independently parse native route records and recompute rank token totals."""
    raw=(root/'pilot-routing.log').read_bytes()
    full=(root/'router.log').read_bytes()
    if (root/'routing-log-span.json').exists():
        span=read(root/'routing-log-span.json');start=span.get('start_byte');end=span.get('end_byte')
        if (type(start) is not int or type(end) is not int or not 0<=start<=end<=len(full) or
            full[start:end]!=raw or hashlib.sha256(raw).hexdigest()!=span.get('sha256')):
            raise ValueError('route snapshot does not match retained log span')
    elif not full.startswith(raw):raise ValueError('route snapshot is not a prefix of retained router log')
    text=re.sub(r'\x1b\[[0-9;]*m','',raw.decode())
    observed={task['id']:[] for task in tasks};ranks=config['expected_ranks']
    for line in text.splitlines():
        if "Consistent hash routing: key='header:x-session-id:" not in line:continue
        m=re.search(r"Consistent hash routing: key='header:x-session-id:([^']+)' -> worker='(http://127\.0\.0\.1:[0-9]+)(?:@([0-9]+))?' \(index=[0-9]+\)",line)
        if not m:raise ValueError('malformed native router selection')
        session,url,rank=m.groups();endpoint=url+'/metrics'
        if session not in observed or endpoint not in ranks:raise ValueError('route outside measured sessions or owned engines')
        if rank is None:
            if set(map(str,ranks[endpoint]))!={'0'}:raise ValueError('missing distributed route rank')
            rank='0'
        if rank not in set(map(str,ranks[endpoint])):raise ValueError('unexpected route rank')
        observed[session].append((endpoint,rank))
    records=[];counts={(url,str(rank)):0 for url,values in ranks.items() for rank in values}
    for task in sorted(tasks,key=lambda t:t['id']):
        choices=observed[task['id']]
        if len(choices)!=(2 if conditioning else 1) or len(set(choices))!=1:raise ValueError('route missing, duplicated or changed')
        endpoint,rank=choices[0];counts[(endpoint,rank)]+=task['input_tokens']
        records.append({'request_id':task['id'],'session_id':task['id'],'endpoint':endpoint,'rank':rank})
    actual={(d['endpoint'],str(d['rank'])):d['queries'] for d in deltas}
    if len(actual)!=len(deltas) or set(actual)!=set(counts):raise ValueError('route counter coverage differs')
    for key,count in counts.items():
        tolerance=max(32 if conditioning else 16,count*.0025)
        if (count==0 and actual[key]!=0) or (count>0 and abs(actual[key]-count)>tolerance):raise ValueError('route and rank query totals disagree')
    proof=read(root/'routing-evidence.json')
    if proof.get('status')!='router_selection_and_rank_counters_verified' or proof.get('requests')!=records:
        raise ValueError('routing receipt differs from raw selections')
    return records


def parse_cache(snapshot, ranks):
    if set(snapshot) != set(ranks): raise ValueError('cache endpoint coverage mismatch')
    found={}
    for url,text in snapshot.items():
        for line in text.splitlines():
            match=LINE.match(line)
            if not match: continue
            engine=ENGINE.search(match.group(2))
            if engine is None: raise ValueError('missing cache engine label')
            key=(url,engine.group(1),match.group(1))
            value=float(match.group(3))
            if key in found or not math.isfinite(value) or value < 0: raise ValueError('invalid cache counter')
            found[key]=value
    expected={(url,str(rank),'vllm:prefix_cache_'+kind+'_total') for url,values in ranks.items() for rank in values for kind in ('queries','hits')}
    if set(found) != expected: raise ValueError('missing cache counter')
    return found

def verify_prompt_provenance(root, tasks, config, cell_id):
    source=Path(root)/'prompt-provenance'; listed={}
    for line in (source/'SHA256SUMS').read_text().splitlines():
        digest,sep,name=line.partition('  '); rel=Path(name)
        if not sep or not re.fullmatch(r'[0-9a-f]{64}',digest) or rel.is_absolute() or '..' in rel.parts or name in listed:
            raise ValueError('invalid prompt provenance manifest')
        listed[name]=digest
    values={'pilot-tasks.json':tasks}
    for name in ('run-config.json','prompt-result.json','manifest.json','checkpoint-receipt.json','image-qualification.json'):
        values[name]=(source/name).read_bytes()
    if any(listed.get(name)!=hashlib.sha256(data).hexdigest() for name,data in values.items()):
        raise ValueError('prompt provenance hash mismatch')
    for name in ('manifest.json','checkpoint-receipt.json','image-qualification.json'):
        if name=='manifest.json' and config.get('prompt_reuse') is not None:continue
        if values[name]!=(Path(root)/name).read_bytes(): raise ValueError('prompt dependency does not match pilot')
    previous=json.loads(values['run-config.json']); result=json.loads(values['prompt-result.json'])
    expected_cell=cell_id;expected_layout=config.get('layout')
    if config.get('prompt_reuse') is not None:
        reuse=config['prompt_reuse'];old=json.loads(values['manifest.json']);new=read(Path(root)/'manifest.json')
        if (not isinstance(reuse,dict) or set(reuse)!={'source_layout','toolkit_manifest','toolkit_sha256'} or
                reuse['source_layout']==config.get('layout') or previous.get('prompt_reuse') is not None or
                config.get('execution_phase','pilot')!='pilot' or previous.get('execution_phase','pilot')!='pilot' or
                previous.get('task_family','pilot')!='pilot' or config.get('task_family','pilot')!='pilot'):
            raise ValueError('invalid direct pilot reuse declaration')
        original=[c for c in old['cells'] if c['id']==previous['cell_id']]
        target=[c for c in new['cells'] if c['id']==cell_id]
        if (len(original)!=1 or len(target)!=1 or original[0]['layout']!=reuse['source_layout'] or
                target[0]['layout']!=config['layout'] or not original[0]['suite'].startswith('pilot_') or
                {k:v for k,v in original[0].items() if k not in ('id','layout')}!=
                {k:v for k,v in target[0].items() if k not in ('id','layout')}):
            raise ValueError('reused pilot shapes differ')
        plan_pair=(old.get('plan_sha256'),new.get('plan_sha256'))
        approved_pair=('fff4c820e94927c3924952a9c5ed9cfaed453e8e1f60d27b1c7118730686947b',
                       'f97c296a68f91530d14558306a850ec424577611366060c981876cf115f6c449')
        if ((plan_pair[0]!=plan_pair[1] and plan_pair!=approved_pair) or
                old.get('matrix_sha256')!=new.get('matrix_sha256') or
                any(old['runtime'].get(k) is None or new['runtime'].get(k) is None or old['runtime'].get(k)!=new['runtime'].get(k) for k in
                    ('image_digest','model_revision','tokenizer_revision','sampling_source','temperature','top_p')) or
                any(old['runtime']['starting_profiles'].get(k)!=new['runtime']['starting_profiles'].get(k)
                    for k in (config['layout'],reuse['source_layout']))):
            raise ValueError('reused pilot encoding dependencies differ')
        old_bytes=(source/'source-toolkit-MANIFEST.json').read_bytes();new_bytes=(source/'target-toolkit-MANIFEST.json').read_bytes()
        if (hashlib.sha256(old_bytes).hexdigest()!=reuse['toolkit_sha256'] or
                reuse['toolkit_sha256']!=old['runtime']['toolkit_manifest_sha256'] or
                hashlib.sha256(new_bytes).hexdigest()!=new['runtime']['toolkit_manifest_sha256']):
            raise ValueError('reused toolkit proof differs')
        old_files=json.loads(old_bytes);new_files=json.loads(new_bytes);names=('reproduce/vllm_prompt.py','reproduce/workload.py')
        if target[0]['cache']=='warm_prefix':names+=('reproduce/warm_pilot_prompt.py',)
        if any(not re.fullmatch(r'[a-f0-9]{64}',str(old_files.get(k))) or old_files.get(k)!=new_files.get(k) for k in names):
            raise ValueError('reused tokenizer generator code differs')
        expected_cell=previous['cell_id'];expected_layout=reuse['source_layout']
        proof={'source_cell_id':expected_cell,'target_cell_id':cell_id,'generator_sha256':{k:old_files[k] for k in names},
               'scope':'unchanged tokenizer inputs shared across pilot layouts; no new generation or inference acceptance'}
        if plan_pair[0]!=plan_pair[1]:
            proof['plan_supersession']={'source_sha256':plan_pair[0],'target_sha256':plan_pair[1],
                'authority':'owner-priority-amendment-20261102','workload_and_acceptance_unchanged':True}
        if read(source/'prompt-reuse.json')!=proof:raise ValueError('pilot reuse binding differs')
    if (previous.get('cell_id')!=expected_cell or previous.get('layout')!=expected_layout or previous.get('image')!=config.get('image') or
        result.get('status')!='passed' or result.get('cleanup')!='passed' or result.get('cell_id')!=expected_cell or
        result.get('tasks')!=len(json.loads(tasks))):
        raise ValueError('prompt provenance contract mismatch')

def verify(root):
    root = Path(root)
    listed = {}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, sep, name = line.partition('  ')
        rel = Path(name)
        if not sep or not re.fullmatch(r'[0-9a-f]{64}',digest) or rel.is_absolute() or '..' in rel.parts or name in listed:
            raise ValueError('invalid evidence checksum entry')
        listed[name] = digest
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('symlink in evidence')
        if path.is_file() and path != root/'SHA256SUMS': actual.add(path.relative_to(root).as_posix())
    if actual != set(listed): raise ValueError('evidence file coverage mismatch')
    for name, expected in listed.items():
        h=hashlib.sha256()
        with (root/name).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        if h.hexdigest() != expected: raise ValueError('evidence checksum mismatch')
    config=read(root/'run-config.json')
    if config.get('execution_phase','pilot')!='pilot':
        raise ValueError('pilot verifier cannot certify a qualification profile')
    meta,result=read(root/'pilot-cell.json'),read(root/'pilot-result.json')
    verify_block_provenance(root,config)
    capture=read(root/'manifest.json').get('runtime',{}).get('capture_schema')
    if capture not in (None,'http-body-chunks/v1'):raise ValueError('unsupported capture schema')
    cell=meta['cell']
    matrix_bytes=(root/'pilot-matrix.json').read_bytes()
    if hashlib.sha256(matrix_bytes).hexdigest() != meta['matrix_sha256'] or matrix_bytes != Path(__file__).with_name('matrix.json').read_bytes():
        raise ValueError('pilot matrix does not match source')
    matrix=json.loads(matrix_bytes)
    tops=matrix['topologies'] + matrix['anchor_only_topologies']
    suite=cell['suite']; layout=cell['layout']; n=cell['input_tokens']; concurrency=cell['concurrency']; cache=cell['cache']; out=cell['output_tokens']
    valid_shape = ((suite=='pilot_smoke' and layout in tops and n==8000 and concurrency==1 and cache=='cold' and out==128) or
                   (suite=='pilot_fit' and layout in matrix['topologies'] and n in (128000,512000,1000000) and concurrency==1 and cache=='cold' and out==512) or
                   (suite=='pilot_concurrency' and layout in matrix['topologies'] and n in (128000,1000000) and concurrency in (8,32) and cache=='cold' and out==512) or
                   (suite=='pilot_warm' and layout in matrix['topologies'] and n==128000 and concurrency==1 and cache=='warm_prefix' and out==512))
    if not valid_shape or cell['id'] != f'{suite}-{layout}-i{n}-c{concurrency}-{cache}-o{out}' or config['layout']!=layout or config.get('cell_id')!=cell['id']:
        raise ValueError('pilot cell does not match planned shape')
    task_bytes=(root/'pilot-tasks.json').read_bytes(); tasks=json.loads(task_bytes)
    if hashlib.sha256(task_bytes).hexdigest() != result['tasks_sha256'] or result['tasks_sha256'] != meta['tasks_sha256'] or result['cell_id'] != cell['id']:
        raise ValueError('pilot contract mismatch')
    verify_prompt_provenance(root, task_bytes, config, cell['id'])
    if len(tasks) != cell['concurrency'] or len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('pilot task coverage mismatch')
    rows=[]
    for task in tasks:
        if not re.fullmatch(r'[A-Za-z0-9]+',task['id']): raise ValueError('unsafe task id')
        session=task['session']; seed=int.from_bytes(hashlib.sha256(json.dumps([suite,n,concurrency,0,session],separators=(',',':')).encode()).digest()[:8],'big') & 0x7fffffff
        body=task['body']; fixed=suite != 'pilot_smoke'
        if (not isinstance(session,int) or not 0<=session<concurrency or task['id']!='s'+str(session) or
            task['seed']!=seed or body.get('seed')!=seed or task['suite']!=suite or task['output_tokens']!=out or
            abs(task['input_tokens']-n)>n*.0025 or task['fixed_output'] != fixed or body.get('model')!='/model' or
            body.get('max_tokens')!=out or body.get('temperature')!=1.0 or body.get('top_p')!=1.0 or
            body.get('stream') is not True or body.get('stream_options')!={'include_usage':True} or
            body.get('chat_template_kwargs')!={'thinking':False,'reasoning_effort':50}):
            raise ValueError('task contract mismatch')
        if fixed and (body.get('min_tokens')!=out or body.get('ignore_eos') is not True): raise ValueError('fixed output mismatch')
        if not fixed and ('min_tokens' in body or body.get('ignore_eos')): raise ValueError('natural stop mismatch')
        marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        if task['expected_answer']!=marker or not any(marker in str(m.get('content',''))[:512] for m in body['messages']):
            raise ValueError('seeded answer missing from prompt')
        row=read(root/'pilot-attempt'/(task['id']+'.json')); raw=row['raw']
        if capture is not None or 'wire_chunks' in raw:verify_wire.verify_complete(raw)
        if row['request_id'] != task['id'] or raw['request_id'] != task['id'] or raw.get('session_id')!=task['id'] or row['start_ns'] != raw['start_ns'] or row['end_ns'] != raw['terminal_ns']:
            raise ValueError('request identity or clock mismatch')
        if (raw.get('http_status')!=200 or not raw.get('events') or raw['events'][-1].get('kind')!='done' or
            sum(item.get('kind')=='done' for item in raw['events'])!=1 or
            any(item.get('kind') not in ('event','done') for item in raw['events'])):
            raise ValueError('invalid terminal stream evidence')
        usage=[]; finishes=[]; text=''; first_output=None; first_visible=None; last_output=None; last_offset=-1
        for item in raw['events']:
            offset=item.get('offset_ns')
            if type(offset) is not int or offset<last_offset or offset>raw['terminal_ns']-raw['start_ns']:
                raise ValueError('invalid event clock')
            last_offset=offset
            if item['kind'] != 'event': continue
            event=item['data']
            if event.get('usage') is not None: usage.append(event['usage'])
            for choice in event.get('choices',[]):
                delta=choice.get('delta') or {}; visible=delta.get('content') or ''
                reasoning=delta.get('reasoning_content') or delta.get('reasoning') or ''; tools=delta.get('tool_calls') or []
                text += visible
                if visible or reasoning or tools:
                    if first_output is None: first_output=offset
                    last_output=offset
                if visible.strip() and first_visible is None: first_visible=offset
                if choice.get('finish_reason') is not None: finishes.append(choice['finish_reason'])
        if not raw['complete'] or len(usage)!=1 or len(finishes)!=1: raise ValueError('invalid stream')
        tokens=usage[0]['completion_tokens']
        prompt=usage[0]['prompt_tokens']
        if type(tokens) is not int or tokens<=0 or type(prompt) is not int or prompt<0:
            raise ValueError('invalid token usage counts')
        clean=('\ufffd' not in text and
               re.search(r'[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]',text) is None and
               not any(ord(c)<32 and c not in '\n\r\t' for c in text))
        valid=(abs(prompt-task['input_tokens']) <= task['input_tokens']*.0025 and bool(text) and clean and finishes[0] in ('stop','length') and
               (tokens==task['output_tokens'] and task['expected_answer'] in text
                if task['fixed_output'] else finishes[0]=='stop' and text.strip()==task['expected_answer']))
        if not valid or row['valid'] is not True or row['completion_tokens']!=tokens:
            raise ValueError('request score mismatch')
        duration=(raw['terminal_ns']-raw['start_ns'])/1e9
        if first_output is None or last_output is None or duration<=0: raise ValueError('missing output timing')
        timing={'first_output_seconds':first_output/1e9,
                'first_visible_seconds':first_visible/1e9 if first_visible is not None else None,
                'stream_window_seconds_per_token':(last_output-first_output)/1e9/(tokens-1) if tokens>1 else None,
                'request_output_tokens_per_second':tokens/duration,
                'scope':'stream-derived; multi-token chunks do not reveal exact token times'}
        if row.get('timing')!=timing: raise ValueError('request timing derivation mismatch')
        rows.append(row)
    if cache=='warm_prefix':
        task=tasks[0]; block=task.get('cache_block_size'); prefix=task.get('prefix_tokens'); condition=task.get('conditioning_body',{})
        instruction=('Begin with the verification code exactly, without a label or introduction. Then write a detailed technical report '
                     'in English about reliable distributed model serving, with explanations, concrete examples and several sections. '
                     'Continue the report in full prose; do not end after the code.')
        if (task.get('prompt_version')!='pilot-warm-report-v2' or task.get('fixed_report_instruction')!=instruction or
                not task['body']['messages'][-1].get('content','').endswith(instruction)):
            raise ValueError('warm measured request lacks its versioned long-form instruction')
        marker=hashlib.sha256(str(task['seed']).encode()).hexdigest()[:16]
        if (not isinstance(block,int) or block<=0 or not isinstance(prefix,int) or prefix!=((n-256)//block)*block or
            task.get('target_prefix_tokens')!=prefix or not re.fullmatch(r'[0-9a-f]{64}',str(task.get('prefix_token_sha256'))) or
            task.get('expected_conditioning_answer')!=marker or not isinstance(condition.get('messages'),list) or not condition['messages'] or
            set(condition)-{'model','messages','stream','stream_options','temperature','top_p','seed','max_tokens','chat_template_kwargs'} or
            not any(marker in str(m.get('content',''))[:512] for m in condition['messages']) or condition.get('model')!='/model' or condition.get('seed')!=task['seed'] or
            condition.get('max_tokens')!=128 or condition.get('stream') is not True or
            condition.get('stream_options')!={'include_usage':True} or condition.get('temperature')!=1.0 or condition.get('top_p')!=1.0 or
            condition.get('chat_template_kwargs')!={'thinking':False,'reasoning_effort':50} or
            condition.get('messages')!=task['body'].get('messages',[])[:len(condition.get('messages',[]))] or
            task['body'].get('messages',[])[len(condition.get('messages',[])):len(condition.get('messages',[]))+1]!=[{'role':'assistant','content':marker}] or
            read(root/'cache-config.json').get('block_size')!=block):
            raise ValueError('warm conditioning contract mismatch')
        cond=read(root/'pilot-conditioning'/(task['id']+'.json')); raw=cond['raw']; usage=[]; finishes=[]; visible='';last_offset=-1
        if capture is not None or 'wire_chunks' in raw:verify_wire.verify_complete(raw)
        if (raw.get('http_status')!=200 or not raw.get('events') or raw['events'][-1].get('kind')!='done' or
            sum(item.get('kind')=='done' for item in raw['events'])!=1 or
            any(item.get('kind') not in ('event','done') for item in raw['events'])):
            raise ValueError('invalid conditioning terminal stream')
        for item in raw['events']:
            offset=item.get('offset_ns')
            if type(offset) is not int or offset<last_offset or offset>raw['terminal_ns']-raw['start_ns']:
                raise ValueError('invalid conditioning event clock')
            last_offset=offset
            if item['kind']!='event': continue
            event=item['data']
            if event.get('usage') is not None: usage.append(event['usage'])
            for choice in event.get('choices',[]):
                visible+=(choice.get('delta') or {}).get('content') or ''
                if choice.get('finish_reason') is not None: finishes.append(choice['finish_reason'])
        if (not raw['complete'] or raw['request_id']!=task['id'] or cond['request_id']!=task['id'] or
            cond['start_ns']!=raw['start_ns'] or cond['end_ns']!=raw['terminal_ns'] or cond['valid'] is not True or
            len(usage)!=1 or type(usage[0].get('prompt_tokens')) is not int or type(usage[0].get('completion_tokens')) is not int or
            usage[0]['completion_tokens']<=0 or abs(usage[0]['prompt_tokens']-prefix)>prefix*.0025 or
            finishes!=['stop'] or visible.strip()!=marker or cond['clock_id']!=rows[0]['clock_id'] or
            cond['end_ns']>rows[0]['start_ns']):
            raise ValueError('warm conditioning response invalid')
    if len({r['clock_id'] for r in rows}) != 1 or any(r['end_ns'] < r['start_ns'] for r in rows):
        raise ValueError('clock mismatch')
    seconds=(max(r['end_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
    if seconds<=0: raise ValueError('invalid duration')
    tokens=sum(r['completion_tokens'] for r in rows)
    expected={'duration_seconds':seconds,'valid_tokens':tokens,'invalid_partial_tokens':0,
              'valid_tokens_per_second':tokens/seconds,'valid_requests':len(rows),'requests':len(rows)}
    if read(root/'pilot-attempt/summary.json')['summary'] != expected: raise ValueError('summary arithmetic mismatch')
    if read(root/'smoke-result.json')['cleanup'] != 'passed': raise ValueError('cleanup evidence missing')
    verify_cleanup(root,config)
    if read(root/'manifest.json').get('runtime',{}).get('kv_budget_schema') is not None or (root/'kv-budget.json').exists():
        verify_kv_budget(root,config)
    startup=read(root/'startup-timing.json'); began=startup.get('launch_started_ns'); ready=startup.get('semantic_ready_ns')
    observed=startup.get('backend_health_observed_seconds',{})
    expected_health={url.removesuffix('/metrics') for url in config['expected_ranks']}
    if (type(began) is not int or type(ready) is not int or began<0 or ready<began or
        set(observed)!=expected_health or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or v>(ready-began)/1e9 for v in observed.values()) or
        any(row['start_ns']<ready for row in rows)):
        raise ValueError('startup timing evidence invalid')
    before=parse_cache(read(root/'cache-before.json'),config['expected_ranks'])
    after_files=sorted(root.glob('cache-after-*.json'))
    if not after_files: raise ValueError('missing cache after snapshot')
    after=parse_cache(read(after_files[-1]),config['expected_ranks'])
    delta=[]
    for url,ranks in config['expected_ranks'].items():
        for rank in ranks:
            query=(url,str(rank),'vllm:prefix_cache_queries_total'); hit=(url,str(rank),'vllm:prefix_cache_hits_total')
            q,h=after[query]-before[query],after[hit]-before[hit]
            if q<0 or h<0 or h>q: raise ValueError('invalid cache deltas')
            delta.append({'endpoint':url,'rank':str(rank),'queries':q,'hits':h})
    # The producer preserves config insertion order; compare by endpoint/rank.
    key=lambda x:(x['endpoint'],x['rank'])
    if sorted(delta,key=key) != sorted(result['attribution']['rank_counter_deltas'],key=key):
        raise ValueError('rank counter attribution mismatch')
    active=[x for x in delta if x['queries']>0]
    if cell['concurrency']==1 and (len(active)!=1 or result['attribution']['selected_rank']!=active[0]):
        raise ValueError('single-request rank attribution mismatch')
    if cache=='warm_prefix':
        condition_before=parse_cache(read(root/'cache-conditioning-before.json'),config['expected_ranks'])
        condition_files=sorted(root.glob('cache-conditioning-after-*.json'))
        if not condition_files: raise ValueError('missing conditioning counters')
        condition_after=parse_cache(read(condition_files[-1]),config['expected_ranks'])
        if condition_after != before: raise ValueError('measured baseline differs from conditioning counters')
        selected=[]
        for url,ranks in config['expected_ranks'].items():
            for rank in ranks:
                q=(url,str(rank),'vllm:prefix_cache_queries_total'); h=(url,str(rank),'vllm:prefix_cache_hits_total')
                queries=condition_after[q]-condition_before[q]; hits=condition_after[h]-condition_before[h]
                if queries<0 or hits<0 or hits>queries: raise ValueError('invalid conditioning counters')
                if queries>0: selected.append((url,str(rank)))
        if (len(selected)!=1 or len(active)!=1 or selected[0]!=(active[0]['endpoint'],active[0]['rank']) or
            result['attribution'].get('conditioning_rank')!=list(selected[0]) or
            result['attribution'].get('hit_fraction')!=active[0]['hits']/active[0]['queries']):
            raise ValueError('warm rank or hit attribution mismatch')
    if concurrency>1 or (root/'routing-evidence.json').exists():
        verify_routes(root,tasks,config,delta,cache=='warm_prefix')
    if result.get('status')=='passed':
        if (result.get('requests_status')!='passed' or result.get('reason') is not None or
            result.get('scope')!='feasibility pilot observation; not scored performance or a frozen-profile qualification'):
            raise ValueError('unsupported pilot acceptance scope')
        verify_single_request_placement(root,dict(config,concurrency=concurrency),result)
    elif result.get('status') != 'invalid' or result.get('requests_status') != 'passed' or result.get('reason') != 'installed backend engagement and per-request rank evidence are not fully verified':
        raise ValueError('unsupported pilot qualification status')
    if suite != 'pilot_smoke':
        # Startup smokes can finish between samples; fit and concurrency work
        # must establish measured-window coverage before registry acceptance.
        coverage=verify_telemetry.verify_window(root,min(r['start_ns'] for r in rows),max(r['end_ns'] for r in rows),config['expected_ranks'])
        if coverage['status']!='pilot_window_coverage_verified':
            raise ValueError('pilot measurement lacks direct in-window telemetry')
    return {'cell_id':cell['id'],'status':result['status'],'requests':len(rows),'valid_tokens':tokens}

if __name__ == '__main__': print(json.dumps(verify(sys.argv[1]),sort_keys=True))
