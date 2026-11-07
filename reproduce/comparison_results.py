"""Derive descriptive336-cell coverage from independently verified fresh captures."""
from datetime import datetime,timezone
import hashlib,json,math,re,statistics
from pathlib import Path
import campaign,fixed_profiles,runtime,verify_fixed_block,verify_fixed_response,verify_telemetry


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def distribution(values):
    values=[v for v in values if v is not None]
    if not values:return dict(n=0,p50=None,min=None,max=None)
    if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid descriptive metric')
    ordered=sorted(values);result=dict(n=len(values),p50=statistics.median(ordered),min=ordered[0],max=ordered[-1])
    if len(values)>=20:result['p95_nearest_rank']=ordered[math.ceil(.95*len(ordered))-1]
    return result


def output_offsets(raw):
    first=visible=last=None
    for event in raw['events']:
        if event.get('kind')!='event':continue
        for choice in event['data'].get('choices',[]):
            text=(choice.get('delta') or {}).get('content')
            if text:
                offset=event['offset_ns'];first=offset if first is None else first;last=offset
                if visible is None and text.strip():visible=offset
    return first,visible,last


def describe(tasks,records,inputs):
    report=verify_fixed_response.population(tasks,{k:v['raw'] for k,v in records.items()},
        inputs['request_ids'],inputs['session_ids'],inputs['clock_id'])
    outcomes=[];by_id={task['id']:task for task in tasks}
    for score in report['rows']:
        raw=records[score['task_id']]['raw'];first,visible,last=output_offsets(raw)
        tokens=score['completion_tokens'];seconds=(raw['terminal_ns']-raw['start_ns'])/1e9
        rate=(tokens-1)/((last-first)/1e9) if score['correct'] and tokens>1 and first is not None and last>first else None
        outcomes.append(dict(task_id=score['task_id'],valid=score['correct'],reasons=score['reasons'],
            expected_input_tokens=by_id[score['task_id']]['input_tokens'],
            target_input_tokens=by_id[score['task_id']]['target_input_tokens'],
            prompt_tokens=score['prompt_tokens'],completion_tokens=tokens,dispatch_ns=raw['start_ns'],
            first_output_ns=raw['start_ns']+first if first is not None else None,
            first_visible_text_ns=raw['start_ns']+visible if visible is not None else None,
            last_output_ns=raw['start_ns']+last if last is not None else None,terminal_ns=raw['terminal_ns'],
            first_output_seconds=first/1e9 if first is not None else None,
            first_visible_text_seconds=visible/1e9 if visible is not None else None,
            completion_seconds=seconds,decode_window_tokens_per_second=rate,
            end_to_end_output_tokens_per_second=tokens/seconds if score['correct'] else None,
            http_status=raw.get('http_status'),complete=raw.get('complete'),content_anomalies=score.get('content_anomalies',[])))
    return dict(requests=report['requests'],valid_requests=report['valid_requests'],
        failed_or_invalid_requests=report['requests']-report['valid_requests'],valid_output_tokens=report['valid_tokens'],
        partial_output_tokens=report['invalid_partial_tokens'],burst_makespan_seconds=report['duration_seconds'],
        dispatch_span_seconds=(max(r['dispatch_ns'] for r in outcomes)-min(r['dispatch_ns'] for r in outcomes))/1e9,
        burst_output_tokens_per_second=report['valid_tokens_per_second'],request_outcomes=outcomes,
        distributions={key:distribution([r[key] for r in outcomes]) for key in
            ('first_output_seconds','first_visible_text_seconds','completion_seconds',
             'decode_window_tokens_per_second','end_to_end_output_tokens_per_second')})


def window_observations(root,start,end):
    memory=[0]*8;active=[];queued=[];per_rank={};count=0
    for sample in verify_telemetry.iter_lines(root,'telemetry'):
        if not start<=sample['monotonic_ns']<=end:continue
        count+=1
        for device in sample['nvml_safety']['devices']:
            memory[device['index']]=max(memory[device['index']],device['used_bytes'])
        totals={'vllm:num_requests_running':0,'vllm:num_requests_waiting':0}
        for endpoint,text in sample['engine_metrics'].items():
            for line in text.splitlines():
                match=re.fullmatch(r'(vllm:num_requests_running|vllm:num_requests_waiting)\{([^}]*)\}\s+([^ ]+)',line)
                if not match:continue
                rank=re.search(r'(?:^|,)\s*engine="([^"]+)"(?:,|$)',match[2])
                if not rank:raise ValueError('observed queue metric lacks rank')
                value=float(match[3]);totals[match[1]]+=value
                key=endpoint+'#rank='+rank[1];row=per_rank.setdefault(key,dict(max_running=0,max_waiting=0))
                name='max_running' if match[1]=='vllm:num_requests_running' else 'max_waiting'
                row[name]=max(row[name],value)
        active.append(totals['vllm:num_requests_running']);queued.append(totals['vllm:num_requests_waiting'])
    return dict(samples=count,max_sampled_device_used_bytes=memory,
        observed_running_requests=distribution(active),observed_waiting_requests=distribution(queued),
        per_logical_rank=per_rank,scope='in-window sampled device memory and active/queued demand; not exact transient or CUDA allocator peaks')


def derive(manifest,roots):
    if manifest.get('scope_id')!='context-concurrency-336-v1':raise ValueError('wrong comparison scope')
    campaign.validate_phase(manifest,'fixed-performance',cell_ids=[c['id'] for c in manifest['cells']],evidence=True)
    profiles=manifest['runtime']['fixed_profiles']
    if set(profiles)!={'2xtp4','4xdep2','2xdep4','dep8'}:raise ValueError('four frozen configuration records required')
    for record in profiles.values():fixed_profiles.validate(record,manifest)
    cells={c['id']:dict(c,accepted=False,repetitions_observed=[],missing_repetitions=[0,1,2],
                      status='not_measured',matched_layouts=[],all_four_matched=False) for c in manifest['cells']}
    seen=set();sources=[]
    for directory in roots:
        if isinstance(directory,dict):
            import capture_archive
            archive_path=capture_archive.validate(directory);key='archive:'+str(archive_path)
            if key in seen:raise ValueError('duplicate capture source')
            seen.add(key)
            with capture_archive.materialize(directory) as temporary:
                partial=derive(manifest,[temporary])
            for observed in partial['cells']:
                cell=cells[observed['id']]
                for rep in observed['repetitions_observed']:
                    if any(old['repetition']==rep['repetition'] for old in cell['repetitions_observed']):
                        raise ValueError('duplicate archive or directory scored repetition')
                    cell['repetitions_observed'].append(rep)
            sources.append({**directory,'archive':str(archive_path),
                'repetitions':partial['measured_repetitions'],'seal':directory['seal_sha256']})
            continue
        root=Path(directory).resolve(strict=True)
        if str(root) in seen:raise ValueError('duplicate capture source')
        seen.add(str(root));checked=verify_fixed_block.verify(root);cfg=read(root/'run-config.json');actual=read(root/'manifest.json')
        if (cfg.get('execution_phase')!='fixed-performance' or
                actual.get('scope_id')!=manifest['scope_id'] or
                any(actual.get(k)!=manifest.get(k) for k in ('plan_sha256','matrix_sha256')) or
                actual['runtime'].get('fixed_profiles')!=profiles):
            raise ValueError('capture is outside the frozen336-cell comparison')
        source_seal=sha(root/'SHA256SUMS')
        for verified in checked['cells']:
            ident=verified['qualification_cell_id'];rep=verified['repetition'];cell=cells.get(ident)
            if (cell is None or verified.get('scored_performance_evidence') is not True or
                    verified['profile_sha256']!=profiles[cell['layout']]['profile_sha256'] or
                    rep not in (0,1,2) or any(r['repetition']==rep for r in cell['repetitions_observed'])):
                raise ValueError('duplicate, historical or changed-profile scored repetition')
            child=root/'cells'/verified['cell_id'];capture=child/'fixed-attribution/requests/measured'
            inputs=read(capture/'fixed-inputs.json');tasks=inputs['tasks']
            records={t['id']:read(capture/(t['id']+'.json')) for t in tasks}
            metrics=describe(tasks,records,inputs)
            if metrics['requests']!=cell['concurrency'] or metrics['valid_requests']!=cell['concurrency']:
                raise ValueError('successful capture cannot hide failed or missing requests')
            start=min(r['dispatch_ns'] for r in metrics['request_outcomes']);end=max(r['terminal_ns'] for r in metrics['request_outcomes'])
            contract=[{k:t.get(k) for k in ('id','body','conditioning_body','input_tokens','output_tokens','prefix_tokens','prefix_token_sha256')} for t in sorted(tasks,key=lambda t:t['id'])]
            # Native attribution was independently checked above. Recompute the
            # achieved fraction from its retained boundary snapshots via that oracle.
            import verify_fixed_attribution
            attribution=verify_fixed_attribution.verify(child/'fixed-attribution')
            cache_population=next(p for p in attribution['populations'] if p['phase']=='measured')
            queries=sum(r['queries'] for r in cache_population['ranks']);hits=sum(r['hits'] for r in cache_population['ranks'])
            conditioning=dict(requests=0,output_tokens=0,makespan_seconds=0)
            if cell['cache']=='warm_prefix':
                conditions=[read(child/'fixed-attribution/requests/conditioning'/(t['id']+'.json')) for t in tasks]
                conditioning=dict(requests=len(conditions),output_tokens=sum(r['completion_tokens'] for r in conditions),
                    makespan_seconds=(max(r['end_ns'] for r in conditions)-min(r['start_ns'] for r in conditions))/1e9)
            measured=dict(repetition=rep,profile_sha256=verified['profile_sha256'],source_run=root.name,
                source_seal=source_seal,task_contract_sha256=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                **metrics,cache=dict(queries=queries,hits=hits,achieved_hit_fraction=hits/queries if queries else None,
                    per_rank=cache_population['ranks']),observations=window_observations(child,start,end),
                cold_cache_label='cache-busted cold; unique early system namespaces, no verified cache flush or reset' if cell['cache']=='cold' else None,
                conditioning=conditioning,warmup_excluded=True)
            cell['repetitions_observed'].append(measured)
        sources.append(dict(root=str(root),seal=source_seal,repetitions=len(checked['cells'])))
    cohorts={}
    for cell in cells.values():
        cell['repetitions_observed'].sort(key=lambda r:r['repetition'])
        cell['missing_repetitions']=[r for r in range(3) if r not in {x['repetition'] for x in cell['repetitions_observed']}]
        cell['accepted']=not cell['missing_repetitions']
        cell['status']='descriptive_measurement_accepted' if cell['accepted'] else 'partial_repetitions' if cell['repetitions_observed'] else 'not_measured'
        if cell['accepted']:
            cell['burst_rate_spread']=distribution([r['burst_output_tokens_per_second'] for r in cell['repetitions_observed']])
            key=tuple(cell[k] for k in ('suite','input_tokens','concurrency','cache','output_tokens'))
            signature=tuple(r['task_contract_sha256'] for r in cell['repetitions_observed'])
            cohorts.setdefault((key,signature),[]).append(cell)
    for group in cohorts.values():
        layouts=sorted(c['layout'] for c in group)
        for cell in group:cell.update(matched_layouts=layouts,all_four_matched=len(layouts)==4)
    cold=sum(c['accepted'] for c in cells.values() if c['suite']=='core')
    warm=sum(c['accepted'] for c in cells.values() if c['suite']=='warm_prefix_panel')
    return dict(schema='fixed-comparison-results/v1',scope_id=manifest['scope_id'],
        observed_at=datetime.now(timezone.utc).isoformat(),plan_sha256=manifest['plan_sha256'],matrix_sha256=manifest['matrix_sha256'],
        generator_sha256=sha(__file__),required_cells=336,required_cold_cells=288,required_warm_cells=48,
        accepted_cells=cold+warm,accepted_cold_cells=cold,accepted_warm_cells=warm,
        all_four_matched_cells=sum(c['all_four_matched'] for c in cells.values()),
        measured_repetitions=sum(len(c['repetitions_observed']) for c in cells.values()),
        measured_requests=sum(r['requests'] for c in cells.values() for r in c['repetitions_observed']),
        complete=cold==288 and warm==48,sources=sources,cells=list(cells.values()),
        limitations=['Finite bursts, not sustained capacity, task-quality equivalence or production reliability.',
            'First output is the first nonempty content chunk; first visible text excludes whitespace. Metadata does not count.',
            'Decode-window rate uses usage tokens and first/last output chunk times; multi-token chunks prevent precise per-token timing.',
            'Repetition spread is retained. No winner is established by unmatched configurations or unresolved variation.',
            'Warm-designated misses and eviction remain warm workload observations, with achieved hit fraction.',
            'Unmeasured or unresolved failed work remains incomplete; historical pilot/tuning results confer no scored credit.'])
