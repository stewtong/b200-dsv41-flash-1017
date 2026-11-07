"""Replay source-pinned worker observations with a physical rank join.

This component verifies observation records only. It cannot accept a model
profile, workload overhead, request population or scored comparison.
"""
import hashlib,json,re
from pathlib import Path
import runtime

COUNTERS=('successful_sample_calls','successful_verification_steps','target_verified_draft_positions','successful_grammar_sample_calls')
SCOPE='successful global V2 target draft positions; mirrored TP rows, aggregate TP0 only; no bonus or accepted-token count'


class UnsettledWorkerCounters(ValueError):
    """Valid asynchronous observations may settle at a quiescent boundary."""


def check_row(row,expected,now_ns,max_age_ns=1_100_000_000):
    if (row.get('schema')!='worker-target-observation/v1' or row.get('identity')!=expected or
            type(row.get('started_ns')) is not int or type(row.get('completed_ns')) is not int or
            not 0<=row['started_ns']<=row['completed_ns']<=now_ns or
            now_ns-row['completed_ns']>max_age_ns):
        raise ValueError('worker observation identity, clock or freshness differs')
    values=row.get('values',{})
    if (set(values)!=set(COUNTERS)|{'kv_layout_canonical_sha256'} or
            any(type(values[k]) is not int or values[k]<0 for k in COUNTERS) or
            values['successful_verification_steps']>values['successful_sample_calls'] or
            values['successful_grammar_sample_calls']>values['successful_sample_calls'] or
            values['target_verified_draft_positions']<values['successful_verification_steps'] or
            not re.fullmatch(r'[a-f0-9]{64}',str(values['kv_layout_canonical_sha256']))):
        raise ValueError('worker counters or resolved KV-layout proof missing')
    if 'allocator_sha256' in expected:
        memory=row.get('allocator',{});v=memory.get('values',{})
        names={'allocated_bytes','reserved_bytes','allocated_peak_bytes_since_engine_reset',
            'reserved_peak_bytes_since_engine_reset','allocation_retries','allocator_ooms'}
        if (memory.get('schema')!='process-cuda-allocator-observation/v1' or memory.get('backend')!='native' or
                type(memory.get('device_index')) is not int or memory['device_index']!=expected.get('local_device_index') or
                memory.get('source_pins')!=expected.get('allocator_source_pins') or
                type(memory.get('started_ns')) is not int or type(memory.get('completed_ns')) is not int or
                not row['started_ns']<=memory['started_ns']<=memory['completed_ns']<=row['completed_ns'] or
                set(v)!=names or any(type(v[k]) is not int or v[k]<0 for k in names) or
                v['allocated_bytes']>v['allocated_peak_bytes_since_engine_reset'] or
                v['reserved_bytes']>v['reserved_peak_bytes_since_engine_reset'] or
                memory.get('peak_scope')!='process-local allocator peaks since the engine last reset them; includes startup; not cell-local peaks or total device memory'):
            raise ValueError('worker allocator observation source, device, clocks or statistic semantics differ')
    elif 'allocator' in row:raise ValueError('undeclared worker allocator observation')
    return values


def join_identity(identity,physical,container_id,contract,contract_sha,source_pins):
    integer_fields=('namespace_pid','replica_id','dp_rank','tp_rank','worker_rank','tp_size','dp_size')
    if (any(type(identity.get(k)) is not int or identity[k]<0 for k in integer_fields) or
            identity.get('namespace_pid')!=physical.get('namespace_pid') or
            identity.get('container_hostname')!=container_id[:12] or
            identity.get('dp_rank')!=physical.get('dp_rank',0) or identity.get('tp_rank')!=physical.get('tp_rank',0) or
            identity.get('boot_id')!=contract['boot_id'] or identity.get('run_id')!=contract['run_id'] or
            identity.get('replica_id')!=contract['replica_id'] or
            identity.get('profile_sha256')!=contract['profile_sha256'] or identity.get('toolkit_sha256')!=contract['toolkit_sha256'] or
            identity.get('extension_sha256')!=contract['extension_sha256'] or identity.get('contract_sha256')!=contract_sha or
            any(identity.get(k)!=contract[k] for k in ('recorder_sha256','runtime_sha256')) or
            identity.get('tp_size')!=contract['tp_size'] or identity.get('dp_size')!=contract['dp_size'] or
            identity.get('expected_gpu_ids')!=contract['expected_gpu_ids'] or identity.get('source_pins')!=source_pins or
            physical.get('gpu_index') not in contract['expected_gpu_ids'] or
            type(physical.get('host_pid')) is not int or type(physical.get('start_ticks')) is not int):
        raise ValueError('worker observation lacks exact source, namespace, container and physical rank binding')
    if 'allocator_sha256' in contract:
        if (identity.get('allocator_sha256')!=contract['allocator_sha256'] or
                identity.get('allocator_source_pins')!=contract['allocator_source_pins'] or
                type(identity.get('local_device_index')) is not int or
                identity['local_device_index']!=contract['expected_gpu_ids'].index(physical['gpu_index'])):
            raise ValueError('worker allocator physical device or source binding differs')
    elif 'allocator_sha256' in identity:raise ValueError('undeclared allocator worker identity')
    return dict(identity)


def read_replica(directory,contract,physical,container_id,source_pins,now_ns,alive=runtime.process_alive):
    root=Path(directory);raw=(root/'contract.json').read_bytes()
    if json.loads(raw)!=contract:raise ValueError('worker observation contract read-back differs')
    sha=hashlib.sha256(raw).hexdigest();workers=physical.get('workers',[])
    expected_dirs={'worker-'+str(p['namespace_pid']) for p in workers}
    ranks={(p.get('dp_rank',0),p.get('tp_rank',0)) for p in workers}
    expected_ranks={(dp,tp) for dp in range(contract['dp_size']) for tp in range(contract['tp_size'])}
    if ({p.name for p in root.glob('worker-*')}!=expected_dirs or len(workers)!=contract['tp_size']*contract['dp_size'] or
            len(expected_dirs)!=len(workers) or len({p.get('gpu_index') for p in workers})!=len(workers) or ranks!=expected_ranks):
        raise ValueError('worker observation physical population incomplete or duplicated')
    result=[]
    for worker in workers:
        folder=root/('worker-'+str(worker['namespace_pid']))
        if folder.is_symlink() or (folder/'failure.json').exists():raise ValueError('worker observation failed or redirected')
        identity=json.loads((folder/'identity.json').read_text())
        expected=join_identity(identity,worker,container_id,contract,sha,source_pins)
        if not alive({'boot_id':contract['boot_id'],'pid':worker['host_pid'],'start_ticks':worker['start_ticks']}):
            raise ValueError('worker observation physical process is no longer live')
        row=json.loads((folder/'live.json').read_text())
        # A live writer may publish after the collection began. Read the clock
        # after the atomic file read; a start timestamp is not a valid upper
        # bound for later worker observations. Offline callers use an exact int.
        observed_ns=now_ns() if callable(now_ns) else now_ns
        values=check_row(row,expected,observed_ns)
        layout=json.loads((folder/'kv-layout.json').read_text());canonical=json.dumps(layout,sort_keys=True,separators=(',',':')).encode()
        if (layout.get('schema')!='resolved-worker-kv-layout/v1' or
                hashlib.sha256(canonical).hexdigest()!=values['kv_layout_canonical_sha256']):
            raise ValueError('worker resolved layout does not match its live counter record')
        result.append({'row':row,'physical_worker':worker,'container_id':container_id,'layout':layout})
    return result


def context(root):
    """Read static evidence once; this does not certify a model profile."""
    root=Path(root);binding=json.loads((root/'worker-observation-binding.json').read_text())
    profile=json.loads((root/'profile-binding.json').read_text())
    placement=json.loads((root/'rank-placement.json').read_text())
    if (binding.get('schema')!='source-pinned-worker-observer/v1' or
            binding.get('spec')!=(profile.get('delta') or {}).get('worker_observation') or
            profile.get('phase')!='qualification' or placement.get('status')!='passed' or
            len(binding.get('contracts',[]))!=len(placement['containers'])):
        raise ValueError('worker sampling source or physical contract differs')
    expected={}
    for item,physical in zip(binding['contracts'],placement['containers']):
        replica=item['replica_id'];folder=root/'worker-observation'/('replica-'+str(replica))
        raw=(folder/'contract.json').read_bytes();contract=json.loads(raw)
        if (hashlib.sha256(raw).hexdigest()!=item['sha256'] or contract!=item['contract'] or
                contract.get('profile_sha256')!=profile['profile_sha256'] or
                len(physical['workers'])!=contract['tp_size']*contract['dp_size']):
            raise ValueError('worker sampling replica contract differs')
        for worker in physical['workers']:
            child=folder/('worker-'+str(worker['namespace_pid']))
            identity=json.loads((child/'identity.json').read_text())
            join_identity(identity,worker,physical['container_id'],contract,item['sha256'],binding['installed_source']['source_pins'])
            layout=json.loads((child/'kv-layout.json').read_text())
            digest=hashlib.sha256(json.dumps(layout,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if layout.get('schema')!='resolved-worker-kv-layout/v1':raise ValueError('worker sampling layout schema differs')
            key=(physical['container_id'],identity['dp_rank'],identity['tp_rank'])
            if key in expected:raise ValueError('duplicate worker sampling rank')
            expected[key]={'identity':identity,'physical_worker':worker,'directory':child,'layout':layout,'layout_sha256':digest}
    if len(expected)!=8 or len({v['physical_worker']['gpu_index'] for v in expected.values()})!=8:
        raise ValueError('worker sampling lacks eight distinct physical GPUs')
    return expected


def check_host_sample(observation,expected,completed_ns):
    if (not isinstance(observation,dict) or observation.get('status')!='physical_worker_observation_sample' or
            type(observation.get('observed_at_ns')) is not int or
            not 0<=completed_ns-observation['observed_at_ns']<=1_100_000_000 or
            not isinstance(observation.get('workers'),list) or len(observation['workers'])!=len(expected)):
        raise ValueError('host worker sample missing or stale')
    result={}
    for item in observation['workers']:
        row=item['row'];identity=row['identity'];key=(item['container_id'],identity['dp_rank'],identity['tp_rank'])
        known=expected.get(key)
        if not known or key in result or item['physical_worker']!=known['physical_worker']:
            raise ValueError('host worker sample has foreign or duplicate physical rank')
        values=check_row(row,known['identity'],completed_ns)
        if (row['completed_ns']>observation['observed_at_ns'] or
                values['kv_layout_canonical_sha256']!=known['layout_sha256']):
            raise ValueError('host worker sample layout or collection clock differs')
        result[key]=row
    if set(result)!=set(expected):raise ValueError('host worker sample population incomplete')
    return result


def context_record(root):
    """Portable static join; the full launcher verifier must bind its sources."""
    root=Path(root);known=context(root)
    binding=json.loads((root/'worker-observation-binding.json').read_text())
    plan=json.loads((root/'launch-plan.json').read_text())
    contracts={r['replica_id']:r for r in binding['contracts']};rows=[];references={}
    for name in ('worker-observation-binding.json','rank-placement.json','profile-binding.json','launch-plan.json'):
        references[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    for key,item in sorted(known.items()):
        identity=item['identity'];replica=identity['replica_id'];contract=contracts[replica]
        endpoint='http://127.0.0.1:'+str(plan['commands'][replica]['host_port'])+'/metrics'
        folder=item['directory']
        for path in (folder/'identity.json',folder/'kv-layout.json',folder.parent/'contract.json'):
            references[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({'container_id':key[0],'endpoint':endpoint,'physical_worker':item['physical_worker'],
            'identity':identity,'layout':item['layout'],'contract':contract['contract'],
            'contract_sha256':contract['sha256']})
    return {'schema':'worker-target-attribution/v1','workers':rows,'source_sha256':references,
        'source_pins':binding['installed_source']['source_pins'],
        'scope':'declared portable worker join; global launch/source/physical evidence requires independent launcher verification'}


def boundary_rows(observation,record,completed_ns):
    expected={}
    for item in record.get('workers',[]):
        identity=item['identity'];key=(item['container_id'],identity['dp_rank'],identity['tp_rank'])
        if key in expected:raise ValueError('duplicate worker attribution context')
        join_identity(identity,item['physical_worker'],item['container_id'],item['contract'],
            item['contract_sha256'],record['source_pins'])
        layout=item['layout'];digest=hashlib.sha256(json.dumps(layout,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        expected[key]={'identity':identity,'physical_worker':item['physical_worker'],'layout_sha256':digest,'layout':layout}
    if (record.get('schema')!='worker-target-attribution/v1' or len(expected)!=8 or
            len({r['physical_worker']['gpu_index'] for r in expected.values()})!=8):
        raise ValueError('worker attribution context lacks eight exact physical workers')
    joined=check_host_sample(observation,expected,completed_ns)
    return [{'row':row,'physical_worker':expected[key]['physical_worker'],'container_id':key[0],
        'layout':expected[key]['layout']} for key,row in sorted(joined.items())]


def native_target_population(record,ranks,speculative_report=None):
    engines={}
    for item in record['workers']:
        key=(item['endpoint'],str(item['identity']['dp_rank']));value=(item['container_id'],item['identity']['dp_rank'])
        if key in engines and engines[key]!=value:raise ValueError('worker target native rank has conflicting engine identity')
        engines[key]=value
    expected={(endpoint,str(rank)) for endpoint,values in ranks.items() for rank in values}
    if set(engines)!=expected:raise ValueError('worker target native rank population differs')
    if speculative_report is None:return {value:0 for value in engines.values()}
    result={}
    for row in speculative_report['ranks']:
        key=(row['endpoint'],row['rank']);count=row['draft_tokens']
        if key not in engines or engines[key] in result or type(count) is not int or count<0:
            raise ValueError('native target proposal population malformed')
        result[engines[key]]=count
    if set(result)!=set(engines.values()):raise ValueError('native target proposal population incomplete')
    return result


def verify_window(root,start_ns,end_ns,host_samples):
    """Check raw per-worker coverage and exact host projections independently.

    Counts describe successful target draft-position observations. This does
    not establish matched overhead, acceptance quality or request attribution.
    """
    if type(start_ns) is not int or type(end_ns) is not int or end_ns<=start_ns:
        raise ValueError('invalid worker sampling window')
    expected=context(root);used={key:{} for key in expected}
    if not host_samples:raise ValueError('worker sampling requires retained host samples')
    for sample in host_samples:
        joined=check_host_sample(sample.get('worker_observations'),expected,sample['monotonic_ns'])
        for key,row in joined.items():
            used[key][row['completed_ns']]=json.dumps(row,sort_keys=True,separators=(',',':'))
    checked=[]
    for key,known in expected.items():
        found={};selected=[];prior=None;following=None;previous=-1
        files=sorted(known['directory'].glob('samples-*.jsonl'))
        if not files:raise ValueError('worker raw sample stream missing')
        for path in files:
            if path.is_symlink():raise ValueError('worker raw sample stream redirected')
            with path.open() as handle:
                for line in handle:
                    row=json.loads(line);stamp=row.get('completed_ns')
                    if type(stamp) is not int or stamp<=previous:raise ValueError('worker raw sample clocks reset or repeat')
                    previous=stamp
                    if stamp in used[key]:found[stamp]=json.dumps(row,sort_keys=True,separators=(',',':'))
                    if stamp<start_ns:prior=row
                    elif stamp<=end_ns:selected.append(row)
                    elif following is None:following=row
        if found!=used[key]:raise ValueError('host worker projection differs from retained raw stream')
        if prior is None or start_ns-prior['completed_ns']>1_100_000_000:
            raise ValueError('worker window lacks fresh opening observation')
        rows=[prior]+selected+([following] if following is not None else [])
        points=[start_ns]+[r['completed_ns'] for r in selected]+[end_ns]
        if max(b-a for a,b in zip(points,points[1:]))>1_100_000_000:
            raise ValueError('worker window sample gap exceeds1.1seconds')
        if following is None and end_ns-rows[-1]['completed_ns']>1_100_000_000:
            raise ValueError('worker window closing coverage missing')
        last=None
        for row in rows:
            values=check_row(row,known['identity'],row['completed_ns'])
            if values['kv_layout_canonical_sha256']!=known['layout_sha256']:
                raise ValueError('worker raw layout differs from retained resolved configuration')
            if last is not None and any(values[k]<last[k] for k in COUNTERS):raise ValueError('worker raw counter reset')
            last=values
        checked.append({'container_id':key[0],'dp_rank':key[1],'tp_rank':key[2],
            'samples_in_window':len(selected),'checked_samples':len(rows)})
    return {'status':'worker_observation_window_coverage_verified','workers':checked,
        'window_seconds':(end_ns-start_ns)/1e9,'scope':SCOPE+'; raw cadence and exact host projection only, no overhead or quality acceptance'}


def boundary_delta(before,after,*,max_age_ns=1_100_000_000):
    def indexed(rows):
        result={}
        for item in rows:
            identity=item['row']['identity'];key=(item['container_id'],identity['dp_rank'],identity['tp_rank'])
            if key in result:raise ValueError('duplicate worker rank boundary')
            result[key]=item
        return result
    old=indexed(before);new=indexed(after)
    if not old or set(old)!=set(new):raise ValueError('worker boundary population changed')
    deltas={};engines={}
    for key,item in new.items():
        prior=old[key]
        if item['physical_worker']!=prior['physical_worker'] or item['row']['identity']!=prior['row']['identity']:
            raise ValueError('worker process incarnation changed across boundary')
        a=check_row(prior['row'],prior['row']['identity'],prior['row']['completed_ns'],max_age_ns)
        b=check_row(item['row'],item['row']['identity'],item['row']['completed_ns'],max_age_ns)
        if (item['row']['completed_ns']<=prior['row']['completed_ns'] or item['layout']!=prior['layout'] or
                a['kv_layout_canonical_sha256']!=b['kv_layout_canonical_sha256']):
            raise ValueError('worker boundary clock or resolved layout changed')
        delta={name:b[name]-a[name] for name in COUNTERS}
        if any(n<0 for n in delta.values()):raise ValueError('worker counter reset')
        deltas[key]=delta;engine=key[:2]
        if engine in engines and engines[engine]!=delta:raise UnsettledWorkerCounters('mirrored TP worker counters disagree at quiescent boundary')
        engines[engine]=delta
    # Read TP0 once per logical engine; never sum replicated TP observations.
    if any((cid,dp,0) not in deltas for cid,dp in engines):raise ValueError('TP0 worker source missing')
    return {'status':'worker_target_boundary_observation_verified',
        'logical_engines':[{'container_id':cid,'dp_rank':dp,**deltas[(cid,dp,0)]} for cid,dp in sorted(engines)],
        'target_verified_draft_positions':sum(deltas[(cid,dp,0)]['target_verified_draft_positions'] for cid,dp in engines),
        'scope':SCOPE+'; request/metrics reconciliation and full qualification remain separate'}


def reconcile_target_positions(before,after,native_proposed,*,speculative):
    """Compare actual target draft positions with grammar-free native proposals.

    Native draft observations count per-request steps. The worker step count
    is a batch population and must never be substituted for that denominator.
    """
    if type(speculative) is not bool:raise ValueError('target reconciliation requires an explicit speculation mode')
    result=boundary_delta(before,after)
    engines={(r['container_id'],r['dp_rank']):r for r in result['logical_engines']}
    if (not isinstance(native_proposed,dict) or set(native_proposed)!=set(engines) or
            any(type(v) is not int or v<0 for v in native_proposed.values())):
        raise ValueError('target/native proposal rank population differs')
    for key,worker in engines.items():
        if worker['successful_grammar_sample_calls']:
            raise ValueError('target/native reconciliation requires observed grammar-free sampling')
        count=worker['target_verified_draft_positions']
        if not speculative and (count or native_proposed[key]):raise ValueError('NoSpec target observation contains draft work')
        if count!=native_proposed[key]:
            raise UnsettledWorkerCounters('target positions differ from grammar-free native proposed drafts')
    return {**result,'status':'worker_target_positions_reconciled',
        'native_proposed_draft_tokens':sum(native_proposed.values()),
        'scope':SCOPE+'; grammar-free native proposal equality; batch steps remain distinct from request draft steps'}
