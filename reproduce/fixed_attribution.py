"""Retain fixed request boundaries, cache/work counters and native route logs.

This runs inside an already owned serving session. It cannot qualify a frozen
profile, physical worker placement, telemetry coverage or shared cleanup.
"""
import hashlib
import json
from pathlib import Path
import time
import cache_evidence,engine_counters,fixed_session,runtime,telemetry,speculation_evidence
import measurement_launch,verify_worker_observations


BOUNDARY_SECONDS=10

def capture(session,url,tasks,directory,timeout,clock_id,namespace,session_ids,expected_ranks,router_log,block_size,speculative_contract=None,worker_root=None,*,client_workers=None,client_cpus=None,prefix_caching=True):
    if not expected_ranks or not tasks or len({t['id'] for t in tasks})!=len(tasks):raise ValueError('explicit attribution population required')
    warm={t.get('cache_designation') for t in tasks}=={'warm_prefix'}
    if type(prefix_caching) is not bool or warm and not prefix_caching:raise ValueError('fixed cache mode differs')
    cache_proof=None
    if not prefix_caching:
        cache_proof=(session.root/'profile-binding.json').read_bytes()
        binding=json.loads(cache_proof)
        if (binding.get('phase')!='qualification' or binding.get('delta',{}).get('enable_prefix_caching') is not False or
                binding.get('record',{}).get('profile_delta')!=binding.get('delta')):
            raise ValueError('cache-off attribution lacks registered profile binding')
    if warm and (type(block_size) is not int or block_size<=0 or any(t.get('cache_block_size')!=block_size for t in tasks)):
        raise ValueError('live cache block size differs from fixed inputs')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False);router_log=Path(router_log)
    route_start=router_log.stat().st_size
    contract={'schema':'fixed-attribution-capture/v1','expected_ranks':expected_ranks,'cache_block_size':block_size,
        'router_start_byte':route_start,'request_namespace':namespace,'session_ids':session_ids,'clock_id':clock_id}
    if cache_proof is not None:
        runtime.atomic_bytes(root/'prefix-cache-profile-binding.json',cache_proof)
        contract.update(prefix_caching=False,prefix_cache_profile_sha256=hashlib.sha256(cache_proof).hexdigest())
    if speculative_contract is not None:
        if (not isinstance(speculative_contract,dict) or set(speculative_contract)!={'max_spec_tokens','placement'} or
                type(speculative_contract['max_spec_tokens']) is not int or not 1<=speculative_contract['max_spec_tokens']<=64 or
                not isinstance(speculative_contract['placement'],dict) or set(speculative_contract['placement'])!=set(session_ids) or
                any(not isinstance(route,list) or len(route)!=2 or route[0] not in expected_ranks or
                    route[1] not in set(map(str,expected_ranks[route[0]])) for route in speculative_contract['placement'].values())):
            raise ValueError('explicit speculative boundary contract differs')
        contract['speculative_contract']=speculative_contract
    worker_context=None
    if worker_root is not None:
        worker_context=verify_worker_observations.context_record(worker_root)
        runtime.atomic_json(root/'worker-context.json',worker_context)
        contract['worker_observation_schema']='worker-target-attribution/v1'
        contract['worker_context_sha256']=hashlib.sha256((root/'worker-context.json').read_bytes()).hexdigest()
    runtime.atomic_json(root/'attribution-contract.json',contract)
    snapshots={};worker_snapshots={};boundaries=[]
    def observe(phase):
        first=time.monotonic_ns();deadline_ns=min(first+BOUNDARY_SECONDS*1_000_000_000,session.deadline_ns-120_000_000_000);index=0
        def remaining():
            seconds=(deadline_ns-time.monotonic_ns())/1e9
            if seconds<=0:raise runtime.RuntimeErrorClosed('engine accounting did not settle within boundary deadline')
            return seconds
        while True:
            session.ensure_healthy()
            metrics={}
            for endpoint in expected_ranks:
                metrics[endpoint]=telemetry.scrape(endpoint,[],timeout=min(3,remaining()))
                remaining()
                session.ensure_healthy()
            for endpoint,ranks in expected_ranks.items():engine_counters.snapshot(metrics[endpoint],ranks)
            cache_evidence.counters(metrics,expected_ranks)
            if speculative_contract is not None:
                for endpoint,ranks in expected_ranks.items():
                    speculation_evidence.snapshot(metrics[endpoint],ranks,speculative_contract['max_spec_tokens'])
            path=root/(phase+'-'+str(index).zfill(3)+'.json');runtime.atomic_json(path,metrics)
            worker_observation=None;worker_rows=None
            if worker_context is not None:
                worker_observation=measurement_launch.collect(worker_root,time.monotonic_ns())
                worker_path=root/(phase+'-'+str(index).zfill(3)+'-worker.json')
                runtime.atomic_json(worker_path,worker_observation)
                worker_rows=verify_worker_observations.boundary_rows(worker_observation,worker_context,time.monotonic_ns())
            remaining()
            if phase.startswith('before_'):break
            prior=snapshots[phase.replace('after_','before_')]
            try:accounting=engine_counters.differences(prior,metrics,expected_ranks)
            except engine_counters.UnsettledPopulationError:
                index+=1;time.sleep(min(.1,remaining()));continue
            finished=sum(sum(r['finished_by_reason'].values()) for r in accounting['ranks'])
            if finished>len(tasks):raise runtime.RuntimeErrorClosed('unexpected requests in fixed accounting boundary')
            if finished==len(tasks):
                if any(r['finished_by_reason'][reason] for r in accounting['ranks'] for reason in ('abort','error','repetition')):
                    raise runtime.RuntimeErrorClosed('failed engine requests in fixed accounting boundary')
                if speculative_contract is not None:
                    folder='conditioning' if phase=='after_conditioning' else 'measured'
                    rows={t['id']:json.loads((root/'requests'/folder/(t['id']+'.json')).read_text()) for t in tasks}
                    try:speculation_evidence.reconcile_requests(prior,metrics,expected_ranks,
                        speculative_contract['max_spec_tokens'],rows,speculative_contract['placement'])
                    except speculation_evidence.UnsettledSpeculationError:
                        index+=1;time.sleep(min(.1,remaining()));continue
                if worker_context is not None:
                    report=(speculation_evidence.verify_delta(prior,metrics,expected_ranks,
                        speculative_contract['max_spec_tokens']) if speculative_contract is not None else None)
                    native=verify_worker_observations.native_target_population(worker_context,expected_ranks,report)
                    try:verify_worker_observations.reconcile_target_positions(
                        worker_snapshots[phase.replace('after_','before_')],worker_rows,native,
                        speculative=speculative_contract is not None)
                    except verify_worker_observations.UnsettledWorkerCounters:
                        index+=1;time.sleep(min(.1,remaining()));continue
                break
            index+=1;time.sleep(min(.1,remaining()))
        snapshots[phase]=metrics
        if worker_context is not None:worker_snapshots[phase]=worker_rows
        remaining()
        boundary={'phase':phase,'started_ns':first,'completed_ns':time.monotonic_ns(),'deadline_ns':deadline_ns,'snapshot_file':path.name,
            'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if worker_context is not None:boundary.update(worker_snapshot_file=worker_path.name,
            worker_snapshot_sha256=hashlib.sha256(worker_path.read_bytes()).hexdigest())
        boundaries.append(boundary)
        runtime.atomic_json(root/'boundaries.json',boundaries)
    try:
        options={} if client_workers is None and client_cpus is None else {'client_workers':client_workers,'client_cpus':client_cpus}
        result=fixed_session.run(session,url,tasks,root/'requests',timeout,clock_id,namespace,session_ids,observe,**options)
        if result['valid_requests']!=len(tasks):raise runtime.RuntimeErrorClosed('fixed request population contains invalid outcomes')
        data=router_log.read_bytes();end=len(data)
        if end<route_start:raise runtime.RuntimeErrorClosed('native router log shrank')
        runtime.atomic_bytes(root/'routing.log',data[route_start:end])
        runtime.atomic_json(root/'routing-span.json',{'start_byte':route_start,'end_byte':end,'sha256':hashlib.sha256(data[route_start:end]).hexdigest()})
        runtime.atomic_json(root/'attribution-result.json',{'status':'captured_cleanup_pending',
            'requests':len(tasks),'scope':'request and aggregate cache/work accounting only; full launcher acceptance remains required'})
        runtime.seal_evidence(root)
        import verify_fixed_attribution
        return verify_fixed_attribution.verify(root,progress=session.ensure_healthy)
    except BaseException as exc:
        runtime.atomic_json(root/'attribution-failure.json',{'status':'failed','error_type':type(exc).__name__,
            'scope':'raw boundary and request records retained; no completed attribution acceptance'})
        raise
