"""A bounded, supervised launch and semantic smoke check; not a benchmark."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import campaign
import cpu_partition
import cache_evidence
import check_fix
import launcher
import qualify_router
import readiness
import runtime
import supervisor
import telemetry
import toolkit_integrity
import process_metrics
import rank_attribution
import backend_evidence
import kv_budget
import profile_registry
import measurement_launch
import nvml_safety_probe

NODE_LOCK = runtime.DEFAULT_NODE_LOCK
INVENTORY_SHA = 'd6c87819bc6575beb3a2f27047d78193bd6de599236e581c01aedd28fd0a370a'
ROUTER_LOCK_SHA = 'af40d763ab7c538cb8fad58c16414dd2ea358b3520549eae90dac2ee2f337bfd'
MODEL_REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())


def await_nvml_measurement(session, root, identity, started_ns):
    """Wait for a fresh measured sample, not a racing safety-only cycle."""
    limit=time.monotonic()+10
    while True:
        session.ensure_healthy()
        current=read(Path(root)/'nvml-live.json')
        if (current.get('sample_started_ns',0)>=started_ns and
                current.get('measurement_mode') is True):
            nvml_safety_probe.validate_live(current,identity,time.monotonic_ns(),measured=True)
            runtime.atomic_json(Path(root)/'nvml-measurement-ready.json',{
                'identity':identity,'requested_after_ns':started_ns,'sample':current})
            return current
        if time.monotonic()>=limit:raise TimeoutError('NVML measured-mode sample timeout')
        time.sleep(.1)

def verify_manifest(config):
    manifest = read(config['manifest'])
    phase=config.get('execution_phase','pilot')
    if phase not in ('pilot','qualification','fixed-performance'):raise ValueError('unknown or unqualified engine execution phase')
    if phase=='fixed-performance':
        selected=sorted({e['cell_id'] for e in config['cells']}) if 'cells' in config else [config['cell_id']]
        campaign.validate_phase(manifest,phase,cell_ids=selected)
    else:campaign.validate_phase(manifest, 'pilot')
    if manifest.get('plan_sha256') != sha(config['plan']):
        raise ValueError('plan content hash mismatch')
    matrix_name='comparison-matrix.json' if manifest.get('scope_id')=='context-concurrency-336-v1' else 'matrix.json'
    if manifest.get('matrix_sha256') != sha(Path(__file__).with_name(matrix_name)):
        raise ValueError('matrix content hash mismatch')
    toolkit=Path(config['toolkit_manifest'])
    if toolkit != Path(__file__).resolve().parents[1]/'MANIFEST.json' or toolkit_integrity.verify(toolkit) != manifest['runtime'].get('toolkit_manifest_sha256'):
        raise ValueError('running toolkit does not match manifest')
    if manifest['runtime'].get('sampling_source') is not None:
        import sampling_source
        if sampling_source.verify(Path(config['model_dir'])/'README.md')!=manifest['runtime']['sampling_source']:
            raise ValueError('sampling source changed after manifest binding')
    if manifest['runtime']['image_digest'] != config['image']:
        raise ValueError('manifest image mismatch')
    profile_registry.resolve(config,manifest)
    return manifest

def session_seconds(config,manifest,*,budget='standard',now=None):
    """Bound a new session without extending its manifest or prior jobs."""
    if budget not in ('standard','pilot-block','qualification-block','fixed-performance-block'):raise ValueError('unknown session budget')
    if budget=='fixed-performance-block' and (config.get('execution_phase')!='fixed-performance' or
        manifest.get('scope_id')!='context-concurrency-336-v1' or
        not isinstance(config.get('cells'),list) or not 1<=len(config['cells'])<=24 or
        any(not isinstance(c,dict) or set(c)!={'cell_id','tasks_file','request_timeout_seconds','repetition'} or
            not str(c['cell_id']).startswith(('core-','warm_prefix_panel-')) for c in config['cells'])):
        raise ValueError('fixed-performance session requires an explicit scoped block')
    if budget=='pilot-block' and (config.get('execution_phase','pilot')!='pilot' or
        not isinstance(config.get('cells'),list) or not 1<=len(config['cells'])<=9 or
        any(not isinstance(c,dict) or set(c) not in ({'cell_id','tasks_file','request_timeout_seconds'},
            {'cell_id','tasks_file','request_timeout_seconds','prompt_reuse'}) or
            'prompt_reuse' in c and not isinstance(c['prompt_reuse'],dict) or
            not str(c['cell_id']).startswith(('pilot_smoke-','pilot_fit-','pilot_warm-','pilot_concurrency-')) for c in config['cells'])):
        raise ValueError('extended session requires an explicit pilot block')
    if budget=='qualification-block' and (config.get('execution_phase')!='qualification' or
        not isinstance(config.get('cells'),list) or not 1<=len(config['cells'])<=24 or
        any(not isinstance(c,dict) or set(c)!={'cell_id','tasks_file','request_timeout_seconds','repetition'} or
            not str(c['cell_id']).startswith(('tuning_trial-','tuning_extra-','anchor-','anchor_greedy-')) for c in config['cells'])):
        raise ValueError('extended qualification session requires an explicit fixed block')
    deadline=datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00'))
    bound=datetime.fromisoformat(manifest['runtime']['operational_deadline'].replace('Z','+00:00'))
    if deadline.tzinfo is None or bound.tzinfo is None or deadline>bound:
        raise ValueError('session deadline exceeds manifest deadline or lacks timezone')
    seconds=(deadline-(now or datetime.now(timezone.utc))).total_seconds()
    # This is a worst-case failure bound, not a runtime estimate. Twenty-four
    # declared cold1M request deadlines plus startup/cleanup fit within52hours.
    limit=187200 if budget=='fixed-performance-block' else 43200 if budget in ('pilot-block','qualification-block') else 14400
    if not 0<seconds<=limit:raise ValueError('session deadline exceeds its phase budget')
    return seconds


def validate(config,*,session_budget='standard'):
    if not os.environ.get('TMUX'): raise runtime.RuntimeErrorClosed('GPU work must run inside tmux')
    launcher.validate_run_id(config.get('run_id'))
    manifest = verify_manifest(config)
    isolated=config.get('isolated_measurement_sampler',False)
    if type(isolated) is not bool:raise ValueError('isolated measurement sampler flag must be boolean')
    if isolated and any(c.get('id')==config.get('cell_id') and c.get('suite')=='sampler_overhead' for c in manifest['cells']):
        raise ValueError('isolated sampler is not an overhead-control arm')
    kv_budget.validate_baseline(config,manifest)
    image_q = read(config['image_qualification'])
    router_q = read(config['router_qualification'])
    weights = read(config['checkpoint_receipt'])
    inventory = Path(config['checkpoint_inventory'])
    inv = read(inventory)
    if image_q['image'] != config['image'] or image_q['source_revision'] != config['source_revision']:
        raise ValueError('image qualification mismatch')
    if image_q.get('entrypoint') != ['vllm','serve']:
        raise ValueError('image entrypoint qualification mismatch')
    drivers=[x.strip() for x in runtime.command(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],timeout=20).splitlines()]
    if len(drivers)!=8 or drivers!=image_q.get('driver_versions'):
        raise ValueError('GPU driver versions changed after image qualification')
    if image_q['fix_checker_sha256'] != sha(Path(__file__).with_name('check_fix.py')):
        raise ValueError('image qualification used a different fix checker')
    if sha(Path(config['image_qualification']).parent/'serve-help.txt') != image_q['help_sha256']:
        raise ValueError('image help evidence mismatch')
    expected_invariants = {n for names in check_fix.REQUIRED.values() for n in names}
    invariants = image_q['installed']['invariants']
    if set(invariants) != expected_invariants or set(invariants.values()) != {'invariants-present'}:
        raise ValueError('incomplete image fix evidence')
    current_image = json.loads(runtime.command(['sudo','-n','docker','image','inspect',config['image']]))
    if (len(current_image) != 1 or current_image[0].get('Id') != image_q['image_id'] or
        config['image'] not in (current_image[0].get('RepoDigests') or []) or
        current_image[0].get('Config',{}).get('Entrypoint') != image_q['entrypoint']):
        raise ValueError('installed image changed after qualification')
    if router_q['lock_sha256'] != ROUTER_LOCK_SHA or router_q['lock_sha256'] != sha(config['router_lock']):
        raise ValueError('router lock mismatch')
    if router_q['router_executable_sha256'] != sha(router_q['router_executable']):
        raise ValueError('router executable changed after qualification')
    router_python = Path(router_q['router_executable']).parent/'python'
    current = runtime.command([str(router_python),'-m','pip','freeze','--all'], timeout=60)
    if qualify_router.versions(current) != router_q['installed_packages']:
        raise ValueError('router environment changed after qualification')
    if sha(Path(config['router_qualification']).parent/'router-help.txt') != router_q['help_sha256']:
        raise ValueError('router help evidence mismatch')
    model = Path(config['model_dir']).resolve(strict=True)
    if str(model) != weights['root'] or weights['inventory_sha256'] != INVENTORY_SHA or weights['inventory_sha256'] != sha(inventory):
        raise ValueError('checkpoint receipt path or inventory mismatch')
    if weights['revision'] != MODEL_REVISION or weights['revision'] != inv['revision'] or weights['files'] != 83 or weights['files'] != len(inv['files']) or weights['bytes'] != 510311045295:
        raise ValueError('checkpoint receipt incomplete')
    if weights.get('file_set_verified') is not True or not isinstance(weights.get('metadata_bytes'), int) or weights['metadata_bytes'] > 1_000_000_000:
        raise ValueError('checkpoint file-set or metadata evidence missing')
    if not config.get('required_metric_names'):
        raise ValueError('metric name contract required')
    seconds=session_seconds(config,manifest,budget=session_budget)
    root = Path(config['run_dir'])
    if (not root.is_absolute() or not root.parent.resolve(strict=True).is_relative_to(Path('/data')) or
        root.name != 'dsv41-' + config['run_id'] or not re.fullmatch(r'dsv41-[a-z0-9-]+', root.name)):
        raise ValueError('run directory must be beneath /data')
    return image_q, router_q, seconds

def await_health(url, session, deadline):
    while time.monotonic() < deadline:
        session.ensure_healthy()
        try:
            if readiness.health(url): return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError('readiness deadline expired')

def record_failure(config, exc, allowed_root='/data', identity_fn=runtime.process_identity):
    root = Path(config.get('run_dir', '/nonexistent'))
    state_file = root/'execution-state.json'
    if root.is_dir() and root.resolve().is_relative_to(Path(allowed_root).resolve()) and re.fullmatch(r'dsv41-[a-z0-9-]+', root.name) and state_file.is_file():
        state = read(state_file)
        if state.get('run_id') == config.get('run_id') and state.get('driver_pid') == os.getpid() and state.get('driver_identity') == identity_fn(os.getpid()):
            runtime.atomic_json(root/'smoke-failure.json', {'status':'failed', 'type':type(exc).__name__, 'message':str(exc)})

def run(config, after_ready=None,*,session_budget='standard'):
    original_affinity = os.sched_getaffinity(0)
    try:
        return _run(config, after_ready,session_budget=session_budget)
    except Exception as exc:
        record_failure(config, exc)
        raise
    finally:
        os.sched_setaffinity(0, original_affinity)

def _run(config, after_ready=None,*,session_budget='standard'):
    image_q, router_q, seconds = validate(config,session_budget=session_budget)
    cpus = cpu_partition.discover()
    os.sched_setaffinity(0, cpus['client_cpus'])
    if os.sched_getaffinity(0) != set(cpus['client_cpus']):
        raise runtime.RuntimeErrorClosed('client CPU affinity mismatch')
    server_cpus = cpu_partition.cpu_list(cpus['server_cpus'])
    if config.get('execution_phase','pilot') in ('qualification','fixed-performance'):
        binding=profile_registry.resolve(config,read(config['manifest']))
        plan=launcher.render(config['layout'],config['image'],config['run_id'],config['model_dir'],config['cache_dir'],server_cpus,overrides=binding['delta'])
    else:
        binding={'phase':'pilot','delta':None,'profile_sha256':launcher.profile_sha256(config['layout']),'record':None}
        plan=launcher.render(config['layout'],config['image'],config['run_id'],config['model_dir'],config['cache_dir'],server_cpus)
    expected_runtime=profile_registry.expected_runtime(binding,config['layout'])
    root = Path(config['run_dir'])
    deadline_ns = time.monotonic_ns() + int(seconds * 1e9)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with supervisor.Supervisor(NODE_LOCK, root, config['run_id'], deadline_ns, manifest_hash=config_hash,
                               phase=binding['phase'] if binding['phase'] in ('qualification','fixed-performance') else 'pilot' if after_ready is not None else 'smoke') as session:
        runtime.atomic_json(root/'run-config.json', config)
        runtime.atomic_json(root/'cpu-partition.json', cpus)
        if binding['phase'] in ('qualification','fixed-performance'):runtime.atomic_json(root/'profile-binding.json',binding)
        for key, name in [('manifest','manifest.json'), ('image_qualification','image-qualification.json'),
                          ('router_qualification','router-qualification.json'),
                          ('checkpoint_receipt','checkpoint-receipt.json')]:
            runtime.atomic_bytes(root/name,Path(config[key]).read_bytes())
        wait_until = time.monotonic() + 15
        while not (root/'watchdog-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic() >= wait_until: raise TimeoutError('watchdog startup timeout')
            time.sleep(0.1)
        ports = [x['host_port'] for x in plan['commands']] + [31100]
        session.preflight(ports)
        retirement = runtime.verify_retirement(config)
        runtime.atomic_json(root/'retirement-preflight.json',retirement)
        collector=session.start_nvml_collector()
        limit=time.monotonic()+20
        while not (root/'nvml-live.json').exists():
            session.ensure_healthy()
            if time.monotonic()>=limit: raise TimeoutError('NVML safety collector startup timeout')
            time.sleep(.1)
        session.activate_nvml_safety(collector)
        observation=(binding.get('delta') or {}).get('worker_observation')
        observation_binding=None
        if observation is not None:
            if binding['phase']!='qualification':raise ValueError('worker observer is forbidden in unchanged pilots')
            import shutil
            import measurement_worker_extension
            source=Path(config['worker_observation_source_evidence'])
            runtime.verify_evidence_manifest(source)
            shutil.copytree(source,root/'worker-source-proof')
            runtime.verify_evidence_manifest(root/'worker-source-proof')
            tp,dp,_=launcher.LAYOUTS[config['layout']]
            allocator_args={}
            if 'allocator_sha256' in observation:
                import allocator_observations
                memory_source=Path(config['worker_allocator_source_evidence'])
                runtime.verify_evidence_manifest(memory_source)
                shutil.copytree(memory_source,root/'worker-allocator-source-proof')
                runtime.verify_evidence_manifest(root/'worker-allocator-source-proof')
                allocator_args={'allocator_evidence':root/'worker-allocator-source-proof',
                    'allocator_pins':allocator_observations.SOURCE_PINS}
            plan,observation_binding=measurement_launch.bind(plan,observation,root,Path(__file__).parent,
                run_id=config['run_id'],profile_sha=binding['profile_sha256'],
                toolkit_sha=sha(config['toolkit_manifest']),deadline_ns=deadline_ns,tp=tp,dp=dp,
                image_id=image_q['image_id'],installed_evidence=root/'worker-source-proof',
                source_pins=measurement_worker_extension.SOURCE_PINS,**allocator_args)
        runtime.atomic_json(root/'launch-plan.json', plan)
        launch_started_ns=time.monotonic_ns(); health_offsets={}
        ids = []
        for item in plan['commands']:
            cid = session.create_start(item['argv']); ids.append(cid)
            placement = runtime.verify_placement(cid, config['run_id'], item['gpu_ids'], item['host_port'], image_q['image_id'], expected_cpus=server_cpus)
            runtime.atomic_json(root/('placement-'+str(item['host_port'])+'.json'), placement)
            if observation_binding is not None:
                row=runtime.inspect_owned(cid,config['run_id'])
                checked=measurement_launch.docker_binding(row,observation_binding,len(ids)-1)
                runtime.atomic_json(root/('worker-docker-binding-'+str(item['host_port'])+'.json'),
                    {'container_id':cid,**checked})
        ready_deadline = min(time.monotonic() + 3600, deadline_ns / 1e9 - 120)
        urls = ['http://127.0.0.1:'+str(item['host_port']) for item in plan['commands']]
        for url in urls:
            await_health(url+'/health', session, ready_deadline)
            health_offsets[url]=(time.monotonic_ns()-launch_started_ns)/1e9
        cache_metrics = {}
        for item,url in zip(plan['commands'],urls):
            path=root/('cache-metrics-'+str(item['host_port'])+'.txt')
            cache_metrics[url+'/metrics']=telemetry.scrape(url+'/metrics', [],capture=lambda text:runtime.atomic_bytes(path,text.encode()))
        runtime.atomic_json(root/'cache-config-metrics.json', cache_metrics)
        if cache_evidence.has_cache_metric(cache_metrics):
            cache_config=cache_evidence.cache_block_size(cache_metrics)
        else:
            startup_logs={}
            for cid in ids:
                path=root/('startup-container-'+cid+'.log')
                runtime.capture_owned_logs(cid,config['run_id'],path,timeout=30)
                startup_logs[cid]=path.read_text(errors='replace')
            tp,dp,_=launcher.LAYOUTS[config['layout']]
            cache_config=cache_evidence.cache_block_size_from_logs(startup_logs,tp*dp)
        runtime.atomic_json(root/'cache-config.json', cache_config)
        argv = plan['router_argv'][:]
        argv[0] = router_q['router_executable']
        session.start_router(argv)
        await_health('http://127.0.0.1:31100/health', session, ready_deadline)
        files = []
        for i, (identity, url) in enumerate(list(zip(ids,urls))+[('router','http://127.0.0.1:31100')]):
            session.ensure_healthy()
            evidence = readiness.known_answer(url+'/v1/chat/completions', '/model', 'smoke'+str(i), timeout=60)
            evidence['identity'] = identity
            path = root/('readiness-'+str(i)+'.json')
            runtime.atomic_json(path, evidence); files.append(path)
            if not evidence['passed']: raise runtime.RuntimeErrorClosed('semantic readiness failed; raw evidence retained')
        measurement_started=time.monotonic_ns()
        runtime.atomic_json(root/'nvml-measurement-mode.json',{'scope':'unscored measured-telemetry qualification'})
        await_nvml_measurement(session,root,collector,measurement_started)
        isolated=config.get('isolated_measurement_sampler',False)
        if type(isolated) is not bool:raise ValueError('isolated measurement sampler flag must be boolean')
        if isolated:
            session.start_sampler([url+'/metrics' for url in urls], config['required_metric_names'], config.get('expected_ranks'), cpus['server_cpus'], collector, isolated=True)
        else:
            session.start_sampler([url+'/metrics' for url in urls], config['required_metric_names'], config.get('expected_ranks'), cpus['server_cpus'], collector)
        sample_deadline = time.monotonic()+15
        while not (root/'engine-heartbeat.json').exists():
            session.ensure_healthy()
            if time.monotonic() >= sample_deadline: raise TimeoutError('engine telemetry startup timeout')
            time.sleep(0.1)
        session.mark_ready(files)
        sample_deadline=time.monotonic()+30
        while session.sampler_stats['count'] < 10:
            session.ensure_healthy()
            if time.monotonic() >= sample_deadline: raise TimeoutError('ten-sample telemetry qualification timed out')
            time.sleep(.1)
        stats=session.sampler_stats
        runtime.atomic_json(root/'telemetry-qualification.json',dict(stats,
            mean_collection_seconds=stats['total_seconds']/stats['count'],
            scope='unscored collection wall time and sampler-thread CPU only; subprocess CPU, workload impact and rank semantics not established'))
        runtime.atomic_json(root/'startup-timing.json', {'launch_started_ns':launch_started_ns,
                            'semantic_ready_ns':time.monotonic_ns(), 'backend_health_observed_seconds':health_offsets,
                            'scope':'elapsed through router and telemetry readiness; backend health times are observations, not exact readiness' })
        runtime.atomic_json(root/'smoke-semantic.json', {'status':'passed', 'scope':'semantic smoke only',
                                                          'rank_coverage':'unverified', 'container_ids':ids})
        tp,dp,expert=launcher.LAYOUTS[config['layout']]
        if (dp > 1 and tp == 1) or (tp > 1 and dp == 1):
            session.ensure_healthy()
            raw_processes=process_metrics.capture(server_cpus=cpus['server_cpus'])
            runtime.atomic_json(root/'rank-process-snapshot.json',raw_processes)
            devices=read(root/'nvml-live.json')['devices']
            mappings=[]
            backends=[]
            for cid,item in zip(ids,plan['commands']):
                runtime.inspect_owned(cid,config['run_id'])
                log=root/('rank-startup-'+cid+'.log')
                runtime.capture_owned_logs(cid,config['run_id'],log,timeout=15)
                top=runtime.command(['sudo','-n','docker','top',cid,'-eo','pid'],timeout=5)
                runtime.atomic_bytes(root/('rank-top-'+cid+'.txt'),top.encode())
                pids=[int(line.strip()) for line in top.splitlines() if line.strip().isdigit()]
                mapper=rank_attribution.map_dp_workers if dp > 1 else rank_attribution.map_tp_workers
                mapped=mapper(log.read_text(errors='replace'),pids,
                    raw_processes['processes'],raw_processes['gpu_processes'],devices,list(range(max(dp,tp))))
                for worker in mapped['workers']:
                    if runtime.process_identity(worker['host_pid'])['start_ticks'] != worker['start_ticks']:
                        raise runtime.RuntimeErrorClosed('Worker process incarnation changed after ownership snapshot')
                if {x['gpu_index'] for x in mapped['workers']} != set(item['gpu_ids']):
                    raise runtime.RuntimeErrorClosed('observed worker placement differs from launch plan')
                mapped['container_id']=cid;mappings.append(mapped)
                selected=expected_runtime['backend'];dtype=expected_runtime['kv_dtype']
                proof=backend_evidence.verify(log.read_text(errors='replace'),selected,tp,dp,
                    dtype if selected=='FLASHMLA_MEGA_ATTN_DSV41' else None,expert,
                    kv_dtype=dtype,expected_spec=expected_runtime['speculative_config'],
                    expected_moe=expected_runtime['moe_backend'])
                proof['container_id']=cid;backends.append(proof)
            session.ensure_healthy()
            runtime.atomic_json(root/'rank-placement.json',{'status':'passed','containers':mappings,
                'scope':'physical worker placement; TP requests span all shards of their selected engine; engine routing and cache hits require separate counters'})
            runtime.atomic_json(root/'backend-selection.json',{'status':'configured_backend_confirmed',
                'containers':backends,'kernel_execution_profiled':False})
            kv_budget.record(root,config)
        else:
            runtime.atomic_json(root/'rank-placement.json',{'status':'unsupported',
                'scope':'mixed TP/DP geometry lacks physical rank proof'})
        if observation_binding is not None:
            # Initial sampler records explicitly lack worker attribution until
            # the real physical mapping exists. Do not release measured work
            # until a completed host sample includes fresh joined worker rows.
            measurement_launch.collect(root,time.monotonic_ns())
            activated=time.monotonic_ns()
            runtime.atomic_json(root/'worker-observation-active.json',{'activated_ns':activated})
            wait=min(time.monotonic()+15,deadline_ns/1e9-120)
            while True:
                session.ensure_healthy();path=root/'worker-observation-last-sampled.json'
                if path.exists() and read(path).get('observed_at_ns',0)>=activated:break
                if time.monotonic()>=wait:raise TimeoutError('worker observer host sample did not acknowledge activation')
                time.sleep(.05)
        if after_ready is not None:
            after_ready(session, root, 'http://127.0.0.1:31100/v1/chat/completions')
    runtime.atomic_json(root/'smoke-result.json', {'status':'passed', 'cleanup':'passed',
                                                   'scope':'semantic smoke only', 'rank_coverage':'unverified',
                                                   'container_ids':ids})
    if after_ready is None: runtime.seal_evidence(root)
