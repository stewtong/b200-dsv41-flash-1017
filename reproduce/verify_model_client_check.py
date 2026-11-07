"""Independently audit an owned model/client comparison, without its runner."""
import hashlib,json,math,os
from pathlib import Path
import routing_map,runtime,verify_engine_profile,verify_fixed_attribution,verify_fixed_response,verify_fixed_task,verify_telemetry
import cache_source_proof

FRONTDOOR='http://127.0.0.1:31100/v1/chat/completions'

def verify(directory,*,completion_required=True):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda name:json.loads((root/name).read_text())
    startup=verify_engine_profile.verify(root);config=read('run-config.json');state=read('execution-state.json')
    design=read('model-client-design.json');binding=read('profile-binding.json');delta=binding.get('delta') or {}
    cache_source_proof.verify(root/'cache-source-proof',read('image-qualification.json')['image_id'])
    source_bytes=(root/'client-toolkit-MANIFEST.json').read_bytes();source_manifest=json.loads(source_bytes)
    if (hashlib.sha256(source_bytes).hexdigest()!=read('manifest.json')['runtime'].get('toolkit_manifest_sha256') or
            not isinstance(source_manifest,dict) or not source_manifest.get('reproduce/model_clients.py')):
        raise ValueError('client source toolkit manifest differs from bound execution')
    n=config.get('client_input_tokens');c=config.get('client_concurrency');cpus=read('cpu-partition.json')['client_cpus']
    identity='clientqual-'+hashlib.sha256(json.dumps([config['layout'],startup['profile_sha256'],n,c],separators=(',',':')).encode()).hexdigest()
    expected={'schema':'owned-model-client-check/v1','identity':identity,'input_tokens':n,'concurrency':c,
        'repetitions':3,'arms':[{'name':'before','workers':1},{'name':'two','workers':2},{'name':'after','workers':1}],
        'cpus':cpus,'prefix_caching':False,'output_tokens':2048,
        'scope':'one declared shape only; no global client ceiling, saturation or scored claim'}
    if (design!=expected or type(n) is not int or n not in (8000,128000,512000,1000000) or
            type(c) is not int or c not in (8,32,128) or delta.get('enable_prefix_caching') is not False or
            delta.get('speculative_config','missing') is not None or config.get('execution_phase')!='qualification' or
            state.get('run_id')!=config.get('run_id') or state.get('phase')!='qualification' or
            not read('cleanup-result.json').get('success') or
            (runtime.process_alive(state['driver_identity']) and state['driver_identity']!=runtime.process_identity(os.getpid())) or
            runtime.process_alive(state['watchdog_identity']) or any(runtime.process_alive(p) for p in state['owned_processes'])):
        raise ValueError('model client design, exact profile, ownership or cleanup differs')
    router=read('router-qualification.json')
    mapping=routing_map.load_verified(root/'routing-map-source',router['lock_sha256'],router['router_executable_sha256'])
    if {f'http://127.0.0.1:{p}/metrics':list(range(mapping['dp'])) for p in mapping['ports']}!=config['expected_ranks']:
        raise ValueError('client comparison rank coverage differs')
    names={'r'+str(r)+'-'+a for r in range(3) for a in ('before','two','after')}
    if {p.name for p in (root/'probes').iterdir()}!=names or {p.name for p in (root/'inputs').iterdir()}!={'0','1','2'}:
        raise ValueError('model client probe/input population differs')
    finished=[];rows=[];last=read('startup-timing.json')['semantic_ready_ns'];last_route=0
    router_bytes=(root/'router.log').read_bytes();all_pids=set();completed=state.get('completed_processes',[])
    for repetition in range(3):
        source=root/'inputs'/str(repetition);runtime.verify_evidence_manifest(source)
        tasks=json.loads((source/'fixed-tasks.json').read_text());generation=json.loads((source/'prompt-result.json').read_text())
        origin=json.loads((source/'run-config.json').read_text());cell=f"core-{config['layout']}-i{n}-c{c}-cold-o2048"
        if (generation.get('status')!='passed' or generation.get('cleanup')!='passed' or generation.get('task_family')!='fixed' or
                generation.get('tasks')!=c or generation.get('cell_id')!=cell or generation.get('repetition')!=repetition or
                origin.get('cell_id')!=cell or origin.get('layout')!=config['layout'] or origin.get('image')!=config['image'] or
                origin.get('repetition')!=repetition or len(tasks)!=c):
            raise ValueError('client input installed generation differs')
        for name in ('manifest.json','checkpoint-receipt.json','image-qualification.json'):
            if (source/name).read_bytes()!=(root/name).read_bytes():raise ValueError('client input dependency changed')
        for task in tasks:
            verify_fixed_task.verify(task)
            if (task.get('suite')!='core' or task.get('target_input_tokens')!=n or task.get('concurrency')!=c or
                    task.get('output_tokens')!=2048 or task.get('cache_designation')!='cold' or task.get('repetition')!=repetition):
                raise ValueError('client input fixed shape differs')
        placement=routing_map.select(mapping,c,repetition);sessions={'s'+str(r['task_session']):r['session_header'] for r in placement}
        rates=[]
        for arm,workers in (('before',1),('two',2),('after',1)):
            label='r'+str(repetition)+'-'+arm;p=root/'probes'/label
            client_source=p/'requests/measured/clients/client-source.py'
            if hashlib.sha256(client_source.read_bytes()).hexdigest()!=source_manifest['reproduce/model_clients.py']:
                raise ValueError('retained client source differs from execution toolkit pin')
            attributed=verify_fixed_attribution.verify(p);verify_fixed_attribution.verify_cache_profile(p,binding)
            inputs=json.loads((p/'requests/measured/fixed-inputs.json').read_text())
            namespace='fx'+hashlib.sha256(json.dumps([config['run_id'],identity+'-'+arm,repetition,startup['profile_sha256']],separators=(',',':')).encode()).hexdigest()[:24]
            if (inputs['tasks']!=tasks or inputs['session_ids']!=sessions or inputs['request_namespace']!=namespace or
                    inputs['clock_id']!=state['driver_identity']['boot_id']+':'+state['run_id']):
                raise ValueError('client probe workload, clock, namespace or affinity differs')
            if any(attributed['session_routes'][r['session_header']]!=[r['endpoint'],r['rank']] for r in placement):
                raise ValueError('client probe native placement differs')
            client=json.loads((p/'requests/measured/clients/model-client-contract.json').read_text())
            if (client.get('workers')!=workers or client.get('cpus')!=cpus or client.get('owner_root')!=config['run_dir'] or
                    client.get('url')!=FRONTDOOR):
                raise ValueError('client probe process count, CPU, owner or route differs')
            for folder in (p/'requests/measured/clients').glob('worker*'):
                receipt=json.loads((folder/'release-receipt.json').read_text());pid=receipt['identity']['pid']
                if pid in all_pids or receipt not in completed:raise ValueError('client probe child reuse or missing parent retirement')
                all_pids.add(pid)
            raw={t['id']:json.loads((p/'requests/measured'/(t['id']+'.json')).read_text())['raw'] for t in tasks}
            population=verify_fixed_response.population(tasks,raw,inputs['request_ids'],sessions,inputs['clock_id'])
            a=min(r['start_ns'] for r in raw.values());z=max(r['terminal_ns'] for r in raw.values())
            if not last<=a<z or population['valid_requests']!=c:raise ValueError('client probes overlap or contain invalid requests')
            last=z
            coverage=verify_telemetry.verify_window(root,a,z,config['expected_ranks'],
                types={**verify_telemetry.PILOT_TYPES,'vllm:request_success_total':'counter'})
            span=json.loads((p/'routing-span.json').read_text());x=span['start_byte'];y=span['end_byte']
            if not last_route<=x<=y<=len(router_bytes) or router_bytes[x:y]!=(p/'routing.log').read_bytes():
                raise ValueError('client probe route span differs from shared log')
            last_route=y
            rates.append(population['valid_tokens_per_second'])
            finished.append({'repetition':repetition,'arm':arm,'workers':workers,'path':'probes/'+label})
            rows.append({'repetition':repetition,'arm':arm,'workers':workers,'requests':c,
                'valid_tokens_per_second':rates[-1],'window_seconds':population['duration_seconds'],'coverage':coverage})
    if read('completed-client-probes.json')!=finished:raise ValueError('client completion ledger differs')
    observations=[]
    for repetition in range(3):
        a,b,z=[r['valid_tokens_per_second'] for r in rows if r['repetition']==repetition]
        drift=z/a-1;ratio=b/math.sqrt(a*z)
        observations.append({'repetition':repetition,'two_to_one_throughput_ratio':ratio,
            'closing_control_drift_fraction':drift,'within_five_percent_control_drift':abs(drift)<=.05})
    result={'status':'passed','cell_id':identity,'probes':rows,'comparisons':observations,
        'controls_stable':all(r['within_five_percent_control_drift'] for r in observations),
        'scope':'unscored matched client observations at declared shape only; cache disabled uniformly; no universal ceiling, saturation or scored acceptance'}
    if completion_required and read('model-client-check-result.json')!=result:raise ValueError('client comparison result differs')
    runtime.verify_evidence_manifest(root);return result
