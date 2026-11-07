"""Audit owned synthetic client/router captures without their dispatcher."""
import base64,hashlib,json,math,os
from pathlib import Path
import runtime,verify_client_fixture

SHAPES=({'concurrency':8,'requests':32,'input_bytes':32000},
        {'concurrency':32,'requests':128,'input_bytes':32000},
        {'concurrency':128,'requests':256,'input_bytes':32000},
        {'concurrency':8,'requests':16,'input_bytes':4_000_000})
LAYOUTS={'2xtp4':(2,1),'4xdep2':(4,2),'2xdep4':(2,4),'dep8':(1,8),'tp8':(1,1),'4xtp2':(4,1)}
SCOPE='synthetic client/router observations only; no real-model ceiling, saturation, tuning or scored acceptance'
# Exact v0.1.14 ChatCompletionRequest defaults, not arbitrary extra fields.
# https://github.com/vllm-project/router/blob/0a6f74ec88c87b5477dac8e24466ab760b9f1f35/src/protocols/spec.rs
NATIVE_DEFAULTS={'logprobs':False,'no_stop_trim':False,'ignore_eos':False,'add_generation_prompt':True,
    'continue_final_message':False,'skip_special_tokens':True,'separate_reasoning':True,'stream_reasoning':True,
    'return_hidden_states':False,'include_reasoning':True}
BODY_CONTRACT={'schema':'native-router-body-transform/v1','router_version':'0.1.14',
    'source_revision':'0a6f74ec88c87b5477dac8e24466ab760b9f1f35',
    'protocol_source_sha256':'baa89f7107a2c28ca40f676fa484b52aae7cac2f8ed64d312e7761deedaeac68',
    'added_defaults':NATIVE_DEFAULTS,
    'scope':'both raw HTTP bodies retained; explicit fields unchanged; only pinned native defaults may be added'}


def audit_arrivals(requests,arrivals,ports,dp,server_cpus,session_routes,expected_bodies):
    seen=set()
    if len(arrivals)!=len(requests) or set(expected_bodies)!=set(requests):raise ValueError('client fixture router lost or retried requests')
    for arrival in arrivals:
        key=arrival.get('request_id');raw=requests.get(key)
        if (raw is None or key in seen or arrival.get('session_id')!=raw['session_id'] or
                arrival.get('backend_port') not in ports or arrival.get('server_cpus')!=sorted(server_cpus) or
                not raw['start_ns']<=arrival['observed_ns']<=raw['terminal_ns']):
            raise ValueError('client fixture backend receipt differs from transmitted wire or CPU placement')
        encoded=arrival.get('body_base64')
        if not isinstance(encoded,str) or len(encoded)>11_000_000:raise ValueError('client fixture backend payload missing or excessive')
        def unique(pairs):
            value={}
            for name,item in pairs:
                if name in value:raise ValueError('duplicate backend JSON field')
                value[name]=item
            return value
        def invalid_constant(value):raise ValueError('non-finite backend JSON')
        try:
            payload=base64.b64decode(encoded,validate=True)
            body=json.loads(payload.decode('utf-8'),object_pairs_hook=unique,parse_constant=invalid_constant)
        except (ValueError,UnicodeError) as exc:raise ValueError('client fixture backend payload malformed') from exc
        original=expected_bodies[key];canonical=json.dumps(original,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if (len(payload)!=arrival.get('body_bytes') or hashlib.sha256(payload).hexdigest()!=arrival.get('body_sha256') or
                len(canonical)!=raw['request_payload_bytes'] or hashlib.sha256(canonical).hexdigest()!=raw['request_body_sha256'] or
                json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False)!=
                    json.dumps({**NATIVE_DEFAULTS,**original},sort_keys=True,separators=(',',':'),allow_nan=False)):
            raise ValueError('client fixture native body transformation differs from exact request and pinned defaults')
        seen.add(key);headers=arrival.get('rank_headers');rank=0
        if dp==1:
            if headers not in ([],['0']):raise ValueError('unexpected TP fixture rank header')
        else:
            if not isinstance(headers,list) or len(headers)!=1 or not headers[0].isdigit() or not 0<=int(headers[0])<dp:
                raise ValueError('client fixture native DP header missing')
            rank=int(headers[0])
        route=(arrival['backend_port'],rank);session=arrival['session_id']
        if session in session_routes and session_routes[session]!=route:raise ValueError('client fixture native affinity changed')
        session_routes[session]=route
    if seen!=set(requests):raise ValueError('client fixture request denominator differs')
    return session_routes


def verify(directory):
    root=Path(directory);runtime.verify_evidence_manifest(root);read=lambda p:json.loads((root/p).read_text())
    config=read('run-config.json');state=read('execution-state.json');cpus=read('cpu-partition.json')
    q=read('router-qualification.json');installed=read('router-runtime-check.json');source=(root/'toolkit-MANIFEST.json').read_bytes()
    if read('router-body-contract.json')!=BODY_CONTRACT:raise ValueError('client fixture native body contract differs')
    if (hashlib.sha256(source).hexdigest()!=config.get('toolkit_manifest_sha256') or
            hashlib.sha256((root/'plan.md').read_bytes()).hexdigest()!=config.get('plan_sha256') or
            installed!={'installed_packages':q['installed_packages'],'executable_sha256':q['router_executable_sha256']} or
            q['installed_packages'].get('vllm-router')!='0.1.14' or
            state.get('run_id')!=config.get('run_id') or state.get('phase')!='client-fixture' or state.get('container_ids')!=[] or
            not cpus.get('client_cpus') or not cpus.get('server_cpus') or set(cpus['client_cpus'])&set(cpus['server_cpus'])):
        raise ValueError('owned client fixture source, CPU or router binding differs')
    if config.get('layout') not in LAYOUTS:raise ValueError('unknown client fixture geometry')
    servers,dp=LAYOUTS[config['layout']];ports=list(range(31000,31000+servers));argv=read('router-argv.json')
    expected=[q['router_executable'],'--host','127.0.0.1','--port','31100','--worker-urls',
        *['http://127.0.0.1:'+str(p) for p in ports],'--policy','consistent_hash','--intra-node-data-parallel-size',str(dp),
        '--request-timeout-secs','90','--disable-retries']
    if argv!=expected:raise ValueError('client fixture front-door path differs')
    router=read('router-identity.json');affinity=read('router-affinity.json')
    if affinity!={'pid':router['pid'],'cpus':sorted(cpus['client_cpus'])}:raise ValueError('client fixture router affinity differs')
    design={'shapes':list(SHAPES),'repetitions':3,'arms':[{'name':'before','workers':1},{'name':'two','workers':2},{'name':'after','workers':1}],
        'events':128,'piece_bytes':32,'probe_timeout_seconds':60,
        'scope':'CPU/client synthetic lower-bound observations; real-model second-client check still required'}
    if read('fixture-design.json')!=design:raise ValueError('client fixture preregistered design differs')
    projection=read('resource-projection.json')
    if (projection.get('projected_additional_bytes')!=10_000_000_000 or
            projection['disk_free_bytes']-projection['projected_additional_bytes']<projection['disk_total_bytes']*.15):
        raise ValueError('client fixture projected storage margin differs')
    completed=read('completed-probes.json');expected_keys=[(shape,repetition,arm,workers)
        for shape in range(4) for repetition in range(3) for arm,workers in (('before',1),('two',2),('after',1))]
    if len(completed)!=len(expected_keys):raise ValueError('client fixture probe population incomplete')
    owned=[router];finished=[];observations={};last=0;session_routes={};source_files=json.loads(source)
    for row,(shape_index,repetition,arm,workers) in zip(completed,expected_keys):
        label='shape'+str(shape_index)+'-r'+str(repetition)+'-'+arm;shape=SHAPES[shape_index];path=Path('probes')/label
        if (row.get('shape_index')!=shape_index or row.get('repetition')!=repetition or row.get('arm')!=arm or
                row.get('path')!=str(path) or row.get('arrivals_file')!=label+'-arrivals.json'):
            raise ValueError('client fixture order or path differs')
        c=read(path/'fixture-contract.json');observation=verify_client_fixture.verify(root/path)
        if (row.get('observation')!=observation or c.get('workers')!=workers or c.get('cpus')!=cpus['client_cpus'] or
                c.get('node_owned') is not True or c.get('url')!='http://127.0.0.1:31100/v1/chat/completions' or
                any(c.get(k)!=v for k,v in shape.items()) or c.get('events')!=128 or c.get('piece_bytes')!=32):
            raise ValueError('client fixture captured workload or observation differs')
        requests={};bodies={};starts=[];ends=[]
        for worker_index in range(workers):
            folder=path/('worker'+str(worker_index));worker=read(folder/'config.json');identity=read(folder/'owned-identity.json')
            if worker.get('worker_source_sha256')!=source_files.get('reproduce/client_fixture.py') or worker.get('owner_root')!=config['run_dir']:
                raise ValueError('client fixture worker source or ownership domain differs')
            release=read(folder/'release-receipt.json')
            if release.get('worker_path')!=str(folder) or release.get('identity')!=identity:
                raise ValueError('client fixture completed ownership path differs')
            finished.append(release);resource=read(folder/'worker-resource.json');starts.append(resource['started_ns']);ends.append(resource['completed_ns'])
            for task in worker['tasks']:
                requests[task['id']]=read(folder/(task['id']+'.json'));bodies[task['id']]=task['body']
        if not last<=min(starts)<max(ends)<state['deadline_ns']-120_000_000_000:
            raise ValueError('client fixture probes overlap or exceed operational margin')
        last=max(ends);arrivals=read(row['arrivals_file']);seen=set()
        audit_arrivals(requests,arrivals,ports,dp,cpus['server_cpus'],session_routes,bodies)
        observations[(shape_index,repetition,arm)]=observation
    if {p.name for p in (root/'probes').iterdir()}!={'shape'+str(s)+'-r'+str(r)+'-'+a for s,r,a,w in expected_keys}:
        raise ValueError('client fixture probe directory population differs')
    all_identities=owned+[r['identity'] for r in finished]
    if (len({json.dumps(x,sort_keys=True) for x in all_identities})!=len(all_identities) or
            sorted(owned,key=lambda x:x['pid'])!=sorted(state['owned_processes'],key=lambda x:x['pid']) or
            state.get('completed_processes')!=finished or
            any(runtime.process_alive(x) for x in [state['watchdog_identity'],*all_identities]) or
            runtime.process_alive(state['driver_identity']) and state['driver_identity']!=runtime.process_identity(os.getpid())):
        raise ValueError('client fixture process population or safe release differs')
    cleanup=read('cleanup-result.json');idle=read('fresh-idle-after-client-fixture.json');outcomes=cleanup.get('outcomes',[])
    if (state.get('stopping') is not True or state.get('watchdog_stop') is not True or cleanup.get('success') is not True or
            cleanup.get('deadline_seconds')!=120 or not 0<=cleanup.get('elapsed_seconds',float('inf'))<=120 or
            len(outcomes)!=len(owned) or {r.get('pid') for r in outcomes}!={r['pid'] for r in owned} or
            any(r.get('status')!='stopped' or 'container_id' in r for r in outcomes) or
            idle.get('compute_processes')!='' or len(idle.get('memory_used_mib',[]))!=8 or
            any(type(n) is not int or not 0<=n<=2048 for n in idle['memory_used_mib']) or
            not cleanup['monotonic_ns']<=idle['monotonic_ns']<=state['deadline_ns']+120_000_000_000 or
            any((root/name).exists() for name in ('watchdog-failure.json','driver-failure.json','primary-failure.json','sampler-failure.json'))):
        raise ValueError('client fixture cleanup or fresh eight-device idle proof differs')
    if read('client-fixture-result.json')!={'status':'captured_verification_pending','probes':36,'scope':SCOPE}:
        raise ValueError('client fixture terminal scope differs')
    ratios=[]
    for shape in range(4):
        for repetition in range(3):
            a,b,c=[observations[(shape,repetition,arm)] for arm in ('before','two','after')]
            baseline=math.sqrt(a['visible_fixture_events_per_second']*c['visible_fixture_events_per_second'])
            ratios.append({'shape_index':shape,'repetition':repetition,'two_to_one_event_rate_ratio':b['visible_fixture_events_per_second']/baseline,
                'closing_one_process_drift_fraction':c['visible_fixture_events_per_second']/a['visible_fixture_events_per_second']-1})
    runtime.verify_evidence_manifest(root)
    return {'status':'synthetic_node_client_path_verified','probes':36,'comparisons':ratios,
            'scope':SCOPE+'; tested rates are lower bounds for this fixture, not a proven client ceiling or proof of real-load headroom'}


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),sort_keys=True))
