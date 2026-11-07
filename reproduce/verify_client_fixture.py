"""Replay fixture request/wire populations and CPU observations independently."""
import hashlib,json,math,re
from pathlib import Path
import runtime,verify_wire


def verify(directory):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda path:json.loads((root/path).read_text())
    c=read('fixture-contract.json');workers=c.get('workers');count=c.get('requests');concurrency=c.get('concurrency')
    tasks=c.get('tasks');deadline=c.get('deadline_ns')
    if (c.get('schema')!='synthetic-client-fixture/v1' or workers not in (1,2) or
            type(count) is not int or type(concurrency) is not int or not workers<=concurrency<=count<=1024 or
            not isinstance(tasks,list) or len(tasks)!=count or [t.get('id') for t in tasks]!=['p'+str(i) for i in range(count)] or
            type(deadline) is not int or type(c.get('node_owned')) is not bool or
            type(c.get('events')) is not int or not 1<=c['events']<=2048 or
            type(c.get('piece_bytes')) is not int or not 1<=c['piece_bytes']<=1024 or
            type(c.get('input_bytes')) is not int or not 1<=c['input_bytes']<=8_000_000):raise ValueError('fixture declared population differs')
    expected_body={'model':'/model','messages':[{'role':'user','content':'a'*c['input_bytes']}],
        'stream':True,'stream_options':{'include_usage':True},'max_tokens':c['events'],'min_tokens':c['events'],
        'ignore_eos':True,'temperature':1.0,'top_p':1.0,'chat_template_kwargs':{'thinking':False,'reasoning_effort':50}}
    if any(t!={'id':'p'+str(i),'session_id':'fixture-session-'+str(i),'body':{**expected_body,'seed':17+i}} for i,t in enumerate(tasks)):
        raise ValueError('fixture declared body or affinity differs')
    if {p.name for p in root.iterdir()}!={'fixture-contract.json','fixture-result.json','SHA256SUMS'}|{'worker'+str(i) for i in range(workers)}:
        raise ValueError('fixture root file population differs')
    rows=[];resources=[]
    for index in range(workers):
        folder=Path('worker'+str(index));config_bytes=(root/folder/'config.json').read_bytes();config=json.loads(config_bytes)
        expected=tasks[index::workers];resource=read(folder/'worker-resource.json');ready=read(folder/'ready.json')
        gate=read(folder/'start.json');result=read(folder/'worker-result.json')
        cpus=c['cpus'][index::workers] if c['cpus'] is not None else None
        if (config.get('schema')!='synthetic-client-worker/v1' or config.get('tasks')!=expected or
                config.get('url')!=c['url'] or config.get('events')!=c['events'] or config.get('piece_bytes')!=c['piece_bytes'] or
                config.get('threads')!=concurrency//workers+(index<concurrency%workers) or config.get('cpus')!=cpus or
                config.get('node_owned')!=c['node_owned'] or config.get('deadline_ns')!=deadline or
                gate!={'config_sha256':hashlib.sha256(config_bytes).hexdigest(),'worker_pid':ready.get('pid')} or
                type(ready.get('pid')) is not int or ready['pid']<=1 or cpus is not None and ready.get('cpus')!=sorted(cpus) or
                resource.get('failed') is not False or resource.get('cpus')!=cpus or
                any(type(resource.get(k)) not in (int,float) or not math.isfinite(resource[k]) or resource[k]<0
                    for k in ('user_cpu_seconds','system_cpu_seconds','max_rss')) or
                resource.get('max_rss_unit') not in ('KiB','bytes') or
                type(resource.get('started_ns')) is not int or type(resource.get('completed_ns')) is not int or
                not 0<=resource['started_ns']<resource['completed_ns']<=deadline):
            raise ValueError('fixture worker ownership, affinity, resource or config binding differs')
        names={'config.json','ready.json','start.json','worker-resource.json','worker-result.json','worker.log'}|{t['id']+'.json' for t in expected}
        if c['node_owned']:
            names.update(('owned-identity.json','exit.json','release-receipt.json'));identity=read(folder/'owned-identity.json')
            if identity.get('pid')!=ready['pid'] or runtime.process_alive(identity):raise ValueError('fixture node worker identity or release differs')
            release=read(folder/'release-receipt.json')
            if (read(folder/'exit.json')!=gate or release.get('identity')!=identity or
                    release.get('config_sha256')!=hashlib.sha256(config_bytes).hexdigest() or
                    release.get('result_sha256')!=hashlib.sha256((root/folder/'worker-result.json').read_bytes()).hexdigest() or
                    release.get('exit_code')!=0 or type(release.get('completed_ns')) is not int or
                    not resource['completed_ns']<=release['completed_ns']<=deadline):
                raise ValueError('fixture exact worker retirement differs')
        if {p.name for p in (root/folder).iterdir()}!=names:raise ValueError('fixture worker file population differs')
        checked=[]
        for task in expected:
            raw=read(folder/(task['id']+'.json'));verify_wire.verify_complete(raw)
            payload=json.dumps(task['body'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            headers={'Content-Type':'application/json','X-Request-Id':task['id'],'X-Session-Id':task['session_id']}
            if (raw.get('request_id')!=task['id'] or raw.get('session_id')!=task['session_id'] or raw.get('request_headers')!=headers or
                    raw.get('request_body_sha256')!=hashlib.sha256(payload).hexdigest() or raw.get('request_payload_bytes')!=len(payload) or
                    raw.get('request_serialization')!='canonical-json-ensure-ascii/v1' or raw.get('http_status')!=200 or
                    not resource['started_ns']<=raw['start_ns']<raw['terminal_ns']<=resource['completed_ns']):
                raise ValueError('fixture transmitted request or clocks differ')
            content=[];usage=[];finishes=[]
            for event in raw['events']:
                if event['kind']!='event':continue
                body=event['data']
                if body.get('usage') is not None:usage.append(body['usage'])
                for choice in body.get('choices',[]):
                    delta=choice.get('delta') or {}
                    if (choice.get('index',0)!=0 or delta.get('role') not in (None,'assistant') or
                            any(delta.get(k) for k in ('reasoning','reasoning_content','tool_calls'))):raise ValueError('fixture output channel differs')
                    piece=delta.get('content')
                    if piece:content.append(piece)
                    if choice.get('finish_reason') is not None:finishes.append(choice['finish_reason'])
            if (usage!=[{'prompt_tokens':1,'completion_tokens':c['events'],'total_tokens':1+c['events']}] or finishes!=['length'] or
                    len(content)!=c['events'] or any(piece!='x'*c['piece_bytes'] for piece in content)):
                raise ValueError('fixture complete event/usage population differs')
            checked.append({'id':task['id'],'start_ns':raw['start_ns'],'end_ns':raw['terminal_ns'],
                'wire_bytes':raw['wire_byte_count'],'request_payload_bytes':raw['request_payload_bytes']})
        if result!={'status':'passed','requests':sorted(checked,key=lambda r:r['id'])}:raise ValueError('fixture worker summary differs from raw')
        rows.extend(checked);resources.append(resource)
    if read('fixture-result.json')!={'status':'captured_verification_pending'} or len(rows)!=count:
        raise ValueError('fixture completion or denominator differs')
    seconds=(max(r['end_ns'] for r in rows)-min(r['start_ns'] for r in rows))/1e9
    wire=sum(r['wire_bytes'] for r in rows);request_bytes=sum(r['request_payload_bytes'] for r in rows)
    runtime.verify_evidence_manifest(root)
    return {'status':'synthetic_client_capture_verified','workers':workers,'concurrency':concurrency,'requests':count,
        'request_window_seconds':seconds,'requests_per_second':count/seconds,
        'response_body_bytes':wire,'response_body_bytes_per_second':wire/seconds,
        'request_payload_bytes':request_bytes,'request_payload_bytes_per_second':request_bytes/seconds,
        'visible_fixture_events':count*c['events'],'visible_fixture_events_per_second':count*c['events']/seconds,
        'worker_cpu_seconds':sum(r['user_cpu_seconds']+r['system_cpu_seconds'] for r in resources),
        'affinity_budget_qualified':c['node_owned'] and c['cpus'] is not None,
        'scope':'bounded synthetic HTTP-body/event and client CPU observations; not model-token throughput, NIC wire rate, maximum capacity, real-model second-client comparison or SLO acceptance'}


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),sort_keys=True))
