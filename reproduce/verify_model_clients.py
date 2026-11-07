"""Independently recompute a fixed-model client capture from raw wire records.

No import of the client capturer or its summary arithmetic. Ownership of GPU
engines, sampler coverage and matched one/two-client conclusions are separate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runtime
import verify_fixed_response
import verify_fixed_task


def verify(directory,*,progress=None):
    if progress is not None and not callable(progress):raise ValueError('verification progress check must be callable')
    if progress is not None:progress()
    root=Path(directory);runtime.verify_evidence_manifest(root)
    c=json.loads((root/'model-client-contract.json').read_text())
    workers=c.get('workers');tasks=c.get('tasks',[]);ids={t.get('id') for t in tasks}
    if (c.get('schema')!='fixed-model-clients/v1' or type(workers) is not int or workers not in (1,2) or
            not workers<=len(tasks)<=128 or len(ids)!=len(tasks) or
            set(c.get('request_ids',{}))!=ids or set(c.get('session_ids',{}))!=ids or
            len(set(c['request_ids'].values()))!=len(tasks) or len(set(c['session_ids'].values()))!=len(tasks) or
            hashlib.sha256((root/'client-source.py').read_bytes()).hexdigest()!=c.get('worker_source_sha256') or
            not isinstance(c.get('clock_id'),str) or not c['clock_id'] or
            type(c.get('deadline_ns')) is not int or (root/'model-client-failure.json').exists()):
        raise ValueError('model client contract, source or population differs')
    for t in tasks:verify_fixed_task.verify(t)
    if {p.name for p in root.glob('worker*')}!={'worker'+str(i) for i in range(workers)}:
        raise ValueError('model client worker population differs')
    records={};resources=[];pids=set()
    for i in range(workers):
        if progress is not None:progress()
        p=root/('worker'+str(i));read=lambda name:json.loads((p/name).read_text())
        config=read('config.json');digest=hashlib.sha256((p/'config.json').read_bytes()).hexdigest()
        selected=tasks[i::workers];keys={t['id'] for t in selected}
        expected={**c,'schema':'fixed-model-client-worker/v1','tasks':selected,'threads':len(selected),
            'request_ids':{t['id']:c['request_ids'][t['id']] for t in selected},
            'session_ids':{t['id']:c['session_ids'][t['id']] for t in selected},
            'cpus':c['cpus'][i::workers] if c['cpus'] is not None else None}
        timeout=config.get('request_timeout_seconds')
        if (config!={**expected,'request_timeout_seconds':timeout} or type(timeout) not in (int,float) or
                not 0<timeout<=7200 or read('worker-result.json')!={'status':'passed','requests':len(selected)} or
                any((p/name).exists() for name in ('abort-request.json','failure.json'))):
            raise ValueError('model client worker config or result differs')
        ready=read('ready.json');gate={'config_sha256':digest,'worker_pid':ready.get('pid')}
        if type(ready.get('pid')) is not int or ready['pid']<=0 or read('start.json')!=gate:
            raise ValueError('model client readiness or gate differs')
        if ready['pid'] in pids:raise ValueError('duplicate model client process')
        pids.add(ready['pid'])
        if expected['cpus'] is not None and ready.get('cpus')!=sorted(expected['cpus']):
            raise ValueError('model client observed CPU affinity differs')
        if c['owner_root'] is not None:
            release=read('release-receipt.json');identity=read('owned-identity.json')
            if (read('exit.json')!=gate or release.get('identity')!=identity or identity.get('pid')!=ready['pid'] or
                    release.get('config_sha256')!=digest or release.get('exit_code')!=0 or
                    release.get('result_sha256')!=hashlib.sha256((p/'worker-result.json').read_bytes()).hexdigest()):
                raise ValueError('model client terminal ownership release differs')
        data=read('worker-resource.json');a=data.get('started_ns');z=data.get('completed_ns')
        if (type(a) is not int or type(z) is not int or not 0<a<z<=c['deadline_ns'] or
                any(type(data.get(k)) not in (int,float) or not math.isfinite(data[k]) or data[k]<0 for k in ('user_cpu_seconds','system_cpu_seconds')) or
                type(data.get('max_rss')) is not int or data['max_rss']<0 or data.get('max_rss_unit') not in ('KiB','bytes')):
            raise ValueError('model client resource clocks or values differ')
        paths=list((p/'capture').glob('s[0-9]*.json'))
        if {x.stem for x in paths}!=keys:raise ValueError('model client raw task coverage differs')
        for path in paths:
            if progress is not None:progress()
            row=json.loads(path.read_text());task=next(t for t in selected if t['id']==path.stem)
            raw=row['raw']
            # The complete population below evaluates every raw response once.
            # Bind each record to this worker's resource window before that
            # shared transport, content and usage recomputation.
            if (row.get('valid') is not True or row.get('clock_id')!=c['clock_id'] or
                    type(raw.get('start_ns')) is not int or type(raw.get('terminal_ns')) is not int or
                    not a<=raw['start_ns']<raw['terminal_ns']<=z):raise ValueError('model client response or resource window differs')
            records[task['id']]=raw
        resources.append(data)
    result=verify_fixed_response.population(tasks,records,c['request_ids'],c['session_ids'],c['clock_id'],progress=progress)
    if result['valid_requests']!=len(tasks):raise ValueError('model client global population differs')
    result.update(status='fixed_model_client_wire_verified',workers=workers,worker_resources=resources,
        scope='raw fixed-model client wire/content/CPU observations; no GPU ownership, telemetry, native routing, measured client ceiling, paired comparison or scored acceptance')
    runtime.verify_evidence_manifest(root);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');a=p.parse_args();print(json.dumps(verify(a.directory),sort_keys=True))
