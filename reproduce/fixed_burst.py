"""Capture one fixed-output burst with distinct wire IDs and independent scoring.

This component cannot accept a scored campaign cell: frozen profile, input
provenance, cache/routing/telemetry and shared cleanup are the launcher's gates.
"""
import hashlib
import json
from pathlib import Path
import re
import pilot_burst
import runtime
import verify_fixed_response
import verify_fixed_task
import fixed_workload


def namespace(run_id,cell_id,repetition,profile_sha256):
    if (not isinstance(run_id,str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',run_id) or
        not isinstance(cell_id,str) or not cell_id or type(repetition) is not int or not 0<=repetition<3 or
        not re.fullmatch(r'[a-f0-9]{64}',str(profile_sha256))):raise ValueError('invalid fixed request namespace')
    return 'fx'+hashlib.sha256(json.dumps([run_id,cell_id,repetition,profile_sha256],separators=(',',':')).encode()).hexdigest()[:24]


def run(session,url,tasks,output_dir,timeout,clock_id,request_namespace,session_ids,*,client_workers=None,client_cpus=None):
    if (not isinstance(request_namespace,str) or not re.fullmatch(r'fx[a-f0-9]{24}',request_namespace) or
        not tasks or len({t.get('prompt_version') for t in tasks})!=1 or any(t.get('prompt_version') not in fixed_workload.EXECUTION_VERSIONS or t.get('fixed_output') is not True or
            t.get('concurrency')!=len(tasks) or t.get('id')!='s'+str(t.get('session')) for t in tasks) or
        len(set(session_ids.values()))!=len(tasks)):
        raise ValueError('fixed capture needs prepared tasks and explicit distinct affinity identities')
    for task in tasks:verify_fixed_task.verify(task)
    if client_workers is not None:
        import model_clients
        return model_clients.capture_fixed(session,url,tasks,output_dir,timeout,clock_id,request_namespace,
            session_ids,workers=client_workers,cpus=client_cpus)
    if client_cpus is not None:raise ValueError('client CPU partition requires explicit subprocess mode')
    ids={t['id']:request_namespace+'s'+str(t['session']) for t in tasks}
    root=Path(output_dir)
    def oracle(task,raw):return verify_fixed_response.evaluate(task,raw,ids[task['id']],session_ids[task['id']])
    summary=pilot_burst.run(session,url,tasks,root,timeout,clock_id,oracle=oracle,request_ids=ids,session_ids=session_ids)
    rows={t['id']:json.loads((root/(t['id']+'.json')).read_text()) for t in tasks}
    report=verify_fixed_response.population(tasks,{key:value['raw'] for key,value in rows.items()},ids,session_ids,clock_id)
    for key in ('requests','valid_requests','valid_tokens','duration_seconds','valid_tokens_per_second','invalid_partial_tokens'):
        if summary[key]!=report[key]:raise runtime.RuntimeErrorClosed('fixed capture summary differs from independent raw evidence')
    for record in report['rows']:
        stored=rows[record['task_id']]
        if stored.get('request_id')!=record['request_id'] or stored.get('valid')!=record['correct']:
            raise runtime.RuntimeErrorClosed('fixed request outcome differs from independent wire scoring')
    runtime.atomic_json(root/'fixed-inputs.json',{'tasks':tasks,'request_ids':ids,'session_ids':session_ids,'clock_id':clock_id,'request_namespace':request_namespace})
    runtime.atomic_json(root/'fixed-verification.json',report)
    runtime.atomic_json(root/'fixed-capture-result.json',{'status':'requests_captured_cleanup_pending' if report['valid_requests']==len(tasks) else 'failed_requests_retained',
        'scope':'raw request capture only; no scored acceptance before full launcher verification'})
    return report
