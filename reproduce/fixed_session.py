"""Condition a fixed workload before its measured burst, within owned serving.

This request capsule excludes tokenizer, profile, rank/cache, telemetry and
shared-cleanup acceptance, which remain mandatory launcher responsibilities.
"""
import json
from pathlib import Path
import fixed_burst,pilot_burst,runtime,verify_fixed_task,verify_fixed_session
import fixed_workload


def run(session,url,tasks,directory,timeout,clock_id,request_namespace,session_ids,boundary_observer=None,*,client_workers=None,client_cpus=None):
    if (not tasks or len({t['id'] for t in tasks})!=len(tasks) or
        {t['id'] for t in tasks}!={'s'+str(i) for i in range(len(tasks))} or
        any(t.get('concurrency')!=len(tasks) for t in tasks) or
        any(t.get('prompt_version') not in fixed_workload.EXECUTION_VERSIONS for t in tasks) or
        any(len({t.get(key) for t in tasks})!=1 for key in ('suite','target_input_tokens','output_tokens','repetition','prompt_version')) or
        set(session_ids)!={t['id'] for t in tasks} or len(set(session_ids.values()))!=len(tasks)):
        raise ValueError('fixed session task and affinity coverage differs')
    cache={t.get('cache_designation') for t in tasks}
    if len(cache)!=1 or not cache.issubset({'cold','warm_prefix'}):raise ValueError('fixed session cache state differs')
    if client_workers is not None:
        if (type(client_workers) is not int or client_workers not in (1,2) or client_workers>len(tasks) or
                not isinstance(client_cpus,list) or len(set(client_cpus))!=len(client_cpus) or len(client_cpus)<client_workers or
                any(type(i) is not int or i<0 for i in client_cpus)):
            raise ValueError('subprocess measurement requires fixed tasks and explicit CPU partition')
    elif client_cpus is not None:raise ValueError('subprocess CPU partition lacks process mode')
    for task in tasks:verify_fixed_task.verify(task)
    # Validate namespace and all affinity header values before creating evidence.
    import re
    if not isinstance(request_namespace,str) or not re.fullmatch(r'fx[a-f0-9]{24}',request_namespace):raise ValueError('invalid fixed session namespace')
    if any(not isinstance(s,str) or not s or not s.isascii() or any(not 33<=ord(c)<=126 for c in s) for s in session_ids.values()):raise ValueError('invalid fixed session affinity')
    if boundary_observer is not None and not callable(boundary_observer):raise ValueError('boundary observer must be callable')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    contract={'schema':'fixed-session-capture/v1','cache':next(iter(cache)),
              'request_namespace':request_namespace,'session_ids':session_ids,'clock_id':clock_id,
              'scheduled_measured_requests':len(tasks)}
    if client_workers is not None:contract['client_processes']={'workers':client_workers,'cpus':client_cpus}
    runtime.atomic_json(root/'fixed-session-contract.json',contract)
    def observe(name):
        session.ensure_healthy()
        if boundary_observer is not None:boundary_observer(name)
    try:
        if cache=={'warm_prefix'}:
            observe('before_conditioning')
            conditions=[{'id':t['id'],'body':t['conditioning_body'],'input_tokens':t['prefix_tokens'],
                         'output_tokens':128,'expected_answer':t['expected_conditioning_answer'],'fixed_output':False} for t in tasks]
            ids={t['id']:request_namespace+'c'+str(t['session']) for t in tasks}
            source={t['id']:t for t in tasks}
            def oracle(task,raw):return verify_fixed_session.conditioning_score(source[task['id']],raw,ids[task['id']],session_ids[task['id']])
            result=pilot_burst.run(session,url,conditions,root/'conditioning',timeout,clock_id,
                oracle=oracle,request_ids=ids,session_ids=session_ids)
            if result['valid_requests']!=len(tasks):raise runtime.RuntimeErrorClosed('fixed conditioning failed')
            runtime.seal_evidence(root/'conditioning')
            observe('after_conditioning')
        observe('before_measured')
        options={} if client_workers is None else {'client_workers':client_workers,'client_cpus':client_cpus}
        measured=fixed_burst.run(session,url,tasks,root/'measured',timeout,clock_id,request_namespace,session_ids,**options)
        runtime.seal_evidence(root/'measured')
        if measured['valid_requests']!=len(tasks):
            raise runtime.RuntimeErrorClosed('fixed measured response oracle rejected a request; raw score retained')
        observe('after_measured')
        runtime.atomic_json(root/'fixed-session-result.json',{'status':'captured_cleanup_pending',
            'measured_requests':measured['requests'],'conditioning_requests':len(tasks) if cache=={'warm_prefix'} else 0,
            'scope':'conditioning excluded from measured makespan and token-rate denominator; achieved cache hits and routing remain separate'})
        runtime.seal_evidence(root)
        return verify_fixed_session.verify(root,progress=session.ensure_healthy)
    except BaseException as exc:
        runtime.atomic_json(root/'fixed-session-failure.json',{'status':'failed','error_type':type(exc).__name__,
            'scope':'all captured request evidence retained; no successful session acceptance'})
        raise
