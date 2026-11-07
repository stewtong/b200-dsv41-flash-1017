"""Supervised natural-stop quality capture; semantic misses stay in the denominator."""
import pilot_burst
from pathlib import Path
import hashlib,json,re,uuid
import runtime
import verify_quality


def run(session, url, tasks, output_dir, timeout, clock_id, repetition, target,
        request_namespace=None,session_ids=None):
    if not tasks:raise ValueError('quality task set is empty')
    concurrency=len(tasks)
    # Contract validation occurs before dispatch using an explicitly failed raw
    # response. It checks task identities and prompt declarations, not tokenized
    # depth provenance; installed-tokenizer generation remains a caller gate.
    for task in tasks:
        verify_quality.evaluate(task,{'request_id':task['id'],'complete':False},repetition,concurrency,target)
    caches={task.get('cache_designation','cold') for task in tasks}
    if len(caches)!=1:raise ValueError('quality population mixes cache states')
    namespace='ql'+uuid.uuid4().hex[:24] if request_namespace is None else request_namespace
    ids={t['id'] for t in tasks}
    affinity={i:namespace+'s'+i[1:] for i in ids} if session_ids is None else session_ids
    if (not isinstance(namespace,str) or not re.fullmatch(r'ql[a-f0-9]{24}',namespace) or
        set(affinity)!=ids or len(set(affinity.values()))!=len(tasks) or
        any(not isinstance(s,str) or not s or not s.isascii() or any(not 33<=ord(c)<=126 for c in s) for s in affinity.values()) or
        not isinstance(clock_id,str) or not clock_id):raise ValueError('quality request namespace or affinity invalid')
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    runtime.atomic_json(root/'quality-capture-contract.json',{
        'schema':'quality-capture/v2','request_namespace':namespace,'session_ids':affinity,
        'clock_id':clock_id,'repetition':repetition,'target':target,'cache':next(iter(caches)),
        'scheduled_measured_requests':len(tasks),
        'tasks_sha256':hashlib.sha256(json.dumps(tasks,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
    measured_ids={t['id']:namespace+'m'+str(t['session']) for t in tasks}
    if caches=={'warm_prefix'}:
        conditioning=[{'id':t['id'],'body':t['conditioning_body'],'input_tokens':t['prefix_tokens'],
                       'output_tokens':128,'expected_answer':t['expected_conditioning_answer'],
                       'fixed_output':False} for t in tasks]
        summary=pilot_burst.run(session,url,conditioning,root/'conditioning',timeout,clock_id,
            request_ids={t['id']:namespace+'c'+str(t['session']) for t in tasks},session_ids=affinity)
        if summary['valid_requests']!=len(tasks):raise runtime.RuntimeErrorClosed('quality conditioning failed')
    def adjudicate(task,raw):
        return verify_quality.evaluate(task,raw,repetition,concurrency,target,require_wire=True,
            request_id=measured_ids[task['id']],session_id=affinity[task['id']])
    return pilot_burst.run(session,url,tasks,root/'measured',timeout,clock_id,oracle=adjudicate,
        request_ids=measured_ids,session_ids=affinity)
