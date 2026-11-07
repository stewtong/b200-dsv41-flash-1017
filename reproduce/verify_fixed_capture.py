"""Audit a sealed fixed-request capture; scored acceptance is a separate check."""
import json
from pathlib import Path
import re
import runtime
import verify_fixed_response


def verify(directory,*,progress=None):
    if progress is not None and not callable(progress):raise ValueError('verification progress check must be callable')
    if progress is not None:progress()
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda name:json.loads((root/name).read_text())
    inputs=read('fixed-inputs.json');tasks=inputs['tasks'];namespace=inputs.get('request_namespace')
    if (not isinstance(namespace,str) or not re.fullmatch(r'fx[a-f0-9]{24}',namespace) or
        len({t['id'] for t in tasks})!=len(tasks) or any(t['id']!='s'+str(t['session']) for t in tasks) or
        {t['id'] for t in tasks}!={'s'+str(i) for i in range(len(tasks))} or
        any(t.get('concurrency')!=len(tasks) for t in tasks) or
        inputs['request_ids']!={t['id']:namespace+'s'+str(t['session']) for t in tasks}):
        raise ValueError('fixed request namespace or logical population differs')
    records={}
    for task in tasks:
        if progress is not None:progress()
        records[task['id']]=read(task['id']+'.json')
    report=verify_fixed_response.population(tasks,{k:v['raw'] for k,v in records.items()},inputs['request_ids'],inputs['session_ids'],inputs['clock_id'],progress=progress)
    if report!=read('fixed-verification.json'):raise ValueError('retained fixed verification differs from raw recomputation')
    summary=read('summary.json')['summary']
    for key in ('requests','valid_requests','valid_tokens','duration_seconds','valid_tokens_per_second','invalid_partial_tokens'):
        if summary[key]!=report[key]:raise ValueError('retained fixed summary differs from raw recomputation')
    for outcome in report['rows']:
        if progress is not None:progress()
        record=records[outcome['task_id']];raw=record['raw'];task=next(t for t in tasks if t['id']==outcome['task_id'])
        # Population already recomputed this complete score from the same raw
        # record, task and bound identities. Preserve all score fields when
        # checking the stored per-request projection.
        score={k:v for k,v in outcome.items() if k not in
            ('task_id','request_id','clock_id','start_ns','end_ns')}
        if (record.get('task_id')!=task['id'] or record.get('request_id')!=outcome['request_id'] or
            record.get('clock_id')!=inputs['clock_id'] or record.get('start_ns')!=raw['start_ns'] or record.get('end_ns')!=raw['terminal_ns'] or
            record.get('valid')!=outcome['correct'] or record.get('completion_tokens')!=outcome['completion_tokens'] or record.get('quality_score')!=score):
            raise ValueError('retained fixed request fields differ from raw evidence')
        if outcome['correct']:
            if not isinstance(record.get('timing'),dict) or any(record['timing'].get(k)!=v for k,v in score['timing'].items() if k!='request_seconds'):
                raise ValueError('retained fixed timing differs from raw clocks')
        elif record.get('timing') is not None:raise ValueError('invalid fixed request has a successful timing')
    expected='requests_captured_cleanup_pending' if report['valid_requests']==len(tasks) else 'failed_requests_retained'
    if read('fixed-capture-result.json').get('status')!=expected:raise ValueError('fixed capture outcome hides invalid requests')
    if (root/'clients').exists():
        import verify_model_clients
        client=verify_model_clients.verify(root/'clients',progress=progress)
        contract=json.loads((root/'clients/model-client-contract.json').read_text())
        if any(contract.get(k)!=inputs[k] for k in ('tasks','request_ids','session_ids','clock_id')):
            raise ValueError('subprocess contract differs from fixed inputs')
        for path in (root/'clients').glob('worker*/capture/s[0-9]*.json'):
            if progress is not None:progress()
            if path.read_bytes()!=(root/path.name).read_bytes():raise ValueError('subprocess raw projection differs')
        for key in ('requests','valid_requests','valid_tokens','duration_seconds','valid_tokens_per_second','invalid_partial_tokens'):
            if client[key]!=report[key]:raise ValueError('subprocess denominator differs from fixed capture')
    runtime.verify_evidence_manifest(root)
    return {'status':'raw_fixed_capture_verified','requests':report['requests'],'valid_requests':report['valid_requests'],
        'valid_tokens_per_second':report['valid_tokens_per_second'],
        'scope':('sealed performance transport/UTF-8/nonempty/usage/channel evidence; no correct-task goodput claim' if tasks[0]['prompt_version']=='fixed-performance-v1' else 'sealed raw transport/length/ASCII/identifier evidence only; excludes tokenizer provenance, cache/rank attribution, frozen profile, telemetry and shared cleanup acceptance')}


def main():
    import sys
    print(json.dumps(verify(sys.argv[1]),sort_keys=True))


if __name__=='__main__':main()
