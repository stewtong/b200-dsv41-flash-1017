"""Reconcile one interrupted pilot without restarting or replaying it."""
import hashlib
import json
from pathlib import Path
import re
import attempts
import pilot_registry
import reconcile
import runtime
import verify_pilot


def recover(attempt, attempts_root=pilot_registry.DEFAULT_ROOT, allowed_root='/data',
            node_lock=runtime.DEFAULT_NODE_LOCK, invoke=runtime.command, alive=runtime.process_alive,
            idle=runtime.verify_gpus_idle, preflight=runtime.live_preflight,
            result_name='pilot-result.json',verifier=None):
    if Path(result_name).name!=result_name or not result_name.endswith('.json'):
        raise ValueError('recovery result must be one explicit JSON file')
    attempt=Path(attempt)
    if attempt.is_symlink() or not re.fullmatch(r'attempt-[0-9a-f]{32}',attempt.name):
        raise runtime.RuntimeErrorClosed('invalid pilot attempt path')
    root=Path(attempts_root).resolve(strict=True)
    if attempt.parent.resolve(strict=True)!=root:raise runtime.RuntimeErrorClosed('attempt outside pilot registry')
    if any((attempt/name).is_symlink() or not (attempt/name).is_file() for name in ('attempt.json','context.json')):
        raise runtime.RuntimeErrorClosed('pilot attempt metadata missing or unsafe')
    row=json.loads((attempt/'attempt.json').read_text());context=json.loads((attempt/'context.json').read_text())
    if row.get('group_id') is not None:
        raise runtime.RuntimeErrorClosed('grouped pilot requires block recovery; do not reconcile it as a standalone cell')
    if row.get('status')!='running' or row.get('attempt_id')!=attempt.name:
        raise runtime.RuntimeErrorClosed('pilot attempt is not unresolved')
    run=Path(context['run_dir'])
    if (not run.is_absolute() or not run.parent.resolve(strict=True).is_relative_to(Path(allowed_root).resolve()) or
        not re.fullmatch(r'dsv41-[a-z0-9-]+',run.name) or run.is_symlink()):
        raise runtime.RuntimeErrorClosed('invalid pilot run path')
    with runtime.NodeLock(node_lock):
        if alive(context['driver_identity']):raise runtime.RuntimeErrorClosed('pilot driver remains alive')
        state=None; ids=[]; ports=[31000,31001,31002,31003,31100]
        if run.exists():
            state_path=run/'execution-state.json'
            if state_path.is_symlink() or not state_path.is_file():raise runtime.RuntimeErrorClosed('run state missing')
            state=json.loads(state_path.read_text())
            if run.name!='dsv41-'+state.get('run_id','') or state.get('driver_identity')!=context['driver_identity']:
                raise runtime.RuntimeErrorClosed('pilot ownership identity mismatch')
            if state.get('watchdog_identity') and alive(state['watchdog_identity']):raise runtime.RuntimeErrorClosed('watchdog remains alive')
            if any(alive(p) for p in state.get('owned_processes',[])):raise runtime.RuntimeErrorClosed('owned process remains alive')
            listing=invoke(['ps','-eo','pid=,args='],timeout=10)
            for line in listing.splitlines():
                fields=line.strip().split()
                if any(Path(fields[i]).name in ('watchdog.py','process_gate.py') and str(run) in fields[i+1:]
                       for i in range(1,len(fields))):
                    raise runtime.RuntimeErrorClosed('unrecorded run process remains alive')
            plan=run/'launch-plan.json'
            if plan.exists():
                parsed=json.loads(plan.read_text());ports=[x['host_port'] for x in parsed['commands']]+[31100]
            ids=state.get('container_ids',[])
            if ids:
                reconcile.reconcile(run,node_lock,allowed_root,invoke,alive,idle,lock_held=True)
            elif state.get('owned_processes'):
                receipt=json.loads((run/'cleanup-result.json').read_text())
                if not receipt.get('success'):raise runtime.RuntimeErrorClosed('owned process cleanup not verified')
        preflight(row['cell_id'],ports,invoke)
        recovery={'attempt_id':attempt.name,'cell_id':row['cell_id'],'run_dir':str(run),
                  'container_ids':ids,'driver_dead':True,'gpu_idle_and_ports_free':True}
        if state is not None:
            result=run/result_name
            if result.exists():
                runtime.verify_evidence_manifest(run)
                verification=(verifier or verify_pilot.verify)(run)
                pilot_registry.finish(attempt,result,verification)
                return {'status':'terminal_recovered','cell_id':row['cell_id']}
            if not (run/'SHA256SUMS').exists():runtime.seal_evidence(run)
            else:runtime.verify_evidence_manifest(run)
            recovery['run_manifest_sha256']=hashlib.sha256((run/'SHA256SUMS').read_bytes()).hexdigest()
        runtime.atomic_json(attempt/'recovery.json',recovery)
        digest=hashlib.sha256((attempt/'recovery.json').read_bytes()).hexdigest()
        attempts.mark_interrupted(attempt,'verified recovery record sha256 '+digest)
        return {'status':'interrupted_reconciled','cell_id':row['cell_id'],'recovery_sha256':digest}
