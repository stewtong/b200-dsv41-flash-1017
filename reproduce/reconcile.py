"""Reconcile a stopped run and remove only its recorded, verified containers."""
import hashlib
from contextlib import nullcontext
import json
from pathlib import Path
import re
import runtime

def recover_failed_cleanup(root, node_lock=runtime.DEFAULT_NODE_LOCK, allowed_root='/data',
                           invoke=runtime.command, alive=runtime.process_alive, idle=runtime.verify_gpus_idle):
    """Record fresh proof after cleanup or its immediate GPU-idle check failed."""
    root = Path(root).resolve(strict=True)
    if not root.is_relative_to(Path(allowed_root).resolve()) or not re.fullmatch(r'dsv41-[a-z0-9-]+',root.name):
        raise runtime.RuntimeErrorClosed('invalid campaign run directory')
    with runtime.NodeLock(node_lock):
        if (root/'SHA256SUMS').exists(): raise runtime.RuntimeErrorClosed('failed cleanup evidence already sealed')
        state=json.loads((root/'execution-state.json').read_text())
        old=json.loads((root/'cleanup-result.json').read_text())
        ids=state['container_ids']
        if root.name!='dsv41-'+state['run_id'] or old.get('success') not in (True,False) or not ids or len(set(ids))!=len(ids) or any(not runtime.re_full_id(x) for x in ids):
            raise runtime.RuntimeErrorClosed('failed cleanup identity invalid')
        if old['success'] and (root/'gpu-idle-after-cleanup.json').exists() and not (root/'cleanup-recovery.json').exists():
            raise runtime.RuntimeErrorClosed('successful cleanup already has GPU idle proof')
        if alive(state['driver_identity']) or alive(state['watchdog_identity']) or any(alive(p) for p in state.get('owned_processes',[])):
            raise runtime.RuntimeErrorClosed('run process still alive')
        results={x.get('container_id'):x for x in old.get('outcomes',[]) if 'container_id' in x}
        if set(results)!=set(ids) or any(results[c].get('status') not in ('failed','stopped') or results[c].get('log_status')!='captured' or not list(root.glob('container-'+c+'-*.log')) for c in ids):
            raise runtime.RuntimeErrorClosed('failed cleanup evidence incomplete')
        idle(invoke)
        for cid in ids:
            if runtime.inspect_owned(cid,state['run_id'],invoke).get('State',{}).get('Running') is not False:
                raise runtime.RuntimeErrorClosed('container not independently verified stopped')
        recovery={'status':'independently_verified_stopped','run_id':state['run_id'],'container_ids':ids,
                  'prior_cleanup_sha256':hashlib.sha256((root/'cleanup-result.json').read_bytes()).hexdigest()}
        path=root/'cleanup-recovery.json'
        if path.exists() and json.loads(path.read_text())!=recovery: raise runtime.RuntimeErrorClosed('cleanup recovery changed')
        if not path.exists(): runtime.atomic_json(path,recovery)
        idle_path=root/'gpu-idle-after-cleanup.json'
        idle_record={'idle':True,'scope':'independent recovery after failed cleanup'}
        if idle_path.exists() and json.loads(idle_path.read_text())!=idle_record: raise runtime.RuntimeErrorClosed('unexpected prior idle receipt')
        if not idle_path.exists(): runtime.atomic_json(idle_path,idle_record)

def reconcile(root, node_lock=runtime.DEFAULT_NODE_LOCK, allowed_root='/data',
              invoke=runtime.command, alive=runtime.process_alive, idle=runtime.verify_gpus_idle, lock_held=False):
    root = Path(root).resolve(strict=True)
    if not root.is_relative_to(Path(allowed_root).resolve()) or not re.fullmatch(r'dsv41-[a-z0-9-]+', root.name):
        raise runtime.RuntimeErrorClosed('invalid campaign run directory')
    with (nullcontext() if lock_held else runtime.NodeLock(node_lock)):
        sealed = (root/'SHA256SUMS').exists()
        state = json.loads((root/'execution-state.json').read_text())
        failures=[root/name for name in ('watchdog-failure.json','driver-failure.json') if (root/name).exists()]
        if any(path.is_symlink() or not isinstance(json.loads(path.read_text()),dict) for path in failures):
            raise runtime.RuntimeErrorClosed('invalid failure evidence')
        failed_watchdog=bool(failures)
        recovered_shutdown = False
        if (root/'cleanup-recovery.json').exists() and (root/'cleanup-result.json').exists():
            prior = json.loads((root/'cleanup-result.json').read_text())
            recovered_shutdown = prior.get('success') is False and json.loads((root/'cleanup-recovery.json').read_text()) == {
                'status':'independently_verified_stopped','run_id':state['run_id'],
                'container_ids':state['container_ids'],
                'prior_cleanup_sha256':hashlib.sha256((root/'cleanup-result.json').read_bytes()).hexdigest()}
        if root.name != 'dsv41-'+state['run_id'] or (not state.get('watchdog_stop') and not failed_watchdog and not recovered_shutdown):
            raise runtime.RuntimeErrorClosed('run identity or shutdown state invalid')
        if alive(state['driver_identity']) or alive(state['watchdog_identity']):
            raise runtime.RuntimeErrorClosed('driver or watchdog still alive')
        if any(alive(p) for p in state.get('owned_processes', [])):
            raise runtime.RuntimeErrorClosed('owned router or client remains alive')
        receipt = json.loads((root/'cleanup-result.json').read_text())
        recovery_path=root/'cleanup-recovery.json'
        recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
        if not receipt.get('success') and recovery != {'status':'independently_verified_stopped','run_id':state['run_id'],
                'container_ids':state['container_ids'],'prior_cleanup_sha256':hashlib.sha256((root/'cleanup-result.json').read_bytes()).hexdigest()}:
            raise runtime.RuntimeErrorClosed('failed cleanup lacks matching independent recovery')
        if not json.loads((root/'gpu-idle-after-cleanup.json').read_text()).get('idle'):
            raise runtime.RuntimeErrorClosed('cleanup or GPU idle proof missing')
        ids = state['container_ids']
        if not ids or len(set(ids)) != len(ids) or any(not runtime.re_full_id(x) for x in ids):
            raise runtime.RuntimeErrorClosed('invalid recorded container set')
        allowed = {'removal-preview.json', 'reconciliation.json'}
        for cid in ids:
            allowed.update({'remove-'+cid+'.intent.json', 'remove-'+cid+'.result.json'})
        if sealed: runtime.verify_evidence_manifest(root, allowed)
        previous = root/'reconciliation.json'
        if previous.exists() and json.loads(previous.read_text()) != {'run_id':state['run_id'], 'status':'complete', 'container_ids':ids}:
            raise runtime.RuntimeErrorClosed('prior reconciliation record changed')
        for cid in ids:
            intent = root/('remove-'+cid+'.intent.json')
            if intent.exists() and json.loads(intent.read_text()) != {'id':cid, 'run_id':state['run_id']}:
                raise runtime.RuntimeErrorClosed('removal intent changed')
            result = root/('remove-'+cid+'.result.json')
            if result.exists() and (not intent.exists() or json.loads(result.read_text()) != {'id':cid, 'status':'absent'}):
                raise runtime.RuntimeErrorClosed('removal result changed')
        results = {x.get('container_id'):x for x in receipt['outcomes'] if 'container_id' in x}
        if set(results) != set(ids) or any(results[c].get('log_status') != 'captured' or (results[c].get('status') != 'stopped' and recovery is None) for c in ids):
            raise runtime.RuntimeErrorClosed('container cleanup coverage incomplete')
        evidence = {}
        for cid in ids:
            files = sorted(root.glob('container-'+cid+'-*.log'))
            if not files: raise runtime.RuntimeErrorClosed('missing captured container log')
            evidence[cid] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        preview = {'run_id':state['run_id'], 'container_ids':ids, 'log_sha256':evidence}
        preview_path = root/'removal-preview.json'
        if preview_path.exists() and json.loads(preview_path.read_text()) != preview:
            raise runtime.RuntimeErrorClosed('removal preview changed')
        if not preview_path.exists(): runtime.atomic_json(preview_path, preview)
        idle(invoke)
        for cid in ids:
            intent = root/('remove-'+cid+'.intent.json')
            result = root/('remove-'+cid+'.result.json')
            present = invoke(['sudo','-n','docker','ps','-aq','--no-trunc','--filter','id='+cid], timeout=30).split()
            if result.exists():
                if not intent.exists() or json.loads(result.read_text()) != {'id':cid, 'status':'absent'} or present:
                    raise runtime.RuntimeErrorClosed('prior removal result does not match current state')
                continue
            if not present:
                if not intent.exists(): raise runtime.RuntimeErrorClosed('container absent without recorded removal intent')
            else:
                if present != [cid]: raise runtime.RuntimeErrorClosed('unexpected container lookup result')
                if not intent.exists(): runtime.atomic_json(intent, {'id':cid, 'run_id':state['run_id']})
                runtime.remove_stopped_owned(cid, state['run_id'], invoke)
            runtime.atomic_json(result, {'id':cid, 'status':'absent'})
        runtime.atomic_json(root/'reconciliation.json', {'run_id':state['run_id'], 'status':'complete', 'container_ids':ids})
        if sealed: runtime.seal_evidence(root)
