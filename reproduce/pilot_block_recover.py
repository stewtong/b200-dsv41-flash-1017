"""Reconcile a dead pilot block without replaying captured or accepted cells."""
import hashlib
import json
from pathlib import Path
import re

import attempts
import pilot_registry
import reconcile
import runtime
import verify_pilot
import verify_pilot_block


def recover(run_dir,attempts_root=pilot_registry.DEFAULT_ROOT,node_lock=runtime.DEFAULT_NODE_LOCK,
            allowed_root='/data',invoke=runtime.command,alive=runtime.process_alive,
            idle=runtime.verify_gpus_idle,preflight=runtime.live_preflight,*,
            result_name='pilot-block-result.json',child_result_name='pilot-result.json',block_verifier=None):
    if any(Path(name).name!=name or not name.endswith('.json') for name in (result_name,child_result_name)):
        raise ValueError('block recovery requires explicit result basenames')
    root=Path(run_dir)
    if (not root.is_absolute() or root.is_symlink() or
        not root.parent.resolve(strict=True).is_relative_to(Path(allowed_root).resolve()) or
        not re.fullmatch(r'dsv41-[a-z0-9-]+',root.name)):
        raise runtime.RuntimeErrorClosed('invalid pilot block recovery root')
    group=root.name.removeprefix('dsv41-');registry=Path(attempts_root)
    with runtime.NodeLock(node_lock):
        rows=[r for r in attempts.load_attempts(registry) if r.get('group_id')==group]
        if not rows or len({r['cell_id'] for r in rows})!=len(rows):
            raise runtime.RuntimeErrorClosed('pilot block attempts missing or ambiguous')
        owners=[r.get('driver_identity') for r in rows]
        if not owners[0] or any(o!=owners[0] for o in owners) or alive(owners[0]):
            raise runtime.RuntimeErrorClosed('pilot block owner remains alive or differs')
        for row in rows:
            path=registry/row['directory'];context=path/'context.json'
            if context.is_symlink():raise runtime.RuntimeErrorClosed('unsafe block attempt context')
            if context.exists():
                data=json.loads(context.read_text())
                if (data.get('run_dir')!=str(root) or data.get('driver_identity')!=owners[0] or
                    data.get('cell_artifact_dir')!=str(root/'cells'/row['cell_id'])):
                    raise runtime.RuntimeErrorClosed('block attempt context differs from ownership')
            elif root.exists():raise runtime.RuntimeErrorClosed('launched block context missing')
        if root.exists():
            state=json.loads((root/'execution-state.json').read_text())
            if (state.get('run_id')!=group or state.get('driver_identity')!=owners[0] or
                alive(state['watchdog_identity']) or any(alive(p) for p in state.get('owned_processes',[]))):
                raise runtime.RuntimeErrorClosed('block process remains alive or identity differs')
            if state.get('container_ids'):
                reconcile.reconcile(root,node_lock,allowed_root,invoke,alive,idle,lock_held=True)
            elif not (root/'SHA256SUMS').exists():runtime.seal_evidence(root)
            runtime.verify_evidence_manifest(root)
        preflight(group,[31000,31001,31002,31003,31100],invoke)
        result=root/result_name
        if result.exists():
            complete=json.loads(result.read_text())
            if (complete.get('status')!='passed' or
                {c.get('cell_id') for c in complete.get('cells',[])}!={r['cell_id'] for r in rows} or
                len(complete['cells'])!=len(rows)):
                raise runtime.RuntimeErrorClosed('block completion contract differs')
            checked={r['cell_id']:r for r in (block_verifier or verify_pilot_block.verify)(root)['cells']}
            if sorted(checked.values(),key=lambda x:x['cell_id'])!=sorted(complete['cells'],key=lambda x:x['cell_id']):
                raise runtime.RuntimeErrorClosed('block verifier differs from completion record')
            for row in rows:
                if row['status']=='running':
                    pilot_registry.finish(registry/row['directory'],root/'cells'/row['cell_id']/child_result_name,checked[row['cell_id']])
                elif row['status']!='passed':raise runtime.RuntimeErrorClosed('completed block has conflicting terminal attempt')
            return {'status':'terminal_block_recovered','cells':len(rows)}
        if any(r['status']=='passed' for r in rows):
            raise runtime.RuntimeErrorClosed('passed cell lacks complete block proof')
        receipt={'group_id':group,'driver_dead':True,'gpu_idle_and_ports_free':True,
                 'run_manifest_sha256':hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest() if root.exists() else None}
        for row in rows:
            if row['status']=='running':
                path=registry/row['directory'];runtime.atomic_json(path/'recovery.json',receipt)
                digest=hashlib.sha256((path/'recovery.json').read_bytes()).hexdigest()
                proof='verified block recovery record sha256 '+digest
                artifact=root/'cells'/row['cell_id']
                if (path/'dispatch-intent.json').exists():attempts.mark_interrupted(path,proof)
                elif artifact.exists():raise runtime.RuntimeErrorClosed('cell artifacts exist without required dispatch intent')
                else:attempts.release_not_dispatched(path,proof)
            elif row['status'] not in ('interrupted','not_dispatched'):
                raise runtime.RuntimeErrorClosed('block attempt status conflicts with interruption')
        return {'status':'interrupted_block_reconciled','cells':len(rows)}
