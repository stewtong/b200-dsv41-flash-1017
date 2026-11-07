"""Persistent attempts: incomplete evidence is preserved, never silently reused."""
import hashlib
import json
from pathlib import Path
import re
import uuid
import runtime

SAFE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$')

def load_attempts(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob('attempt-*')):
        if path.is_symlink() or not path.is_dir():
            rows.append({'directory':path.name,'status':'unverified_orphan','cell_id':None});continue
        metadata = path / 'attempt.json'
        try:
            if metadata.is_symlink() or not metadata.is_file(): raise ValueError('missing metadata')
            row = json.loads(metadata.read_text())
            if (not isinstance(row,dict) or row.get('attempt_id')!=path.name or
                not isinstance(row.get('cell_id'),str) or not SAFE.fullmatch(row['cell_id']) or
                row.get('status') not in ('running','passed','failed','timed_out','invalid','interrupted','not_dispatched') or
                (row.get('method_sha256') is not None and not re.fullmatch(r'[0-9a-f]{64}',str(row['method_sha256'])))):
                raise ValueError('invalid metadata')
            row['directory'] = path.name;rows.append(row)
        except (ValueError,TypeError,OSError):
            rows.append({'directory':path.name,'status':'unverified_orphan','cell_id':None})
    return rows

def begin(root, cell_id, retry=False, retry_reason=None, method_sha256=None):
    root = Path(root)
    if root.is_symlink(): raise runtime.RuntimeErrorClosed('attempt root is a symlink')
    root.mkdir(parents=True, exist_ok=True)
    if not SAFE.fullmatch(cell_id): raise ValueError('invalid cell ID')
    if method_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}',method_sha256): raise ValueError('invalid method hash')
    with runtime.NodeLock(root/'attempts.lock'):
        existing = load_attempts(root)
        if any(a.get('status') == 'unverified_orphan' for a in existing):
            raise runtime.RuntimeErrorClosed('unverified orphan needs reconciliation')
        if any(a.get('status') == 'running' for a in existing):
            raise runtime.RuntimeErrorClosed('incomplete attempt needs reconciliation')
        terminal = [a for a in existing if a.get('cell_id') == cell_id and a.get('status') in ('passed', 'failed', 'timed_out', 'invalid', 'interrupted')]
        # Stew approved one third/final startup attempt for these exact cells
        # on2026-11-02; every other cell keeps the ordinary two-attempt cap.
        final_startups={'pilot_smoke-4xdep2-i8000-c1-cold-o128','pilot_smoke-4xtp2-i8000-c1-cold-o128'}
        limit=3 if cell_id in final_startups else 2
        if any(a['status'] == 'passed' for a in terminal) or len(terminal)>=limit or (terminal and not retry):
            raise runtime.RuntimeErrorClosed('cell already terminal; a single explicit changed-method retry is required')
        if retry and (not terminal or not isinstance(retry_reason,str) or not 12<=len(retry_reason.strip())<=500 or '\n' in retry_reason or '\r' in retry_reason):
            raise runtime.RuntimeErrorClosed('retry requires a prior attempt and a recorded changed method or hypothesis')
        if retry and (not method_sha256 or any(not prior.get('method_sha256') for prior in terminal) or method_sha256 in {prior['method_sha256'] for prior in terminal}):
            raise runtime.RuntimeErrorClosed('retry requires a changed recorded method hash')
        name = 'attempt-' + uuid.uuid4().hex
        path = root/name
        path.mkdir()
        runtime.atomic_json(path/'attempt.json', {'cell_id':cell_id, 'status':'running', 'attempt_id':name,
                                                       'retry_reason':retry_reason if retry else None,'method_sha256':method_sha256})
        return path

def mark_interrupted(path, reconciliation):
    """Record externally verified reconciliation; caller owns process checks."""
    path = Path(path)
    if not reconciliation: raise ValueError('reconciliation evidence required')
    with runtime.NodeLock(path.parent/'attempts.lock'):
        row = json.loads((path/'attempt.json').read_text())
        if row['status'] != 'running': raise runtime.RuntimeErrorClosed('attempt not running')
        row.update(status='interrupted', reconciliation_evidence=reconciliation)
        runtime.atomic_json(path/'attempt.json', row)

def finish(path, status, evidence_files):
    path = Path(path)
    if status not in ('passed', 'failed', 'timed_out', 'invalid'):
        raise ValueError('invalid attempt terminal status')
    with runtime.NodeLock(path.parent/'attempts.lock'):
        row = json.loads((path/'attempt.json').read_text())
        if row['status'] != 'running': raise runtime.RuntimeErrorClosed('attempt already terminal')
        checksums = {}
        for name in evidence_files:
            if Path(name).name != name: raise ValueError('invalid evidence path')
            evidence=path/name
            if evidence.is_symlink() or not evidence.is_file(): raise ValueError('unsafe evidence file')
            checksums[name] = hashlib.sha256(evidence.read_bytes()).hexdigest()
        if not checksums: raise ValueError('terminal attempt needs evidence')
        row.update(status=status, evidence_sha256=checksums)
        runtime.atomic_json(path/'attempt.json', row)
        return row


def invalidate_pass(path, expected_attempt_sha256, audit):
    """Withdraw an erroneous pass without erasing its prior terminal evidence.

    The caller independently verifies the audit and reconciles live ownership.
    This operation never dispatches work or increases the two-attempt ceiling.
    """
    path=Path(path)
    if path.is_symlink() or not path.is_dir():raise ValueError('unsafe attempt path')
    if (not isinstance(audit,dict) or audit.get('status')!='invalid' or
        audit.get('ownership_released') is not True or not isinstance(audit.get('reason'),str) or
        not 12<=len(audit['reason'])<=1000 or
        not isinstance(audit.get('source_manifest_sha256'),str) or
        not re.fullmatch(r'[0-9a-f]{64}',audit['source_manifest_sha256'])):
        raise ValueError('independent invalidation audit required')
    with runtime.NodeLock(path.parent/'attempts.lock'):
        raw=(path/'attempt.json').read_bytes();row=json.loads(raw)
        if (row.get('status')!='passed' or row.get('attempt_id')!=path.name or
            audit.get('cell_id')!=row.get('cell_id') or
            hashlib.sha256(raw).hexdigest()!=expected_attempt_sha256):
            raise runtime.RuntimeErrorClosed('passed attempt identity changed')
        for name,digest in row.get('evidence_sha256',{}).items():
            file=path/name
            if Path(name).name!=name or file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
                raise runtime.RuntimeErrorClosed('prior terminal evidence changed')
        if not row.get('evidence_sha256'):raise ValueError('prior terminal evidence missing')
        prior=path/'pre-audit-passed-attempt.json';receipt=path/'invalidation-audit.json'
        encoded=(json.dumps(audit,indent=2,sort_keys=True)+'\n').encode()
        for file,data in ((prior,raw),(receipt,encoded)):
            if file.exists():
                if file.is_symlink() or file.read_bytes()!=data:raise runtime.RuntimeErrorClosed('invalidation receipt differs')
            else:runtime.atomic_bytes(file,data)
        hashes=dict(row['evidence_sha256'])
        hashes.update({file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in (prior,receipt)})
        row.update(status='invalid',prior_terminal_status='passed',evidence_sha256=hashes,
                   invalidation_reason=audit['reason'])
        runtime.atomic_json(path/'attempt.json',row)
        return row


def begin_block(root, cells, group_id, owner_identity, *, retry=False, retry_reason=None,retry_cells=None,rebaseline_cells=None):
    """Reserve all distinct cells before a shared engine session can launch.

    Reservations stay running through capture and shared cleanup. A crash during
    reservation leaves explicit unresolved attempts, blocking further dispatch.
    A retry block applies the same one-retry rule to every selected cell before
    reserving any new attempt. Mixed blocks require an exact per-cell retry map;
    default/global retry behavior still requires uniform eligibility.
    """
    root=Path(root)
    if type(retry) is not bool:raise ValueError('retry must be boolean')
    if retry and (not isinstance(retry_reason,str) or not 12<=len(retry_reason.strip())<=500 or '\n' in retry_reason or '\r' in retry_reason):
        raise ValueError('block retry requires a recorded changed method')
    if root.is_symlink():raise runtime.RuntimeErrorClosed('attempt root is a symlink')
    if not isinstance(group_id,str) or not SAFE.fullmatch(group_id):raise ValueError('invalid block ID')
    if not cells or any(not isinstance(c,dict) or set(c)!={'cell_id','method_sha256'} for c in cells):
        raise ValueError('block needs exact cell and method identities')
    if len({c['cell_id'] for c in cells})!=len(cells):raise ValueError('duplicate block cell')
    for cell in cells:
        if (not isinstance(cell['cell_id'],str) or not SAFE.fullmatch(cell['cell_id']) or
            not isinstance(cell['method_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',cell['method_sha256'])):
            raise ValueError('invalid block cell identity')
    if retry_cells is not None:
        if (retry or retry_reason is not None or not isinstance(retry_cells,dict) or not retry_cells or
                not set(retry_cells).issubset({c['cell_id'] for c in cells}) or
                any(not isinstance(reason,str) or not 12<=len(reason.strip())<=500 or '\n' in reason or '\r' in reason for reason in retry_cells.values())):
            raise ValueError('mixed block requires exact explicit per-cell changed-method declarations')
    if rebaseline_cells is not None:
        import tp4_rebaseline
        if (retry or retry_reason is not None or not isinstance(rebaseline_cells,dict) or
                not rebaseline_cells or not set(rebaseline_cells).issubset({c['cell_id'] for c in cells}) or
                set(rebaseline_cells).intersection(retry_cells or {})):
            raise ValueError('named rebaselines require separate exact declarations')
        for ident,value in rebaseline_cells.items():tp4_rebaseline.declaration(ident,value)
        tp4_rebaseline.reservation_context(cells,group_id,rebaseline_cells)
    if (not isinstance(owner_identity,dict) or set(owner_identity)!={'boot_id','pid','start_ticks'} or
        owner_identity!=runtime.process_identity(__import__('os').getpid())):
        raise runtime.RuntimeErrorClosed('block reservation requires this exact live driver')
    root.mkdir(parents=True,exist_ok=True)
    with runtime.NodeLock(root/'attempts.lock'):
        existing=load_attempts(root)
        if any(a.get('status') in ('unverified_orphan','running') for a in existing):
            raise runtime.RuntimeErrorClosed('incomplete attempt needs reconciliation')
        selected={c['cell_id'] for c in cells}
        if not retry and retry_cells is None and rebaseline_cells is None and any(a.get('cell_id') in selected and a.get('status')!='not_dispatched' for a in existing):
            raise runtime.RuntimeErrorClosed('block cannot replay a prior cell; reconcile and use explicit changed-method retry')
        if retry:
            for cell in cells:
                old=[a for a in existing if a.get('cell_id')==cell['cell_id'] and a.get('status')!='not_dispatched']
                if (len(old)!=1 or old[0]['status'] not in ('failed','timed_out','invalid','interrupted') or
                    not old[0].get('method_sha256') or old[0]['method_sha256']==cell['method_sha256']):
                    raise runtime.RuntimeErrorClosed('block retry needs one reconciled prior failure and a changed method for every cell')
        rebaseline_proofs={}
        if retry_cells is not None or rebaseline_cells is not None:
            for cell in cells:
                old=[a for a in existing if a.get('cell_id')==cell['cell_id'] and a.get('status')!='not_dispatched']
                if cell['cell_id'] in (rebaseline_cells or {}):
                    rebaseline_proofs[cell['cell_id']]=tp4_rebaseline.validate_history(root,
                        cell['cell_id'],rebaseline_cells[cell['cell_id']],old,cell['method_sha256'])
                elif cell['cell_id'] in (retry_cells or {}):
                    if (len(old)!=1 or old[0]['status'] not in ('failed','timed_out','invalid','interrupted') or
                            not old[0].get('method_sha256') or old[0]['method_sha256']==cell['method_sha256']):
                        raise runtime.RuntimeErrorClosed('per-cell retry needs one reconciled failure and a changed method')
                elif old:raise runtime.RuntimeErrorClosed('undeclared prior attempt in mixed block')
        if any(a.get('group_id')==group_id for a in existing):
            raise runtime.RuntimeErrorClosed('block identity already used')
        result=[]
        for cell in cells:
            name='attempt-'+uuid.uuid4().hex;path=root/name;path.mkdir()
            metadata=dict(cell,status='running',attempt_id=name,
                retry_reason=retry_cells.get(cell['cell_id']) if retry_cells is not None else retry_reason if retry else None,
                group_id=group_id,driver_identity=owner_identity)
            if cell['cell_id'] in rebaseline_proofs:metadata['rebaseline_authorization']=rebaseline_proofs[cell['cell_id']]
            runtime.atomic_json(path/'attempt.json',metadata)
            result.append(path)
        return result


def capture_pending(path, evidence_files, group_id, owner_identity):
    """Retain bounded cell artifacts without declaring cleanup or acceptance."""
    path=Path(path)
    with runtime.NodeLock(path.parent/'attempts.lock'):
        row=json.loads((path/'attempt.json').read_text())
        if (row.get('status')!='running' or row.get('group_id')!=group_id or
            row.get('driver_identity')!=owner_identity or
            owner_identity!=runtime.process_identity(__import__('os').getpid())):
            raise runtime.RuntimeErrorClosed('capture requires the same live block owner')
        if (path/'capture-pending.json').exists():raise runtime.RuntimeErrorClosed('cell capture already retained')
        hashes={}
        for name in evidence_files:
            file=path/name
            if Path(name).name!=name or file.is_symlink() or not file.is_file():raise ValueError('unsafe captured artifact')
            hashes[name]=hashlib.sha256(file.read_bytes()).hexdigest()
        if not hashes:raise ValueError('capture evidence missing')
        runtime.atomic_json(path/'capture-pending.json',{'status':'captured_cleanup_pending',
            'group_id':group_id,'evidence_sha256':hashes})
        return hashes


def release_not_dispatched(path,reconciliation):
    """Release only a reconciled block reservation with no dispatch intent."""
    path=Path(path)
    if not reconciliation:raise ValueError('reconciliation evidence required')
    with runtime.NodeLock(path.parent/'attempts.lock'):
        row=json.loads((path/'attempt.json').read_text())
        if (row.get('status')!='running' or not row.get('group_id') or
            any((path/name).exists() for name in ('dispatch-intent.json','capture.json','capture-pending.json'))):
            raise runtime.RuntimeErrorClosed('reservation may have dispatched; preserve as an interrupted attempt')
        row.update(status='not_dispatched',reconciliation_evidence=reconciliation)
        runtime.atomic_json(path/'attempt.json',row)
