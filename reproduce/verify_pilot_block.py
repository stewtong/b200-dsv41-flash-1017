"""Independently verify pilot cells and their one shared engine session."""
import hashlib
import json
from pathlib import Path
import re
import sys
import verify_pilot
import verify_cancellation


def verify(root):
    root=Path(root);listed={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest,sep,name=line.partition('  ');rel=Path(name)
        if not sep or not re.fullmatch(r'[0-9a-f]{64}',digest) or rel.is_absolute() or '..' in rel.parts or name in listed:
            raise ValueError('invalid pilot block checksum entry')
        listed[name]=digest
    files={}
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('symlink in pilot block')
        if path.is_file() and path!=root/'SHA256SUMS':files[path.relative_to(root).as_posix()]=path
    if set(files)!=set(listed):raise ValueError('pilot block file coverage differs')
    for name,path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=listed[name]:raise ValueError('pilot block checksum differs')
    config=json.loads((root/'run-config.json').read_text());result=json.loads((root/'pilot-block-result.json').read_text())
    cells=config.get('cells',[]);ids=[c['cell_id'] for c in cells]
    if (not ids or len(set(ids))!=len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+',i) for i in ids) or
        {p.name for p in (root/'cells').iterdir()}!=set(ids) or result.get('status')!='passed'):
        raise ValueError('pilot block cell population differs')
    checked=[]
    for ident in ids:
        cell=root/'cells'/ident;checked.append(verify_pilot.verify(cell))
        proof=json.loads((cell/'block-provenance.json').read_text())
        for name,digest in proof['shared_file_sha256'].items():
            source=root/('run-config.json' if name=='shared-run-config.json' else name)
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
                raise ValueError('cell shared records differ from actual block session')
    if (len(result.get('cells',[]))!=len(checked) or
        sorted(result['cells'],key=lambda r:r['cell_id'])!=sorted(checked,key=lambda r:r['cell_id'])):
        raise ValueError('pilot block summary differs from independently verified cells')
    controls=config.get('qualify_cancellation',False)
    if type(controls) is not bool:raise ValueError('invalid cancellation qualification declaration')
    if controls:
        shapes=[json.loads((root/'cells'/ident/'pilot-cell.json').read_text())['cell'] for ident in ids]
        if [(c['suite'],c['input_tokens'],c['concurrency'],c['cache']) for c in shapes]!=[('pilot_fit',128000,1,'cold'),('pilot_fit',512000,1,'cold')]:
            raise ValueError('cancellation control precedes required fit qualification')
        task=json.loads((root/'cells'/ids[1]/'pilot-tasks.json').read_text())[0]
        last_end=max(json.loads((root/'cells'/ident/'pilot-attempt'/(t['id']+'.json')).read_text())['end_ns']
            for ident in ids for t in json.loads((root/'cells'/ident/'pilot-tasks.json').read_text()))
        for mode in ('prefill','decode'):
            directory=root/('cancellation-'+mode);verify_cancellation.verify(directory)
            contract=json.loads((directory/'control-input.json').read_text())
            if (contract['run_id']!=config['run_id'] or contract['mode']!=mode or
                contract['expected_ranks']!=config['expected_ranks'] or
                json.loads((directory/'source-task.json').read_text())!=task or
                json.loads((directory/'before.json').read_text())['started_ns']<last_end):
                raise ValueError('cancellation source task or shared engine population differs')
            last_end=json.loads((directory/'post-cancellation-semantic.json').read_text())['raw']['terminal_ns']
    return {'status':'passed','cells':checked,'scope':'unscored pilot cells; one verified shared startup and cleanup'}


if __name__=='__main__':print(json.dumps(verify(sys.argv[1]),sort_keys=True))
