"""Verify a downloaded checkpoint against the pinned public file inventory."""
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
import sys

def verify_file(row, root, safety_check=None):
    base = Path(root).resolve(strict=True)
    rel = PurePosixPath(row['path'])
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe inventory path')
    path = base.joinpath(*rel.parts)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(base) or not path.is_file():
        raise ValueError('missing, unsafe or non-file checkpoint entry: ' + str(rel))
    if path.stat().st_size != row['size']:
        raise ValueError('size mismatch: ' + str(rel))
    algorithm = hashlib.sha256() if row['lfs_sha256'] else hashlib.sha1()
    if not row['lfs_sha256']:
        algorithm.update(('blob %d\0' % row['size']).encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            if safety_check is not None: safety_check()
            algorithm.update(chunk)
    expected = row['lfs_sha256'] or row['git_blob']
    if not expected or algorithm.hexdigest() != expected:
        raise ValueError('hash mismatch: ' + str(rel))

def verify(inventory, root, safety_check=None):
    data = json.loads(Path(inventory).read_text())
    seen = set()
    for row in data['files']:
        if row['path'] in seen: raise ValueError('duplicate inventory path')
        seen.add(row['path'])
        verify_file(row, root, safety_check)
    base=Path(root).resolve(strict=True); actual=set(); metadata_bytes=0
    errors=[]
    for directory,dirs,files in os.walk(base,topdown=True,followlinks=False,onerror=lambda exc:errors.append(type(exc).__name__)):
        for name in dirs+files:
            if safety_check is not None: safety_check()
            path=Path(directory)/name
            if path.is_symlink(): raise ValueError('symlink in checkpoint tree')
            mode=path.lstat().st_mode
            if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode): raise ValueError('special file in checkpoint tree')
            if stat.S_ISREG(mode):
                rel=path.relative_to(base).as_posix()
                if rel.endswith('.incomplete'): raise ValueError('incomplete checkpoint download remains')
                if rel.startswith(('.cache/huggingface/','.hf-client/')):
                    metadata_bytes+=path.stat().st_size
                else: actual.add(rel)
    if errors: raise ValueError('checkpoint tree enumeration failed')
    if actual!=seen: raise ValueError('checkpoint file-set mismatch')
    if metadata_bytes>1_000_000_000: raise ValueError('checkpoint metadata exceeds safety allowance')
    return {'revision':data['revision'], 'files':len(seen), 'bytes':sum(x['size'] for x in data['files']),
            'metadata_bytes':metadata_bytes, 'file_set_verified':True}

if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1], sys.argv[2]), sort_keys=True))
