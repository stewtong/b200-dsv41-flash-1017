"""Verify the staged launcher source against its extracted archive manifest."""
import hashlib
import json
from pathlib import Path


def verify(path):
    path=Path(path)
    if path.is_symlink() or path.name!='MANIFEST.json': raise ValueError('invalid toolkit manifest path')
    data=path.read_bytes(); entries=json.loads(data);root=path.parent.resolve(strict=True)
    if not isinstance(entries,dict) or not entries: raise ValueError('empty toolkit manifest')
    for name,expected in entries.items():
        rel=Path(name); item=root/rel
        if (rel.is_absolute() or '..' in rel.parts or item.is_symlink() or not item.is_file() or
            hashlib.sha256(item.read_bytes()).hexdigest()!=expected): raise ValueError('toolkit source hash mismatch')
    for item in root.rglob('*'):
        if item.is_symlink(): raise ValueError('toolkit contains symlink')
        if (item.is_file() and item!=root/'MANIFEST.json'
                and item.relative_to(root).as_posix() not in entries):
            raise ValueError('unlisted toolkit source file')
    return hashlib.sha256(data).hexdigest()
