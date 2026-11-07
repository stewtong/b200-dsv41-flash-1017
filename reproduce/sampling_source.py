"""Verify sampling against the pinned checkpoint's actual model-card bytes."""
import hashlib
import json
from pathlib import Path
import re


def verify(path):
    lock=json.loads((Path(__file__).resolve().parents[1]/'configs/sampling-source.json').read_text())
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('sampling source missing or unsafe')
    data=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if (lock.get('revision')!='1bae0e1088993b9b82de08f59de5be5a34e42170' or
        len(data)!=lock['bytes'] or hashlib.sha256(data).hexdigest()!=lock['sha256'] or blob!=lock['git_blob']):
        raise ValueError('sampling source differs from pinned checkpoint')
    text=data.decode('utf-8');values={}
    for key in ('temperature','top_p'):
        found=re.findall(r'^\|\s*`'+key+r'`\s*\|\s*([0-9.]+)\s*\|\s*$',text,re.M)
        if len(found)!=1 or float(found[0])!=lock[key] or lock[key]!=1.0:
            raise ValueError('sampling recommendation missing or conflicting')
        values[key]=float(found[0])
    return dict(lock,status='pinned_model_card_verified',**{'parameters':values})
