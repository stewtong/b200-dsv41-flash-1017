"""Materialize one checksum-bound capture archive with bounded local storage."""
from contextlib import contextmanager
import hashlib,os,re,shutil,tarfile,tempfile
from pathlib import Path
import runtime

FIELDS={'archive','archive_sha256','seal_sha256','root_name'}

def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate(source):
    if (not isinstance(source,dict) or set(source)!=FIELDS or
            any(not isinstance(source[k],str) for k in FIELDS) or
            not re.fullmatch(r'dsv41-[a-z0-9-]+',source['root_name']) or
            any(not re.fullmatch(r'[a-f0-9]{64}',source[k]) for k in ('archive_sha256','seal_sha256'))):
        raise ValueError('capture archive requires exact archive/root/seal identities')
    path=Path(source['archive'])
    if path.is_symlink() or not path.is_file():raise ValueError('capture archive must be a regular file')
    return path.resolve(strict=True)

@contextmanager
def materialize(source,*,reserve_bytes=50_000_000_000,max_bytes=64*1024**3):
    """Yield one verified tree; always remove only this owned temporary tree."""
    path=validate(source)
    if type(reserve_bytes) is not int or reserve_bytes<0 or type(max_bytes) is not int or max_bytes<=0:
        raise ValueError('invalid archive materialization budget')
    with path.open('rb') as payload:
        if digest(payload)!=source['archive_sha256']:raise ValueError('capture archive checksum differs')
        payload.seek(0)
        with tempfile.TemporaryDirectory(prefix='dsv41-capture-read-') as directory:
            base=Path(directory);root=base/source['root_name'];seen=set();regular=set();stored=0
            with tarfile.open(fileobj=payload,mode='r|*') as archive:
                for member in archive:
                    name=member.name.rstrip('/');parts=name.split('/')
                    if (not name or name in seen or parts[0]!=source['root_name'] or
                            any(p in ('','.','..') for p in parts) or Path(name).is_absolute() or
                            any('.tmp.' in p for p in parts)):
                        raise ValueError('unsafe or duplicate capture archive member')
                    seen.add(name);target=base.joinpath(*parts)
                    if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
                    if len(parts)<2:raise ValueError('capture archive root must be a directory')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if member.islnk():
                        link=member.linkname.split('/')
                        if (member.size!=0 or member.linkname not in regular or not link or link[0]!=source['root_name'] or
                                any(p in ('','.','..') for p in link) or parts[-1]=='SHA256SUMS' or link[-1]=='SHA256SUMS'):
                            raise ValueError('capture hard link must target prior internal raw evidence')
                        os.link(base.joinpath(*link),target);regular.add(name);continue
                    if not member.isfile():raise ValueError('capture archive permits only regular files and internal hard links')
                    if member.size<0 or stored+member.size>max_bytes:
                        raise ValueError('capture archive exceeds materialization budget')
                    if parts==[source['root_name'],'SHA256SUMS'] and member.size>32*1024**2:
                        raise ValueError('capture checksum manifest exceeds metadata budget')
                    if shutil.disk_usage(base).free-member.size<reserve_bytes:
                        raise ValueError('capture materialization would cross local disk reserve')
                    reader=archive.extractfile(member);count=0
                    if reader is None:raise ValueError('capture archive payload missing')
                    with target.open('xb') as output:
                        for chunk in iter(lambda:reader.read(8*1024*1024),b''):
                            if shutil.disk_usage(base).free-len(chunk)<reserve_bytes:
                                raise ValueError('capture materialization crossed disk reserve')
                            output.write(chunk);count+=len(chunk)
                    if count!=member.size:raise ValueError('truncated capture archive member')
                    stored+=count;regular.add(name)
            if not (root/'SHA256SUMS').is_file() or hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest()!=source['seal_sha256']:
                raise ValueError('capture archive seal differs')
            runtime.verify_evidence_manifest(root)
            payload.seek(0)
            if digest(payload)!=source['archive_sha256']:raise ValueError('capture archive changed during materialization')
            yield root
