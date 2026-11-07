"""Exact installed cache-manager source provenance for cache-off controls."""
import hashlib,json
from pathlib import Path
import runtime

RELATIVE_PATH='v1/core/kv_cache_manager.py'
PINNED_SHA256='b5d5e05187fcdad166e09e85ed87dae632d67cf4d2709002e150e2976dfad310'


def verify(directory,image_id):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    read=lambda name:json.loads((root/name).read_text())
    data=read('installed-source.json');result=read('source-result.json');state=read('execution-state.json')
    container=read('source-container.json');files=data.get('files')
    if (not isinstance(image_id,str) or not image_id.startswith('sha256:') or
            result.get('status')!='captured' or result.get('image_id')!=image_id or container.get('image_id')!=image_id or
            result.get('files')!=1 or not isinstance(files,list) or len(files)!=1 or
            data.get('expected_source_pins')!={RELATIVE_PATH:PINNED_SHA256} or
            state.get('phase')!='installed-source-capture' or state.get('stopping') is not True or
            state.get('watchdog_stop') is not True or state.get('container_ids')!=[container.get('id')] or
            not read('cleanup-result.json').get('success') or not read('gpu-idle-after-cleanup.json').get('idle') or
            any(runtime.process_alive(p) for p in [state['driver_identity'],state['watchdog_identity'],*state['owned_processes']]) or
            any((root/name).exists() for name in ('primary-failure.json','watchdog-failure.json','source-failure.json','driver-failure.json'))):
        raise ValueError('cache source image, file population or released ownership differs')
    argv=container.get('argv',[])
    if ('--gpus' in argv or '--mount' in argv or '--read-only' not in argv or
            '--network' not in argv or argv[argv.index('--network')+1]!='none'):
        raise ValueError('cache source capture was not isolated CPU-only work')
    row=files[0];source=row.get('source')
    if (row.get('relative_path')!=RELATIVE_PATH or row.get('sha256')!=PINNED_SHA256 or not isinstance(source,str) or
            hashlib.sha256(source.encode()).hexdigest()!=PINNED_SHA256 or
            not 0<len(source.encode())<=8_000_000 or result.get('source_bytes')!=len(source.encode()) or
            data.get('source_bytes')!=len(source.encode())):
        raise ValueError('installed cache source differs from exact pinned public bytes')
    runtime.verify_evidence_manifest(root)
    return {'status':'pinned_installed_cache_source_verified','image_id':image_id,
        'relative_path':RELATIVE_PATH,'source_sha256':PINNED_SHA256,
        'evidence_manifest_sha256':hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest(),
        'scope':'installed source provenance only; cache behavior and model/client performance remain separate'}
