"""Bind unchanged pilot input bytes across layouts without rerunning generation."""
import hashlib
import json
from pathlib import Path
import re
import toolkit_integrity

GENERATORS=('reproduce/vllm_prompt.py','reproduce/workload.py')
RUNTIME_PINS=('image_digest','model_revision','tokenizer_revision','sampling_source','temperature','top_p')
# The November 2 owner amendment adds sequencing/resource discipline only.
# Its exact diff preserves the original workload, acceptance and safety body.
PRIORITY_PLAN_PAIR=('fff4c820e94927c3924952a9c5ed9cfaed453e8e1f60d27b1c7118730686947b',
                    'f97c296a68f91530d14558306a850ec424577611366060c981876cf115f6c449')


def compare(config,source_config,source_manifest,target_manifest,old_toolkit,new_toolkit):
    reuse=config.get('prompt_reuse')
    if (not isinstance(reuse,dict) or set(reuse)!={'source_layout','toolkit_manifest','toolkit_sha256'} or
            not re.fullmatch(r'[a-f0-9]{64}',str(reuse['toolkit_sha256'])) or
            reuse['source_layout']==config['layout'] or
            source_config.get('layout')!=reuse['source_layout'] or source_config.get('prompt_reuse') is not None or
            config.get('execution_phase','pilot')!='pilot' or source_config.get('execution_phase','pilot')!='pilot' or
            source_config.get('task_family','pilot')!='pilot' or config.get('task_family','pilot')!='pilot'):
        raise ValueError('explicit direct cross-layout pilot input reuse required')
    old=source_manifest['runtime'];new=target_manifest['runtime']
    plan_pair=(source_manifest.get('plan_sha256'),target_manifest.get('plan_sha256'))
    if ((plan_pair[0]!=plan_pair[1] and plan_pair!=PRIORITY_PLAN_PAIR) or
            source_manifest.get('matrix_sha256')!=target_manifest.get('matrix_sha256') or
            any(old.get(k) is None or new.get(k) is None or old.get(k)!=new.get(k) for k in RUNTIME_PINS) or
            any(old['starting_profiles'].get(k)!=new['starting_profiles'].get(k)
                for k in (reuse['source_layout'],config['layout']))):
        raise ValueError('pilot reuse dependencies or starting profiles differ')
    original=[c for c in source_manifest['cells'] if c['id']==source_config['cell_id']]
    target=[c for c in target_manifest['cells'] if c['id']==config['cell_id']]
    if (len(original)!=1 or len(target)!=1 or not original[0]['suite'].startswith('pilot_') or
            original[0]['layout']!=reuse['source_layout'] or target[0]['layout']!=config['layout'] or
            {k:v for k,v in original[0].items() if k not in ('id','layout')}!=
            {k:v for k,v in target[0].items() if k not in ('id','layout')}):
        raise ValueError('reused pilot workload shape differs')
    generators=GENERATORS+(('reproduce/warm_pilot_prompt.py',) if target[0]['cache']=='warm_prefix' else ())
    for name in generators:
        if (not re.fullmatch(r'[a-f0-9]{64}',str(old_toolkit.get(name))) or
                old_toolkit.get(name)!=new_toolkit.get(name)):
            raise ValueError('pilot generator source differs')
    proof={'source_cell_id':original[0]['id'],'target_cell_id':target[0]['id'],
            'generator_sha256':{name:old_toolkit[name] for name in generators},
            'scope':'unchanged tokenizer inputs shared across pilot layouts; no new generation or inference acceptance'}
    if plan_pair[0]!=plan_pair[1]:
        proof['plan_supersession']={'source_sha256':plan_pair[0],'target_sha256':plan_pair[1],
            'authority':'owner-priority-amendment-20261102','workload_and_acceptance_unchanged':True}
    return proof


def load(config,source_root,allowed_cache='/data/hf-cache'):
    source_root=Path(source_root)
    reuse=config['prompt_reuse']
    if not isinstance(reuse,dict) or set(reuse)!={'source_layout','toolkit_manifest','toolkit_sha256'}:
        raise ValueError('explicit direct pilot reuse declaration required')
    old_path=Path(reuse['toolkit_manifest']);new_path=Path(config['toolkit_manifest'])
    for path in (old_path,new_path):
        if (path.is_symlink() or path.parent.is_symlink() or path.name!='MANIFEST.json' or
                path.parent.resolve().parent!=Path(allowed_cache).resolve() or
                not re.fullmatch(r'dsv41-toolkit-[0-9]{8}T[0-9]{4}Z',path.parent.name)):
            raise ValueError('pilot reuse requires immutable campaign source manifests')
    old_bytes=old_path.read_bytes();new_bytes=new_path.read_bytes()
    if toolkit_integrity.verify(old_path)!=reuse['toolkit_sha256']:
        raise ValueError('original tokenizer toolkit pin differs')
    target=json.loads(Path(config['manifest']).read_text());source=json.loads((source_root/'manifest.json').read_text())
    if (hashlib.sha256(old_bytes).hexdigest()!=source['runtime'].get('toolkit_manifest_sha256') or
            toolkit_integrity.verify(new_path)!=target['runtime'].get('toolkit_manifest_sha256')):
        raise ValueError('reused generator source does not match recorded manifest')
    binding=compare(config,json.loads((source_root/'run-config.json').read_text()),source,target,
                    json.loads(old_bytes),json.loads(new_bytes))
    return binding,old_bytes,new_bytes
