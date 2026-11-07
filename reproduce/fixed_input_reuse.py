"""Retain exact fixed-input generator lineage across compatible runtime packets."""
import hashlib,json,re
from pathlib import Path
import toolkit_integrity

GENERATORS=('fixed_workload.py','workload.py','verify_fixed_task.py')
PINS=('image_digest','model_revision','tokenizer_revision','sampling_source','temperature','top_p')


def compare(config,origin,old_manifest,new_manifest,old_toolkit,new_toolkit):
    reuse=config.get('fixed_input_reuse')
    if (not isinstance(reuse,dict) or set(reuse)!={'source_layout','toolkit_manifest','toolkit_sha256'} or
        not re.fullmatch(r'[a-f0-9]{64}',str(reuse.get('toolkit_sha256'))) or
        origin.get('layout')!=reuse['source_layout'] or origin.get('fixed_input_reuse') is not None or
        origin.get('task_family')!='fixed' or origin.get('fixed_prompt_version')!='fixed-performance-v1'):
        raise ValueError('explicit direct fixed-performance input lineage required')
    old=old_manifest['runtime'];new=new_manifest['runtime']
    if (any(not re.fullmatch(r'[a-f0-9]{64}',str(old_manifest.get(k))) or old_manifest.get(k)!=new_manifest.get(k) for k in ('plan_sha256','matrix_sha256')) or
        any(old.get(k) is None or new.get(k) is None or old[k]!=new[k] for k in PINS)):
        raise ValueError('fixed input plan, encoder or sampling dependencies differ')
    digests={}
    for name in GENERATORS:
        key='reproduce/'+name;digest=old_toolkit.get(key)
        if not re.fullmatch(r'[a-f0-9]{64}',str(digest)) or new_toolkit.get(key)!=digest:
            raise ValueError('fixed generator source differs')
        digests[name]=digest
    return {'schema':'fixed-performance-input-lineage/v1','source_layout':reuse['source_layout'],'target_layout':config['layout'],
        'plan_sha256':old_manifest['plan_sha256'],'matrix_sha256':old_manifest['matrix_sha256'],
        'generator_sha256':digests,'scope':'unchanged installed tokenizer tasks reused; target engine/cache/routing and performance remain separately verified'}


def load(config,source):
    source=Path(source);reuse=config.get('fixed_input_reuse')
    if not isinstance(reuse,dict) or set(reuse)!={'source_layout','toolkit_manifest','toolkit_sha256'}:
        raise ValueError('explicit fixed input reuse declaration required')
    old_path=Path(reuse['toolkit_manifest']);new_path=Path(config['toolkit_manifest'])
    for path in (old_path,new_path):
        if (path.is_symlink() or path.parent.is_symlink() or path.name!='MANIFEST.json' or
            not re.fullmatch(r'dsv41-toolkit-[0-9]{8}T[0-9]{4}Z',path.parent.name) or
            path.parent.resolve().parent!=new_path.parent.resolve().parent):
            raise ValueError('fixed reuse requires sibling immutable campaign toolkits')
    old_bytes=old_path.read_bytes();new_bytes=new_path.read_bytes()
    old_manifest=json.loads((source/'manifest.json').read_text());new_manifest=json.loads(Path(config['manifest']).read_text())
    if (toolkit_integrity.verify(old_path)!=reuse['toolkit_sha256'] or
        hashlib.sha256(old_bytes).hexdigest()!=old_manifest['runtime'].get('toolkit_manifest_sha256') or
        toolkit_integrity.verify(new_path)!=new_manifest['runtime'].get('toolkit_manifest_sha256')):
        raise ValueError('fixed reuse toolkit does not match retained source manifest')
    proof=compare(config,json.loads((source/'run-config.json').read_text()),old_manifest,new_manifest,json.loads(old_bytes),json.loads(new_bytes))
    files={'source-toolkit-MANIFEST.json':old_bytes,'target-toolkit-MANIFEST.json':new_bytes}
    for name,digest in proof['generator_sha256'].items():
        data=(old_path.parent/'reproduce'/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('fixed generator byte binding differs')
        files[name]=data
    return proof,files


def verify(config,source,target_manifest,directory):
    """Check retained generator bytes and manifests without a node filesystem."""
    source=Path(source);root=Path(directory);reuse=config.get('fixed_input_reuse')
    if not isinstance(reuse,dict):raise ValueError('fixed reuse declaration missing')
    old_bytes=(root/'source-toolkit-MANIFEST.json').read_bytes();new_bytes=(root/'target-toolkit-MANIFEST.json').read_bytes()
    old_manifest=json.loads((source/'manifest.json').read_text())
    if (hashlib.sha256(old_bytes).hexdigest()!=reuse.get('toolkit_sha256') or
        hashlib.sha256(old_bytes).hexdigest()!=old_manifest['runtime'].get('toolkit_manifest_sha256') or
        hashlib.sha256(new_bytes).hexdigest()!=target_manifest['runtime'].get('toolkit_manifest_sha256')):
        raise ValueError('retained fixed reuse toolkit hash differs')
    proof=compare(config,json.loads((source/'run-config.json').read_text()),old_manifest,target_manifest,json.loads(old_bytes),json.loads(new_bytes))
    if json.loads((root/'lineage.json').read_text())!=proof:raise ValueError('fixed input lineage receipt differs')
    if {p.name for p in root.iterdir()}!=set(GENERATORS)|{'source-toolkit-MANIFEST.json','target-toolkit-MANIFEST.json','lineage.json'}:
        raise ValueError('fixed input lineage file coverage differs')
    for name,digest in proof['generator_sha256'].items():
        if (root/name).is_symlink() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('retained fixed generator bytes differ')
    return proof
