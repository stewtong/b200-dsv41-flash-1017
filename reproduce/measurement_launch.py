"""Bind an explicitly registered worker observer to its exact owned launch.

This is opt-in instrumentation, never part of the unchanged pilot profiles.
Source parity, live cadence and matched total overhead remain separate checks.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import runtime

FILES={'extension_sha256':'measurement_worker_extension.py',
       'recorder_sha256':'worker_observations.py','runtime_sha256':'runtime.py'}
SCHEMA='source-pinned-worker-observer/v1'

def source_files(spec):
    return dict(FILES,**({'allocator_sha256':'allocator_observations.py'} if 'allocator_sha256' in spec else {}))


def validate(spec):
    if (not isinstance(spec,dict) or set(spec) not in ({'schema',*FILES},{'schema',*FILES,'allocator_sha256'}) or spec['schema']!=SCHEMA or
            any(not re.fullmatch(r'[a-f0-9]{64}',str(spec[k])) for k in source_files(spec))):
        raise ValueError('worker observation requires an exact registered source contract')


def template(plan,spec):
    validate(spec);result=copy.deepcopy(plan)
    for index,command in enumerate(result['commands']):
        argv=command['argv']
        if '--worker-extension-cls' in argv:raise ValueError('worker extension already declared')
        image_index=argv.index('--model')-1
        if '@sha256:' not in argv[image_index]:raise ValueError('worker observer image position differs')
        docker=['--mount','type=bind,src=/measurement-source,dst=/campaign-source,readonly',
            '--mount',f'type=bind,src=/measurement-output/replica-{index},dst=/campaign-measurement',
            '-e','PYTHONPATH=/campaign-source','-e','PYTHONDONTWRITEBYTECODE=1',
            '-e','DSV41_MEASUREMENT_ROOT=/campaign-measurement',
            '-e','DSV41_MEASUREMENT_EXTENSION_SHA256='+spec['extension_sha256'],
            '-e',f'DSV41_MEASUREMENT_CONTRACT_SHA256=contract-{index}']
        argv[image_index:image_index]=docker
        argv.extend(['--worker-extension-cls','measurement_worker_extension.CampaignMeasurementExtension'])
    return result


def verify_installed_source(directory,image_id,source_pins):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    data=json.loads((root/'installed-source.json').read_text())
    result=json.loads((root/'source-result.json').read_text())
    container=json.loads((root/'source-container.json').read_text())
    if (result.get('status')!='captured' or result.get('image_id')!=image_id or
            container.get('image_id')!=image_id or '--gpus' in container.get('argv',[]) or
            not json.loads((root/'cleanup-result.json').read_text()).get('success')):
        raise ValueError('worker observer installed-source capture or image differs')
    files={row['relative_path']:row for row in data['files']}
    if len(files)!=len(data['files']):raise ValueError('duplicate installed worker source')
    for relative,digest in source_pins.items():
        row=files.get(relative)
        if (not row or row['sha256']!=digest or
                hashlib.sha256(row['source'].encode()).hexdigest()!=digest):
            raise ValueError('worker observer installed source parity missing')
    return {'path':str(root),'manifest_sha256':hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest(),
            'image_id':image_id,'source_pins':dict(source_pins)}


def bind(plan,spec,root,source_root,*,run_id,profile_sha,toolkit_sha,deadline_ns,
         tp,dp,image_id,installed_evidence,source_pins,boot_id=None,allocator_evidence=None,allocator_pins=None):
    """Run after Supervisor owns root, before any engine create/start."""
    validate(spec);root=Path(root);source_root=Path(source_root)
    if (not root.is_dir() or root.is_symlink() or not root.is_absolute() or
            not source_root.is_absolute() or source_root.is_symlink() or
            any(c in str(source_root)+str(root) for c in (',','\n')) or
            any(not re.fullmatch(r'[a-f0-9]{64}',str(v)) for v in (profile_sha,toolkit_sha)) or
            type(tp) is not int or type(dp) is not int or tp*dp not in (2,4,8) or
            type(deadline_ns) is not int or not time.monotonic_ns()<deadline_ns<=time.monotonic_ns()+86_400_000_000_000):
        raise ValueError('worker observation ownership, source, deadline or rank geometry differs')
    for key,name in source_files(spec).items():
        path=source_root/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=spec[key]:
            raise ValueError('worker observer source bytes differ from preregistration')
    proof=verify_installed_source(installed_evidence,image_id,source_pins)
    allocator_proof=None
    if 'allocator_sha256' in spec:
        if not allocator_evidence or not allocator_pins:raise ValueError('allocator installed-source proof missing')
        allocator_proof=verify_installed_source(allocator_evidence,image_id,{'torch/'+k:v for k,v in allocator_pins.items()})
    retained_source=root/'worker-observer-source';retained_source.mkdir(exist_ok=False)
    for name in source_files(spec).values():runtime.atomic_bytes(retained_source/name,(source_root/name).read_bytes())
    output=root/'worker-observation';output.mkdir(exist_ok=False)
    boot_id=boot_id or Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    result=copy.deepcopy(plan);contracts=[]
    for index,command in enumerate(result['commands']):
        directory=output/('replica-'+str(index));directory.mkdir()
        contract={'schema':'owned-worker-observation/v1','run_id':run_id,'replica_id':index,
            'boot_id':boot_id,'deadline_ns':deadline_ns,'extension_sha256':spec['extension_sha256'],
            'recorder_sha256':spec['recorder_sha256'],'runtime_sha256':spec['runtime_sha256'],
            'profile_sha256':profile_sha,'toolkit_sha256':toolkit_sha,'tp_size':tp,'dp_size':dp,
            'expected_gpu_ids':command['gpu_ids']}
        if allocator_proof is not None:
            contract.update(allocator_sha256=spec['allocator_sha256'],allocator_source_pins=allocator_pins)
        if len(command['gpu_ids'])!=tp*dp:raise ValueError('worker observation device population differs')
        runtime.atomic_json(directory/'contract.json',contract)
        digest=hashlib.sha256((directory/'contract.json').read_bytes()).hexdigest()
        substitutions={'type=bind,src=/measurement-source,dst=/campaign-source,readonly':
            f'type=bind,src={source_root},dst=/campaign-source,readonly',
            f'type=bind,src=/measurement-output/replica-{index},dst=/campaign-measurement':
            f'type=bind,src={directory},dst=/campaign-measurement',
            f'DSV41_MEASUREMENT_CONTRACT_SHA256=contract-{index}':'DSV41_MEASUREMENT_CONTRACT_SHA256='+digest}
        if any(command['argv'].count(key)!=1 for key in substitutions):
            raise ValueError('registered worker observer placeholders differ')
        command['argv']=[substitutions.get(arg,arg) for arg in command['argv']]
        contracts.append({'replica_id':index,'directory':str(directory),'contract':contract,'sha256':digest})
    binding={'schema':SCHEMA,'spec':spec,'source_root':str(source_root),'installed_source':proof,
        'contracts':contracts,'scope':'declared observer launch only; no live measurement or overhead acceptance'}
    if allocator_proof is not None:binding['allocator_installed_source']=allocator_proof
    runtime.atomic_json(root/'worker-observation-binding.json',binding)
    return result,binding


def docker_binding(row,binding,index):
    """Check observed Docker mounts/env without retaining unrelated env values."""
    item=binding['contracts'][index];mounts={m['Destination']:m for m in row.get('Mounts',[])}
    expected={'/campaign-source':(binding['source_root'],False),
              '/campaign-measurement':(item['directory'],True)}
    for destination,(source,writable) in expected.items():
        mount=mounts.get(destination,{})
        if mount.get('Type')!='bind' or mount.get('Source')!=source or mount.get('RW') is not writable:
            raise ValueError('observed worker mount differs from exact owned binding')
    env=row.get('Config',{}).get('Env',[])
    wanted={'PYTHONPATH':'/campaign-source','PYTHONDONTWRITEBYTECODE':'1',
        'DSV41_MEASUREMENT_ROOT':'/campaign-measurement',
        'DSV41_MEASUREMENT_EXTENSION_SHA256':binding['spec']['extension_sha256'],
        'DSV41_MEASUREMENT_CONTRACT_SHA256':item['sha256']}
    for key,value in wanted.items():
        if [v for v in env if v.startswith(key+'=')]!=[key+'='+value]:
            raise ValueError('observed worker environment differs from registered contract')
    return {'mounts':{k:{'source':v[0],'writable':v[1]} for k,v in expected.items()},'environment':wanted}


def collect(root,now_ns):
    """Read only exact owned workers. Static layout bytes remain stored once."""
    import verify_worker_observations
    root=Path(root);binding=json.loads((root/'worker-observation-binding.json').read_text())
    placement=json.loads((root/'rank-placement.json').read_text())
    if placement.get('status')!='passed' or len(placement['containers'])!=len(binding['contracts']):
        raise ValueError('worker observer physical binding is incomplete')
    workers=[]
    for item,physical in zip(binding['contracts'],placement['containers']):
        rows=verify_worker_observations.read_replica(root/'worker-observation'/('replica-'+str(item['replica_id'])),
            item['contract'],physical,physical['container_id'],binding['installed_source']['source_pins'],time.monotonic_ns)
        workers.extend({key:value for key,value in row.items() if key!='layout'} for row in rows)
    if len(workers)!=8:raise ValueError('worker observer does not cover eight physical workers')
    completed=time.monotonic_ns()
    if not 0<=completed-now_ns<=1_000_000_000:raise ValueError('worker observation collection exceeded one-second budget')
    for item in workers:verify_worker_observations.check_row(item['row'],item['row']['identity'],completed)
    return {'status':'physical_worker_observation_sample','observed_at_ns':completed,'workers':workers}
