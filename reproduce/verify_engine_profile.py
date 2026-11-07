"""Verify registered candidate startup separately from tuning/adoption claims."""
import copy
import ast
import hashlib
import json
from pathlib import Path
import re

import runtime
import verify_pilot
import verify_wire


def read(path):return json.loads(Path(path).read_text())


def verify(root):
    root=Path(root);runtime.verify_evidence_manifest(root)
    config=read(root/'run-config.json');manifest=read(root/'manifest.json');state=read(root/'execution-state.json')
    binding=read(root/'profile-binding.json')
    phase=config.get('execution_phase')
    if phase not in ('qualification','fixed-performance') or state.get('phase')!=phase:
        raise ValueError('candidate startup requires an explicit qualification phase')
    if phase=='fixed-performance':
        import fixed_profiles
        record=fixed_profiles.validate(manifest['runtime'].get('fixed_profiles',{}).get(config['layout']),manifest)
    else:record=manifest['runtime'].get('qualification_profiles',{}).get(config.get('profile_id'))
    if not isinstance(record,dict):raise ValueError('candidate definition missing from bound manifest')
    digest=hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!='registration_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    template=record['template'];profile_sha=hashlib.sha256(json.dumps(template,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (record.get('schema')!=('frozen-fixed-profile/v1' if phase=='fixed-performance' else 'registered-tuning-profile/v1') or record.get('id')!=config['profile_id'] or
        record.get('layout')!=config['layout'] or record.get('registration_sha256')!=digest or
        record.get('profile_sha256')!=profile_sha or record.get('status')!=('frozen_fixed_configuration' if phase=='fixed-performance' else 'preregistered_unqualified') or
        record.get('plan_sha256')!=manifest.get('plan_sha256') or record.get('matrix_sha256')!=manifest.get('matrix_sha256') or
        record.get('image_digest')!=config['image'] or record.get('model_revision')!=manifest['runtime']['model_revision'] or
        record.get('tokenizer_revision')!=manifest['runtime']['tokenizer_revision'] or
        binding!={'phase':phase,'delta':record['profile_delta'],'profile_sha256':profile_sha,'record':record}):
        raise ValueError('candidate registration, source or profile binding differs')
    cpus=read(root/'cpu-partition.json');server=','.join(map(str,cpus['server_cpus']))
    if not cpus['server_cpus'] or set(cpus['server_cpus'])&set(cpus['client_cpus']):raise ValueError('candidate CPU partition overlaps')
    expected=copy.deepcopy(template);expected.pop('cpu_policy')
    replacements={'example/image@sha256:'+'0'*64:config['image'],'run_id=profile':'run_id='+config['run_id'],
        'type=bind,src=/model-source,dst=/model,readonly':'type=bind,src='+config['model_dir']+',dst=/model,readonly',
        'type=bind,src=/cache-source,dst=/root/.cache/vllm':'type=bind,src='+config['cache_dir']+',dst=/root/.cache/vllm'}
    expected['run_id']=config['run_id']
    observation=record['profile_delta'].get('worker_observation')
    observation_binding=None
    if observation is not None:
        observation_binding=read(root/'worker-observation-binding.json')
        source_root=str(Path(config['toolkit_manifest']).parent/'reproduce')
        if (observation_binding.get('schema')!='source-pinned-worker-observer/v1' or
                observation_binding.get('spec')!=observation or observation_binding.get('source_root')!=source_root or
                len(observation_binding.get('contracts',[]))!=len(expected['commands'])):
            raise ValueError('worker observer registration or owned binding differs')
        proof=root/'worker-source-proof';runtime.verify_evidence_manifest(proof)
        source_files={'extension_sha256':'measurement_worker_extension.py',
            'recorder_sha256':'worker_observations.py','runtime_sha256':'runtime.py'}
        if 'allocator_sha256' in observation:source_files['allocator_sha256']='allocator_observations.py'
        for key,name in source_files.items():
            if hashlib.sha256((root/'worker-observer-source'/name).read_bytes()).hexdigest()!=observation[key]:
                raise ValueError('worker observer retained source differs from registration')
        tree=ast.parse((root/'worker-observer-source'/'measurement_worker_extension.py').read_text())
        declarations=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='SOURCE_PINS' for t in n.targets)]
        if len(declarations)!=1 or not declarations[0]:raise ValueError('worker observer source pins unavailable')
        source_data=read(proof/'installed-source.json');source_result=read(proof/'source-result.json')
        pins=observation_binding['installed_source']
        if (pins.get('path')!=config['run_dir']+'/worker-source-proof' or
                pins.get('manifest_sha256')!=hashlib.sha256((proof/'SHA256SUMS').read_bytes()).hexdigest() or
                pins.get('image_id')!=read(root/'image-qualification.json')['image_id'] or
                source_result.get('image_id')!=pins['image_id'] or source_result.get('status')!='captured' or
                pins.get('source_pins')!=declarations[0]):
            raise ValueError('worker observer installed proof differs')
        files={f['relative_path']:f for f in source_data['files']}
        for name,digest in pins['source_pins'].items():
            if name not in files or files[name]['sha256']!=digest or hashlib.sha256(files[name]['source'].encode()).hexdigest()!=digest:
                raise ValueError('worker observer installed source bytes differ')
        if 'allocator_sha256' in observation:
            tree=ast.parse((root/'worker-observer-source/allocator_observations.py').read_text())
            declared=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
                any(isinstance(t,ast.Name) and t.id=='SOURCE_PINS' for t in n.targets)]
            if len(declared)!=1 or not declared[0]:raise ValueError('allocator source pins missing')
            memory_proof=root/'worker-allocator-source-proof';runtime.verify_evidence_manifest(memory_proof)
            memory=observation_binding.get('allocator_installed_source',{})
            if (memory.get('path')!=config['run_dir']+'/worker-allocator-source-proof' or
                    memory.get('manifest_sha256')!=hashlib.sha256((memory_proof/'SHA256SUMS').read_bytes()).hexdigest() or
                    memory.get('image_id')!=pins['image_id'] or
                    memory.get('source_pins')!={'torch/'+k:v for k,v in declared[0].items()} or
                    read(memory_proof/'source-result.json').get('image_id')!=pins['image_id'] or
                    read(memory_proof/'source-result.json').get('status')!='captured'):
                raise ValueError('allocator installed source proof differs')
            memory_files={f['relative_path']:f for f in read(memory_proof/'installed-source.json')['files']}
            for name,digest in memory['source_pins'].items():
                f=memory_files.get(name,{})
                if f.get('sha256')!=digest or hashlib.sha256(f.get('source','').encode()).hexdigest()!=digest:
                    raise ValueError('allocator installed source bytes differ')
    elif (root/'worker-observation-binding.json').exists():
        raise ValueError('undeclared worker observer binding')
    for command in expected['commands']:
        argv=command['argv'];command['argv']=[replacements.get(a,a) for a in argv]
        command['argv'][argv.index('--cpuset-cpus')+1]=server
    if observation_binding is not None:
        for index,command in enumerate(expected['commands']):
            item=observation_binding['contracts'][index]
            contract=read(root/'worker-observation'/('replica-'+str(index))/'contract.json')
            digest=hashlib.sha256((root/'worker-observation'/('replica-'+str(index))/'contract.json').read_bytes()).hexdigest()
            if (item!={'replica_id':index,'directory':config['run_dir']+'/worker-observation/replica-'+str(index),
                    'contract':contract,'sha256':digest} or contract.get('schema')!='owned-worker-observation/v1' or
                    contract.get('run_id')!=config['run_id'] or contract.get('replica_id')!=index or
                    contract.get('profile_sha256')!=profile_sha or contract.get('toolkit_sha256')!=manifest['runtime']['toolkit_manifest_sha256'] or
                    contract.get('extension_sha256')!=observation['extension_sha256'] or
                    contract.get('boot_id')!=state['driver_identity']['boot_id'] or contract.get('deadline_ns')!=state['deadline_ns'] or
                    contract.get('expected_gpu_ids')!=command['gpu_ids']):
                raise ValueError('worker observer contract identity or owned deadline differs')
            for flag,key in [('--tensor-parallel-size','tp_size'),('--data-parallel-size','dp_size')]:
                value=int(command['argv'][command['argv'].index(flag)+1]) if flag in command['argv'] else 1
                if type(contract.get(key)) is not int or contract.get(key)!=value:raise ValueError('worker observer rank contract differs')
            expected_contract={'schema':'owned-worker-observation/v1','run_id':config['run_id'],'replica_id':index,
                'boot_id':state['driver_identity']['boot_id'],'deadline_ns':state['deadline_ns'],
                'extension_sha256':observation['extension_sha256'],'recorder_sha256':observation['recorder_sha256'],
                'runtime_sha256':observation['runtime_sha256'],'profile_sha256':profile_sha,
                'toolkit_sha256':manifest['runtime']['toolkit_manifest_sha256'],
                'tp_size':contract['tp_size'],'dp_size':contract['dp_size'],'expected_gpu_ids':command['gpu_ids']}
            if 'allocator_sha256' in observation:
                expected_contract.update(allocator_sha256=observation['allocator_sha256'],allocator_source_pins=declared[0])
            if json.dumps(contract,sort_keys=True)!=json.dumps(expected_contract,sort_keys=True):
                raise ValueError('worker observer complete contract differs from registered source or owner')
            changes={'type=bind,src=/measurement-source,dst=/campaign-source,readonly':
                'type=bind,src='+observation_binding['source_root']+',dst=/campaign-source,readonly',
                f'type=bind,src=/measurement-output/replica-{index},dst=/campaign-measurement':
                'type=bind,src='+item['directory']+',dst=/campaign-measurement',
                f'DSV41_MEASUREMENT_CONTRACT_SHA256=contract-{index}':'DSV41_MEASUREMENT_CONTRACT_SHA256='+digest}
            if any(command['argv'].count(value)!=1 for value in changes):raise ValueError('worker observer template placeholders differ')
            command['argv']=[changes.get(arg,arg) for arg in command['argv']]
    plan=read(root/'launch-plan.json')
    if plan!=expected:raise ValueError('actual launch arguments differ from preregistration')
    image=read(root/'image-qualification.json')
    if (image.get('image')!=config['image'] or image.get('source_revision')!=config.get('source_revision') or
        image.get('image')!=manifest['runtime']['image_digest']):raise ValueError('candidate qualified image differs')
    expected_runtime=None
    for i,command in enumerate(plan['commands']):
        argv=command['argv'];port=31000+i
        placement=read(root/f'placement-{port}.json')
        if (placement.get('container_id')!=state['container_ids'][i] or placement.get('gpu_ids')!=command['gpu_ids'] or
            placement.get('host_port')!=port or placement.get('image_id')!=image['image_id'] or
            placement.get('cpuset_cpus')!=server or not placement.get('effective_cpus') or
            not set(placement['effective_cpus']).issubset(cpus['server_cpus'])):
            raise ValueError('actual Docker image, device or CPU placement differs')
        if observation_binding is not None:
            item=observation_binding['contracts'][i]
            expected_observer={'container_id':state['container_ids'][i],
                'mounts':{'/campaign-source':{'source':observation_binding['source_root'],'writable':False},
                    '/campaign-measurement':{'source':item['directory'],'writable':True}},
                'environment':{'PYTHONPATH':'/campaign-source','PYTHONDONTWRITEBYTECODE':'1',
                    'DSV41_MEASUREMENT_ROOT':'/campaign-measurement',
                    'DSV41_MEASUREMENT_EXTENSION_SHA256':observation['extension_sha256'],
                    'DSV41_MEASUREMENT_CONTRACT_SHA256':item['sha256']}}
            if read(root/f'worker-docker-binding-{port}.json')!=expected_observer:
                raise ValueError('actual worker observer Docker binding differs')
        if (argv[:3]!=['docker','run','-d'] or '--privileged' in argv or
            argv[argv.index('--restart')+1]!='no' or argv[argv.index('-p')+1]!=f'127.0.0.1:{port}:8000' or
            argv[argv.index('--gpus')+1]!='"device='+','.join(map(str,command['gpu_ids']))+'"' or
            'campaign=dsv41-1017' not in argv or 'run_id='+config['run_id'] not in argv):
            raise ValueError('candidate resource or ownership arguments differ')
        attention=json.loads(argv[argv.index('--attention-config')+1])
        current={'backend':attention['backend'],'kv_dtype':argv[argv.index('--kv-cache-dtype')+1],
            'speculative_config':json.loads(argv[argv.index('--speculative-config')+1]) if '--speculative-config' in argv else None,
            'moe_backend':json.loads(argv[argv.index('--kernel-config')+1]).get('moe_backend') if '--kernel-config' in argv else None}
        if expected_runtime is not None and current!=expected_runtime:raise ValueError('candidate replicas differ')
        expected_runtime=current
    if expected_runtime is None:raise ValueError('candidate engines absent')
    verify_pilot.verify_single_request_placement(root,config,{'attribution':{}},expected_profile=expected_runtime,require_selected_rank=False)
    verify_pilot.verify_cleanup(root,config)
    if manifest['runtime'].get('kv_budget_schema') or (root/'kv-budget.json').exists():verify_pilot.verify_kv_budget(root,config)
    expected_ids=state['container_ids']+['router'];qualified=[]
    for i,identity in enumerate(expected_ids):
        evidence=read(root/f'readiness-{i}.json');raw=evidence['raw'];verify_wire.verify_complete(raw)
        text=''.join((c.get('delta') or {}).get('content') or '' for e in raw['events'] if e['kind']=='event' for c in e['data'].get('choices',[]))
        finishes=[c['finish_reason'] for e in raw['events'] if e['kind']=='event' for c in e['data'].get('choices',[]) if c.get('finish_reason') is not None]
        usage=[e['data']['usage'] for e in raw['events'] if e['kind']=='event' and e['data'].get('usage') is not None]
        if (evidence.get('identity')!=identity or raw['request_id']!='smoke'+str(i) or text.strip()!='READY' or
            finishes!=['stop'] or len(usage)!=1 or any(type(usage[0].get(k)) is not int or usage[0][k]<=0 for k in ('prompt_tokens','completion_tokens'))):
            raise ValueError('candidate semantic startup is unproven')
        qualified.append(identity)
    runtime.verify_evidence_manifest(root)
    return {'status':'registered_startup_verified','profile_sha256':profile_sha,'registration_sha256':digest,
        'semantic_identities':qualified,
        'scope':'exact declared argv, raw installed backend/KV/speculation, physical workers and semantic startup; scalar effects, host allocations, kernel trace, quality, performance and adoption remain separate'}
