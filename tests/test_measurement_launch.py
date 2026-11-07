"""Persisted observer launch contracts and rejected source/mount tampering."""
import copy,hashlib,json,sys,tempfile,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import launcher,measurement_launch as launch,runtime


def fixture(directory,with_allocator=False):
    root=Path(directory);source=root/'toolkit'/'reproduce';source.mkdir(parents=True)
    content='# public installed fixture\n';pins={'v1/fixture.py':hashlib.sha256(content.encode()).hexdigest()}
    for key,name in launch.FILES.items():
        (source/name).write_text('SOURCE_PINS='+repr(pins)+'\n' if key=='extension_sha256' else '# public observer fixture\n')
    spec={'schema':launch.SCHEMA,**{key:hashlib.sha256((source/name).read_bytes()).hexdigest() for key,name in launch.FILES.items()}}
    proof=root/'worker-source-proof';proof.mkdir()
    runtime.atomic_json(proof/'installed-source.json',{'files':[{'relative_path':'v1/fixture.py','source':content,'sha256':pins['v1/fixture.py']}]})
    runtime.atomic_json(proof/'source-result.json',{'status':'captured','image_id':'fixture-image'})
    runtime.atomic_json(proof/'source-container.json',{'image_id':'fixture-image','argv':['docker','run','--read-only']})
    runtime.atomic_json(proof/'cleanup-result.json',{'success':True});runtime.seal_evidence(proof)
    if with_allocator:
        memory_text='# public installed allocator fixture\n'
        memory_pins={'cuda/memory.py':hashlib.sha256(memory_text.encode()).hexdigest()}
        path=source/'allocator_observations.py';path.write_text('SOURCE_PINS='+repr(memory_pins)+'\n')
        spec['allocator_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        memory=root/'worker-allocator-source-proof';memory.mkdir()
        runtime.atomic_json(memory/'installed-source.json',{'files':[{'relative_path':'torch/cuda/memory.py','source':memory_text,'sha256':memory_pins['cuda/memory.py']}]})
        runtime.atomic_json(memory/'source-result.json',{'status':'captured','image_id':'fixture-image'})
        runtime.atomic_json(memory/'source-container.json',{'image_id':'fixture-image','argv':['docker','run','--read-only']})
        runtime.atomic_json(memory/'cleanup-result.json',{'success':True});runtime.seal_evidence(memory)
    return source,spec,proof,pins


class MeasurementLaunchTests(unittest.TestCase):
    def test_opt_in_allocator_requires_retained_registered_source_and_installed_parity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,spec,proof,pins=fixture(root)
            memory_source=source/'allocator_observations.py';memory_source.write_text('# public allocator fixture\n')
            spec['allocator_sha256']=hashlib.sha256(memory_source.read_bytes()).hexdigest()
            memory_proof=root/'memory-proof';memory_proof.mkdir();text='# public installed memory fixture\n'
            memory_pins={'cuda/memory.py':hashlib.sha256(text.encode()).hexdigest()}
            runtime.atomic_json(memory_proof/'installed-source.json',{'files':[{'relative_path':'torch/cuda/memory.py','source':text,'sha256':memory_pins['cuda/memory.py']}]})
            runtime.atomic_json(memory_proof/'source-result.json',{'status':'captured','image_id':'fixture-image'})
            runtime.atomic_json(memory_proof/'source-container.json',{'image_id':'fixture-image','argv':['docker','run','--read-only']})
            runtime.atomic_json(memory_proof/'cleanup-result.json',{'success':True});runtime.seal_evidence(memory_proof)
            plan=launcher.render('2xtp4','example/image@sha256:'+'a'*64,'fixture','/model','/cache',overrides={'worker_observation':spec})
            args=dict(run_id='fixture',profile_sha='b'*64,toolkit_sha='c'*64,deadline_ns=time.monotonic_ns()+10_000_000_000,
                tp=4,dp=1,image_id='fixture-image',installed_evidence=proof,source_pins=pins,boot_id='fixture-boot')
            with self.assertRaisesRegex(ValueError,'allocator installed-source proof missing'):launch.bind(plan,spec,root,source,**args)
            bound,binding=launch.bind(plan,spec,root,source,**args,allocator_evidence=memory_proof,allocator_pins=memory_pins)
            self.assertEqual(binding['contracts'][0]['contract']['allocator_sha256'],spec['allocator_sha256'])
            self.assertEqual(binding['contracts'][0]['contract']['allocator_source_pins'],memory_pins)
            self.assertEqual((root/'worker-observer-source/allocator_observations.py').read_bytes(),memory_source.read_bytes())

    def test_persisted_registered_template_contracts_and_observed_docker_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,spec,proof,pins=fixture(root)
            image='example/image@sha256:'+'a'*64
            before=launcher.render('2xtp4',image,'fixture','/model','/cache','0,1')
            plan=launcher.render('2xtp4',image,'fixture','/model','/cache','0,1',overrides={'worker_observation':spec})
            self.assertNotEqual(launcher.profile_sha256('2xtp4'),launcher.profile_sha256('2xtp4',{'worker_observation':spec}))
            self.assertEqual([c['gpu_ids'] for c in plan['commands']],[c['gpu_ids'] for c in before['commands']])
            bound,binding=launch.bind(plan,spec,root,source,run_id='fixture',profile_sha='b'*64,
                toolkit_sha='c'*64,deadline_ns=time.monotonic_ns()+10_000_000_000,tp=4,dp=1,
                image_id='fixture-image',installed_evidence=proof,source_pins=pins,boot_id='fixture-boot')
            self.assertEqual(json.loads((root/'worker-observation-binding.json').read_text()),binding)
            for index,command in enumerate(bound['commands']):
                item=binding['contracts'][index];path=Path(item['directory'])/'contract.json'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item['sha256'])
                self.assertEqual(json.loads(path.read_text())['expected_gpu_ids'],command['gpu_ids'])
                argv=command['argv'];self.assertLess(argv.index('PYTHONPATH=/campaign-source'),argv.index(image))
                self.assertEqual(argv[-2:],['--worker-extension-cls','measurement_worker_extension.CampaignMeasurementExtension'])
                row={'Mounts':[{'Destination':'/campaign-source','Type':'bind','Source':str(source),'RW':False},
                    {'Destination':'/campaign-measurement','Type':'bind','Source':item['directory'],'RW':True}],
                    'Config':{'Env':[argv[i+1] for i,a in enumerate(argv) if a=='-e']}}
                checked=launch.docker_binding(row,binding,index)
                self.assertEqual(checked['environment']['DSV41_MEASUREMENT_CONTRACT_SHA256'],item['sha256'])
                changed=copy.deepcopy(row);changed['Mounts'][0]['RW']=True
                with self.assertRaisesRegex(ValueError,'mount differs'):launch.docker_binding(changed,binding,index)
                changed=copy.deepcopy(row);changed['Config']['Env'].append('PYTHONPATH=/other')
                with self.assertRaisesRegex(ValueError,'environment differs'):launch.docker_binding(changed,binding,index)
            self.assertEqual(plan['commands'][0]['argv'].count('DSV41_MEASUREMENT_CONTRACT_SHA256=contract-0'),1)

    def test_changed_sources_and_unproven_installed_bytes_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source,spec,proof,pins=fixture(root)
            plan=launcher.render('4xdep2','example/image@sha256:'+'a'*64,'fixture','/model','/cache',
                overrides={'worker_observation':spec})
            args=dict(run_id='fixture',profile_sha='b'*64,toolkit_sha='c'*64,
                deadline_ns=time.monotonic_ns()+10_000_000_000,tp=1,dp=2,
                image_id='fixture-image',installed_evidence=proof,source_pins=pins,boot_id='fixture-boot')
            for name,changed in [('runtime.py',b'# changed fixture\n')]:
                path=source/name;old=path.read_bytes();path.write_bytes(changed)
                with self.assertRaisesRegex(ValueError,'source bytes differ'):launch.bind(plan,spec,root,source,**args)
                self.assertFalse((root/'worker-observation').exists());path.write_bytes(old)
            with self.assertRaisesRegex(ValueError,'parity missing'):
                launch.bind(plan,spec,root,source,**dict(args,source_pins={'v1/missing.py':'f'*64}))
            self.assertFalse((root/'worker-observation').exists())
            for value in (None,{},dict(spec,extension_sha256='bad'),dict(spec,extra=True)):
                with self.assertRaises(ValueError):launch.validate(value)


if __name__=='__main__':unittest.main()
