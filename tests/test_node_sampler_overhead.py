"""Source/arm preparation and hand-calculated comparison controls, no GPU claim."""
import copy,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_workload,node_sampler_overhead as producer,runtime,verify_sampler_overhead as verifier
import attempts,threading
from types import SimpleNamespace
import test_node_fixed_qualification as input_fixture
from test_vllm_prompt import FakeTokenizer


class SamplerOverheadTests(unittest.TestCase):
    def fixture(self,root):
        config,_=input_fixture.FixedQualificationTests().fixture(root)
        manifest=json.loads(Path(config['manifest']).read_text())
        cell=next(c for c in manifest['cells'] if c['suite']=='sampler_overhead' and c['layout']=='2xdep4' and c['parameters']['arm']=='control_before')
        config.update(cell_id=cell['id'],profile_id='control');config.pop('tasks_file')
        files=[]
        for repetition in range(3):
            source=root/('dsv41-overhead-input-r'+str(repetition));source.mkdir()
            for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
                (source/name).write_bytes(Path(config[key]).read_bytes())
            origin={'cell_id':'core-2xdep4-i8000-c8-cold-o2048','layout':'2xdep4','image':config['image'],'task_family':'fixed','repetition':repetition}
            runtime.atomic_json(source/'run-config.json',origin)
            runtime.atomic_json(source/'prompt-result.json',{'status':'passed','cleanup':'passed','task_family':'fixed',
                'cell_id':origin['cell_id'],'repetition':repetition,'tasks':8})
            tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',8,repetition,i) for i in range(8)]
            runtime.atomic_json(source/'fixed-tasks.json',tasks);runtime.seal_evidence(source);files.append(str(source/'fixed-tasks.json'))
        config['tasks_files']=files
        return config

    def test_declared_arm_binds_three_distinct_installed_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.fixture(root);rows,budget=producer.prepare(config,root)
            self.assertEqual(len(rows),3);self.assertEqual(sum(len(r['prepared']['tasks']) for r in rows),24)
            self.assertEqual(budget,5400);self.assertEqual(len({r['identity'] for r in rows}),3)
            total=dict(config,overhead_contract='total-worker-and-host/v1')
            self.assertEqual(len(producer.prepare(total,root)[0]),3)
            self.assertEqual(len(producer.prepare(dict(total,retry_repetitions=[0],retry_reason='explicit no-label instruction repair'),root)[0]),3)
            for bad in ([],[True],[3],[0,0],[[0]]):
                with self.subTest(retries=bad),self.assertRaises(ValueError):
                    producer.prepare(dict(total,retry_repetitions=bad,retry_reason='explicit no-label instruction repair'),root)
            with self.assertRaisesRegex(ValueError,'unknown sampler overhead'):
                producer.prepare(dict(config,overhead_contract='invented'),root)
            manifest=json.loads(Path(config['manifest']).read_text())
            challenger=next(c for c in manifest['cells'] if c['suite']=='sampler_overhead' and c['layout']=='2xdep4' and c['parameters']['arm']=='challenger')
            with self.assertRaisesRegex(ValueError,'observer only in the full arm'):
                producer.prepare(dict(total,cell_id=challenger['id']),root)
            for change in ({'execution_phase':'scored'},{'tasks_files':config['tasks_files'][:2]},
                           {'tasks_files':[config['tasks_files'][0]]*3},{'profile_id':'candidate'},
                           {'request_timeout_seconds':True},{'request_timeout_seconds':1801}):
                with self.subTest(change=change),self.assertRaises(ValueError):producer.prepare(dict(config,**change),root)
            path=Path(config['tasks_files'][1]).parent/'prompt-result.json';record=json.loads(path.read_text())
            record['repetition']=0;runtime.atomic_json(path,record);runtime.seal_evidence(path.parent)
            with self.assertRaisesRegex(ValueError,'generation'):producer.prepare(config,root)

    def test_matched_comparison_is_paired_and_signed_with_fresh_disjoint_owners(self):
        with tempfile.TemporaryDirectory() as d:
            roots=[];arms=[]
            for index,(arm,rate) in enumerate([('control_before',100),('challenger',90),('control_after',100)]):
                root=Path(d)/arm;root.mkdir();roots.append(root)
                runtime.atomic_json(root/'run-config.json',{'layout':'fixture','image':'fixture','toolkit_manifest':'fixture','profile_id':'fixture'})
                runtime.atomic_json(root/'manifest.json',{'fixture':'identical manifest'})
                runtime.atomic_json(root/'execution-state.json',{'driver_identity':{'boot_id':'fixture','pid':index+1,'start_ticks':index+1},'container_ids':['container'+str(index)]})
                runtime.atomic_json(root/'cleanup-result.json',{'monotonic_ns':index*100+90})
                runtime.atomic_json(root/'startup-timing.json',{'launch_started_ns':index*100+1})
                arms.append({'cells':[{'arm':arm,'profile_sha256':'fixture','input_body_sha256':['fixture'],
                    'valid_tokens_per_second':rate} for _ in range(3)]})
            with patch.object(verifier,'verify',side_effect=arms):result=verifier.compare(*roots)
            self.assertTrue(result['attribution_qualified'])
            self.assertAlmostEqual(result['repetitions'][0]['incremental_overhead_fraction'],.1)
            changed=copy.deepcopy(arms);changed[1]['cells'][0]['valid_tokens_per_second']=110
            with patch.object(verifier,'verify',side_effect=changed):result=verifier.compare(*roots)
            self.assertAlmostEqual(result['repetitions'][0]['incremental_overhead_fraction'],-.1)
            changed=copy.deepcopy(arms);changed[2]['cells'][1]['valid_tokens_per_second']=120
            with patch.object(verifier,'verify',side_effect=changed):self.assertFalse(verifier.compare(*roots)['attribution_qualified'])
            changed=copy.deepcopy(arms);changed[1]['cells'][0]['input_body_sha256']=['different']
            with patch.object(verifier,'verify',side_effect=changed),self.assertRaisesRegex(ValueError,'transmitted workload'):verifier.compare(*roots)
            runtime.atomic_json(roots[1]/'execution-state.json',{'driver_identity':{'boot_id':'fixture','pid':2,'start_ticks':2},'container_ids':['container0']})
            with patch.object(verifier,'verify',side_effect=arms),self.assertRaisesRegex(ValueError,'reused engine'):verifier.compare(*roots)

    def test_total_overhead_changes_only_observer_and_keeps_fresh_matched_controls(self):
        with tempfile.TemporaryDirectory() as d:
            roots=[];arms=[];bindings=[]
            for index,(arm,rate) in enumerate([('control_before',100),('challenger',90),('control_after',100)]):
                root=Path(d)/arm;root.mkdir();roots.append(root);profile=('c' if index==1 else 'b')*64
                delta={'max_num_batched_tokens':8192}
                if index==1:delta['worker_observation']={'fixture':True}
                binding={'profile_sha256':profile,'record':{'profile_delta':delta,'control_profile_sha256':'b'*64}}
                bindings.append(binding);runtime.atomic_json(root/'profile-binding.json',binding)
                runtime.atomic_json(root/'run-config.json',{'layout':'fixture','image':'fixture','toolkit_manifest':'fixture',
                    'profile_id':'observed' if index==1 else 'base','overhead_contract':'total-worker-and-host/v1'})
                runtime.atomic_json(root/'manifest.json',{'scope':'isolated paired arithmetic fixture'})
                runtime.atomic_json(root/'execution-state.json',{'driver_identity':{'boot_id':'fixture','pid':index+1,'start_ticks':1},'container_ids':['container'+str(index)]})
                runtime.atomic_json(root/'cleanup-result.json',{'monotonic_ns':index*100+90})
                runtime.atomic_json(root/'startup-timing.json',{'launch_started_ns':index*100+1})
                arms.append({'cells':[{'arm':arm,'profile_sha256':profile,'input_body_sha256':['fixture'],
                    'valid_tokens_per_second':rate} for _ in range(3)]})
            with patch.object(verifier,'verify',side_effect=arms):result=verifier.compare(*roots)
            self.assertTrue(result['attribution_qualified']);self.assertIn('worker observer',result['scope'])
            self.assertAlmostEqual(result['repetitions'][0]['incremental_overhead_fraction'],.1)
            changed=copy.deepcopy(bindings[1]);changed['record']['profile_delta']['max_num_batched_tokens']=16384
            runtime.atomic_json(roots[1]/'profile-binding.json',changed)
            with patch.object(verifier,'verify',side_effect=arms),self.assertRaisesRegex(ValueError,'beyond declared instrumentation'):
                verifier.compare(*roots)
            runtime.atomic_json(roots[1]/'profile-binding.json',bindings[1])
            changed=copy.deepcopy(arms);changed[1]['cells'][0]['input_body_sha256']=['other']
            with patch.object(verifier,'verify',side_effect=changed),self.assertRaisesRegex(ValueError,'transmitted workload'):
                verifier.compare(*roots)

    def test_arm_reserves_once_persists_all_repetitions_and_finishes_after_shared_verification(self):
        self.arm_lifecycle(interrupt=False)

    def test_explicit_single_retry_and_two_fresh_repetitions_share_one_session(self):
        self.arm_lifecycle(interrupt=False,mixed=True)

    def test_interrupted_arm_keeps_capture_and_intents_without_automatic_replay(self):
        self.arm_lifecycle(interrupt=True)

    def arm_lifecycle(self,interrupt,mixed=False):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);config=self.fixture(base);prepared,budget=producer.prepare(config,base)
            root=Path(config['run_dir']);registry=base/'registry';mapping={'ports':[31000,31001],'dp':4}
            runtime.atomic_json(base/'routing-fixture.json',{'scope':'isolated placement; no actual router claim'})
            route=base/'route';route.mkdir();runtime.atomic_json(route/'fixture.json',{'scope':'isolated routing helper'})
            runtime.seal_evidence(route);config['routing_map_evidence']=str(route)
            calls=[];owner={'boot_id':'fixture','pid':123,'start_ticks':456}
            prior=None;prior_bytes=None
            if mixed:
                config.update(retry_repetitions=[0],retry_reason='explicit no-label instruction repair')
                prior=attempts.begin(registry,prepared[0]['identity'],method_sha256='a'*64)
                (prior/'raw.json').write_text('{}');attempts.finish(prior,'invalid',['raw.json'])
                prior_bytes=(prior/'attempt.json').read_bytes()
            def capture(session,dest,endpoint,child,inputs,selected,placement,router_log,cache_config):
                calls.append(child['repetition'])
                rows=[r for r in attempts.load_attempts(registry) if r.get('group_id')==config['run_id']]
                self.assertEqual(len(rows),3);self.assertTrue(all(r['status']=='running' for r in rows))
                runtime.atomic_json(dest/'fixture-captured.json',{'repetition':child['repetition']})
                if interrupt and child['repetition']==1:raise RuntimeError('intentional capture interruption')
            def run(c,callback):
                root.mkdir();runtime.atomic_json(root/'execution-state.json',{'run_id':c['run_id'],
                    'driver_identity':owner,'watchdog_identity':{'pid':1,'boot_id':'fixture','start_ticks':1},
                    'owned_processes':[],'container_ids':[]})
                def begin(cell,until):
                    runtime.atomic_json(root/'sampler-control-applied.json',{'cell_id':cell,'mode':'essential','completed_ns':2})
                    return {'mode':'essential','selected_ns':1}
                session=SimpleNamespace(deadline_ns=__import__('time').monotonic_ns()+10_000_000_000_000,
                    begin_sampler_control=begin,ensure_healthy=lambda:None)
                callback(session,root,'declared-fixture')
            checks=[{'status':'passed','cell_id':r['identity'],'scope':'isolated registry/projection; physical and wire verification excluded'} for r in prepared]
            complete={'status':'passed','cells':checks,'scope':'isolated registry/projection; physical and wire verification excluded'}
            with patch.object(producer,'prepare',return_value=(prepared,budget)), \
                 patch.object(producer.node_smoke,'validate',return_value=({}, {'lock_sha256':'fixture','router_executable_sha256':'fixture'},12000)), \
                 patch.object(producer.routing_map,'load_verified',return_value=mapping), \
                 patch.object(producer.routing_map,'select',return_value=[]), \
                 patch.object(producer.runtime,'DEFAULT_NODE_LOCK',base/'node.lock'), \
                 patch.object(producer.runtime,'live_preflight'),patch.object(producer,'DEFAULT_ROOT',registry), \
                 patch.object(producer.runtime,'process_identity',return_value=owner), \
                 patch.object(producer.node_smoke,'run',side_effect=run), \
                 patch.object(producer.node_fixed_qualification,'capture_prepared',side_effect=capture), \
                 patch.object(verifier,'verify',return_value=complete):
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError,'intentional capture interruption'):producer.run(config)
                else:self.assertEqual(producer.run(config),complete)
            rows=[r for r in attempts.load_attempts(registry) if r.get('group_id')==config['run_id']]
            if mixed:
                self.assertEqual((prior/'attempt.json').read_bytes(),prior_bytes)
                self.assertEqual(len(attempts.load_attempts(registry)),4)
                self.assertEqual(sum(r['retry_reason'] is not None for r in rows),1)
            if interrupt:
                self.assertEqual(calls,[0,1]);self.assertTrue(all(r['status']=='running' for r in rows))
                self.assertEqual(sum((registry/r['directory']/'capture.json').exists() for r in rows),1)
                self.assertEqual(sum((registry/r['directory']/'dispatch-intent.json').exists() for r in rows),2)
                self.assertTrue(all((registry/r['directory']/'failure.json').exists() for r in rows))
                self.assertFalse((root/'sampler-overhead-result.json').exists())
            else:
                self.assertEqual(calls,[0,1,2]);self.assertTrue(all(r['status']=='passed' for r in rows))
                runtime.verify_evidence_manifest(root)
                for r in rows:
                    for name,digest in r['evidence_sha256'].items():
                        self.assertEqual(__import__('hashlib').sha256((registry/r['directory']/name).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
