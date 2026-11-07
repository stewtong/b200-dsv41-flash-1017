"""Declared inputs and owned registry lifecycle; no GPU or live model claim."""
import copy,hashlib,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import attempts,node_model_client_check as producer,profile_registry,runtime,verify_model_client_check
import test_node_sampler_overhead as inputs_fixture
import test_cache_source_proof as source_fixture


@patch.object(producer.cache_source_proof,'PINNED_SHA256',source_fixture.FIXTURE_PIN)
class ModelClientCheckTests(unittest.TestCase):
    def fixture(self,root):
        c=inputs_fixture.SamplerOverheadTests().fixture(root)
        m=json.loads(Path(c['manifest']).read_text())
        ids=[x['id'] for x in m['cells'] if x['suite']=='tuning_trial' and x['layout']==c['layout']][:1]
        record=profile_registry.register('client-cache-off',c['layout'],{'speculative_config':None,'enable_prefix_caching':False},m,
            hypothesis='Isolate one versus two clients with prefix reuse disabled uniformly.',deciding_cells=ids,protected_tail='p95_request_seconds')
        m['runtime']['qualification_profiles']['client-cache-off']=record
        toolkit=root/'fixture-toolkit-MANIFEST.json'
        runtime.atomic_json(toolkit,{'reproduce/model_clients.py':hashlib.sha256(Path(producer.__file__).with_name('model_clients.py').read_bytes()).hexdigest()})
        c['toolkit_manifest']=str(toolkit);m['runtime']['toolkit_manifest_sha256']=hashlib.sha256(toolkit.read_bytes()).hexdigest()
        runtime.atomic_json(Path(c['manifest']),m)
        image=json.loads(Path(c['image_qualification']).read_text());image['image_id']=source_fixture.IMAGE
        runtime.atomic_json(Path(c['image_qualification']),image)
        c['cache_source_evidence']=str(source_fixture.fixture(root/'fixture-cache-source'))
        for f in c['tasks_files']:
            source=Path(f).parent;runtime.atomic_json(source/'manifest.json',m)
            runtime.atomic_json(source/'image-qualification.json',image);runtime.seal_evidence(source)
        c.update(profile_id='client-cache-off',client_input_tokens=8000,client_concurrency=8,request_timeout_seconds=900)
        return c

    def test_preparation_requires_exact_three_sources_and_explicit_cache_off_profile(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.fixture(root);p,budget=producer.prepare(c,root)
            self.assertEqual(budget,8100);self.assertEqual(len(p['inputs']),3);self.assertEqual(p['concurrency'],8)
            self.assertEqual(p['identity'],producer.identity(c['layout'],p['binding']['profile_sha256'],8000,8))
            for change in ({'execution_phase':'scored'},{'profile_id':'control'},{'client_input_tokens':True},
                           {'client_concurrency':1},{'tasks_files':c['tasks_files'][:2]},
                           {'tasks_files':[c['tasks_files'][0]]*3},{'request_timeout_seconds':True}):
                with self.subTest(change=change),self.assertRaises(ValueError):producer.prepare(dict(c,**change),root)
            source=Path(c['tasks_files'][1]).parent;result=json.loads((source/'prompt-result.json').read_text())
            result['repetition']=0;runtime.atomic_json(source/'prompt-result.json',result);runtime.seal_evidence(source)
            with self.assertRaisesRegex(ValueError,'generation'):producer.prepare(c,root)

    def test_one_owned_attempt_covers_nine_ordered_probes_and_reconciles_before_pass(self):
        self.lifecycle(False)

    def test_interrupted_probe_keeps_raw_and_running_attempt_without_replay(self):
        self.lifecycle(True)

    def lifecycle(self,interrupt):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);c=self.fixture(base);p,budget=producer.prepare(c,base);root=Path(c['run_dir']);registry=base/'registry'
            route=base/'route';route.mkdir();runtime.atomic_json(route/'fixture.json',{'scope':'isolated placement fixture'});runtime.seal_evidence(route)
            c['routing_map_evidence']=str(route);mapping={'ports':[31000,31001],'dp':4}
            owner={'boot_id':'fixture','pid':123,'start_ticks':456};seen=[]
            placement=[{'task_session':i,'session_header':'unit'+str(i),'endpoint':'http://127.0.0.1:31000/metrics','rank':str(i%4)} for i in range(8)]
            def capture(session,endpoint,tasks,dest,*args,**kwargs):
                seen.append((Path(dest).name,kwargs['client_workers']));self.assertIs(kwargs['prefix_caching'],False)
                self.assertEqual(kwargs['client_cpus'],[0,1]);self.assertEqual(attempts.load_attempts(registry)[0]['status'],'running')
                Path(dest).mkdir(parents=True);(Path(dest)/'requests/measured').mkdir(parents=True)
                runtime.atomic_json(dest/'fixture-capture.json',{'scope':'lifecycle only; wire/metrics verifier mocked'})
                if interrupt and len(seen)==2:raise RuntimeError('intentional model-client interruption')
                for t in tasks:runtime.atomic_json(dest/'requests/measured'/(t['id']+'.json'),
                    {'raw':{'start_ns':time.monotonic_ns()-100000,'terminal_ns':time.monotonic_ns()}})
                return {'session_routes':{r['session_header']:[r['endpoint'],r['rank']] for r in placement}}
            def run(config,callback):
                root.mkdir();runtime.atomic_json(root/'cpu-partition.json',{'client_cpus':[0,1]});runtime.atomic_json(root/'cache-config.json',{'block_size':128})
                runtime.atomic_json(root/'image-qualification.json',{'image_id':source_fixture.IMAGE})
                (root/'router.log').write_text('')
                session=SimpleNamespace(root=root,run_id=c['run_id'],deadline_ns=time.monotonic_ns()+10_000_000_000_000,ensure_healthy=lambda:None)
                callback(session,root,'http://127.0.0.1:31100/v1/chat/completions')
            result={'status':'passed','cell_id':p['identity'],'scope':'registry lifecycle fixture; independent acceptance mocked'}
            with patch.object(producer,'prepare',return_value=(p,budget)), \
                 patch.object(producer.node_smoke,'validate',return_value=({}, {'lock_sha256':'fixture','router_executable_sha256':'fixture'},14000)), \
                 patch.object(producer.routing_map,'load_verified',return_value=mapping), \
                 patch.object(producer.routing_map,'select',return_value=placement), \
                 patch.object(producer.runtime,'process_identity',return_value=owner), \
                 patch.object(producer.runtime,'DEFAULT_NODE_LOCK',base/'node.lock'), \
                 patch.object(producer.runtime,'live_preflight'), \
                 patch.object(producer,'DEFAULT_ROOT',registry), \
                 patch.object(producer.node_smoke,'run',side_effect=run), \
                 patch.object(producer.node_pilot,'clock_domain',return_value='fixture:clock'), \
                 patch.object(producer.fixed_attribution,'capture',side_effect=capture), \
                 patch('verify_telemetry.verify_window',return_value={'scope':'isolated lifecycle fixture'}), \
                 patch.object(verify_model_client_check,'verify',return_value=result):
                if interrupt:
                    with self.assertRaisesRegex(RuntimeError,'intentional'):producer.run(c)
                else:self.assertEqual(producer.run(c),result)
                with self.assertRaises(runtime.RuntimeErrorClosed):producer.run(c)
            rows=attempts.load_attempts(registry);self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['status'],'running' if interrupt else 'passed')
            if interrupt:
                self.assertTrue((registry/rows[0]['directory']/'failure.json').exists())
                self.assertTrue((root/'probes/r0-two/fixture-capture.json').exists())
            else:self.assertEqual(seen,[(f'r{r}-{a}',w) for r in range(3) for a,w in [('before',1),('two',2),('after',1)]])
