from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,launcher,profile_registry,readiness,runtime,verify_engine_profile,verify_pilot
from test_node_pilot import fixture_cleanup
import test_pilot_qualification as physical_fixture
from test_measurement_launch import fixture as observation_fixture
import measurement_launch


class EngineProfileVerificationTests(unittest.TestCase):
    def test_frozen_nospec_phase_checks_actual_startup_cli_and_record_integrity(self):
        from test_fixed_profiles import manifest_fixture
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                events=[{'choices':[{'delta':{'content':'READY'}}]},
                    {'choices':[{'delta':{},'finish_reason':'stop'}]},
                    {'choices':[],'usage':{'prompt_tokens':16,'completion_tokens':1}}]
                data=(''.join('data: '+json.dumps(row)+'\n\n' for row in events)+'data: [DONE]\n\n').encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);physical_fixture.PilotQualificationTests().fixture(root)
                manifest=manifest_fixture();record=manifest['runtime']['fixed_profiles']['4xdep2']
                cfg=dict(run_id='fixture',layout='4xdep2',execution_phase='fixed-performance',
                    profile_id=record['id'],image=manifest['runtime']['image_digest'],source_revision='c'*40,
                    model_dir='/fixture-model',cache_dir='/fixture-cache')
                fixture_cleanup(root,cfg)
                def put(name,value):(root/name).write_text(json.dumps(value))
                state=json.loads((root/'execution-state.json').read_text());state['phase']='fixed-performance'
                put('execution-state.json',state);put('manifest.json',manifest)
                put('profile-binding.json',profile_registry.resolve(cfg,manifest))
                put('cpu-partition.json',dict(server_cpus=[0,1],client_cpus=[2,3]))
                put('image-qualification.json',dict(image=cfg['image'],source_revision=cfg['source_revision'],image_id='fixture-image'))
                plan=launcher.render(cfg['layout'],cfg['image'],cfg['run_id'],cfg['model_dir'],cfg['cache_dir'],'0,1')
                put('launch-plan.json',plan)
                for i,cid in enumerate(state['container_ids']):
                    put(f'placement-{31000+i}.json',dict(container_id=cid,gpu_ids=plan['commands'][i]['gpu_ids'],
                        host_port=31000+i,image_id='fixture-image',cpuset_cpus='0,1',effective_cpus=[0,1]))
                for i,identity in enumerate(state['container_ids']+['router']):
                    response=readiness.known_answer(f'http://127.0.0.1:{server.server_port}/v1/chat/completions','/model','smoke'+str(i),5)
                    response['identity']=identity;put('readiness-'+str(i)+'.json',response)
                runtime.seal_evidence(root)
                checked=verify_engine_profile.verify(root)
                self.assertEqual(checked['profile_sha256'],record['profile_sha256'])
                output=io.StringIO()
                with patch.object(sys,'argv',['campaign.py','verify-engine-profile','--run-dir',str(root)]),redirect_stdout(output):campaign.main()
                self.assertEqual(json.loads(output.getvalue()),checked)
                manifest['runtime']['fixed_profiles']['4xdep2']['fixed_cell_ids'].pop()
                put('manifest.json',manifest);runtime.seal_evidence(root)
                with self.assertRaises(ValueError):verify_engine_profile.verify(root)
        finally:server.shutdown();server.server_close();thread.join()

    def test_registered_startup_raw_cli_and_resealed_argv_spec_identity_tampering(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                events=[{'choices':[{'delta':{'content':'READY'}}]}, {'choices':[{'delta':{},'finish_reason':'stop'}]},
                        {'choices':[],'usage':{'prompt_tokens':16,'completion_tokens':1}}]
                data=(''.join('data: '+json.dumps(row)+'\n\n' for row in events)+'data: [DONE]\n\n').encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);physical_fixture.PilotQualificationTests().fixture(root)
                m=campaign.ledger(Path(campaign.__file__).with_name('matrix.json'));m['plan_sha256']='a'*64;m['runtime']['image_digest']='example/image@sha256:'+'b'*64
                ids=[c['id'] for c in m['cells'] if c['suite']=='tuning_trial' and c['layout']=='4xdep2' and c['parameters']['lever']=='dspark_static' and c['parameters']['input']==8000 and c['parameters']['concurrency']==1]
                spec={'method':'dspark','num_speculative_tokens':5,'rejection_sample_method':'block','enable_adaptive_verification':False}
                record=profile_registry.register('dep2-static-k5','4xdep2',{'speculative_config':spec},m,
                    hypothesis='Static K5 may improve useful throughput at matched anchors.',deciding_cells=ids,protected_tail='p95_ttft_seconds')
                m['runtime']['qualification_profiles']={record['id']:record}
                cfg={'run_id':'fixture','layout':'4xdep2','execution_phase':'qualification','profile_id':record['id'],
                    'image':m['runtime']['image_digest'],'source_revision':'c'*40,'model_dir':'/fixture-model','cache_dir':'/fixture-cache'}
                fixture_cleanup(root,cfg)
                def put(name,value):(root/name).write_text(json.dumps(value))
                state=json.loads((root/'execution-state.json').read_text());state['phase']='qualification';put('execution-state.json',state)
                put('manifest.json',m);put('profile-binding.json',profile_registry.resolve(cfg,m));put('cpu-partition.json',{'server_cpus':[0,1],'client_cpus':[2,3]})
                put('image-qualification.json',{'image':cfg['image'],'source_revision':cfg['source_revision'],'image_id':'fixture-image'})
                plan=launcher.render(cfg['layout'],cfg['image'],cfg['run_id'],cfg['model_dir'],cfg['cache_dir'],'0,1',overrides=record['profile_delta']);put('launch-plan.json',plan)
                for i,cid in enumerate(state['container_ids']):
                    put(f'placement-{31000+i}.json',{'container_id':cid,'gpu_ids':plan['commands'][i]['gpu_ids'],
                        'host_port':31000+i,'image_id':'fixture-image','cpuset_cpus':'0,1','effective_cpus':[0,1]})
                    p=root/('rank-startup-'+cid+'.log');line,*rest=p.read_text().splitlines();installed=json.loads(line.split('--args-json ',1)[1]);installed['speculative_config']=spec
                    p.write_text('Launching Rust frontend: fixture --args-json '+json.dumps(installed)+'\n'+'\n'.join(rest))
                for i,identity in enumerate(state['container_ids']+['router']):
                    data=readiness.known_answer(f'http://127.0.0.1:{server.server_port}/v1/chat/completions','/model','smoke'+str(i),5);data['identity']=identity;put(f'readiness-{i}.json',data)
                runtime.seal_evidence(root);checked=verify_engine_profile.verify(root)
                self.assertEqual(checked['status'],'registered_startup_verified');self.assertEqual(checked['profile_sha256'],record['profile_sha256'])
                output=io.StringIO()
                with patch.object(sys,'argv',['campaign.py','verify-engine-profile','--run-dir',str(root)]),redirect_stdout(output):campaign.main()
                self.assertEqual(json.loads(output.getvalue()),checked)
                with self.assertRaisesRegex(ValueError,'cannot certify'):verify_pilot.verify(root)
                source,observer,proof,pins=observation_fixture(root,with_allocator=True)
                import ast
                memory_pins=ast.literal_eval(ast.parse((source/'allocator_observations.py').read_text()).body[0].value)
                m['runtime']['toolkit_manifest_sha256']='d'*64
                instrumented=profile_registry.register('dep2-static-k5-observed','4xdep2',
                    {'speculative_config':spec,'worker_observation':observer},m,
                    hypothesis='Observe verified target positions without modifying model outputs.',
                    deciding_cells=ids,protected_tail='p95_ttft_seconds',control=record)
                m['runtime']['qualification_profiles'][instrumented['id']]=instrumented
                cfg.update(profile_id=instrumented['id'],run_dir=str(root),toolkit_manifest=str(source.parent/'MANIFEST.json'))
                state.update(driver_identity={'boot_id':'fixture-boot'},deadline_ns=time.monotonic_ns()+10_000_000_000)
                put('execution-state.json',state);put('run-config.json',cfg);put('manifest.json',m)
                put('profile-binding.json',profile_registry.resolve(cfg,m))
                template=launcher.render(cfg['layout'],cfg['image'],cfg['run_id'],cfg['model_dir'],cfg['cache_dir'],
                    '0,1',overrides=instrumented['profile_delta'])
                plan,observed=measurement_launch.bind(template,observer,root,source,run_id='fixture',
                    profile_sha=instrumented['profile_sha256'],toolkit_sha='d'*64,deadline_ns=state['deadline_ns'],
                    tp=1,dp=2,image_id='fixture-image',installed_evidence=proof,source_pins=pins,boot_id='fixture-boot',
                    allocator_evidence=root/'worker-allocator-source-proof',allocator_pins=memory_pins)
                put('launch-plan.json',plan)
                for i,command in enumerate(plan['commands']):
                    item=observed['contracts'][i];argv=command['argv']
                    docker={'Mounts':[{'Destination':'/campaign-source','Type':'bind','Source':str(source),'RW':False},
                        {'Destination':'/campaign-measurement','Type':'bind','Source':item['directory'],'RW':True}],
                        'Config':{'Env':[argv[j+1] for j,a in enumerate(argv) if a=='-e']}}
                    put(f'worker-docker-binding-{31000+i}.json',{'container_id':state['container_ids'][i],
                        **measurement_launch.docker_binding(docker,observed,i)})
                runtime.seal_evidence(root)
                self.assertEqual(verify_engine_profile.verify(root)['profile_sha256'],instrumented['profile_sha256'])
                output=io.StringIO()
                with patch.object(sys,'argv',['campaign.py','verify-engine-profile','--run-dir',str(root)]),redirect_stdout(output):campaign.main()
                self.assertEqual(json.loads(output.getvalue())['profile_sha256'],instrumented['profile_sha256'])
                for name,change in [('worker-observation-binding.json',lambda v:v['installed_source'].update(source_pins={})),
                    ('worker-observation-binding.json',lambda v:v['allocator_installed_source'].update(source_pins={})),
                    ('worker-observation/replica-0/contract.json',lambda v:v.update(allocator_source_pins={})),
                    ('worker-docker-binding-31000.json',lambda v:v['mounts']['/campaign-source'].update(writable=True)),
                    ('worker-observation/replica-0/contract.json',lambda v:v.update(deadline_ns=v['deadline_ns']+1))]:
                    path=root/name;old=path.read_bytes();value=json.loads(old);change(value);put(name,value);runtime.seal_evidence(root)
                    with self.subTest(observer_tamper=name),self.assertRaises(ValueError):verify_engine_profile.verify(root)
                    path.write_bytes(old);runtime.seal_evidence(root)
                for name,change in [('launch-plan.json',lambda v:v['commands'][0]['argv'].append('--unregistered')),
                    ('placement-31000.json',lambda v:v.update(image_id='other-image'))]:
                    path=root/name;old=path.read_text();value=json.loads(old);change(value);put(name,value);runtime.seal_evidence(root)
                    with self.assertRaises(ValueError):verify_engine_profile.verify(root)
                    path.write_text(old);runtime.seal_evidence(root)
                p=root/('rank-startup-'+state['container_ids'][0]+'.log');p.write_text(p.read_text().replace('"block"','"synthetic"'));runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'real-block'):verify_engine_profile.verify(root)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
