"""Synthetic shared ownership with actual loopback HTTP and independent checks."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import sys,tempfile,threading,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import attempts,campaign,node_pilot,node_pilot_block,pilot_registry,runtime,verify_pilot_block,workload
from test_node_pilot import fixture_provenance,fixture_cleanup
import test_pilot_qualification as physical_fixture


class PilotBlockTests(unittest.TestCase):
    def shared_fixture(self,base):
        root=base/'dsv41-shared-fixture';dest=root/'cells/cell0';dest.mkdir(parents=True)
        owner={'boot_id':'fixture','pid':4321,'start_ticks':10}
        collector={'boot_id':'fixture','pid':4323,'start_ticks':12}
        state={'run_id':'shared-fixture','driver_identity':owner,
               'watchdog_identity':{'boot_id':'fixture','pid':4322,'start_ticks':11},
               'owned_processes':[collector],'nvml_identity':collector,
               'stopping':True,'watchdog_stop':True,'cleanup_started_ns':5}
        runtime.atomic_json(root/'run-config.json',{'run_id':'shared-fixture'})
        runtime.atomic_json(root/'execution-state.json',state)
        runtime.atomic_json(root/'smoke-result.json',{'status':'passed'})
        runtime.atomic_json(root/'cleanup-result.json',{'success':True})
        runtime.atomic_json(root/'gpu-idle-after-cleanup.json',{'idle':True})
        runtime.atomic_json(root/'nvml-live.json',{'pid':collector['pid'],'measurement_mode':False})
        (root/'telemetry-0000.jsonl').write_bytes(b'{}\n'*340000)
        return root,dest,owner,collector

    def test_reprojecting_same_inode_streams_leaves_no_temporary_alias(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            root,dest,owner,_=self.shared_fixture(Path(directory))
            raw=root/'telemetry-0000.jsonl';before=raw.read_bytes()
            with patch.object(runtime,'process_alive',return_value=False):
                first=node_pilot_block.project_shared(root,dest,owner)
                self.assertTrue(os.path.samefile(raw,dest/raw.name))
                second=node_pilot_block.project_shared(root,dest,owner)
            self.assertEqual(first,second);self.assertEqual(raw.read_bytes(),before)
            self.assertFalse(any('.tmp.' in p.name for p in root.rglob('*')))
            runtime.seal_evidence(dest);runtime.verify_evidence_manifest(dest)

    def test_owned_interrupted_snapshot_is_retained_without_replacing_canonical_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root,dest,owner,collector=self.shared_fixture(Path(directory))
            temporary=root/('nvml-live.json.tmp.'+str(collector['pid']))
            data=json.dumps({'pid':collector['pid'],'monotonic_ns':8,'measurement_mode':False}).encode()
            temporary.write_bytes(data);head=(root/'nvml-live.json').read_bytes()
            with patch.object(runtime,'process_alive',return_value=False):
                node_pilot_block.project_shared(root,dest,owner)
            receipt=json.loads((root/'interrupted-owned-metadata.json').read_text())
            self.assertFalse(temporary.exists());self.assertEqual((root/receipt['retained_path']).read_bytes(),data)
            self.assertEqual((root/'nvml-live.json').read_bytes(),head)
            self.assertEqual(receipt['sha256'],hashlib.sha256(data).hexdigest())
            runtime.seal_evidence(root);runtime.verify_evidence_manifest(root)
            # Retaining a known snapshot never permits ignoring another temporary file.
            unknown=root/'unowned.json.tmp.9999';unknown.write_bytes(b'preserve unknown')
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'temporary evidence remains'):
                runtime.seal_evidence(root)
            self.assertEqual(unknown.read_bytes(),b'preserve unknown')

    def test_atomic_snapshot_retention_rejects_live_unsafe_and_failed_measurement_states(self):
        for case in ('live_writer','failed_cleanup','measured_fault','symlink','wrong_writer'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                base=Path(directory);root,dest,owner,collector=self.shared_fixture(base)
                temporary=root/('nvml-live.json.tmp.'+str(collector['pid']))
                data=json.dumps({'pid':9999 if case=='wrong_writer' else collector['pid'],
                                 'monotonic_ns':8,'measurement_mode':False}).encode()
                if case=='symlink':
                    outside=base/'preserved-fixture';outside.write_bytes(data);temporary.symlink_to(outside)
                else:temporary.write_bytes(data)
                if case=='failed_cleanup':runtime.atomic_json(root/'cleanup-result.json',{'success':False})
                if case=='measured_fault':runtime.atomic_json(root/'nvml-cadence-failure.json',{'intentional_fixture':True})
                head=(root/'nvml-live.json').read_bytes()
                with patch.object(runtime,'process_alive',side_effect=lambda ident:case=='live_writer' and ident==collector):
                    with self.assertRaises(runtime.RuntimeErrorClosed):node_pilot_block.project_shared(root,dest,owner)
                self.assertEqual(temporary.read_bytes(),data)
                self.assertEqual((root/'nvml-live.json').read_bytes(),head)
                self.assertFalse((root/'postprocessing-forensics').exists())

    def test_failed_cancellation_after_fit_captures_never_marks_block_passed(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'dsv41-cancel-fixture';order=[]
            rows=[]
            for n in (128000,512000):
                ident=f'pilot_fit-2xtp4-i{n}-c1-cold-o512'
                rows.append({'cell':{'id':ident,'suite':'pilot_fit','input_tokens':n,'concurrency':1,'cache':'cold'},
                             'config':{'cell_id':ident},'tasks':[{'body':{}}],'task_bytes':b'[]','attribution':{}})
            cfg={'run_id':'cancel-fixture','run_dir':str(root),'layout':'2xtp4','qualify_cancellation':True,
                 'expected_ranks':{'http://127.0.0.1:31000/metrics':[0]},'cells':[]}
            owner={'boot_id':'fixture','pid':os_pid(),'start_ticks':1}
            class Session:
                run_id='cancel-fixture'
                def ensure_healthy(self):pass
            def node(config,callback,**kwargs):root.mkdir();callback(Session(),root,'http://127.0.0.1:31100/v1/chat/completions')
            def capture(_s,_root,_url,_config,cell,*_args,**_kwargs):order.append(cell['input_tokens'])
            def cancel(*args,**kwargs):order.append(args[5]);raise RuntimeError('intentional missing work-absence proof')
            with patch.object(node_pilot_block,'prepare',return_value=(rows,b'{}',10)), \
                 patch.object(node_pilot_block,'prior_cells',return_value=[('pilot_smoke',8000,1)]), \
                 patch.object(node_pilot_block.node_smoke,'validate',return_value=({}, {},10000)), \
                 patch.object(node_pilot_block.node_smoke,'run',side_effect=node), \
                 patch.object(node_pilot,'capture_cell',side_effect=capture), \
                 patch.object(node_pilot,'clock_domain',return_value='fixture'), \
                 patch.object(node_pilot_block.cancellation_control,'run',side_effect=cancel), \
                 patch.object(pilot_registry,'DEFAULT_ROOT',base/'registry'), \
                 patch.object(pilot_registry,'method_fingerprint',return_value='a'*64), \
                 patch.object(runtime,'process_identity',return_value=owner):
                with self.assertRaisesRegex(RuntimeError,'work-absence'):node_pilot_block.run(cfg)
            self.assertEqual(order,[128000,512000,'prefill'])
            self.assertFalse((root/'pilot-block-result.json').exists())
            self.assertEqual({r['status'] for r in attempts.load_attempts(base/'registry')},{'running'})
            self.assertTrue(all((p/'failure.json').exists() for p in (base/'registry').glob('attempt-*')))

    def test_prerequisite_order_requires_successful_lower_rungs(self):
        def row(suite,n,c=1):return {'cell':{'suite':suite,'input_tokens':n,'concurrency':c}}
        good=[row('pilot_fit',128000),row('pilot_fit',512000),row('pilot_fit',1000000),
              row('pilot_concurrency',1000000,8),row('pilot_concurrency',1000000,32)]
        node_pilot_block.validate_order(good,[('pilot_smoke',8000,1)])
        for bad,prior in [([good[1]],[('pilot_smoke',8000,1)]),([good[-1]],[('pilot_smoke',8000,1),('pilot_fit',1000000,1)]),(good,[])]:
            with self.assertRaisesRegex(ValueError,'prerequisite'):node_pilot_block.validate_order(bad,prior)

    def test_two_cells_share_one_startup_and_cleanup_and_cannot_replay(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'dsv41-fixture';owner={'boot_id':'fixture','pid':os_pid(),'start_ticks':1}
            tasks={};entries=[]
            for n in (128000,512000):
                seed=workload.seed_for('pilot_fit',n,1,0,0);marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
                task={'id':'s0','session':0,'suite':'pilot_fit','seed':seed,'expected_answer':marker,
                      'input_tokens':n,'output_tokens':512,'fixed_output':True,'prompt_sha256':'a'*64,
                      'body':workload.request_body('/model',[{'role':'user','content':marker}],n,512,seed,fixed_output=True)}
                f=base/f'tasks-{n}.json';f.write_text(json.dumps([task]));tasks[seed]=task
                entries.append({'cell_id':f'pilot_fit-4xdep2-i{n}-c1-cold-o512','tasks_file':str(f),'request_timeout_seconds':5})
            config={'matrix':str(Path(node_pilot.__file__).with_name('matrix.json')),'layout':'4xdep2',
                    'run_id':'fixture','run_dir':str(root),'image':'fixture','model_dir':'/fixture',
                    'manifest':str(base/'manifest.json'),'cells':entries,
                    'required_metric_names':list(node_pilot.PILOT_METRICS),
                    'expected_ranks':{f'http://127.0.0.1:{31000+i}/metrics':[0,1] for i in range(4)}}
            (base/'manifest.json').write_text('{}');queries=[0];calls=[]
            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));task=tasks[body['seed']]
                    queries[0]+=task['input_tokens']
                    with (root/'router.log').open('a') as f:f.write("Consistent hash routing: key='header:x-session-id:s0' -> worker='http://127.0.0.1:31000@0' (index=0)\n")
                    data=('data: '+json.dumps({'choices':[{'delta':{'content':task['expected_answer']+' report'},'finish_reason':'length'}]})+'\n\n'+
                          'data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':task['input_tokens'],'completion_tokens':512}})+'\n\ndata: [DONE]\n\n').encode()
                    self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                def log_message(self,*args):pass
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            class Session:
                deadline_ns=time.monotonic_ns()+600_000_000_000;run_id='fixture'
                def ensure_healthy(self):pass
                def abort(self,reason):raise AssertionError(reason)
            def node(cfg,callback,**kwargs):
                calls.append('startup');root.mkdir();physical_fixture.PilotQualificationTests().fixture(root)
                state=json.loads((root/'execution-state.json').read_text());state['driver_identity']=owner;state['watchdog_identity']=dict(owner,pid=owner['pid']+1)
                (root/'execution-state.json').write_text(json.dumps(state));(root/'run-config.json').write_text(json.dumps(cfg))
                (root/'startup-timing.json').write_text(json.dumps({'launch_started_ns':0,'semantic_ready_ns':1,
                    'backend_health_observed_seconds':{u.removesuffix('/metrics'):0 for u in cfg['expected_ranks']}}))
                for name in ('manifest.json','checkpoint-receipt.json','image-qualification.json'):(root/name).write_text('{}')
                (root/'router.log').write_text('')
                callback(Session(),root,f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
                fixture_cleanup(root,cfg);(root/'smoke-result.json').write_text('{"cleanup":"passed"}');calls.append('cleanup')
            def scrape(url,names):
                return ''.join(f'vllm:prefix_cache_queries_total{{engine="{r}"}} {queries[0] if url.endswith("31000/metrics") and r==0 else 0}\nvllm:prefix_cache_hits_total{{engine="{r}"}} 0\n' for r in (0,1))
            try:
                with patch.object(node_pilot,'verified_tasks',side_effect=lambda c:Path(c['tasks_file']).read_bytes()), \
                     patch.object(node_pilot,'copy_prompt_provenance',side_effect=fixture_provenance), \
                     patch.object(node_pilot,'clock_domain',return_value='fixture'), \
                     patch.object(node_pilot.telemetry,'scrape',side_effect=scrape), \
                     patch.object(node_pilot_block.verify_pilot.verify_telemetry,'verify_window',return_value={'status':'pilot_window_coverage_verified'}), \
                     patch.object(node_pilot_block,'prior_cells',return_value=[('pilot_smoke',8000,1)]), \
                     patch.object(node_pilot_block.node_smoke,'validate',return_value=({}, {},7200)), \
                     patch.object(node_pilot_block.node_smoke,'run',side_effect=node), \
                     patch.object(pilot_registry,'DEFAULT_ROOT',base/'registry'), \
                     patch.object(pilot_registry,'method_fingerprint',side_effect=lambda c,h:hashlib.sha256(c['cell_id'].encode()).hexdigest()), \
                     patch.object(runtime,'process_identity',return_value=owner):
                    config_file=base/'block-config.json';config_file.write_text(json.dumps(config))
                    with patch.object(sys,'argv',['campaign.py','node-pilot-block','--config',str(config_file)]):campaign.main()
                    checked=verify_pilot_block.verify(root)
                    self.assertEqual(len(checked['cells']),2);self.assertEqual(calls,['startup','cleanup'])
                    self.assertEqual({r['status'] for r in attempts.load_attempts(base/'registry')},{'passed'})
                    with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'replay'):node_pilot_block.run(config)
                    import pilot_block_recover,pilot_recover
                    pending=sorted((base/'registry').glob('attempt-*'))[-1]
                    record=json.loads((pending/'attempt.json').read_text());record['status']='running'
                    (pending/'attempt.json').write_text(json.dumps(record))
                    with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'grouped pilot'):
                        pilot_recover.recover(pending,attempts_root=base/'registry',allowed_root=base)
                    with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'owner remains alive'):
                        pilot_block_recover.recover(root,attempts_root=base/'registry',node_lock=base/'node.lock',allowed_root=base,alive=lambda identity:True)
                    with patch.object(pilot_block_recover.reconcile,'reconcile'):
                        recovered=pilot_block_recover.recover(root,attempts_root=base/'registry',node_lock=base/'node.lock',allowed_root=base,alive=lambda identity:False,preflight=lambda *_args:None)
                    self.assertEqual(recovered,{'status':'terminal_block_recovered','cells':2})
                    self.assertEqual({r['status'] for r in attempts.load_attempts(base/'registry')},{'passed'})
                receipt=root/'cleanup-result.json';data=json.loads(receipt.read_text());data['success']=False;receipt.write_text(json.dumps(data));runtime.seal_evidence(root)
                with patch.object(node_pilot_block.verify_pilot.verify_telemetry,'verify_window',return_value={'status':'pilot_window_coverage_verified'}):
                    with self.assertRaisesRegex(ValueError,'actual block session'):verify_pilot_block.verify(root)
            finally:server.shutdown();server.server_close();thread.join()


def os_pid():
    import os
    return os.getpid()
