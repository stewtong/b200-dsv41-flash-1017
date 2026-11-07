"""Actual subprocess/HTTP boundary for paired clients, without model or GPU claims."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import copy,json,os,subprocess,sys,tempfile,threading,time,unittest,multiprocessing
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,fixed_burst,fixed_session,fixed_workload,model_clients,runtime,verify_fixed_capture,verify_fixed_session,verify_model_clients
import watchdog
from test_vllm_prompt import FakeTokenizer


class BurstFixtureServer(ThreadingHTTPServer):
    # The fixture must admit its declared eight simultaneous loopback clients.
    request_queue_size=32


def pressure_server(pipe):
    """Synthetic SSE source in a separate process from the control-plane test."""
    gate=threading.Barrier(128,timeout=20)
    class Server(ThreadingHTTPServer):request_queue_size=256
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            gate.wait()
            self.send_response(200);self.end_headers()
            event={'id':'synthetic-pressure','object':'chat.completion.chunk','created':0,'model':'fixture',
                'choices':[{'index':0,'delta':{'content':'word '},'finish_reason':None}]}
            frame=('data: '+json.dumps(event)+'\n\n').encode()
            tail=[{'choices':[{'delta':{},'finish_reason':'length'}]},
                {'choices':[],'usage':{'prompt_tokens':8000,'completion_tokens':2048}}]
            self.wfile.write(frame*2048+(''.join('data: '+json.dumps(x)+'\n\n' for x in tail)+'data: [DONE]\n\n').encode())
        def log_message(self,*args):pass
    server=Server(('127.0.0.1',0),Handler);pipe.send(server.server_port);pipe.close();server.serve_forever()


class ModelClientsTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os,'sched_getaffinity'),'owned128-stream control-pressure fixture requires Linux')
    def test_owned128_streams_keep_control_progress_and_exact_child_release(self):
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',128,0,i,
            prompt_version='fixed-performance-v1') for i in range(128)]
        ctx=multiprocessing.get_context('fork');read,write=ctx.Pipe(duplex=False)
        server=ctx.Process(target=pressure_server,args=(write,));server.start();write.close()
        try:
            self.assertTrue(read.poll(10));port=read.recv();read.close()
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);runtime.atomic_json(root/'execution-state.json',{'phase':'qualification','run_id':'fixture',
                    'driver_identity':runtime.process_identity(os.getpid()),'owned_processes':[],'container_ids':[],'stopping':False})
                checks=[]
                class Session:
                    def ensure_healthy(self):
                        checks.append(time.monotonic_ns())
                        watchdog.inspect_ownership(self.root,json.loads((self.root/'execution-state.json').read_text()),None,time.monotonic())
                    def abort(self,reason):raise AssertionError(reason)
                session=Session();session.root=root;session.run_id='fixture';session.deadline_ns=time.monotonic_ns()+300_000_000_000
                cpus=sorted(os.sched_getaffinity(0))[:16];sessions={t['id']:'pressure'+t['id'] for t in tasks}
                session.ensure_healthy()
                report=fixed_session.run(session,f'http://127.0.0.1:{port}/v1/chat/completions',tasks,
                    root/'capture',120,'fixture-clock','fx'+'5'*24,sessions,client_workers=1,client_cpus=cpus)
                session.ensure_healthy()
                gap=max((b-a)/1e9 for a,b in zip(checks,checks[1:]))
                self.assertLess(gap,9,'control progress exceeded the unchanged log-check freshness guard')
                self.assertEqual(report['valid_requests'],128)
                state=json.loads((root/'execution-state.json').read_text());self.assertEqual(state['owned_processes'],[])
                self.assertEqual(len(state['completed_processes']),1)
                self.assertFalse(runtime.process_alive(state['completed_processes'][0]['identity']))
                print(json.dumps({'fixture':'128-synthetic-2048-event-streams','control_checks':len(checks),
                    'max_control_gap_seconds':gap,'valid_requests':128,'gpu_requests':0}),flush=True)
        finally:
            if server.is_alive():server.terminate()
            server.join(10);self.assertFalse(server.is_alive())

    def test_two_actual_clients_preserve_global_requests_cpu_and_wire_cli(self):
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',8,0,i) for i in range(8)]
        by_seed={t['seed']:t for t in tasks};received=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));task=by_seed[body['seed']]
                received.append((self.headers['X-Request-Id'],self.headers['X-Session-Id'],body))
                self.send_response(200);self.end_headers()
                content=('wrong' if self.headers['X-Request-Id'].startswith('bad') else task['expected_answer'])+' public fixture report'
                data=[{'choices':[{'delta':{'content':content},'finish_reason':'length'}]},
                    {'choices':[],'usage':{'prompt_tokens':task['input_tokens'],'completion_tokens':2048}}]
                self.wfile.write((''.join('data: '+json.dumps(x)+'\n\n' for x in data)+'data: [DONE]\n\n').encode())
            def log_message(self,*args):pass
        server=BurstFixtureServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                for workers in (1,2):
                    root=Path(d)/('capture'+str(workers));ids={t['id']:'wire'+str(workers)+t['id'] for t in tasks};sessions={t['id']:'affinity'+t['id'] for t in tasks}
                    model_clients.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',tasks,root,ids,sessions,'fixture-clock',workers=workers,timeout=10)
                    report=verify_model_clients.verify(root);self.assertEqual(report['workers'],workers);self.assertEqual(report['valid_tokens'],8*2048)
                    cli=subprocess.run([sys.executable,'-B',verify_model_clients.__file__,str(root)],capture_output=True,text=True,timeout=10)
                    self.assertEqual(cli.returncode,0,cli.stderr);self.assertEqual(json.loads(cli.stdout)['requests'],8)
                    standard=Path(d)/('standard'+str(workers))
                    namespace=fixed_burst.namespace('fixture-clients','cold',workers-1,'a'*64)
                    combined=fixed_burst.run(None,f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                        tasks,standard,10,'fixture-clock',namespace,sessions,client_workers=workers)
                    self.assertEqual(combined['valid_requests'],8)
                    self.assertEqual(verify_fixed_capture.verify(standard)['valid_requests'],8)
                    cli=subprocess.run([sys.executable,'-B',verify_fixed_capture.__file__,str(standard)],capture_output=True,text=True,timeout=10)
                    self.assertEqual(cli.returncode,0,cli.stderr)
                    projection=standard/'s0.json';original=projection.read_bytes();projection.write_bytes(original+b'\n');runtime.seal_evidence(standard)
                    with self.assertRaisesRegex(ValueError,'projection'):verify_fixed_capture.verify(standard)
                    projection.write_bytes(original);runtime.seal_evidence(standard)
                    path=root/'worker0/ready.json';old=path.read_bytes();data=json.loads(old);data['pid']+=1;path.write_text(json.dumps(data));runtime.seal_evidence(root)
                    with self.assertRaisesRegex(ValueError,'gate'):verify_model_clients.verify(root)
                    path.write_bytes(old);runtime.seal_evidence(root)
                    raw_path=root/'worker0/capture/s0.json';original_raw=raw_path.read_bytes()
                    for field,value in (('start_ns',True),('start_ns','invalid'),('terminal_ns',False),('terminal_ns',0)):
                        changed=json.loads(original_raw);changed['raw'][field]=value
                        raw_path.write_text(json.dumps(changed));runtime.seal_evidence(root)
                        with self.assertRaises(ValueError):verify_model_clients.verify(root)
                    raw_path.write_bytes(original_raw);runtime.seal_evidence(root)
                    self.assertEqual(verify_model_clients.verify(root),report)
                    path=root/'worker0/capture/s0.json';data=json.loads(path.read_text());data['raw']['request_body_sha256']='f'*64;path.write_text(json.dumps(data));runtime.seal_evidence(root)
                    with self.assertRaises(ValueError):verify_model_clients.verify(root)
                self.assertEqual(len(received),32);self.assertEqual(len({r[0] for r in received}),32)
                for ident,session,body in received:self.assertEqual(body,by_seed[body['seed']]['body'])
                failed=Path(d)/'failed';ids={t['id']:'bad'+t['id'] for t in tasks}
                with self.assertRaises(RuntimeError):
                    model_clients.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',tasks,failed,ids,sessions,'fixture-clock',workers=2,timeout=10)
                self.assertTrue((failed/'model-client-failure.json').exists())
                self.assertTrue(list(failed.glob('worker*/capture/s[0-9]*.json')))
                self.assertLessEqual(len(received),40)
        finally:server.shutdown();server.server_close();thread.join()

    def test_invalid_geometry_and_route_fail_before_any_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'untouched'
            for url in ('http://example.com:80/v1/chat/completions','http://127.0.0.1:1/other'):
                with self.assertRaises(ValueError):model_clients.capture(url,[],root,{},{} ,'fixture',workers=2)
            self.assertFalse(root.exists())

    def test_warm_measured_subprocess_preserves_fixed_wire_contract(self):
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',2,0,i,
            'warm_prefix',128,prompt_version='fixed-performance-v1') for i in range(2)]
        by_seed={t['seed']:t for t in tasks}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));t=by_seed[body['seed']]
                frames=[{'choices':[{'delta':{'content':'public fixture'},'finish_reason':'length'}]},
                    {'choices':[],'usage':{'prompt_tokens':t['input_tokens'],'completion_tokens':2048}}]
                self.send_response(200);self.end_headers()
                self.wfile.write((''.join('data: '+json.dumps(x)+'\n\n' for x in frames)+'data: [DONE]\n\n').encode())
            def log_message(self,*args):pass
        server=BurstFixtureServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'capture';namespace='fx'+'3'*24;sessions={t['id']:'warm'+t['id'] for t in tasks}
                report=fixed_burst.run(None,f'http://127.0.0.1:{server.server_port}/v1/chat/completions',tasks,
                    root,10,'fixture-clock',namespace,sessions,client_workers=1)
                self.assertEqual(report['valid_requests'],2)
                self.assertEqual(verify_fixed_capture.verify(root)['valid_requests'],2)
                client=json.loads((root/'clients/model-client-contract.json').read_text())
                self.assertEqual(client['tasks'],tasks)
                self.assertEqual(client['workers'],1)
        finally:server.shutdown();server.server_close();thread.join()

    @unittest.skipUnless(hasattr(os,'sched_getaffinity'),'actual owned model-client CPU affinity requires Linux')
    def test_owned_clients_remain_live_until_exact_terminal_retirement(self):
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',2,0,i) for i in range(2)]
        by_seed={t['seed']:t for t in tasks}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));t=by_seed[body['seed']]
                conditioning=body['max_tokens']==128
                frames=[{'choices':[{'delta':{'content':t['expected_answer']+' public fixture'},'finish_reason':'length'}]},
                    {'choices':[],'usage':{'prompt_tokens':t['prefix_tokens'] if conditioning else t['input_tokens'],
                        'completion_tokens':body['max_tokens']}}]
                self.send_response(200);self.end_headers();self.wfile.write((''.join('data: '+json.dumps(x)+'\n\n' for x in frames)+'data: [DONE]\n\n').encode())
            def log_message(self,*args):pass
        server=BurstFixtureServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);runtime.atomic_json(root/'execution-state.json',{'phase':'qualification','run_id':'fixture',
                    'driver_identity':runtime.process_identity(os.getpid()),'owned_processes':[],'container_ids':[],'stopping':False})
                class Session:
                    def ensure_healthy(self):watchdog.inspect_ownership(self.root,json.loads((self.root/'execution-state.json').read_text()),None,time.monotonic())
                    def abort(self,reason):raise AssertionError(reason)
                session=Session();session.root=root;session.run_id='fixture';session.deadline_ns=time.monotonic_ns()+180_000_000_000
                cpus=sorted(os.sched_getaffinity(0));self.assertGreaterEqual(len(cpus),2)
                ids={t['id']:'owned'+t['id'] for t in tasks};sessions={t['id']:'affinity'+t['id'] for t in tasks}
                model_clients.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',tasks,root/'capture',ids,sessions,'fixture-clock',workers=2,cpus=cpus,session=session,timeout=15)
                verify_model_clients.verify(root/'capture');state=json.loads((root/'execution-state.json').read_text())
                self.assertEqual(state['owned_processes'],[]);self.assertEqual(len(state['completed_processes']),2)
                self.assertTrue(all(not runtime.process_alive(r['identity']) for r in state['completed_processes']))
                namespace=fixed_burst.namespace('fixture','owned-model-clients',0,'a'*64)
                boundaries=[]
                result=fixed_session.run(session,f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    tasks,root/'standard',15,'fixture-clock',namespace,sessions,boundaries.append,
                    client_workers=2,client_cpus=cpus)
                self.assertEqual(result['valid_requests'],2);self.assertEqual(boundaries,['before_measured','after_measured'])
                self.assertEqual(verify_fixed_session.verify(root/'standard')['valid_requests'],2)
                cli=subprocess.run([sys.executable,'-B',campaign.__file__,'verify-fixed-session','--run-dir',str(root/'standard')],
                    capture_output=True,text=True,timeout=10)
                self.assertEqual(cli.returncode,0,cli.stderr)
                state=json.loads((root/'execution-state.json').read_text());self.assertEqual(state['owned_processes'],[])
                self.assertEqual(len(state['completed_processes']),4)
                warm=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',2,0,i,
                    'warm_prefix',128,prompt_version='fixed-performance-v1') for i in range(2)]
                by_seed.clear();by_seed.update({t['seed']:t for t in warm});boundaries=[]
                result=fixed_session.run(session,f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    warm,root/'warm',15,'fixture-clock','fx'+'4'*24,sessions,boundaries.append,
                    client_workers=1,client_cpus=cpus)
                self.assertEqual(result['valid_requests'],2);self.assertEqual(result['conditioning_requests'],2)
                self.assertEqual(boundaries,['before_conditioning','after_conditioning','before_measured','after_measured'])
                self.assertEqual(verify_fixed_session.verify(root/'warm')['valid_requests'],2)
                state=json.loads((root/'execution-state.json').read_text());self.assertEqual(state['owned_processes'],[])
                self.assertEqual(len(state['completed_processes']),5)
                contract=root/'standard/fixed-session-contract.json';original=contract.read_bytes();row=json.loads(original)
                row['client_processes']['workers']=1;contract.write_text(json.dumps(row));runtime.seal_evidence(root/'standard')
                with self.assertRaisesRegex(ValueError,'mode or CPU'):verify_fixed_session.verify(root/'standard')
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
