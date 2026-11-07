"""Actual local one/two-process HTTP capture; not node affinity or GPU proof."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import base64,copy,hashlib,json,os,subprocess,sys,tempfile,threading,time,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import client_fixture as capture,runtime,verify_client_fixture as audit
import client_fixture_backend,watchdog


class ClientFixtureTests(unittest.TestCase):
    def test_terminal_worker_release_is_atomic_with_watchdog_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);child=root/'probes/p0/worker0';child.mkdir(parents=True)
            identity={'boot_id':'fixture','pid':42,'start_ticks':1};alive=[True]
            runtime.atomic_json(child/'owned-identity.json',identity)
            runtime.atomic_json(child/'worker-result.json',{'status':'passed','requests':[]})
            original={'phase':'client-fixture','stopping':False,'owned_processes':[identity]}
            runtime.atomic_json(root/'execution-state.json',original)
            class Session:
                def ensure_healthy(self):pass
            session=Session();session.root=root
            test=self
            class Process:
                pid=42
                def poll(self):return None if alive[0] else 0
                def wait(self,timeout):
                    test.assertEqual(json.loads((child/'exit.json').read_text()),{'config_sha256':'a'*64,'worker_pid':42})
                    # A watchdog cannot inspect an exited but not yet retired child.
                    with test.assertRaisesRegex(runtime.RuntimeErrorClosed,'campaign lock held'):
                        with runtime.NodeLock(root/'cleanup.lock'):pass
                    alive[0]=False;return 0
            with patch.object(runtime,'process_alive',side_effect=lambda _:alive[0]):
                capture.finish_owned_workers(session,[(Process(),child,'a'*64)])
            state=json.loads((root/'execution-state.json').read_text());receipt=json.loads((child/'release-receipt.json').read_text())
            self.assertEqual(state['owned_processes'],[]);self.assertEqual(state['completed_processes'],[receipt])
            self.assertEqual(receipt['identity'],identity);self.assertEqual(receipt['worker_path'],'probes/p0/worker0')
            self.assertEqual(receipt['result_sha256'],hashlib.sha256((child/'worker-result.json').read_bytes()).hexdigest())
            runtime.atomic_json(root/'execution-state.json',original)
            with patch.object(runtime,'process_alive',return_value=False),self.assertRaisesRegex(RuntimeError,'live ownership'):
                capture.finish_owned_workers(session,[(Process(),child,'a'*64)])
            self.assertEqual(json.loads((root/'execution-state.json').read_text()),original)

    @unittest.skipUnless(hasattr(os,'sched_getaffinity'),'actual owned worker CPU affinity requires Linux')
    def test_owned_http_workers_remain_live_until_registered_terminal_release(self):
        backend=client_fixture_backend.Backend(0,16,8)
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);identity=runtime.process_identity(os.getpid())
                runtime.atomic_json(root/'execution-state.json',{'phase':'client-fixture','run_id':'fixture',
                    'driver_identity':identity,'owned_processes':[],'container_ids':[],'stopping':False})
                class Session:
                    def ensure_healthy(self):
                        state=json.loads((self.root/'execution-state.json').read_text())
                        watchdog.inspect_ownership(self.root,state,None,time.monotonic())
                session=Session();session.root=root;cpus=sorted(os.sched_getaffinity(0))
                self.assertGreaterEqual(len(cpus),2)
                for index in range(2):
                    report=capture.capture(f'http://127.0.0.1:{backend.port}/v1/chat/completions',root/'probes'/str(index),
                        workers=2,concurrency=4,requests=8,events=16,piece_bytes=8,input_bytes=128,timeout=15,
                        cpus=cpus,session=session)
                    self.assertTrue(report['affinity_budget_qualified']);session.ensure_healthy()
                state=json.loads((root/'execution-state.json').read_text())
                self.assertEqual(state['owned_processes'],[]);self.assertEqual(len(state['completed_processes']),4)
                self.assertTrue(all(not runtime.process_alive(r['identity']) for r in state['completed_processes']))
        finally:backend.close()

    def test_production_fixture_backend_retains_actual_request_body_receipts(self):
        backend=client_fixture_backend.Backend(0,16,8)
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'capture'
                result=capture.capture(f'http://127.0.0.1:{backend.port}/v1/chat/completions',root,
                    workers=2,concurrency=4,requests=8,events=16,piece_bytes=8,input_bytes=128,timeout=15)
                self.assertEqual(result['requests'],8)
                with backend.lock:receipts=list(backend.records)
                self.assertEqual(len(receipts),8);self.assertEqual(len({r['request_id'] for r in receipts}),8)
                raws={json.loads(p.read_text())['request_id']:json.loads(p.read_text()) for p in root.glob('worker*/p[0-9]*.json')}
                for receipt in receipts:
                    raw=raws[receipt['request_id']]
                    backend_bytes=base64.b64decode(receipt['body_base64'],validate=True)
                    self.assertEqual(hashlib.sha256(backend_bytes).hexdigest(),receipt['body_sha256'])
                    self.assertEqual(len(backend_bytes),receipt['body_bytes'])
                    self.assertEqual(receipt['body_sha256'],raw['request_body_sha256'])
                    self.assertEqual(receipt['body_bytes'],raw['request_payload_bytes'])
                    self.assertLessEqual(raw['start_ns'],receipt['observed_ns']);self.assertLessEqual(receipt['observed_ns'],raw['terminal_ns'])
        finally:backend.close()

    def test_one_and_two_actual_client_processes_replay_raw_bytes_and_reject_forgery(self):
        received=[];lock=threading.Lock();bad=[False];events=16;piece_bytes=8
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:received.append((self.headers['X-Request-Id'],self.headers['X-Session-Id']))
                frames=[{'choices':[{'index':0,'delta':{'content':'x'*piece_bytes}}]} for _ in range(events)]
                frames += [{'choices':[{'index':0,'delta':{},'finish_reason':'length'}]},
                    {'choices':[],'usage':{'prompt_tokens':1,'completion_tokens':events,'total_tokens':events+1}}]
                data=(''.join('data: '+json.dumps(f)+'\n\n' for f in frames)+('' if bad[0] else 'data: [DONE]\n\n')).encode()
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        class Server(ThreadingHTTPServer):request_queue_size=64
        server=Server(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
                for workers in (1,2):
                    received.clear();dest=root/('workers'+str(workers))
                    report=capture.capture(url,dest,workers=workers,concurrency=4,requests=8,events=events,piece_bytes=piece_bytes,input_bytes=128,timeout=15)
                    self.assertEqual(report['requests'],8);self.assertEqual(report['workers'],workers)
                    self.assertEqual(report['visible_fixture_events'],128);self.assertGreater(report['response_body_bytes_per_second'],0)
                    self.assertFalse(report['affinity_budget_qualified'])
                    self.assertCountEqual(received,[('p'+str(i),'fixture-session-'+str(i)) for i in range(8)])
                    cli=subprocess.run([sys.executable,'-B',audit.__file__,str(dest)],capture_output=True,text=True,timeout=10)
                    self.assertEqual(cli.returncode,0,cli.stderr);self.assertEqual(json.loads(cli.stdout),report)
                dest=root/'workers2';path=dest/'worker0/p0.json';original=path.read_bytes();row=json.loads(original)
                row['request_headers']['X-Session-Id']='forged';runtime.atomic_json(path,row);runtime.seal_evidence(dest)
                with self.assertRaisesRegex(ValueError,'transmitted request'):audit.verify(dest)
                path.write_bytes(original);runtime.seal_evidence(dest)
                path=dest/'worker0/worker-result.json';value=json.loads(path.read_text());value['requests'].pop()
                runtime.atomic_json(path,value);runtime.seal_evidence(dest)
                with self.assertRaisesRegex(ValueError,'summary differs'):audit.verify(dest)
                bad[0]=True
                with self.assertRaises(RuntimeError):
                    capture.capture(url,root/'truncated',workers=2,concurrency=4,requests=8,events=events,piece_bytes=piece_bytes,input_bytes=128,timeout=15)
                self.assertTrue((root/'truncated/fixture-failure.json').exists())
                self.assertFalse((root/'truncated/fixture-result.json').exists())
        finally:server.shutdown();server.server_close();thread.join()

    def test_malformed_or_excessive_work_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'capture'
            for kwargs in ({'workers':3},{'concurrency':0},{'requests':0},
                           {'input_bytes':8_000_000,'requests':1024},{'events':2048,'piece_bytes':1024,'requests':1024}):
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):capture.capture('unused',root,**kwargs)
                self.assertFalse(root.exists())


if __name__=='__main__':unittest.main()
