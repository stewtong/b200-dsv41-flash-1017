"""Actual HTTP ordering, identities, failure retention and verifier CLI."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import copy,json,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_burst,fixed_session,fixed_workload,runtime,verify_fixed_session
from test_vllm_prompt import FakeTokenizer


class FixedSessionTests(unittest.TestCase):
    def test_conditioning_precedes_measured_burst_and_is_separate_in_actual_cli(self):
        seen=[];bad=[False];bad_prefix=[False];tasks_by_seed={}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));task=tasks_by_seed[body['seed']]
                cond='min_tokens' not in body
                seen.append({'conditioning':cond,'request_id':self.headers['X-Request-Id'],'session_id':self.headers['X-Session-Id']})
                text=('wrong' if bad[0] else task['expected_conditioning_answer']) if cond else task['expected_answer']+' technical report'
                if not cond and bad_prefix[0]:text='Verification code: '+text
                frames=[{'choices':[{'index':0,'delta':{'role':'assistant','content':text},'finish_reason':'stop' if cond else 'length'}]},
                        {'choices':[],'usage':{'prompt_tokens':task['prefix_tokens'] if cond else task['input_tokens'],'completion_tokens':8 if cond else task['output_tokens']}}]
                data=(''.join('data: '+json.dumps(f)+'\n\n' for f in frames)+'data: [DONE]\n\n').encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        class Server(ThreadingHTTPServer):request_queue_size=64
        server=Server(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        class Session:
            aborted=False
            def ensure_healthy(self):pass
            def abort(self,reason):self.aborted=True
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
                tasks=[fixed_workload.build(FakeTokenizer(),'/model',10000,2048,'core',2,0,i,'warm_prefix',128) for i in range(2)]
                tasks_by_seed.update({t['seed']:t for t in tasks});ids={t['id']:'unit'+str(t['session']) for t in tasks}
                ns=fixed_burst.namespace('fixture','warm-cell',0,'a'*64);session=Session()
                with self.assertRaisesRegex(ValueError,'task and affinity coverage'):
                    fixed_session.run(session,url,tasks[:1],root/'partial',5,'fixture',ns,{'s0':'unit0'})
                self.assertFalse((root/'partial').exists());self.assertEqual(seen,[])
                result=fixed_session.run(session,url,tasks,root/'warm',5,'fixture',ns,ids)
                self.assertEqual((result['requests'],result['conditioning_requests'],result['valid_requests']),(2,2,2))
                self.assertEqual(result['conditioning_tokens'],16);self.assertFalse(session.aborted)
                self.assertEqual([r['conditioning'] for r in seen],[True,True,False,False])
                self.assertEqual(len({r['request_id'] for r in seen}),4)
                self.assertEqual({r['session_id'] for r in seen[:2]},{r['session_id'] for r in seen[2:]})
                cli=[sys.executable,'-B',str(Path(fixed_session.__file__).with_name('campaign.py')),'verify-fixed-session','--run-dir',str(root/'warm')]
                output=subprocess.run(cli,capture_output=True,text=True,timeout=10)
                self.assertEqual(output.returncode,0,output.stderr);self.assertEqual(json.loads(output.stdout)['requests'],2)
                # Resealing cannot make a forged conditioning affinity legitimate.
                path=root/'warm/conditioning/s0.json';row=json.loads(path.read_text());row['raw']['session_id']='forged';path.write_text(json.dumps(row))
                runtime.seal_evidence(path.parent);runtime.seal_evidence(root/'warm')
                failed=subprocess.run(cli,capture_output=True,text=True,timeout=10);self.assertNotEqual(failed.returncode,0)
                # A failed acknowledgement never starts the measured burst.
                before=len(seen);bad[0]=True;one=[fixed_workload.build(FakeTokenizer(),'/model',10000,2048,'core',1,0,0,'warm_prefix',128)]
                tasks_by_seed.update({t['seed']:t for t in one});session=Session()
                with self.assertRaises(runtime.RuntimeErrorClosed):
                    fixed_session.run(session,url,one,root/'bad',5,'fixture',fixed_burst.namespace('fixture','bad-cell',0,'a'*64),{'s0':'unit0'})
                self.assertTrue(session.aborted);self.assertTrue(all(x['conditioning'] for x in seen[before:]))
                self.assertFalse((root/'bad/measured').exists());self.assertTrue((root/'bad/conditioning/s0.json').exists())
                self.assertEqual(json.loads((root/'bad/fixed-session-failure.json').read_text())['status'],'failed')
                bad[0]=False;cold=[fixed_workload.build(FakeTokenizer(),'/model',10000,2048,'core',1,0,0)]
                tasks_by_seed.update({t['seed']:t for t in cold});before=len(seen)
                r=fixed_session.run(Session(),url,cold,root/'cold',5,'fixture',fixed_burst.namespace('fixture','cold-cell',0,'a'*64),{'s0':'unit0'})
                self.assertEqual(r['conditioning_requests'],0);self.assertFalse(any(x['conditioning'] for x in seen[before:]))
                bad_prefix[0]=True;boundaries=[];session=Session()
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'fixed measured response oracle rejected'):
                    fixed_session.run(session,url,cold,root/'bad-prefix',5,'fixture',fixed_burst.namespace('fixture','bad-prefix',0,'a'*64),{'s0':'unit0'},boundary_observer=boundaries.append)
                self.assertEqual(boundaries,['before_measured']);self.assertTrue(session.aborted)
                score=json.loads((root/'bad-prefix/measured/fixed-verification.json').read_text())['rows'][0]
                self.assertEqual(score['reasons'],['leading_identifier'])
                self.assertTrue((root/'bad-prefix/measured/s0.json').exists())
        finally:server.shutdown();server.server_close();thread.join()
