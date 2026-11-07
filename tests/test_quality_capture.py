from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys,tempfile,threading,unittest,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import quality_capture,quality_tasks,verify_wire,verify_quality_capture

class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return '|'.join(m['role']+':'+m['content'] for m in messages)
    def encode(self,text,add_special_tokens=False):return list(text)

class Session:
    def __init__(self):self.aborts=[]
    def ensure_healthy(self):pass
    def abort(self,reason):self.aborts.append(reason)

class QualityCaptureTests(unittest.TestCase):
    def capture(self,malformed=False):
        tasks=[quality_tasks.build(Tokenizer(),1000,128,2,0,i) for i in range(2)]
        byseed={t['seed']:t for t in tasks};calls=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                task=byseed[body['seed']]
                calls.append((self.headers['X-Request-Id'],self.headers['X-Session-Id']))
                answer=task['expected_answer'] if task['id']=='s0' else 'wrong'
                if isinstance(answer,list):answer=json.dumps(answer)
                events=[{'choices':[{'delta':{'content':answer},'finish_reason':'stop'}]},
                        {'choices':[],'usage':{'prompt_tokens':task['input_tokens'],'completion_tokens':2}}]
                raw=(b'data: {bad}\n\n' if malformed else
                     ''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode()+b'data: [DONE]\n\n')
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        session=Session()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'capture'
                summary=quality_capture.run(session,f'http://127.0.0.1:{server.server_port}/v1/chat/completions',tasks,root,2,'fixture',0,1000)
                rows=[json.loads((root/'measured'/(t['id']+'.json')).read_text()) for t in tasks]
                status=json.loads((root/'measured/summary.json').read_text())['status']
                self.assertEqual(len({r for r,s in calls}),2)
                checked=verify_quality_capture.verify(root,tasks,0,1000)
                self.assertEqual(checked['requests'],2)
                task_file=Path(d)/'tasks.json';task_file.write_text(json.dumps(tasks))
                cli=subprocess.run([sys.executable,'-B',str(Path(verify_quality_capture.__file__)),str(root),str(task_file),'--repetition','0','--target','1000'],capture_output=True,text=True,check=True)
                self.assertEqual(json.loads(cli.stdout),checked)
                original=(root/'measured/s0.json').read_bytes();changed=json.loads(original)
                changed['quality_score']['correct']=not changed['quality_score']['correct']
                (root/'measured/s0.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'quality score'):
                    verify_quality_capture.verify(root,tasks,0,1000)
                (root/'measured/s0.json').write_bytes(original)
                return summary,rows,status,session.aborts
        finally:server.shutdown();server.server_close();thread.join()

    def test_wrong_answer_is_retained_without_cancelling_quality_population(self):
        summary,rows,status,aborts=self.capture()
        self.assertEqual(summary['requests'],2);self.assertEqual(summary['valid_requests'],1)
        self.assertEqual([r['valid'] for r in rows],[True,False]);self.assertFalse(aborts)
        self.assertEqual(status,'captured_with_invalid_requests_cleanup_pending')
        for row in rows:verify_wire.verify_complete(row['raw'])
        self.assertFalse(rows[1]['quality_score']['correct'])

    def test_malformed_transport_aborts_but_keeps_raw_denominator(self):
        summary,rows,status,aborts=self.capture(True)
        self.assertEqual(summary['requests'],2);self.assertEqual(summary['valid_requests'],0)
        self.assertTrue(aborts);self.assertEqual(status,'failed')
        for row in rows:
            verify_wire.retained_bytes(row['raw'])
            self.assertTrue(row['fatal_transport_or_content'])

    def test_warm_conditioning_precedes_measurement_and_is_not_in_quality_denominator(self):
        tasks=[quality_tasks.build(Tokenizer(),8000,128,2,0,i,'warm_prefix',128) for i in range(2)]
        byseed={t['seed']:t for t in tasks};calls=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                task=byseed[body['seed']];condition=len(body['messages'])==2
                calls.append((task['id'],condition,self.headers['X-Request-Id'],self.headers['X-Session-Id']))
                answer=task['expected_conditioning_answer'] if condition else task['expected_answer']
                if isinstance(answer,list):answer=json.dumps(answer)
                prompt=task['prefix_tokens'] if condition else task['input_tokens']
                events=[{'choices':[{'delta':{'content':answer},'finish_reason':'stop'}]},
                        {'choices':[],'usage':{'prompt_tokens':prompt,'completion_tokens':2}}]
                raw=''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode()+b'data: [DONE]\n\n'
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();session=Session()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'warm'
                summary=quality_capture.run(session,f'http://127.0.0.1:{server.server_port}/v1/chat/completions',tasks,root,2,'fixture',0,8000)
                self.assertEqual(summary['requests'],2);self.assertEqual(summary['valid_requests'],2)
                self.assertEqual([c[1] for c in calls],[True,True,False,False]);self.assertFalse(session.aborts)
                self.assertEqual(len({c[2] for c in calls}),4)
                for task in tasks:
                    selected=[c for c in calls if c[0]==task['id']]
                    self.assertEqual(len({c[3] for c in selected}),1)
                checked=verify_quality_capture.verify(root,tasks,0,8000)
                self.assertEqual(checked['correct'],2);self.assertEqual(checked['requests'],2)
                task_file=Path(d)/'tasks.json';task_file.write_text(json.dumps(tasks))
                cli=subprocess.run([sys.executable,'-B',str(Path(verify_quality_capture.__file__)),str(root),str(task_file),'--repetition','0','--target','8000'],capture_output=True,text=True,check=True)
                self.assertEqual(json.loads(cli.stdout),checked)
                path=root/'conditioning/s0.json';original=path.read_bytes();row=json.loads(original);row['valid']=False;path.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError,'conditioning identity'):verify_quality_capture.verify(root,tasks,0,8000)
                path.write_bytes(original)
                for directory,mutation in (
                    ('conditioning',lambda r:r['raw'].__setitem__('request_body_sha256','0'*64)),
                    ('measured',lambda r:r['raw']['request_headers'].__setitem__('X-Session-Id','foreign-affinity')),
                    ('measured',lambda r:r['raw'].__setitem__('request_payload_bytes',1)),
                ):
                    path=root/directory/'s0.json';original=path.read_bytes();row=json.loads(original);mutation(row)
                    self.assertNotEqual(row,json.loads(original));path.write_text(json.dumps(row))
                    with self.assertRaisesRegex(ValueError,'transmitted request binding'):verify_quality_capture.verify(root,tasks,0,8000)
                    path.write_bytes(original)
                # A replayed conditioning response cannot become measured work by
                # changing its outer score or logical task label.
                path=root/'measured/s0.json';original=path.read_bytes()
                path.write_bytes((root/'conditioning/s0.json').read_bytes())
                with self.assertRaisesRegex(ValueError,'quality identity'):verify_quality_capture.verify(root,tasks,0,8000)
                path.write_bytes(original)
                contract_path=root/'quality-capture-contract.json';original=contract_path.read_bytes();contract=json.loads(original)
                contract['session_ids']['s0']='foreign-affinity';contract_path.write_text(json.dumps(contract))
                with self.assertRaises(ValueError):verify_quality_capture.verify(root,tasks,0,8000)
                contract_path.write_bytes(original)
                self.assertEqual(verify_quality_capture.verify(root,tasks,0,8000),checked)
        finally:server.shutdown();server.server_close();thread.join()
