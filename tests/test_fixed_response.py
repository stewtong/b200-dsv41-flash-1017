from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,fixed_burst,fixed_workload,runtime,stream,verify_fixed_capture,verify_fixed_response
from test_vllm_prompt import FakeTokenizer


class FixedResponseTests(unittest.TestCase):
    def test_actual_http_raw_recomputation_denominators_cli_and_request_fingerprints(self):
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',2,0,i) for i in range(2)]
        by_seed={t['seed']:t for t in tasks};received=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));task=by_seed[body['seed']]
                ident=self.headers['X-Request-Id'];received.append((ident,self.headers['X-Session-Id']))
                if ident.startswith('rejected'):
                    text=b'owned fixture admission boundary';self.send_response(429);self.send_header('Content-Length',str(len(text)));self.end_headers();self.wfile.write(text);return
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                bad=ident.startswith('bad') or self.headers['X-Session-Id']=='bad-affinity'
                if bad:time.sleep(.03)
                content=('wrong' if bad else task['expected_answer'])+' fixture report'
                if ident.startswith('unicode'):content+=chr(0x2014)
                if ident.startswith('filler') or self.headers['X-Session-Id']=='filler-affinity':
                    content=task['expected_answer']+' clear facts and examples'*64
                chunks=[{'choices':[{'index':0,'delta':{'role':'assistant','content':content},'finish_reason':'length'}]},
                    {'choices':[],'usage':{'prompt_tokens':task['input_tokens'],'completion_tokens':2048}}]
                for chunk in chunks:self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
            def log_message(self,*a):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions';rows={};ids={'s0':'good-owned-0','s1':'bad-owned-1'};sessions={'s0':'affinity-0','s1':'affinity-1'}
            for task in tasks:rows[task['id']]=stream.capture(url,task['body'],ids[task['id']],2,session_id=sessions[task['id']])
            report=verify_fixed_response.population(tasks,rows,ids,sessions,'fixture-clock')
            unicode_raw=stream.capture(url,tasks[0]['body'],'unicode-owned',2,session_id='affinity-0')
            unicode_score=verify_fixed_response.evaluate(tasks[0],unicode_raw,'unicode-owned','affinity-0')
            self.assertFalse(unicode_score['correct']);self.assertTrue(unicode_score['fatal_content'])
            self.assertIn('ascii_content',unicode_score['reasons'])
            filler=stream.capture(url,tasks[0]['body'],'filler-owned',2,session_id='affinity-0')
            score=verify_fixed_response.evaluate(tasks[0],filler,'filler-owned','affinity-0')
            self.assertFalse(score['correct']);self.assertTrue(score['fatal_content'])
            self.assertEqual(score['reasons'],['periodic_filler'])
            for body in (' word'*128,(' '+' '.join('word'+str(i) for i in range(100)))*5):
                self.assertTrue(verify_fixed_response.obvious_periodic_filler(tasks[0]['expected_answer']+body,tasks[0]['expected_answer']))
            substantive=tasks[0]['expected_answer']+' '+' '.join('item'+str(i) for i in range(256))
            self.assertFalse(verify_fixed_response.obvious_periodic_filler(substantive,tasks[0]['expected_answer']))
            self.assertFalse(verify_fixed_response.obvious_periodic_filler('short report','code'))
            self.assertEqual(report['requests'],2);self.assertEqual(report['valid_requests'],1);self.assertEqual(report['valid_tokens'],2048)
            self.assertEqual(report['duration_seconds'],(rows['s1']['terminal_ns']-rows['s0']['start_ns'])/1e9)
            self.assertTrue(report['rows'][1]['fatal_content']);self.assertEqual(received[:2],[('good-owned-0','affinity-0'),('bad-owned-1','affinity-1')])
            rejected=stream.capture(url,tasks[0]['body'],'rejected-owned',2,session_id='affinity-0')
            result=verify_fixed_response.evaluate(tasks[0],rejected,'rejected-owned','affinity-0')
            self.assertFalse(result['correct']);self.assertFalse(result['fatal_content']);self.assertIn('transport',result['reasons'])
            for key,value in (('request_body_sha256','0'*64),('request_payload_bytes',0),('request_headers',{})):
                forged=copy.deepcopy(rows['s0']);forged[key]=value
                with self.assertRaises(ValueError):verify_fixed_response.evaluate(tasks[0],forged,ids['s0'],sessions['s0'])
            forged=copy.deepcopy(rows['s0']);forged['events'][0]['data']['choices'][0]['delta']['content']='forged'
            with self.assertRaises(ValueError):verify_fixed_response.evaluate(tasks[0],forged,ids['s0'],sessions['s0'])
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);task_path=root/'task.json';raw_path=root/'raw.json'
                task_path.write_text(json.dumps(tasks[0],sort_keys=True));raw_path.write_text(json.dumps(rows['s0'],sort_keys=True))
                argv=[sys.executable,'-B',verify_fixed_response.__file__,str(task_path),str(raw_path),'--request-id',ids['s0'],'--session-id',sessions['s0']]
                good=subprocess.run(argv,capture_output=True,text=True,timeout=5)
                self.assertEqual(good.returncode,0,good.stderr);self.assertTrue(json.loads(good.stdout)['correct'])
                raw_path.write_text(json.dumps(filler,sort_keys=True))
                rejected_filler=subprocess.run(argv[:-3]+['filler-owned','--session-id','affinity-0'],capture_output=True,text=True,timeout=5)
                self.assertEqual(rejected_filler.returncode,1,rejected_filler.stderr)
                self.assertEqual(json.loads(rejected_filler.stdout)['reasons'],['periodic_filler'])
                task_path.write_text(json.dumps(tasks[1],sort_keys=True));raw_path.write_text(json.dumps(rows['s1'],sort_keys=True))
                bad=subprocess.run(argv[:-3]+[ids['s1'],'--session-id',sessions['s1']],capture_output=True,text=True,timeout=5)
                self.assertEqual(bad.returncode,1,bad.stderr);self.assertFalse(json.loads(bad.stdout)['correct'])
            class Session:
                def __init__(self):self.aborts=[]
                def ensure_healthy(self):pass
                def abort(self,reason):self.aborts.append(reason)
            with tempfile.TemporaryDirectory() as d:
                session=Session();ident=fixed_burst.namespace('fixture','core',0,'a'*64)
                report=fixed_burst.run(session,url,tasks,Path(d)/'good',2,'fixture-clock',ident,sessions)
                self.assertEqual(report['valid_requests'],2);self.assertEqual(session.aborts,[])
                self.assertTrue(all(r['request_id'].startswith(ident) for r in report['rows']))
                runtime.seal_evidence(Path(d)/'good')
                audited=verify_fixed_capture.verify(Path(d)/'good');self.assertEqual(audited['valid_requests'],2)
                path=Path(d)/'good/s0.json';original=path.read_bytes()
                for field in ('extra','correct','timing','scope'):
                    changed=json.loads(original)
                    score=changed['quality_score']
                    if field=='extra':score['unrecognized_score_field']=True
                    elif field=='correct':score['correct']=not score['correct']
                    elif field=='timing':score['timing']['first_output_seconds']+=1
                    else:score.pop('scope')
                    path.write_text(json.dumps(changed));runtime.seal_evidence(Path(d)/'good')
                    with self.assertRaisesRegex(ValueError,'request fields'):verify_fixed_capture.verify(Path(d)/'good')
                path.write_bytes(original);runtime.seal_evidence(Path(d)/'good')
                self.assertEqual(verify_fixed_capture.verify(Path(d)/'good'),audited)
                cli=subprocess.run([sys.executable,'-B',campaign.__file__,'verify-fixed-capture','--run-dir',str(Path(d)/'good')],capture_output=True,text=True,timeout=5)
                self.assertEqual(cli.returncode,0,cli.stderr);self.assertEqual(json.loads(cli.stdout),audited)
                session=Session();failed=fixed_burst.run(session,url,tasks,Path(d)/'bad',2,'fixture-clock',
                    fixed_burst.namespace('fixture-next','core',0,'a'*64),{'s0':'affinity-0','s1':'bad-affinity'})
                self.assertEqual(failed['requests'],2);self.assertLess(failed['valid_requests'],2);self.assertEqual(len(session.aborts),1)
                self.assertEqual(json.loads((Path(d)/'bad/fixed-capture-result.json').read_text())['status'],'failed_requests_retained')
                runtime.seal_evidence(Path(d)/'bad');self.assertLess(verify_fixed_capture.verify(Path(d)/'bad')['valid_requests'],2)
                session=Session();filler_report=fixed_burst.run(session,url,tasks,Path(d)/'filler',2,'fixture-clock',
                    fixed_burst.namespace('fixture-filler','core',0,'a'*64),{'s0':'filler-affinity','s1':'affinity-1'})
                self.assertEqual(len(session.aborts),1);self.assertLess(filler_report['valid_requests'],2)
                self.assertIn('periodic_filler',next(r for r in filler_report['rows'] if r['task_id']=='s0')['reasons'])
                self.assertTrue((Path(d)/'filler/s0.json').exists())
                changed=json.loads((Path(d)/'good/fixed-verification.json').read_text());changed['valid_requests']=0
                (Path(d)/'good/fixed-verification.json').write_text(json.dumps(changed));runtime.seal_evidence(Path(d)/'good')
                with self.assertRaises(ValueError):verify_fixed_capture.verify(Path(d)/'good')
        finally:server.shutdown();server.server_close();thread.join()
