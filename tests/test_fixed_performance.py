"""Performance acceptance differs from legacy correctness without hiding failures."""
import json
from pathlib import Path
import sys,tempfile,threading,time,unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_workload,verify_fixed_task,verify_fixed_capture,fixed_burst,stream,verify_fixed_response,fixed_session,verify_fixed_session
from test_vllm_prompt import FakeTokenizer

class PerformanceTests(unittest.TestCase):
 def test_actual_capture_and_independent_recompute_keep_unicode_and_repetition(self):
  for cache in ('cold','warm_prefix'):
   t=fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,2,0,cache,128 if cache=='warm_prefix' else None,prompt_version='fixed-performance-v1')
   verify_fixed_task.verify(t,FakeTokenizer())
  task=fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,2,0,prompt_version='fixed-performance-v1')
  self.assertNotIn('ASCII',task['body']['messages'][0]['content'])
  warm=fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,2,0,'warm_prefix',128,prompt_version='fixed-performance-v1')
  mode={'value':'valid'}
  class Handler(BaseHTTPRequestHandler):
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
    content='Report '+chr(0x2014)+' '+chr(0xFF1A)+' '+'clear facts and examples '*80
    if mode['value']=='empty':content=''
    delta={'content':content}
    if mode['value']=='channel':delta['reasoning_content']='unexpected'
    tokens=body['max_tokens']-1 if mode['value']=='short' else body['max_tokens']
    if mode['value']=='valid':
     for prefix in ({'choices':[{'delta':{'role':'assistant'}}]},{'choices':[{'delta':{'content':' '}}]}):
      self.wfile.write(('data: '+json.dumps(prefix)+'\n\n').encode());self.wfile.flush();time.sleep(.005)
    for row in ({'choices':[{'index':0,'delta':delta,'finish_reason':'length'}]}, {'choices':[],'usage':{'prompt_tokens':warm['prefix_tokens'] if body['max_tokens']==128 else task['input_tokens'],'completion_tokens':tokens}}):
     self.wfile.write(('data: '+json.dumps(row)+'\n\n').encode())
    self.wfile.write(b'data: [DONE]\n\n')
   def log_message(self,*args):pass
  class Session:
   def ensure_healthy(self):pass
   def request_stop(self,*args):pass
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
  try:
   with tempfile.TemporaryDirectory() as d:
    result=fixed_burst.run(Session(),f'http://127.0.0.1:{server.server_port}/v1/chat/completions',[task],Path(d)/'capture',timeout=3,clock_id='fixture-clock',request_namespace='fx'+'1'*24,session_ids={'s0':'affinity-0'})
    import runtime
    runtime.seal_evidence(Path(d)/'capture')
    proof=verify_fixed_capture.verify(Path(d)/'capture')
    self.assertEqual(proof['valid_requests'],1)
    report=json.loads((Path(d)/'capture/fixed-verification.json').read_text())
    self.assertTrue(report['rows'][0]['performance_valid'])
    stored=json.loads((Path(d)/'capture/s0.json').read_text())
    self.assertTrue(stored['valid']);self.assertFalse(stored['fatal_transport_or_content'])
    self.assertFalse(stored['quality_score']['fatal_content'])
    self.assertIn('periodic_filler',report['rows'][0]['content_anomalies'])
    self.assertIn('no correct-task goodput',proof['scope'])
    score_path=Path(d)/'capture/s0.json';original_score=score_path.read_bytes()
    for field in ('performance_valid','fatal_content','content_anomalies','extra'):
     changed=json.loads(original_score);quality=changed['quality_score']
     if field=='extra':quality['unrecognized_score_field']=True
     elif field=='content_anomalies':quality['content_anomalies']=[]
     else:quality[field]=not quality[field]
     score_path.write_text(json.dumps(changed));runtime.seal_evidence(Path(d)/'capture')
     with self.assertRaisesRegex(ValueError,'request fields'):verify_fixed_capture.verify(Path(d)/'capture')
    score_path.write_bytes(original_score);runtime.seal_evidence(Path(d)/'capture')
    self.assertEqual(verify_fixed_capture.verify(Path(d)/'capture'),proof)
    import comparison_results
    inputs=json.loads((Path(d)/'capture/fixed-inputs.json').read_text())
    described=comparison_results.describe([task],{'s0':stored},inputs)
    timing=described['request_outcomes'][0]
    self.assertLess(timing['first_output_ns'],timing['first_visible_text_ns'])
    self.assertEqual(stored['quality_score']['timing']['first_visible_seconds'],timing['first_visible_text_seconds'])
    self.assertGreater(stored['quality_score']['timing']['first_visible_seconds'],stored['quality_score']['timing']['first_output_seconds'])
    self.assertEqual(described['valid_output_tokens'],2048)
    self.assertEqual(described['burst_makespan_seconds'],(stored['raw']['terminal_ns']-stored['raw']['start_ns'])/1e9)
    mode['value']='valid'
    fixed_session.run(Session(),f'http://127.0.0.1:{server.server_port}/v1/chat/completions',[warm],Path(d)/'warm',3,'fixture-clock','fx'+'2'*24,{'s0':'warm-affinity'})
    warm_proof=verify_fixed_session.verify(Path(d)/'warm')
    self.assertEqual(warm_proof['conditioning_requests'],1)
    self.assertEqual(warm_proof['conditioning_tokens'],128)
    for case,reason in (('empty','nonempty_content'),('short','fixed_output_usage'),('channel','unexpected_output_channel')):
     mode['value']=case
     raw=stream.capture(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',task['body'],'negative',3,session_id='negative-affinity')
     score=verify_fixed_response.evaluate(task,raw,'negative','negative-affinity')
     self.assertFalse(score['performance_valid']);self.assertIn(reason,score['reasons'])
     failed=comparison_results.describe([task],{'s0':{'raw':raw}},dict(request_ids={'s0':'negative'},session_ids={'s0':'negative-affinity'},clock_id='fixture-clock'))
     self.assertEqual(failed['failed_or_invalid_requests'],1);self.assertEqual(failed['valid_output_tokens'],0)
     self.assertGreater(failed['burst_makespan_seconds'],0)
  finally:server.shutdown();server.server_close();thread.join()
