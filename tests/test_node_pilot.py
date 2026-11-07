from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_pilot
import verify_pilot
import workload
import warm_pilot_prompt

MATRIX = Path(__file__).resolve().parents[1]/'reproduce/matrix.json'


def fixture_provenance(config, root, task_bytes):
    dest=root/'prompt-provenance'; dest.mkdir()
    previous={'cell_id':config['cell_id'],'layout':config['layout'],'image':config.get('image')}
    result={'status':'passed','cleanup':'passed','cell_id':config['cell_id'],'tasks':len(json.loads(task_bytes))}
    values={'run-config.json':json.dumps(previous).encode(),'prompt-result.json':json.dumps(result).encode(),
            'manifest.json':b'{}','checkpoint-receipt.json':b'{}','image-qualification.json':b'{}'}
    lines=[hashlib.sha256(task_bytes).hexdigest()+'  pilot-tasks.json\n']
    for name,data in values.items():
        (dest/name).write_bytes(data);lines.append(hashlib.sha256(data).hexdigest()+'  '+name+'\n')
        if name in ('manifest.json','checkpoint-receipt.json','image-qualification.json'):(root/name).write_bytes(data)
    (dest/'SHA256SUMS').write_text(''.join(lines))

def fixture_cleanup(root,config):
    root=Path(root);config['run_id']='fixture'
    (root/'run-config.json').write_text(json.dumps(config))
    state=json.loads((root/'execution-state.json').read_text()) if (root/'execution-state.json').exists() else {'container_ids':['a'*64,'b'*64]}
    state.update(run_id='fixture',stopping=True,watchdog_stop=True,owned_processes=[])
    (root/'execution-state.json').write_text(json.dumps(state))
    (root/'cleanup-result.json').write_text(json.dumps({'success':True,'deadline_seconds':120,'elapsed_seconds':1.,
        'outcomes':[{'container_id':cid,'status':'stopped','log_status':'captured'} for cid in state['container_ids']]}))
    (root/'gpu-idle-after-cleanup.json').write_text('{"idle":true}')

class PilotContractTests(unittest.TestCase):
    def test_pilot_requires_matched_seed_and_natural_stopping(self):
        seed = workload.seed_for('pilot_smoke',8000,1,0,0)
        marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        body = workload.request_body('/model',[{'role':'user','content':marker}],8000,128,seed,fixed_output=False)
        task = {'id':'s0','body':body,'input_tokens':8000,'output_tokens':128,'fixed_output':False,
                'prompt_sha256':'a'*64,'session':0,'suite':'pilot_smoke','seed':seed,'expected_answer':marker}
        cell='pilot_smoke-2xtp4-i8000-c1-cold-o128'
        self.assertEqual(node_pilot.prepare(MATRIX,cell,[task])['concurrency'], 1)
        body['seed'] += 1
        with self.assertRaises(ValueError): node_pilot.prepare(MATRIX,cell,[task])

    def test_warm_is_not_mislabeled(self):
        with self.assertRaises(ValueError):
            node_pilot.prepare(MATRIX,'pilot_warm-2xtp4-i128000-c1-warm_prefix-o512',[])

    def test_pilot_callback_persists_raw_request(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                answer=body['messages'][0]['content']
                self.send_response(200); self.end_headers()
                event={'choices':[{'delta':{'content':answer},'finish_reason':'stop'}]}
                self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
                self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":8000,"completion_tokens":1}}\n\n')
                self.wfile.write(b'data: [DONE]\n\n')
            def log_message(self,*args): pass
        class Session:
            run_id='fixture'; deadline_ns=time.monotonic_ns()+300_000_000_000
            def ensure_healthy(self): pass
            def abort(self, reason): raise AssertionError(reason)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                seed=workload.seed_for('pilot_smoke',8000,1,0,0)
                marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
                task={'id':'s0','body':workload.request_body('/model',[{'role':'user','content':marker}],8000,128,seed,fixed_output=False),
                      'input_tokens':8000,'output_tokens':128,'fixed_output':False,'prompt_sha256':'a'*64,
                      'session':0,'suite':'pilot_smoke','seed':seed,'expected_answer':marker}
                tasks=Path(d)/'tasks.json'; tasks.write_text(json.dumps([task]))
                cfg={'matrix':str(MATRIX),'cell_id':'pilot_smoke-2xtp4-i8000-c1-cold-o128',
                     'tasks_file':str(tasks),'layout':'2xtp4','request_timeout_seconds':5,
                     'run_dir':d,
                     'required_metric_names':list(node_pilot.PILOT_METRICS),
                     'expected_ranks':{'http://127.0.0.1:31000/metrics':[0], 'http://127.0.0.1:31001/metrics':[0]}}
                def fake_node(config, callback):
                    (Path(d)/'run-config.json').write_text(json.dumps(config))
                    (Path(d)/'startup-timing.json').write_text(json.dumps({'launch_started_ns':0,'semantic_ready_ns':1,'backend_health_observed_seconds':{'http://127.0.0.1:31000':0,'http://127.0.0.1:31001':0}}))
                    (Path(d)/'router.log').write_text("Consistent hash routing: key='header:x-session-id:s0' -> worker='http://127.0.0.1:31000' (index=0)\n")
                    callback(Session(),Path(d),f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
                    (Path(d)/'smoke-result.json').write_text('{"cleanup":"passed"}');fixture_cleanup(Path(d),config)
                calls=[]
                def fake_scrape(url, names):
                    calls.append(url); q=8000 if len(calls)>2 and '31000' in url else 0
                    return f'vllm:prefix_cache_queries_total{{engine="0"}} {q}\nvllm:prefix_cache_hits_total{{engine="0"}} 0\n'
                with patch.object(node_pilot,'verified_tasks',side_effect=lambda c:Path(c['tasks_file']).read_bytes()), \
                     patch.object(node_pilot,'copy_prompt_provenance',side_effect=fixture_provenance), \
                     patch.object(node_pilot.node_smoke,'validate',return_value=({}, {}, 600)), \
                     patch.object(node_pilot.pilot_registry,'begin',return_value=Path(d)/'registry'), \
                     patch.object(node_pilot.pilot_registry,'finish'), \
                     patch.object(node_pilot.pilot_registry,'fail'), \
                     patch.object(node_pilot.node_smoke,'run',side_effect=fake_node), patch.object(node_pilot,'clock_domain',return_value='fixture'), \
                     patch.object(node_pilot.telemetry,'scrape',side_effect=fake_scrape):
                    node_pilot.run(cfg)
                self.assertEqual(json.loads((Path(d)/'pilot-attempt/summary.json').read_text())['status'],'requests_passed_cleanup_pending')
                self.assertEqual(json.loads((Path(d)/'pilot-result.json').read_text())['status'],'invalid')
                self.assertTrue((Path(d)/'pilot-attempt/s0.json').exists())
                self.assertEqual(verify_pilot.verify(d)['requests'],1)
                (Path(d)/'pilot-attempt/s0.json').write_text('{}')
                with self.assertRaises(ValueError): verify_pilot.verify(d)
        finally:
            server.shutdown(); server.server_close(); thread.join()

class WarmPilotTests(unittest.TestCase):
    def test_conditioning_rank_and_hits_are_independently_verified(self):
        seed=workload.seed_for('pilot_warm',128000,1,0,0)
        marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        prefix=((128000-256)//64)*64
        initial=[{'role':'system','content':'follow instructions'}, {'role':'user','content':'Remember '+marker}]
        messages=initial+[{'role':'assistant','content':marker},{'role':'user','content':warm_pilot_prompt.INSTRUCTION}]
        task={'id':'s0','suite':'pilot_warm','session':0,'seed':seed,'input_tokens':128000,'output_tokens':512,
              'prompt_version':warm_pilot_prompt.VERSION,'fixed_report_instruction':warm_pilot_prompt.INSTRUCTION,
              'fixed_output':True,'prompt_sha256':'a'*64,'prefix_token_sha256':'b'*64,'prefix_tokens':prefix,
              'target_prefix_tokens':prefix,'cache_block_size':64,'expected_conditioning_answer':marker,'expected_answer':marker,
              'conditioning_body':workload.request_body('/model',initial,prefix,128,seed,fixed_output=False),
              'body':workload.request_body('/model',messages,128000,512,seed,fixed_output=True)}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                condition=body['max_tokens']==128
                self.send_response(200); self.end_headers()
                event={'choices':[{'delta':{'content':marker},'finish_reason':'stop' if condition else 'length'}]}
                usage={'choices':[],'usage':{'prompt_tokens':prefix if condition else 128000,'completion_tokens':1 if condition else 512}}
                self.wfile.write(('data: '+json.dumps(event)+'\n\ndata: '+json.dumps(usage)+'\n\ndata: [DONE]\n\n').encode())
            def log_message(self,*args): pass
        class Session:
            run_id='fixture'; deadline_ns=time.monotonic_ns()+300_000_000_000
            def ensure_healthy(self): pass
            def abort(self,reason): raise AssertionError(reason)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);tasks=root/'tasks.json';tasks.write_text(json.dumps([task]))
                cfg={'matrix':str(MATRIX),'cell_id':'pilot_warm-2xtp4-i128000-c1-warm_prefix-o512','tasks_file':str(tasks),
                     'layout':'2xtp4','request_timeout_seconds':5,'run_dir':d,
                     'required_metric_names':list(node_pilot.PILOT_METRICS),
                     'expected_ranks':{'http://127.0.0.1:31000/metrics':[0], 'http://127.0.0.1:31001/metrics':[0]}}
                def fake_node(config,callback):
                    (root/'run-config.json').write_text(json.dumps(config));(root/'cache-config.json').write_text('{"block_size":64}')
                    (root/'startup-timing.json').write_text(json.dumps({'launch_started_ns':0,'semantic_ready_ns':1,'backend_health_observed_seconds':{'http://127.0.0.1:31000':0,'http://127.0.0.1:31001':0}}))
                    (root/'router.log').write_text("Consistent hash routing: key='header:x-session-id:s0' -> worker='http://127.0.0.1:31000' (index=0)\n"*2)
                    callback(Session(),root,f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
                    (root/'smoke-result.json').write_text('{"cleanup":"passed"}');fixture_cleanup(root,config)
                calls=[]
                def scrape(url,names):
                    calls.append(url); active='31000' in url
                    q=(0 if len(calls)<=2 else prefix if len(calls)<=4 else prefix+128000) if active else 0
                    h=102400 if active and len(calls)>4 else 0
                    return f'vllm:prefix_cache_queries_total{{engine="0"}} {q}\nvllm:prefix_cache_hits_total{{engine="0"}} {h}\n'
                with patch.object(node_pilot,'verified_tasks',side_effect=lambda c:Path(c['tasks_file']).read_bytes()), \
                     patch.object(node_pilot,'copy_prompt_provenance',side_effect=fixture_provenance), \
                     patch.object(node_pilot.node_smoke,'validate',return_value=({}, {}, 600)), \
                     patch.object(node_pilot.pilot_registry,'begin',return_value=Path(d)/'registry'), \
                     patch.object(node_pilot.pilot_registry,'finish'), \
                     patch.object(node_pilot.pilot_registry,'fail'), \
                     patch.object(node_pilot.node_smoke,'run',side_effect=fake_node), patch.object(node_pilot,'clock_domain',return_value='fixture'), \
                     patch.object(verify_pilot.verify_telemetry,'verify_window',return_value={'status':'pilot_window_coverage_verified'}), \
                     patch.object(node_pilot.telemetry,'scrape',side_effect=scrape):
                    node_pilot.run(cfg)
                result=json.loads((root/'pilot-result.json').read_text())
                self.assertEqual(result['attribution']['hit_fraction'],.8)
                with self.assertRaisesRegex(ValueError,'missing telemetry records'):
                    verify_pilot.verify(root)
                with patch.object(verify_pilot.verify_telemetry,'verify_window',return_value={'status':'pilot_window_coverage_verified'}) as window:
                    self.assertEqual(verify_pilot.verify(root)['requests'],1)
                    self.assertEqual(window.call_count,1)
                row_path=root/'pilot-attempt/s0.json'; original_row=row_path.read_bytes(); altered=json.loads(original_row)
                altered['timing']['first_output_seconds']+=1; row_path.write_text(json.dumps(altered));node_pilot.runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'timing derivation mismatch'):verify_pilot.verify(root)
                row_path.write_bytes(original_row);node_pilot.runtime.seal_evidence(root)
                altered=json.loads(original_row);altered['raw']['events'].pop();row_path.write_text(json.dumps(altered));node_pilot.runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'terminal stream|retained wire chunks'):verify_pilot.verify(root)
                row_path.write_bytes(original_row);node_pilot.runtime.seal_evidence(root)
                proof=root/'prompt-provenance/image-qualification.json'; original=proof.read_bytes();proof.write_text('{"changed":true}')
                node_pilot.runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'prompt provenance hash mismatch'): verify_pilot.verify(root)
                proof.write_bytes(original);node_pilot.runtime.seal_evidence(root)
                condition=root/'pilot-conditioning/s0.json'; data=json.loads(condition.read_text())
                data['raw']['events'][0]['data']['choices'][0]['delta']['content']='wrong'
                condition.write_text(json.dumps(data)); node_pilot.runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'conditioning response invalid|retained wire chunks'):verify_pilot.verify(root)
        finally:
            server.shutdown();server.server_close();thread.join()

class PromptProvenanceTests(unittest.TestCase):
    def test_requires_sealed_prompt_and_matching_dependency_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); root=base/'dsv41-prompt';root.mkdir()
            config={'tasks_file':str(root/'pilot-tasks.json'),'cell_id':'cell','layout':'dep8','image':'image'}
            for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
                external=base/(key+'.json');external.write_text('{"fixture":true}')
                config[key]=str(external);(root/name).write_bytes(external.read_bytes())
            (root/'run-config.json').write_text(json.dumps({'cell_id':'cell','layout':'dep8','image':'image'}))
            (root/'prompt-result.json').write_text(json.dumps({'status':'passed','cleanup':'passed','cell_id':'cell','tasks':1}))
            (root/'pilot-tasks.json').write_text('[{"id":"s0"}]')
            node_pilot.runtime.seal_evidence(root)
            self.assertEqual(json.loads(node_pilot.verified_tasks(config,d)),[{'id':'s0'}])
            Path(config['image_qualification']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'dependency evidence changed'):node_pilot.verified_tasks(config,d)
            Path(config['image_qualification']).write_text('{"fixture":true}')
            (root/'pilot-tasks.json').write_text('[]')
            with self.assertRaisesRegex(node_pilot.runtime.RuntimeErrorClosed,'checksum mismatch'):node_pilot.verified_tasks(config,d)
