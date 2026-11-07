from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_pilot
import runtime
import verify_pilot
import workload
from test_node_pilot import fixture_provenance,fixture_cleanup,MATRIX
import test_pilot_qualification as physical_fixture


class ConcurrentPilotTests(unittest.TestCase):
    def test_full_capture_routes_physical_join_and_independent_concurrent_verification(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                marker=hashlib.sha256(str(body['seed']).encode()).hexdigest()[:16]
                events=[{'choices':[{'delta':{'content':marker+' fixture report'},'finish_reason':None}]},
                        {'choices':[{'delta':{},'finish_reason':'length'}]},
                        {'choices':[],'usage':{'prompt_tokens':128000,'completion_tokens':512}}]
                for event in events:self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode());self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
            def log_message(self,*args):pass
        class Session:
            run_id='fixture';deadline_ns=time.monotonic_ns()+300_000_000_000
            def ensure_healthy(self):pass
            def abort(self,reason):
                failed=[json.loads(p.read_text()) for p in (root/'pilot-attempt').glob('s*.json') if json.loads(p.read_text()).get('valid') is False]
                raise AssertionError((reason,failed))
        class Server(ThreadingHTTPServer):
            request_queue_size=64
        server=Server(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);tasks=[]
                for i in range(8):
                    seed=workload.seed_for('pilot_concurrency',128000,8,0,i);marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
                    tasks.append({'id':'s'+str(i),'session':i,'suite':'pilot_concurrency','seed':seed,'expected_answer':marker,
                                  'input_tokens':128000,'output_tokens':512,'fixed_output':True,'prompt_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
                                  'body':workload.request_body('/model',[{'role':'user','content':marker}],128000,512,seed,fixed_output=True)})
                task_file=root/'tasks.json';task_file.write_text(json.dumps(tasks))
                cfg={'matrix':str(MATRIX),'cell_id':'pilot_concurrency-4xdep2-i128000-c8-cold-o512','layout':'4xdep2',
                     'tasks_file':str(task_file),'run_dir':d,'request_timeout_seconds':5,
                     'required_metric_names':list(node_pilot.PILOT_METRICS),
                     'expected_ranks':{f'http://127.0.0.1:{31000+i}/metrics':[0,1] for i in range(4)}}
                def node(config,callback):
                    physical_fixture.PilotQualificationTests().fixture(root)
                    (root/'run-config.json').write_text(json.dumps(config))
                    (root/'startup-timing.json').write_text(json.dumps({'launch_started_ns':0,'semantic_ready_ns':1,
                        'backend_health_observed_seconds':{url.removesuffix('/metrics'):0 for url in config['expected_ranks']}}))
                    (root/'router.log').write_text(''.join(f"Consistent hash routing: key='header:x-session-id:s{i}' -> worker='http://127.0.0.1:{31000+i//2}@{i%2}' (index={i})\n" for i in range(8)))
                    callback(Session(),root,f'http://127.0.0.1:{server.server_port}/v1/chat/completions')
                    (root/'smoke-result.json').write_text('{"cleanup":"passed"}');fixture_cleanup(root,config)
                calls=[]
                def scrape(url,names):
                    calls.append(url);q=128000 if len(calls)>4 else 0
                    return ''.join(f'vllm:prefix_cache_queries_total{{engine="{i}"}} {q}\nvllm:prefix_cache_hits_total{{engine="{i}"}} 0\n' for i in (0,1))
                with patch.object(node_pilot,'verified_tasks',return_value=task_file.read_bytes()), \
                     patch.object(node_pilot,'copy_prompt_provenance',side_effect=fixture_provenance), \
                     patch.object(node_pilot.node_smoke,'validate',return_value=({}, {},600)), \
                     patch.object(node_pilot.node_smoke,'run',side_effect=node), \
                     patch.object(node_pilot,'clock_domain',return_value='fixture'), \
                     patch.object(node_pilot.telemetry,'scrape',side_effect=scrape), \
                     patch.object(verify_pilot.verify_telemetry,'verify_window',return_value={'status':'pilot_window_coverage_verified'}), \
                     patch.object(node_pilot.pilot_registry,'begin',return_value=root/'registry'), \
                     patch.object(node_pilot.pilot_registry,'finish') as finish, \
                     patch.object(node_pilot.pilot_registry,'fail'):
                    node_pilot.run(cfg)
                with patch.object(verify_pilot.verify_telemetry,'verify_window',return_value={'status':'pilot_window_coverage_verified'}):
                    self.assertEqual(verify_pilot.verify(root),{'cell_id':cfg['cell_id'],'status':'passed','requests':8,'valid_tokens':4096})
                self.assertEqual(finish.call_args.args[2]['status'],'passed')
                path=root/'pilot-routing.log';path.write_text(path.read_text().replace('@1','@0'));runtime.seal_evidence(root)
                with self.assertRaises(ValueError):verify_pilot.verify(root)
        finally:server.shutdown();server.server_close();thread.join()
