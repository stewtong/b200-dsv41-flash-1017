"""Real owned HTTP/subprocess capture; GPU startup and 1Hz are isolated fixtures."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,threading,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import engine_counters,node_model_client_check as producer,profile_registry,runtime,verify_model_client_check as verifier,watchdog
import test_node_model_client_check as inputs_fixture
import test_cache_source_proof as source_fixture


@unittest.skipUnless(hasattr(os,'sched_getaffinity'),'owned model client comparison requires Linux process identity/affinity')
@patch.object(producer.cache_source_proof,'PINNED_SHA256',source_fixture.FIXTURE_PIN)
class ModelClientCheckLinuxTests(unittest.TestCase):
    def test_nine_real_http_probes_verify_raw_lifecycle_and_resealed_forgery(self):
        # All metrics are a deterministic HTTP fixture. Physical GPU startup,
        # source placement and sampler windows are explicitly mocked below.
        state={r:{'prompt':0,'output':0,'length':0} for r in range(2)};lock=threading.Lock();route=[None];tasks={}
        def metrics():
            types={};lines=[]
            for rank,s in state.items():
                for (name,_,dimension),(kind,integer) in engine_counters.contract([rank]).items():
                    base=name.removesuffix('_total') if kind=='counter' else name.rsplit('_',1)[0];types[base]=kind
                    label=f'model_name="/model",engine="{rank}"'
                    if dimension:label+=',source="'+dimension+'"' if name=='vllm:prompt_tokens_by_source_total' else ',finished_reason="'+dimension+'"'
                    value=0
                    if name=='vllm:prompt_tokens_total':value=s['prompt']
                    elif name=='vllm:generation_tokens_total':value=s['output']
                    elif name=='vllm:prompt_tokens_by_source_total':value=s['prompt'] if dimension=='local_compute' else 0
                    elif name=='vllm:request_success_total':value=s['length'] if dimension=='length' else 0
                    elif name.endswith('_count'):value=s['length']
                    elif name=='vllm:request_prompt_tokens_sum':value=s['prompt']
                    elif name=='vllm:request_generation_tokens_sum':value=s['output']
                    elif name=='vllm:request_prefill_time_seconds_sum':value=s['length']*.2
                    elif name=='vllm:request_decode_time_seconds_sum':value=s['length']*.1
                    lines.append(name+'{'+label+'} '+str(value))
                for kind in ('queries','hits'):
                    name='vllm:prefix_cache_'+kind;types[name]='counter';lines.append(name+'_total{model_name="/model",engine="'+str(rank)+'"} 0')
            return '\n'.join('# TYPE '+k+' '+v for k,v in types.items())+'\n'+'\n'.join(lines)+'\n'
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with lock:data=metrics().encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));task=tasks[body['seed']];rank=task['session']%2
                with lock:
                    s=state[rank];s['prompt']+=task['input_tokens'];s['output']+=2048;s['length']+=1
                    with route[0].open('a') as f:f.write("Consistent hash routing: key='header:x-session-id:"+self.headers['X-Session-Id']+"' -> worker='http://127.0.0.1:"+str(self.server.server_port)+'@'+str(rank)+"' (index="+str(rank)+')\n')
                frames=[{'choices':[{'delta':{'content':task['expected_answer']+' fixture technical report'},'finish_reason':'length'}]},
                    {'choices':[],'usage':{'prompt_tokens':task['input_tokens'],'completion_tokens':2048}}]
                data=(''.join('data: '+json.dumps(x)+'\n\n' for x in frames)+'data: [DONE]\n\n').encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                base=Path(d);c=inputs_fixture.ModelClientCheckTests().fixture(base);prepared,budget=producer.prepare(c,base);root=Path(c['run_dir'])
                for item in prepared['inputs']:tasks.update({t['seed']:t for t in item['tasks']})
                url=f'http://127.0.0.1:{server.server_port}';endpoint=url+'/metrics';c['expected_ranks']={endpoint:[0,1]}
                route_source=base/'route';route_source.mkdir();runtime.atomic_json(route_source/'fixture.json',{'scope':'isolated native route geometry'});runtime.seal_evidence(route_source);c['routing_map_evidence']=str(route_source)
                placement=[{'task_session':i,'session_header':'unit'+str(i),'endpoint':endpoint,'rank':str(i%2)} for i in range(8)]
                cpus=sorted(os.sched_getaffinity(0))[:2];self.assertEqual(len(cpus),2)
                owner=runtime.process_identity(os.getpid());registry=base/'registry'
                class Session:
                    def ensure_healthy(self):watchdog.inspect_ownership(self.root,json.loads((self.root/'execution-state.json').read_text()),None,time.monotonic())
                    def abort(self,reason):raise AssertionError(reason)
                def run(config,callback):
                    root.mkdir();session=Session();session.root=root;session.run_id=c['run_id'];session.deadline_ns=time.monotonic_ns()+14_000_000_000_000
                    runtime.atomic_json(root/'execution-state.json',{'run_id':c['run_id'],'phase':'qualification','driver_identity':owner,
                        'watchdog_identity':{'boot_id':owner['boot_id'],'pid':2_000_000_000,'start_ticks':1},'owned_processes':[],'container_ids':[],'stopping':False})
                    for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
                        runtime.atomic_bytes(root/name,Path(c[key]).read_bytes())
                    runtime.atomic_json(root/'run-config.json',c);runtime.atomic_json(root/'profile-binding.json',prepared['binding'])
                    runtime.atomic_json(root/'cpu-partition.json',{'client_cpus':cpus});runtime.atomic_json(root/'cache-config.json',{'block_size':128})
                    runtime.atomic_json(root/'startup-timing.json',{'semantic_ready_ns':time.monotonic_ns()})
                    runtime.atomic_json(root/'router-qualification.json',{'lock_sha256':'fixture','router_executable_sha256':'fixture'})
                    route[0]=root/'router.log';route[0].write_text('');callback(session,root,url+'/v1/chat/completions')
                    runtime.atomic_json(root/'cleanup-result.json',{'success':True,'scope':'no GPU process was launched'})
                with patch.object(producer,'prepare',return_value=(prepared,budget)), \
                     patch.object(producer.node_smoke,'validate',return_value=({}, {'lock_sha256':'fixture','router_executable_sha256':'fixture'},14000)), \
                     patch.object(producer.routing_map,'load_verified',return_value={'ports':[server.server_port],'dp':2}), \
                     patch.object(producer.routing_map,'select',return_value=placement), \
                     patch.object(producer.runtime,'DEFAULT_NODE_LOCK',base/'node.lock'),patch.object(producer.runtime,'live_preflight'), \
                     patch.object(producer,'DEFAULT_ROOT',registry),patch.object(producer.node_smoke,'run',side_effect=run), \
                     patch('verify_telemetry.verify_window',return_value={'scope':'isolated sampler fixture; no1Hzclaim'}), \
                     patch.object(verifier.verify_engine_profile,'verify',return_value={'profile_sha256':prepared['binding']['profile_sha256']}), \
                     patch.object(verifier,'FRONTDOOR',url+'/v1/chat/completions'):
                    result=producer.run(c);self.assertEqual(result['status'],'passed');self.assertEqual(len(result['probes']),9)
                    self.assertEqual(len(result['comparisons']),3);self.assertEqual(verifier.verify(root),result)
                    self.assertEqual(attempts_status:=json.loads(next(registry.glob('attempt-*/attempt.json')).read_text())['status'],'passed')
                    self.assertEqual(len(json.loads((root/'execution-state.json').read_text())['completed_processes']),12)
                    source=root/'probes/r0-two/requests/measured/clients/client-source.py';original=source.read_bytes()
                    source.write_bytes(original+b'\n')
                    for p in (source.parent,source.parent.parent,source.parent.parent.parent,root/'probes/r0-two',root):runtime.seal_evidence(p)
                    with self.assertRaisesRegex(ValueError,'execution toolkit pin'):verifier.verify(root)
                    source.write_bytes(original)
                    for p in (source.parent,source.parent.parent,source.parent.parent.parent,root/'probes/r0-two',root):runtime.seal_evidence(p)
                    path=root/'probes/r0-two/requests/fixed-session-contract.json';row=json.loads(path.read_text());row['client_processes']['workers']=1;runtime.atomic_json(path,row)
                    # Reseal the changed leaf and each enclosing artifact.
                    for p in (path.parent,root/'probes/r0-two',root):runtime.seal_evidence(p)
                    with self.assertRaisesRegex(ValueError,'mode or CPU'):verifier.verify(root)
        finally:server.shutdown();server.server_close();thread.join()
