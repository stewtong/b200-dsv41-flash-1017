"""Actual HTTP capsule, engine accounting, native affinity and independent CLI."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import copy,hashlib,json,re,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import engine_counters,fixed_attribution,fixed_burst,fixed_workload,runtime,verify_fixed_attribution
import node_fixed_qualification as qualified
import verify_worker_observations
import test_speculative_boundary as spec_fixture
from test_vllm_prompt import FakeTokenizer


class FixedAttributionTests(unittest.TestCase):
    def test_late_serial_metrics_reads_cannot_open_a_measurement_window(self):
        clock=[1_000_000_000];calls=[]
        class Session:
            deadline_ns=200_000_000_000
            def ensure_healthy(self):pass
        def scrape(endpoint,names,timeout):
            calls.append((endpoint,timeout))
            # A provider can return after its socket timeout. The whole-cycle
            # clock, rather than that timeout alone, must reject a late result.
            clock[0]+=9_000_000_000 if len(calls)==1 else 2_000_000_000
            return 'late metrics'
        def capsule(*args):
            args[-1]('before_measured')
            self.fail('late boundary allowed request dispatch')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'router.log';log.write_text('')
            with patch.object(fixed_attribution,'time',SimpleNamespace(monotonic_ns=lambda:clock[0])), \
                 patch.object(fixed_attribution.telemetry,'scrape',side_effect=scrape), \
                 patch.object(fixed_attribution.fixed_session,'run',side_effect=capsule):
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'boundary deadline'):
                    fixed_attribution.capture(Session(),'unused',[{'id':'s0','cache_designation':'cold'}],
                        root/'capture',5,'fixture','fx'+'a'*24,{'s0':'unit0'},
                        {'http://127.0.0.1:31000/metrics':[0],'http://127.0.0.1:31001/metrics':[0]},log,128)
            self.assertEqual([timeout for _,timeout in calls],[3,1])
            self.assertFalse((root/'capture/boundaries.json').exists())
            self.assertEqual(json.loads((root/'capture/attribution-failure.json').read_text())['status'],'failed')

    def test_http_capture_and_cli_reject_forged_routes_boundaries_and_counter_work(self):
        state={};tasks_by_seed={};lock=threading.Lock();route_path=[None];spec_mode=[False];cache_enabled=[True]
        def export():
            types={};lines=[]
            for rank in (0,1):
                row=state[rank]
                for (name,_,dimension),(kind,integer) in engine_counters.contract([rank]).items():
                    base=name.removesuffix('_total') if kind=='counter' else name.rsplit('_',1)[0];types[base]=kind
                    label='model_name="/model",engine="'+str(rank)+'"'
                    if dimension:label+=',source="'+dimension+'"' if name=='vllm:prompt_tokens_by_source_total' else ',finished_reason="'+dimension+'"'
                    value=0
                    if name=='vllm:prompt_tokens_total':value=row['prompt']
                    elif name=='vllm:prompt_tokens_cached_total':value=row['cached']
                    elif name=='vllm:generation_tokens_total':value=row['output']
                    elif name=='vllm:prompt_tokens_by_source_total':value={'local_compute':row['prompt']-row['cached'],'local_cache_hit':row['cached'],'external_kv_transfer':0}[dimension]
                    elif name=='vllm:request_success_total':value=row[dimension]
                    elif name.endswith('_count'):value=row['stop']+row['length']
                    elif name=='vllm:request_prompt_tokens_sum':value=row['prompt']
                    elif name=='vllm:request_generation_tokens_sum':value=row['output']
                    elif name=='vllm:request_prefill_time_seconds_sum':value=(row['stop']+row['length'])*.2
                    elif name=='vllm:request_decode_time_seconds_sum':value=(row['stop']+row['length'])*.1
                    lines.append(name+'{'+label+'} '+str(value))
                for kind,field in [('queries','prompt'),('hits','cached')]:
                    name='vllm:prefix_cache_'+kind;types[name]='counter';lines.append(name+'_total{model_name="/model",engine="'+str(rank)+'"} '+str(row[field] if cache_enabled[0] else 0))
            result='\n'.join('# TYPE '+n+' '+k for n,k in types.items())+'\n'+'\n'.join(lines)+'\n'
            if spec_mode[0]:result+=spec_fixture.metrics([state[rank]['stop']+state[rank]['length'] for rank in (0,1)])+'\n'
            return result
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with lock:data=export().encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));task=tasks_by_seed[body['seed']]
                cond='min_tokens' not in body;rank=int(self.headers['X-Session-Id'][-1]);tokens=8 if cond else task['output_tokens'];prompt=task['prefix_tokens'] if cond else task['input_tokens']
                with lock:
                    s=state[rank];s['prompt']+=prompt;s['output']+=tokens;s['stop' if cond else 'length']+=1
                    if not cond and task['cache_designation']=='warm_prefix':s['cached']+=task['prefix_tokens']
                    with route_path[0].open('a') as f:f.write("Consistent hash routing: key='header:x-session-id:"+self.headers['X-Session-Id']+"' -> worker='http://127.0.0.1:"+str(self.server.server_port)+'@'+str(rank)+"' (index="+str(rank)+')\n')
                text=task['expected_answer']+('' if cond else ' technical report')
                frames=[{'choices':[{'index':0,'delta':{'role':'assistant','content':text},'finish_reason':'stop' if cond else 'length'}]},
                        {'choices':[],'usage':{'prompt_tokens':prompt,'completion_tokens':tokens}}]
                if spec_mode[0]:frames[-1]['metrics']=spec_fixture.wire('fixture')['raw']['events'][0]['data']['metrics']
                data=(''.join('data: '+json.dumps(f)+'\n\n' for f in frames)+'data: [DONE]\n\n').encode()
                self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        class Server(ThreadingHTTPServer):request_queue_size=64
        server=Server(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        class Session:
            deadline_ns=time.monotonic_ns()+60_000_000_000+120_000_000_000
            def ensure_healthy(self):pass
            def abort(self,reason):raise AssertionError(reason)
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);log=root/'router.log';log.write_text('');route_path[0]=log
                for rank in (0,1):state[rank]={key:0 for key in ['prompt','cached','output',*engine_counters.REASONS]}
                url=f'http://127.0.0.1:{server.server_port}';ranks={url+'/metrics':[0,1]}
                for cache in ('cold','warm_prefix'):
                    tasks=[fixed_workload.build(FakeTokenizer(),'/model',10000,2048,'core',2,0,i,cache,128) for i in range(2)]
                    tasks_by_seed.update({t['seed']:t for t in tasks});ids={t['id']:'unit'+str(t['session']) for t in tasks}
                    result=fixed_attribution.capture(Session(),url+'/v1/chat/completions',tasks,root/cache,5,'fixture',fixed_burst.namespace('fixture',cache,0,'a'*64),ids,ranks,log,128)
                    self.assertEqual(result['requests'],2);self.assertEqual(result['conditioning_requests'],2 if cache=='warm_prefix' else 0)
                    self.assertEqual(len(result['session_routes']),2)
                    if cache=='warm_prefix':self.assertGreater(sum(r['hits'] for r in result['populations'][-1]['ranks']),0)
                    cli=[sys.executable,'-B',str(Path(fixed_attribution.__file__).with_name('campaign.py')),'verify-fixed-attribution','--run-dir',str(root/cache)]
                    checked=subprocess.run(cli,capture_output=True,text=True,timeout=10);self.assertEqual(checked.returncode,0,checked.stderr)
                # Exercise the new excluded-warmup producer and independent audit
                # with real retained HTTP/native counters, before scored dispatch.
                import kernel_warmup,verify_kernel_warmup,routing_map
                from test_kernel_warmup import KernelWarmupTests
                preparation=root/'warmup-preparation';preparation.mkdir()
                manifest,warm_cfg,warm_tasks,measured_tasks=KernelWarmupTests().fixture(preparation)
                warm_cfg.update(run_id='fixture',expected_ranks=ranks)
                tasks_by_seed.update({t['seed']:t for t in warm_tasks})
                records=[]
                for i in range(128):
                    calibration_session='grid'+str((i//2)*10+i%2)
                    for rep in (0,1):records.append(dict(session_id=calibration_session,
                        request_id=calibration_session+'-r'+str(rep),backend_port=server.server_port,rank_headers=[str(i%2)]))
                mapping=routing_map.build(records,[server.server_port],2)
                prepared=kernel_warmup.prepare(warm_cfg,manifest,measured_tasks,root)
                runtime.atomic_json(root/'cache-config.json',dict(block_size=128))
                runtime.atomic_json(root/'execution-state.json',dict(driver_identity=dict(boot_id='fixture-boot')))
                owned=Session();owned.root=root;owned.run_id='fixture'
                with patch.object(kernel_warmup.node_pilot,'clock_domain',return_value='fixture-boot:fixture'):
                    kernel_warmup.run(owned,root,url+'/v1/chat/completions',warm_cfg,manifest,prepared,mapping)
                audited=verify_kernel_warmup.verify(root/'kernel-warmup',measured_tasks,manifest,warm_cfg,mapping,time.monotonic_ns())
                self.assertEqual(audited['requests'],8);self.assertEqual(audited['scored_requests'],0)
                for rank in (0,1):state[rank]={key:0 for key in ['prompt','cached','output',*engine_counters.REASONS]}
                cache_enabled[0]=False
                binding={'phase':'qualification','delta':{'enable_prefix_caching':False},
                    'record':{'profile_delta':{'enable_prefix_caching':False},'template':{'commands':[{'argv':['--disable-prefix-caching']}]}}}
                runtime.atomic_json(root/'profile-binding.json',binding);owned=Session();owned.root=root
                cold=[fixed_workload.build(FakeTokenizer(),'/model',10000,2048,'core',2,1,i) for i in range(2)]
                tasks_by_seed.update({t['seed']:t for t in cold});cold_ids={t['id']:'unit'+str(t['session']) for t in cold}
                result=fixed_attribution.capture(owned,url+'/v1/chat/completions',cold,root/'cache-off',5,'fixture',
                    fixed_burst.namespace('fixture','cache-off',1,'a'*64),cold_ids,ranks,log,128,prefix_caching=False)
                self.assertTrue(all(r['queries']==r['hits']==0 for r in result['populations'][0]['ranks']))
                verify_fixed_attribution.verify_cache_profile(root/'cache-off',binding)
                with self.assertRaisesRegex(ValueError,'parent profile'):
                    verify_fixed_attribution.verify_cache_profile(root/'cache-off',{'delta':{'enable_prefix_caching':True}})
                cli=[sys.executable,'-B',str(Path(fixed_attribution.__file__).with_name('campaign.py')),'verify-fixed-attribution','--run-dir',str(root/'cache-off')]
                checked=subprocess.run(cli,capture_output=True,text=True,timeout=10);self.assertEqual(checked.returncode,0,checked.stderr)
                target=root/'cache-off';bounds=json.loads((target/'boundaries.json').read_text());snapshot=target/bounds[-1]['snapshot_file']
                values=json.loads(snapshot.read_text());old=values[url+'/metrics'];values[url+'/metrics']=old.replace('vllm:prefix_cache_queries_total{model_name="/model",engine="0"} 0',
                    'vllm:prefix_cache_queries_total{model_name="/model",engine="0"} 1')
                self.assertNotEqual(old,values[url+'/metrics']);runtime.atomic_json(snapshot,values)
                bounds[-1]['snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest();runtime.atomic_json(target/'boundaries.json',bounds);runtime.seal_evidence(target)
                with self.assertRaisesRegex(ValueError,'cache counters'):verify_fixed_attribution.verify(target)
                cache_enabled[0]=True
                for rank in (0,1):state[rank]={key:0 for key in ['prompt','cached','output',*engine_counters.REASONS]}
                spec_mode[0]=True
                spec_tasks=[fixed_workload.build(FakeTokenizer(),'/model',10000,2048,'core',2,2,i,'cold',128) for i in range(2)]
                tasks_by_seed.update({t['seed']:t for t in spec_tasks});spec_ids={t['id']:'unit'+str(t['session']) for t in spec_tasks}
                contract={'max_spec_tokens':3,'placement':{t['id']:[url+'/metrics',str(t['session'])] for t in spec_tasks}}
                # Declared mixed TP/DP metadata isolates the portable rank join.
                # HTTP/native counters are real fixtures; GPU/source ownership
                # is not asserted by this component fixture.
                layout={'schema':'resolved-worker-kv-layout/v1','scope':'synthetic rank-join fixture'}
                layout_sha=hashlib.sha256(json.dumps(layout,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                wc={'run_id':'fixture','replica_id':0,'boot_id':'fixture-boot','profile_sha256':'b'*64,
                    'toolkit_sha256':'c'*64,'extension_sha256':'d'*64,'recorder_sha256':'e'*64,'runtime_sha256':'f'*64,
                    'tp_size':4,'dp_size':2,'expected_gpu_ids':list(range(8))}
                worker_context={'schema':'worker-target-attribution/v1','source_pins':{'public-fixture.py':'a'*64},
                    'source_sha256':{},'scope':'synthetic join; no installed source or GPU claim','workers':[]}
                for i in range(8):
                    identity={**wc,'namespace_pid':500+i,'container_hostname':('f'*64)[:12],
                        'dp_rank':i//4,'tp_rank':i%4,'worker_rank':i,'contract_sha256':'a'*64,
                        'source_pins':worker_context['source_pins']}
                    worker_context['workers'].append({'container_id':'f'*64,'endpoint':url+'/metrics','identity':identity,
                        'physical_worker':{'namespace_pid':500+i,'host_pid':600+i,'start_ticks':100+i,'gpu_index':i,'dp_rank':i//4,'tp_rank':i%4},
                        'layout':layout,'contract':wc,'contract_sha256':'a'*64})
                def worker_observation(*args):
                    observation={'status':'physical_worker_observation_sample','workers':[]}
                    with lock:counts=[state[rank]['stop']+state[rank]['length'] for rank in (0,1)]
                    for item in worker_context['workers']:
                        count=counts[item['identity']['dp_rank']];stamp=time.monotonic_ns()
                        values={'successful_sample_calls':1+count*4,'successful_verification_steps':count*4,
                            'target_verified_draft_positions':count*12,'successful_grammar_sample_calls':0,
                            'kv_layout_canonical_sha256':layout_sha}
                        observation['workers'].append({'container_id':item['container_id'],'physical_worker':item['physical_worker'],
                            'row':{'schema':'worker-target-observation/v1','identity':item['identity'],
                                'started_ns':stamp,'completed_ns':stamp,'values':values}})
                    observation['observed_at_ns']=time.monotonic_ns();return observation
                with patch.object(verify_worker_observations,'context_record',return_value=worker_context), \
                     patch.object(fixed_attribution.measurement_launch,'collect',side_effect=worker_observation):
                    observed=fixed_attribution.capture(Session(),url+'/v1/chat/completions',spec_tasks,root/'speculative',5,'fixture',
                        fixed_burst.namespace('fixture','speculative',2,'a'*64),spec_ids,ranks,log,128,
                        speculative_contract=contract,worker_root=root/'declared-worker-fixture')
                self.assertEqual(observed['populations'][-1]['speculative_accounting']['requests'],2)
                self.assertEqual(observed['populations'][-1]['target_observation_accounting']['target_verified_draft_positions'],24)
                checked=subprocess.run([sys.executable,'-B',str(Path(fixed_attribution.__file__).with_name('campaign.py')),
                    'verify-fixed-attribution','--run-dir',str(root/'speculative')],capture_output=True,text=True,timeout=10)
                self.assertEqual(checked.returncode,0,checked.stderr)
                target_spec=root/'speculative';bounds=json.loads((target_spec/'boundaries.json').read_text());sp=target_spec/bounds[-1]['snapshot_file']
                wp=target_spec/bounds[-1]['worker_snapshot_file'];old_worker=wp.read_bytes();old_bounds=(target_spec/'boundaries.json').read_bytes()
                forged=json.loads(old_worker)
                for item in forged['workers']:
                    if item['row']['identity']['dp_rank']==0:item['row']['values']['target_verified_draft_positions']+=1
                runtime.atomic_json(wp,forged);bounds[-1]['worker_snapshot_sha256']=hashlib.sha256(wp.read_bytes()).hexdigest()
                runtime.atomic_json(target_spec/'boundaries.json',bounds);runtime.seal_evidence(target_spec)
                with self.assertRaisesRegex(ValueError,'target positions differ'):verify_fixed_attribution.verify(target_spec)
                wp.write_bytes(old_worker);(target_spec/'boundaries.json').write_bytes(old_bounds);runtime.seal_evidence(target_spec)
                bounds=json.loads(old_bounds)
                raw_sp=sp.read_bytes();values=json.loads(raw_sp)
                values[url+'/metrics']=values[url+'/metrics'].replace('vllm:spec_decode_num_drafts_total{model_name="/model",engine="0"} 4',
                    'vllm:spec_decode_num_drafts_total{model_name="/model",engine="0"} 5')
                self.assertNotEqual(json.loads(raw_sp),values);sp.write_text(json.dumps(values))
                bounds[-1]['snapshot_sha256']=__import__('hashlib').sha256(sp.read_bytes()).hexdigest()
                runtime.atomic_json(target_spec/'boundaries.json',bounds);runtime.seal_evidence(target_spec)
                with self.assertRaisesRegex(ValueError,'completed request steps per rank'):verify_fixed_attribution.verify(target_spec)
                spec_mode[0]=False
                # Exercise the dispatcher/shared-block capture helper through
                # actual HTTP. Source/map records are declared fixtures; this
                # does not substitute for installed-tokenizer or GPU evidence.
                tasks=[fixed_workload.build(FakeTokenizer(),'/model',10000,2048,'core',2,1,i,'cold',128) for i in range(2)]
                tasks_by_seed.update({t['seed']:t for t in tasks})
                source=root/'source-fixture';source.mkdir();task_bytes=json.dumps(tasks).encode();(source/'fixed-tasks.json').write_bytes(task_bytes)
                runtime.atomic_json(source/'fixture.json',{'scope':'isolated HTTP capture helper; no generator provenance claim'});runtime.seal_evidence(source)
                route=root/'route-fixture';route.mkdir();runtime.atomic_json(route/'fixture.json',{'scope':'placement supplied explicitly to HTTP helper'});runtime.seal_evidence(route)
                dest=root/'dispatcher-helper';dest.mkdir();runtime.atomic_json(root/'cache-config.json',{'block_size':128})
                cfg={'run_id':'http-fixed-helper','cell_id':'declared-fixture','repetition':1,'request_timeout_seconds':5,
                     'matrix':str(Path(fixed_workload.__file__).with_name('matrix.json')),
                     'routing_map_evidence':str(route),'expected_ranks':ranks,'layout':'fixture-geometry'}
                prepared={'source':source,'source_manifest_sha256':__import__('hashlib').sha256((source/'SHA256SUMS').read_bytes()).hexdigest(),
                          'task_bytes':task_bytes,'tasks':tasks,'cell':{'id':'declared-fixture'},'attempt_cell_id':'declared-fixture',
                          'binding':{'profile_sha256':'a'*64}}
                placement=[{'task_session':i,'session_header':'unit'+str(i),'endpoint':url+'/metrics','rank':str(i)} for i in range(2)]
                owned=Session();owned.run_id='http-fixed-helper'
                with patch.object(qualified.node_pilot,'clock_domain',return_value='fixture-helper-clock'), \
                     patch.object(qualified.profile_registry,'expected_runtime',return_value={'speculative_config':None}):
                    qualified.capture_prepared(owned,dest,url+'/v1/chat/completions',cfg,prepared,
                        {'mapping_sha256':'declared-fixture'},placement,log,root/'cache-config.json')
                self.assertEqual(verify_fixed_attribution.verify(dest/'fixed-attribution')['requests'],2)
                self.assertEqual(json.loads((dest/'fixed-qualification-plan.json').read_text())['repetition'],1)
                target=root/'warm_prefix';route=target/'routing.log';original=route.read_bytes();span=target/'routing-span.json';oldspan=span.read_bytes()
                route.write_bytes(original.replace(b'unit0',b'unit9'));record=json.loads(oldspan);record['sha256']=__import__('hashlib').sha256(route.read_bytes()).hexdigest();span.write_text(json.dumps(record));runtime.seal_evidence(target)
                with self.assertRaisesRegex(ValueError,'unknown'):verify_fixed_attribution.verify(target)
                route.write_bytes(original);span.write_bytes(oldspan)
                path=target/'boundaries.json';old=path.read_bytes();rows=json.loads(old)
                rows[-1]['started_ns']=rows[-2]['completed_ns'];rows[-1]['deadline_ns']=rows[-1]['started_ns']+10_000_000_000
                path.write_text(json.dumps(rows));runtime.seal_evidence(target)
                with self.assertRaisesRegex(ValueError,'outside'):verify_fixed_attribution.verify(target)
                path.write_bytes(old);runtime.seal_evidence(target)
                rows=json.loads(old);rows[-1]['deadline_ns']=rows[-1]['completed_ns']
                path.write_text(json.dumps(rows));runtime.seal_evidence(target)
                with self.assertRaisesRegex(ValueError,'boundary clocks'):verify_fixed_attribution.verify(target)
                path.write_bytes(old);runtime.seal_evidence(target)
                snapshot=target/rows[-1]['snapshot_file'];old_snapshot=snapshot.read_bytes();metrics=json.loads(old_snapshot)
                endpoint=next(iter(metrics));metrics[endpoint]=re.sub(r'(vllm:request_generation_tokens_sum\{model_name="/model",engine="0"\}) \S+',r'\g<1> 999999',metrics[endpoint]);snapshot.write_text(json.dumps(metrics))
                rows=json.loads(old);rows[-1]['snapshot_sha256']=__import__('hashlib').sha256(snapshot.read_bytes()).hexdigest();path.write_text(json.dumps(rows));runtime.seal_evidence(target)
                with self.assertRaisesRegex(ValueError,'output work'):verify_fixed_attribution.verify(target)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
