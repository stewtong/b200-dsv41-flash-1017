"""Qualification grouping/registry fixtures; no GPU or physical-placement claim."""
import copy,hashlib,io,json,sys,tempfile,unittest
from contextlib import ExitStack,redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import attempts,campaign,fixed_workload,node_fixed_block as block,node_fixed_qualification as qualified
import pilot_block_recover,routing_map,runtime,verify_fixed_block,verify_fixed_qualification
import test_node_fixed_qualification as prep_fixture
from test_vllm_prompt import FakeTokenizer


class FixedBlockTests(unittest.TestCase):
    def fixture(self,base):
        config,source=prep_fixture.FixedQualificationTests().fixture(base)
        config['cell_id']=config['cell_id'].replace('challenger','control_before');config['profile_id']='control'
        second=base/'dsv41-fixed-input-fixture-r1';second.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name!='SHA256SUMS':(second/path.name).write_bytes(path.read_bytes())
        for name in ('run-config.json','prompt-result.json'):
            path=second/name;row=json.loads(path.read_text());row['repetition']=1;path.write_text(json.dumps(row))
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,1,0)]
        (second/'fixed-tasks.json').write_text(json.dumps(tasks));runtime.seal_evidence(second)
        entries=[{'cell_id':config['cell_id'],'tasks_file':str(path/'fixed-tasks.json'),
                  'request_timeout_seconds':1800,'repetition':rep} for rep,path in enumerate((source,second))]
        for key in ('cell_id','tasks_file','request_timeout_seconds','repetition'):config.pop(key,None)
        config.update(run_id='fixed-block-fixture',run_dir=str(base/'dsv41-fixed-block-fixture'),cells=entries)
        route=base/'synthetic-route-source';route.mkdir();(route/'fixture.json').write_text('{"synthetic":"routing loader is isolated in this registry/projection fixture"}')
        runtime.seal_evidence(route);config['routing_map_evidence']=str(route)
        return config

    def test_preparation_keeps_distinct_repetitions_and_fresh_comparison_arms(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);config=self.fixture(base);rows,budget=block.prepare(config,base)
            self.assertEqual(len(rows),2);self.assertEqual(budget,3640)
            too_many=copy.deepcopy(config);too_many['cells']*=13
            with self.assertRaisesRegex(ValueError,'twenty-four'):block.prepare(too_many,base)
            self.assertEqual(len({r['identity'] for r in rows}),2)
            for mutate,reason in (
                (lambda c:c['cells'].__setitem__(1,copy.deepcopy(c['cells'][0])),'duplicate'),
                (lambda c:c['cells'][1].__setitem__('cell_id',c['cells'][1]['cell_id'].replace('control_before','control_after')),'mixes'),
                (lambda c:c['cells'][1].__setitem__('repetition',True),'repetition'),
                (lambda c:c['cells'][1].__setitem__('profile_id','candidate'),'fields')):
                bad=copy.deepcopy(config);mutate(bad);self.assertNotEqual(json.dumps(bad,sort_keys=True),json.dumps(config,sort_keys=True))
                with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason):block.prepare(bad,base)

    def execute_fixture(self,base,fail_second=False,mixed=False,fail_verification=False):
        config=self.fixture(base);root=Path(config['run_dir']);registry=base/'registry'
        owner={'boot_id':'fixture','pid':123,'start_ticks':456};calls=[]
        records=[]
        for i in range(128):
            unit=i%8
            for repeat in (0,1):records.append({'session_id':'grid'+str(i),'request_id':'grid'+str(i)+'-r'+str(repeat),
                'backend_port':31000+unit//4,'rank_headers':[str(unit%4)]})
        mapping=routing_map.build(records,[31000,31001],4)
        def put(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        def capture(session,dest,endpoint,child,prepared,*args):
            calls.append(child['repetition'])
            if fail_second and len(calls)==2:raise RuntimeError('declared fixture interruption')
            directory=dest/'fixed-attribution';directory.mkdir()
            index=len(calls)-1
            put(directory/'requests/measured/s0.json',{'start_ns':100+index*100,'end_ns':150+index*100})
            put(directory/'routing-span.json',{'start_byte':index*2,'end_byte':index*2+2})
            (directory/'routing.log').write_bytes(b'A\n' if not index else b'B\n')
        def engine(cfg,callback,*,session_budget):
            self.assertEqual(session_budget,'qualification-block')
            root.mkdir();put(root/'run-config.json',cfg)
            (root/'manifest.json').write_bytes(Path(cfg['manifest']).read_bytes())
            put(root/'execution-state.json',{'run_id':cfg['run_id'],'driver_identity':owner,
                'watchdog_identity':owner,'owned_processes':[],'container_ids':[],'stopping':True,'watchdog_stop':True})
            (root/'router.log').write_bytes(b'A\nB\n');(root/'telemetry-0000.jsonl').write_text('{"synthetic":true}\n')
            callback(SimpleNamespace(ensure_healthy=lambda:None),root,'synthetic-url')
            put(root/'smoke-result.json',{'status':'passed'})
        def verify_child(path):
            result=json.loads((Path(path)/'fixed-qualification-result.json').read_text())
            if fail_verification and result['repetition']==1:
                raise ValueError('synthetic invalid captured repetition')
            return {**result,'requests':1,'valid_tokens':2048,'fixture_scope':'isolated child verifier stub; no live physical/HTTP validation'}
        if mixed:
            planned,_=block.prepare(config,base);ident=planned[0]['identity']
            prior=attempts.begin(registry,ident,method_sha256='0'*64)
            attempts.mark_interrupted(prior,'fixture-only independently reconciled interrupted control')
            config['retry_cells']={ident:'Use the explicitly versioned performance workload after retained legacy correctness failure; keep the original scheduled identity.'}
        original_prepare=block.prepare
        with ExitStack() as stack:
            stack.enter_context(patch.object(block,'prepare',side_effect=lambda cfg:original_prepare(cfg,base)))
            stack.enter_context(patch.object(block.node_smoke,'validate',return_value=({}, {'lock_sha256':'fixture','router_executable_sha256':'fixture'},10000)))
            stack.enter_context(patch.object(block.routing_map,'load_verified',return_value=mapping))
            stack.enter_context(patch.object(runtime,'DEFAULT_NODE_LOCK',base/'node.lock'))
            stack.enter_context(patch.object(runtime,'live_preflight'))
            stack.enter_context(patch.object(runtime,'process_identity',return_value=owner))
            stack.enter_context(patch.object(runtime,'process_alive',return_value=False))
            stack.enter_context(patch.object(qualified,'DEFAULT_ROOT',registry))
            stack.enter_context(patch.object(qualified,'capture_prepared',side_effect=capture))
            stack.enter_context(patch.object(block.node_smoke,'run',side_effect=engine))
            stack.enter_context(patch.object(verify_fixed_qualification,'verify',side_effect=verify_child))
            if fail_second:
                with self.assertRaisesRegex(RuntimeError,'fixture interruption'):block.run(config)
                self.assertFalse((root/'fixed-qualification-block-result.json').exists())
                rows=attempts.load_attempts(registry);self.assertEqual([r['status'] for r in rows],['running','running'])
                self.assertTrue(all((registry/r['directory']/'failure.json').exists() for r in rows))
                checked=pilot_block_recover.recover(root,attempts_root=registry,node_lock=base/'node.lock',allowed_root=base,
                    alive=lambda ident:False,preflight=lambda *args:None,result_name='fixed-qualification-block-result.json',
                    child_result_name='fixed-qualification-result.json',block_verifier=verify_fixed_block.verify)
                self.assertEqual(checked['status'],'interrupted_block_reconciled')
                self.assertTrue(all(r['status']=='interrupted' for r in attempts.load_attempts(registry)))
            elif fail_verification:
                with self.assertRaisesRegex(ValueError,'invalid captured repetition'):block.run(config)
                self.assertFalse((root/'fixed-qualification-block-result.json').exists())
                terminal=attempts.load_attempts(registry)
                self.assertEqual(len(terminal),2)
                self.assertTrue(all(row['status']=='running' for row in terminal))
                self.assertTrue(all((registry/row['directory']/'failure.json').is_file() for row in terminal))
            else:
                checked=block.run(config);self.assertEqual(len(checked['cells']),2)
                terminal=attempts.load_attempts(registry)
                self.assertEqual(sum(r['status']=='passed' for r in terminal),2)
                self.assertEqual(sum(r['status']=='interrupted' for r in terminal),int(mixed))
                if mixed:
                    retried=[r for r in terminal if r['cell_id']==ident and r['status']=='passed']
                    self.assertEqual(len(retried),1);self.assertEqual(retried[0]['retry_reason'],config['retry_cells'][ident])
                output=io.StringIO()
                with patch.object(sys,'argv',['campaign.py','verify-fixed-block','--run-dir',str(root)]),redirect_stdout(output):campaign.main()
                self.assertEqual(json.loads(output.getvalue())['status'],'passed')
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'replay|per-cell retry'):block.run(config)
                cell=root/'cells'/checked['cells'][0]['cell_id'];path=cell/'fixed-attribution/routing-span.json'
                old=path.read_bytes();span=json.loads(old);span['start_byte']=1;put(path,span)
                runtime.seal_evidence(cell);runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'router span'):verify_fixed_block.verify(root)
                path.write_bytes(old);runtime.seal_evidence(cell);runtime.seal_evidence(root)
                self.assertEqual(verify_fixed_block.verify(root)['status'],'passed')
            self.assertEqual(calls,[0,1])

    def test_failed_parallel_verification_keeps_failure_records_without_group_success(self):
        with tempfile.TemporaryDirectory() as d:self.execute_fixture(Path(d),fail_verification=True)

    def test_mixed_retry_and_fresh_repetitions_keep_underlying_history(self):
        with tempfile.TemporaryDirectory() as d:self.execute_fixture(Path(d),mixed=True)

    def test_group_cli_and_resealed_native_span_tampering(self):
        with tempfile.TemporaryDirectory() as d:self.execute_fixture(Path(d))

    def test_capture_interruption_preserves_every_attempt_and_recovers_without_replay(self):
        with tempfile.TemporaryDirectory() as d:self.execute_fixture(Path(d),True)


if __name__=='__main__':unittest.main()
