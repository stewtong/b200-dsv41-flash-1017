"""Source/rank/freshness joins and TP deduplication from actual recorder files."""
import copy,hashlib,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import runtime,worker_observations as producer,verify_worker_observations as audit
from test_worker_observations import config


class VerifyWorkerObservationTests(unittest.TestCase):
    def test_allocator_snapshot_requires_registered_source_device_and_nested_clock(self):
        import allocator_observations
        with tempfile.TemporaryDirectory() as d:
            identity={'allocator_sha256':'a'*64,'allocator_source_pins':dict(allocator_observations.SOURCE_PINS),'local_device_index':2}
            stats=dict(zip(allocator_observations.FIELDS.values(),(10,20,30,40,0,0)))
            reader=lambda:allocator_observations.read(lambda device:stats,2,'native')
            recorder=producer.Recorder(Path(d)/'worker',identity,time.monotonic_ns()+10_000_000_000,allocator_reader=reader)
            recorder.record_layout(config());row=recorder.snapshot()
            audit.check_row(row,identity,row['completed_ns'])
            for field,value in [('device_index',1),('backend','cudaMallocAsync'),('started_ns',row['started_ns']-1),('source_pins',{})]:
                changed=copy.deepcopy(row);changed['allocator'][field]=value
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,'allocator observation'):
                    audit.check_row(changed,identity,changed['completed_ns'])
            changed=copy.deepcopy(row);del changed['allocator']
            with self.assertRaisesRegex(ValueError,'allocator observation'):audit.check_row(changed,identity,row['completed_ns'])
            changed=copy.deepcopy(row);changed['allocator']['values']['allocated_bytes']=True
            with self.assertRaisesRegex(ValueError,'allocator observation'):audit.check_row(changed,identity,row['completed_ns'])

    def fixture(self,root):
        root=Path(root);cid='f'*64;pins={'public-fixture.py':'a'*64}
        contract={'run_id':'fixture','replica_id':0,'boot_id':'public-fixture-boot','profile_sha256':'b'*64,
            'recorder_sha256':'e'*64,'runtime_sha256':'f'*64,
            'toolkit_sha256':'c'*64,'extension_sha256':'d'*64,'tp_size':2,'dp_size':1,'expected_gpu_ids':[0,1]}
        runtime.atomic_json(root/'contract.json',contract);digest=hashlib.sha256((root/'contract.json').read_bytes()).hexdigest()
        physical={'workers':[]};recorders=[]
        for tp in range(2):
            worker={'namespace_pid':500+tp,'host_pid':600+tp,'start_ticks':100+tp,'gpu_index':tp,'tp_rank':tp}
            physical['workers'].append(worker)
            identity={**contract,'namespace_pid':500+tp,'container_hostname':cid[:12],'tp_rank':tp,'dp_rank':0,
                'contract_sha256':digest,'source_pins':pins,'worker_rank':tp}
            recorder=producer.Recorder(root/('worker-'+str(500+tp)),identity,time.monotonic_ns()+10_000_000_000)
            recorder.record_layout(config());recorder.observe_sample(lambda *args:None,None,SimpleNamespace(num_draft_tokens=3,num_reqs=1),None)
            runtime.atomic_json(recorder.root/'live.json',recorder.snapshot());recorders.append(recorder)
        return cid,pins,contract,physical,recorders

    def test_actual_recorders_join_physical_ranks_and_count_logical_target_work_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cid,pins,contract,physical,recorders=self.fixture(root)
            before=audit.read_replica(root,contract,physical,cid,pins,time.monotonic_ns(),alive=lambda _:True)
            live_clock_calls=[]
            def live_clock():live_clock_calls.append(True);return time.monotonic_ns()
            audit.read_replica(root,contract,physical,cid,pins,live_clock,alive=lambda _:True)
            self.assertEqual(len(live_clock_calls),2)
            for recorder in recorders:
                recorder.observe_sample(lambda *args:None,None,SimpleNamespace(num_draft_tokens=4,num_reqs=1),None)
                runtime.atomic_json(recorder.root/'live.json',recorder.snapshot())
            after=audit.read_replica(root,contract,physical,cid,pins,time.monotonic_ns(),alive=lambda _:True)
            result=audit.boundary_delta(before,after)
            self.assertEqual(result['target_verified_draft_positions'],4)
            self.assertEqual(len(result['logical_engines']),1)
            self.assertEqual(result['logical_engines'][0]['successful_verification_steps'],1)
            reconciled=audit.reconcile_target_positions(before,after,{(cid,0):4},speculative=True)
            self.assertEqual(reconciled['native_proposed_draft_tokens'],4)
            self.assertEqual(reconciled['target_verified_draft_positions'],4)
            with self.assertRaises(audit.UnsettledWorkerCounters):
                audit.reconcile_target_positions(before,after,{(cid,0):3},speculative=True)
            with self.assertRaisesRegex(ValueError,'NoSpec'):
                audit.reconcile_target_positions(before,after,{(cid,0):0},speculative=False)
            with self.assertRaisesRegex(ValueError,'population differs'):
                audit.reconcile_target_positions(before,after,{(cid,0):True},speculative=True)
            grammar=copy.deepcopy(after)
            for item in grammar:item['row']['values']['successful_grammar_sample_calls']=1
            with self.assertRaisesRegex(ValueError,'grammar-free'):
                audit.reconcile_target_positions(before,grammar,{(cid,0):4},speculative=True)
            altered=copy.deepcopy(after);altered[1]['row']['values']['target_verified_draft_positions']+=1
            with self.assertRaisesRegex(ValueError,'mirrored TP'):audit.boundary_delta(before,altered)
            altered=copy.deepcopy(after);altered[0]['row']['values']['target_verified_draft_positions']=2
            with self.assertRaisesRegex(ValueError,'counter reset'):audit.boundary_delta(before,altered)
            with self.assertRaisesRegex(ValueError,'population changed'):audit.boundary_delta(before,after[:1])

    def test_stale_foreign_dead_or_changed_worker_records_do_not_qualify(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cid,pins,contract,physical,recorders=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'no longer live'):
                audit.read_replica(root,contract,physical,cid,pins,time.monotonic_ns(),alive=lambda _:False)
            with self.assertRaisesRegex(ValueError,'freshness'):
                audit.read_replica(root,contract,physical,cid,pins,time.monotonic_ns()+2_000_000_000,alive=lambda _:True)
            with self.assertRaisesRegex(ValueError,'source, namespace'):
                audit.read_replica(root,contract,physical,'e'*64,pins,time.monotonic_ns(),alive=lambda _:True)
            changed=copy.deepcopy(physical);changed['workers'][1]['gpu_index']=0
            with self.assertRaisesRegex(ValueError,'population'):
                audit.read_replica(root,contract,changed,cid,pins,time.monotonic_ns(),alive=lambda _:True)
            path=recorders[0].root/'kv-layout.json';layout=json.loads(path.read_text());layout['device_pool_blocks']=1
            runtime.atomic_json(path,layout)
            with self.assertRaisesRegex(ValueError,'resolved layout'):
                audit.read_replica(root,contract,physical,cid,pins,time.monotonic_ns(),alive=lambda _:True)

    def test_raw_eight_worker_window_and_exact_host_projection_reject_gaps_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bindings=[];containers=[];host=[];raw_paths=[]
            stamps=[10_500_000_000+i*500_000_000 for i in range(6)]
            for replica in range(4):
                folder=root/'worker-observation'/('replica-'+str(replica));folder.mkdir(parents=True)
                _,pins,c,physical,recorders=self.fixture(folder);cid=hex(replica+10)[2:]*64
                c.update(replica_id=replica,expected_gpu_ids=[replica*2,replica*2+1])
                runtime.atomic_json(folder/'contract.json',c);digest=hashlib.sha256((folder/'contract.json').read_bytes()).hexdigest()
                bindings.append({'replica_id':replica,'contract':c,'sha256':digest})
                physical['container_id']=cid;containers.append(physical)
                for tp,(worker,recorder) in enumerate(zip(physical['workers'],recorders)):
                    worker.update(gpu_index=replica*2+tp,host_pid=600+replica*2+tp)
                    recorder.identity.update(replica_id=replica,expected_gpu_ids=c['expected_gpu_ids'],
                        container_hostname=cid[:12],contract_sha256=digest)
                    runtime.atomic_json(recorder.root/'identity.json',recorder.identity)
                    rows=[]
                    for index,stamp in enumerate(stamps):
                        row=recorder.snapshot();row.update(started_ns=stamp-1000,completed_ns=stamp)
                        row['values'].update(successful_sample_calls=index+1,successful_verification_steps=index+1,
                            target_verified_draft_positions=3*(index+1));rows.append(row)
                    path=recorder.root/'samples-000000.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows));raw_paths.append(path)
                    for index in range(1,5):
                        while len(host)<index:host.append({'monotonic_ns':stamps[len(host)+1]+10_000_000,
                            'worker_observations':{'status':'physical_worker_observation_sample',
                            'observed_at_ns':stamps[len(host)+1]+5_000_000,'workers':[]}})
                        host[index-1]['worker_observations']['workers'].append({'row':rows[index],
                            'physical_worker':worker,'container_id':cid})
            runtime.atomic_json(root/'worker-observation-binding.json',{'schema':'source-pinned-worker-observer/v1',
                'spec':{'fixture':True},'contracts':bindings,'installed_source':{'source_pins':pins}})
            runtime.atomic_json(root/'profile-binding.json',{'phase':'qualification','profile_sha256':'b'*64,
                'delta':{'worker_observation':{'fixture':True}}})
            runtime.atomic_json(root/'rank-placement.json',{'status':'passed','containers':containers})
            runtime.atomic_json(root/'launch-plan.json',{'commands':[{'host_port':31000+i} for i in range(4)]})
            start=11_050_000_000;end=12_550_000_000
            result=audit.verify_window(root,start,end,host)
            self.assertEqual(result['status'],'worker_observation_window_coverage_verified');self.assertEqual(len(result['workers']),8)
            portable=audit.context_record(root)
            ranks={'http://127.0.0.1:'+str(31000+i)+'/metrics':[0] for i in range(4)}
            native={'ranks':[{'endpoint':endpoint,'rank':'0','draft_tokens':9} for endpoint in ranks]}
            proposed=audit.native_target_population(portable,ranks,native)
            opened=audit.boundary_rows(host[0]['worker_observations'],portable,host[0]['monotonic_ns'])
            closed=audit.boundary_rows(host[-1]['worker_observations'],portable,host[-1]['monotonic_ns'])
            self.assertEqual(audit.reconcile_target_positions(opened,closed,proposed,speculative=True)['target_verified_draft_positions'],36)
            self.assertEqual(audit.native_target_population(portable,ranks),{key:0 for key in proposed})
            altered=copy.deepcopy(host);altered[0]['worker_observations']['workers'].pop()
            with self.assertRaisesRegex(ValueError,'missing or stale'):audit.verify_window(root,start,end,altered)
            altered=copy.deepcopy(host);altered[0]['worker_observations']['workers'][0]['row']['values']['target_verified_draft_positions']+=1
            with self.assertRaisesRegex(ValueError,'projection differs'):audit.verify_window(root,start,end,altered)
            for path in raw_paths:
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                path.write_text(''.join(json.dumps(row)+'\n' for i,row in enumerate(rows) if i not in (2,3)))
            with self.assertRaisesRegex(ValueError,'sample gap'):audit.verify_window(root,start,end,[host[0],host[-1]])


if __name__=='__main__':unittest.main()
