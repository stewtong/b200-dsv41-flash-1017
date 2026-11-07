import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import kv_budget
import runtime
import json
import tempfile
import verify_pilot
import node_pilot_block


def line(rank=None,tokens='2,097,152',concurrency='2.00'):
    name='EngineCore' if rank is None else 'EngineCore_DP'+str(rank)
    return '('+name+' pid=42) GPU KV cache size: '+tokens+' tokens, Maximum concurrency for 1,048,576 tokens per request: '+concurrency+'x'


class KvBudgetTests(unittest.TestCase):
    def test_frozen_phase_preserves_raw_kv_verification_and_rejects_profile_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cid='a'*64
            cfg=dict(run_id='fixed-fixture',layout='dep8',image='fixture',
                     execution_phase='fixed-performance',profile_id='fixed-nospec-dep8')
            record=dict(schema='frozen-fixed-profile/v1',id=cfg['profile_id'],layout='dep8',
                        status='frozen_fixed_configuration',profile_sha256='f'*64)
            manifest={'runtime':{'fixed_profiles':{'dep8':record}}}
            runtime.atomic_json(root/'run-config.json',cfg)
            runtime.atomic_json(root/'manifest.json',manifest)
            runtime.atomic_json(root/'execution-state.json',dict(run_id=cfg['run_id'],container_ids=[cid]))
            runtime.atomic_json(root/'launch-plan.json',{'commands':[{'gpu_ids':list(range(8))}]})
            (root/('rank-startup-'+cid+'.log')).write_text('\n'.join(line(i) for i in range(8)))
            kv_budget.record(root,cfg)
            verify_pilot.verify_kv_budget(root,cfg)
            manifest['runtime']['fixed_profiles']['dep8']['profile_sha256']='e'*64
            runtime.atomic_json(root/'manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'receipt differs'):verify_pilot.verify_kv_budget(root,cfg)

    def test_group_aware_capacity_and_expected_engine_population(self):
        rows=kv_budget.parse('\n'.join(line(i) for i in range(4)),[0,1,2,3])
        self.assertEqual([r['logical_rank'] for r in rows],[0,1,2,3])
        self.assertEqual(rows[0]['equivalent_full_context_tokens'],2_097_152)
        self.assertEqual(rows[0]['max_concurrency_at_limit'],2)
        self.assertEqual(kv_budget.parse(line(),[0])[0]['logical_rank'],0)

    def test_missing_duplicate_malformed_or_inconsistent_pool_never_passes(self):
        for text in (line(0),line(0)+'\n'+line(0),line(0,tokens='2,,097,152')+'\n'+line(1),
                     line(0,concurrency='3.00')+'\n'+line(1)):
            with self.subTest(text=text),self.assertRaises(ValueError):kv_budget.parse(text,[0,1])

    def test_two_percent_drift_is_a_pre_dispatch_failure_for_the_same_profile(self):
        before={'schema':'group-aware-kv-capacity/v1','layout':'2xtp4','image':'fixture',
                'profile_sha256':'a'*64,'installed_source_sha256':kv_budget.SOURCE_SHA256,
                'replicas':[{'gpu_ids':[0,1,2,3],'engines':[{'logical_rank':0,'pool':'GPU',
                    'max_request_tokens':1048576,'equivalent_full_context_tokens':100000}]}]}
        after=copy.deepcopy(before);after['replicas'][0]['engines'][0]['equivalent_full_context_tokens']=102000
        self.assertEqual(kv_budget.compare(after,before)['status'],'within_two_percent')
        after['replicas'][0]['engines'][0]['equivalent_full_context_tokens']=102001
        with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'over two percent'):kv_budget.compare(after,before)
        after=copy.deepcopy(before);after['profile_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'exact starting profile'):kv_budget.compare(after,before)
        after=copy.deepcopy(before);after['replicas'][0]['gpu_ids']=[4,5,6,7]
        with self.assertRaisesRegex(ValueError,'physical coverage'):kv_budget.compare(after,before)

    def test_portable_raw_baseline_excerpt_and_independent_verifier_reject_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            def fixture(name):
                root=Path(d)/name;root.mkdir();cid='a'*64
                config={'run_id':name,'layout':'tp8','image':'fixture'}
                runtime.atomic_json(root/'run-config.json',config)
                runtime.atomic_json(root/'execution-state.json',{'run_id':name,'container_ids':[cid]})
                runtime.atomic_json(root/'launch-plan.json',{'commands':[{'gpu_ids':list(range(8))}]})
                runtime.atomic_json(root/'manifest.json',{'runtime':{'starting_profiles':{'tp8':'f'*64}}})
                (root/('rank-startup-'+cid+'.log')).write_text(line()+'\n')
                return root,config
            old,_=fixture('old');runtime.seal_evidence(old)
            root,config=fixture('current');config['kv_baseline_evidence']=str(old)
            manifest=json.loads((root/'manifest.json').read_text())
            kv_budget.validate_baseline(config,manifest)
            wrong=copy.deepcopy(manifest);wrong['runtime']['starting_profiles']['tp8']='e'*64
            with self.assertRaisesRegex(ValueError,'before startup'):kv_budget.validate_baseline(config,wrong)
            kv_budget.record(root,config);verify_pilot.verify_kv_budget(root,config)
            budget=json.loads((root/'kv-budget.json').read_text());budget['replicas'][0]['engines'][0]['equivalent_full_context_tokens']+=1
            runtime.atomic_json(root/'kv-budget.json',budget)
            with self.assertRaisesRegex(ValueError,'receipt differs'):verify_pilot.verify_kv_budget(root,config)
            budget['replicas'][0]['engines'][0]['equivalent_full_context_tokens']-=1;runtime.atomic_json(root/'kv-budget.json',budget)
            path=root/'kv-baseline-source'/('rank-startup-'+'a'*64+'.log');path.write_text(path.read_text().replace('2,097,152','2,097,153'))
            with self.assertRaisesRegex(ValueError,'integrity differs'):verify_pilot.verify_kv_budget(root,config)

    def test_shared_cell_projection_keeps_the_complete_raw_baseline_for_independent_verification(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);owner={'pid':123,'start_ticks':1,'boot_id':'fixture'};cid='a'*64
            def fixture(name):
                root=base/name;root.mkdir();cfg={'run_id':name,'layout':'tp8','image':'fixture'}
                runtime.atomic_json(root/'run-config.json',cfg)
                runtime.atomic_json(root/'execution-state.json',{'run_id':name,'container_ids':[cid],'driver_identity':owner})
                runtime.atomic_json(root/'launch-plan.json',{'commands':[{'gpu_ids':list(range(8))}]})
                runtime.atomic_json(root/'manifest.json',{'runtime':{'starting_profiles':{'tp8':'f'*64}}})
                (root/('rank-startup-'+cid+'.log')).write_text(line()+'\n');return root,cfg
            old,_=fixture('old');runtime.seal_evidence(old)
            root,cfg=fixture('current');cfg['kv_baseline_evidence']=str(old);runtime.atomic_json(root/'run-config.json',cfg)
            kv_budget.record(root,cfg)
            cell=root/'cells/cell';cell.mkdir(parents=True);runtime.atomic_json(cell/'run-config.json',cfg)
            hashes=node_pilot_block.project_shared(root,cell,owner)
            self.assertIn('kv-baseline-source/SOURCE_SHA256SUMS',hashes)
            verify_pilot.verify_kv_budget(cell,cfg)
            copied=cell/'kv-baseline-source'/('rank-startup-'+cid+'.log');copied.write_text(copied.read_text().replace('2,097,152','2,097,153'))
            with self.assertRaisesRegex(ValueError,'integrity differs'):verify_pilot.verify_kv_budget(cell,cfg)
