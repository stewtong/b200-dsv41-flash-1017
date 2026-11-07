"""Excluded warmup provenance/order controls; raw HTTP attribution tests are separate."""
import copy,hashlib,json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_burst,fixed_workload,kernel_warmup,node_fixed_qualification,routing_map,runtime,verify_kernel_warmup
from test_vllm_prompt import FakeTokenizer
from test_fixed_profiles import manifest_fixture


class KernelWarmupTests(unittest.TestCase):
    def fixture(self,base):
        manifest=manifest_fixture();source=base/'source';source.mkdir()
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'anchor',8,0,i,
            prompt_version='fixed-performance-v1') for i in range(8)]
        measured=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,0,0,
            prompt_version='fixed-performance-v1')]
        cell='anchor-2xtp4-i8000-c8-cold-o2048'
        for name,data in [('manifest.json',manifest),('run-config.json',dict(task_family='fixed',
                fixed_prompt_version='fixed-performance-v1',image=manifest['runtime']['image_digest'],cell_id=cell)),
                ('prompt-result.json',dict(status='passed',cleanup='passed',tasks=8,cell_id=cell)),
                ('cleanup-result.json',dict(success=True)),('reconciliation.json',dict(status='complete')),
                ('fixed-tasks.json',tasks)]:runtime.atomic_json(source/name,data)
        runtime.seal_evidence(source)
        cfg=dict(execution_phase='fixed-performance',layout='2xtp4',run_id='fixed-fixture',
            image=manifest['runtime']['image_digest'],expected_ranks={'http://127.0.0.1:31000/metrics':[0],
                'http://127.0.0.1:31001/metrics':[0]},kernel_warmup=dict(tasks_file=str(source/'fixed-tasks.json'),
                source_manifest_sha256=hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest(),request_timeout_seconds=240))
        return manifest,cfg,tasks,measured

    def test_preparation_requires_exact_excluded_source_and_disjoint_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);manifest,cfg,tasks,measured=self.fixture(base)
            prepared=kernel_warmup.prepare(cfg,manifest,measured,base)
            self.assertEqual(prepared['budget_seconds'],260)
            for change in ({'kernel_warmup':None},{'execution_phase':'qualification'},
                    {'kernel_warmup':dict(cfg['kernel_warmup'],source_manifest_sha256='0'*64)},
                    {'kernel_warmup':dict(cfg['kernel_warmup'],request_timeout_seconds=1801)}):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    kernel_warmup.prepare(dict(cfg,**change),manifest,measured,base)
            with self.assertRaisesRegex(ValueError,'overlaps'):
                kernel_warmup.prepare(cfg,manifest,tasks,base)
            with self.assertRaisesRegex(ValueError,'shared block'):
                node_fixed_qualification.run(cfg)

    def test_independent_audit_rejects_resealed_overlap_boot_order_and_score_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);manifest,cfg,tasks,measured=self.fixture(base)
            import shutil
            root=base/'kernel-warmup';root.mkdir();shutil.copytree(base/'source',root/'input-source')
            runtime.atomic_json(base/'execution-state.json',{'driver_identity':{'boot_id':'fixture-boot'}})
            records=[]
            for i in range(128):
                for rep in (0,1):records.append(dict(session_id='grid'+str(i),request_id='grid'+str(i)+'-r'+str(rep),
                    backend_port=31000+i%2,rank_headers=[]))
            mapping=routing_map.build(records,[31000,31001],1);placement=routing_map.select(mapping,8,0)
            capture=root/'capture';folder=capture/'requests/measured';folder.mkdir(parents=True)
            namespace=fixed_burst.namespace(cfg['run_id'],'excluded-kernel-warmup',0,manifest['runtime']['profile_hashes']['2xtp4'])
            runtime.atomic_json(folder/'fixed-inputs.json',dict(tasks=tasks,request_namespace=namespace,clock_id='fixture-boot:'+cfg['run_id']))
            runtime.atomic_json(capture/'attribution-contract.json',dict(expected_ranks=cfg['expected_ranks']))
            for i in range(8):runtime.atomic_json(folder/('s'+str(i)+'.json'),dict(start_ns=100+i,end_ns=200+i))
            outcome=dict(schema='excluded-kernel-warmup/v1',status='captured_cleanup_pending',
                source_seal=cfg['kernel_warmup']['source_manifest_sha256'],mapping_sha256=mapping['mapping_sha256'],
                placement=placement,started_ns=90,completed_ns=210,scored_requests=0,
                scope='one excluded8K/c8 kernel warmup; no claim that every later context/graph shape is warmed',
                cold_cache_label='cache-busted cold; unique early system namespaces, no verified cache flush or reset')
            runtime.atomic_json(root/'warmup-result.json',outcome);runtime.seal_evidence(root)
            attributed=dict(session_routes={r['session_header']:[r['endpoint'],r['rank']] for r in placement})
            # Only raw HTTP/native accounting is stubbed here; its existing integration
            # suite exercises wire, cache and route tampering independently.
            with patch.object(verify_kernel_warmup.verify_fixed_attribution,'verify',return_value=attributed):
                checked=verify_kernel_warmup.verify(root,measured,manifest,cfg,mapping,220)
                self.assertEqual(checked['scored_requests'],0)
                with self.assertRaisesRegex(ValueError,'timing'):
                    verify_kernel_warmup.verify(root,measured,manifest,cfg,mapping,205)
                with self.assertRaisesRegex(ValueError,'overlap'):
                    verify_kernel_warmup.verify(root,tasks,manifest,cfg,mapping,220)
                changed=dict(outcome,scored_requests=8);runtime.atomic_json(root/'warmup-result.json',changed);runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'exclusion'):
                    verify_kernel_warmup.verify(root,measured,manifest,cfg,mapping,220)
                runtime.atomic_json(root/'warmup-result.json',outcome);runtime.seal_evidence(root)
                runtime.atomic_json(base/'execution-state.json',{'driver_identity':{'boot_id':'other-boot'}})
                with self.assertRaisesRegex(ValueError,'another run'):
                    verify_kernel_warmup.verify(root,measured,manifest,cfg,mapping,220)


if __name__=='__main__':unittest.main()
