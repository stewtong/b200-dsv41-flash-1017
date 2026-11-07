"""Actual recorder files and non-mutating adapter behavior; no GPU claim."""
import copy,hashlib,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import measurement_worker_extension as adapter
import worker_observations as observations


def config():
    spec=SimpleNamespace(block_size=128,page_size_bytes=8192,dtype='fp8',dcp_sharded=False,is_draft=False,
        cache_role='indexer',cache_dtype_str='mxfp4',is_index_group_leader=True)
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['model.indexer'],kv_cache_spec=spec,
        role='default',host_resident=False)],kv_cache_tensors=[SimpleNamespace(size=819200,layers=['model.indexer'],host_resident=False)],
        num_blocks=100,kv_cache_layout='LBH',hisparse_host_num_blocks=None,hisparse_host_block_stride=None,
        hisparse_shared_host_pool=False,kv_tp_replicas=1)


class WorkerObservationTests(unittest.TestCase):
    def test_actual_imported_recorder_and_runtime_bytes_must_match_registered_sources(self):
        root=Path(adapter.__file__).parent
        c={'recorder_sha256':hashlib.sha256(Path(observations.__file__).read_bytes()).hexdigest(),
            'runtime_sha256':hashlib.sha256(Path(observations.runtime.__file__).read_bytes()).hexdigest()}
        adapter.verify_imported_modules(c,root)
        with self.assertRaisesRegex(ValueError,'imported module differs'):
            adapter.verify_imported_modules(dict(c,runtime_sha256='f'*64),root)
        with patch.object(observations,'__file__',str(root/'foreign-module.py')):
            with self.assertRaisesRegex(ValueError,'imported module differs'):adapter.verify_imported_modules(c,root)

    def test_exact_contract_rejects_changed_content_geometry_and_expired_bound(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);extension_sha='e'*64
            value={'schema':'owned-worker-observation/v1','run_id':'fixture','replica_id':0,
                'boot_id':'public-fixture-boot','deadline_ns':time.monotonic_ns()+1_000_000_000,
                'extension_sha256':extension_sha,'profile_sha256':'b'*64,'toolkit_sha256':'c'*64,
                'recorder_sha256':'d'*64,'runtime_sha256':'f'*64,
                'tp_size':2,'dp_size':1,'expected_gpu_ids':[0,1]}
            path=root/'contract.json'
            def write(v):
                path.write_text(json.dumps(v));return hashlib.sha256(path.read_bytes()).hexdigest()
            digest=write(value);self.assertEqual(adapter.contract(root,digest,extension_sha),value)
            with self.assertRaisesRegex(ValueError,'content differs'):adapter.contract(root,'f'*64,extension_sha)
            for changed in (dict(value,expected_gpu_ids=[0,0]),dict(value,deadline_ns=time.monotonic_ns()-1),
                            dict(value,extension_sha256='d'*64),dict(value,tp_size=4)):
                with self.subTest(changed=changed),self.assertRaisesRegex(ValueError,'contract identity'):
                    adapter.contract(root,write(changed),extension_sha)
            self.assertEqual([p.name for p in root.iterdir()],['contract.json'])

    def test_success_only_counters_keep_outputs_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            recorder=observations.Recorder(Path(d)/'worker',{'tp_rank':0,'dp_rank':0},time.monotonic_ns()+10_000_000_000)
            returned=object();hidden=object();grammar=object();batch=SimpleNamespace(num_draft_tokens=5,num_reqs=2)
            def original(h,b,g):
                self.assertIs(h,hidden);self.assertIs(b,batch);self.assertIs(g,grammar);return returned
            self.assertIs(recorder.observe_sample(original,hidden,batch,grammar),returned)
            def failed(*args):raise RuntimeError('fixture')
            with self.assertRaisesRegex(RuntimeError,'fixture'):recorder.observe_sample(failed,hidden,batch,grammar)
            self.assertEqual(recorder.snapshot()['values']['target_verified_draft_positions'],5)
            self.assertEqual(recorder.snapshot()['values']['successful_verification_steps'],1)
            self.assertEqual(recorder.snapshot()['values']['successful_grammar_sample_calls'],1)
            self.assertEqual(batch.num_draft_tokens,5)
            batch.num_draft_tokens=0;recorder.observe_sample(original,hidden,batch,grammar)
            self.assertEqual(recorder.snapshot()['values']['successful_sample_calls'],2)
            recorder.error='fixture'
            with self.assertRaisesRegex(RuntimeError,'persistence failed'):recorder.observe_sample(original,hidden,batch,grammar)

    def test_actual_emitter_preserves_source_identity_layout_and_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'worker';recorder=observations.Recorder(root,{'tp_rank':0,'dp_rank':1},time.monotonic_ns()+10_000_000_000,period=.05)
            value=recorder.record_layout(config());self.assertEqual(value['groups'][0]['layer_specs']['model.indexer']['cache_role'],'indexer')
            recorder.start()
            try:
                deadline=time.monotonic()+2
                while not (root/'live.json').exists():
                    if time.monotonic()>deadline:raise AssertionError('recorder did not publish')
                    time.sleep(.01)
                live=json.loads((root/'live.json').read_text());self.assertEqual(live['identity'],{'tp_rank':0,'dp_rank':1})
                self.assertLessEqual(live['started_ns'],live['completed_ns'])
                raw=json.dumps(json.loads((root/'kv-layout.json').read_text()),sort_keys=True,separators=(',',':')).encode()
                self.assertEqual(live['values']['kv_layout_canonical_sha256'],hashlib.sha256(raw).hexdigest())
            finally:recorder.close()
            self.assertFalse(json.loads((root/'closed.json').read_text())['failed'])
            self.assertTrue(list(root.glob('samples-*.jsonl')))
            changed=config();changed.num_blocks=101
            with self.assertRaisesRegex(RuntimeError,'layout changed'):recorder.record_layout(changed)

    def test_hooks_keep_original_signatures_results_and_actual_resolved_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            recorder=observations.Recorder(Path(d)/'worker',{},time.monotonic_ns()+10_000_000_000)
            returned=object()
            class Runner:
                def sample(self,hidden_states,input_batch,grammar_output):return returned
                def initialize_kv_cache(self,kv_cache_config,is_profiling=False,kv_cache_allocation_context=None):
                    self.kv_cache_config=kv_cache_config;return returned
            runner=Runner();adapter.install_hooks(SimpleNamespace(model_runner=runner),recorder)
            self.assertIs(runner.initialize_kv_cache(config(),is_profiling=True),returned)
            self.assertIsNone(recorder.layout_sha256)
            self.assertIs(runner.initialize_kv_cache(config()),returned);self.assertIsNotNone(recorder.layout_sha256)
            self.assertIs(runner.sample(None,SimpleNamespace(num_draft_tokens=3,num_reqs=1),None),returned)
            self.assertEqual(recorder.target_verified_positions,3)

    def test_missing_or_changed_pinned_sources_fail_before_adapter_installation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'worker.py').write_text('# public fixture\n');digest=hashlib.sha256((root/'worker.py').read_bytes()).hexdigest()
            self.assertEqual(adapter.verify_sources(root,{'worker.py':digest}),{'worker.py':digest})
            (root/'worker.py').write_text('# changed fixture\n')
            with self.assertRaisesRegex(ValueError,'source mismatch'):adapter.verify_sources(root,{'worker.py':digest})
            with self.assertRaises(FileNotFoundError):adapter.verify_sources(root,{'missing.py':digest})
            self.assertEqual(sorted(p.name for p in root.iterdir()),['worker.py'])


if __name__=='__main__':unittest.main()
