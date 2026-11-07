import copy
import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,fixed_workload,node_fixed_qualification as qualified,profile_registry,runtime
from test_vllm_prompt import FakeTokenizer


class FixedQualificationTests(unittest.TestCase):
    def fixture(self,root):
        manifest=campaign.ledger(Path(campaign.__file__).with_name('matrix.json'));manifest['plan_sha256']='a'*64
        manifest['runtime'].update(image_digest='example/image@sha256:'+'b'*64,toolkit_manifest_sha256='c'*64)
        cells=[c['id'] for c in manifest['cells'] if c['suite']=='tuning_trial' and c['layout']=='2xdep4' and
            c['parameters']['lever']=='dspark_static' and c['parameters']['input']==8000 and c['parameters']['concurrency']==1]
        args=dict(hypothesis='Fixture declares matched static speculation and a no-speculation control.',deciding_cells=cells,protected_tail='p95_ttft_seconds')
        control=profile_registry.register('control','2xdep4',{},manifest,**args)
        candidate=profile_registry.register('candidate','2xdep4',{'speculative_config':{'method':'dspark','num_speculative_tokens':5,'rejection_sample_method':'block'},'per_request_spec_decode_metrics':'detailed'},manifest,**args)
        manifest['runtime']['qualification_profiles']={'control':control,'candidate':candidate}
        config={'execution_phase':'qualification','profile_id':'candidate','trial_profile_id':'candidate','layout':'2xdep4',
            'cell_id':next(c for c in cells if 'challenger' in c),'repetition':0,'request_timeout_seconds':1800,
            'matrix':str(Path(campaign.__file__).with_name('matrix.json')),'image':manifest['runtime']['image_digest'],
            'expected_ranks':{'http://127.0.0.1:31000/metrics':[0,1,2,3],'http://127.0.0.1:31001/metrics':[0,1,2,3]},
            'run_id':'fixed-fixture','run_dir':str(root/'dsv41-fixed-fixture'),'deadline_utc':'2026-11-03T00:00:00+00:00'}
        for key,name,value in [('manifest','manifest.json',manifest),('checkpoint_receipt','checkpoint-receipt.json',{'synthetic':True}),
            ('image_qualification','image-qualification.json',{'synthetic':True})]:
            path=root/name;path.write_text(json.dumps(value));config[key]=str(path)
        source=root/'dsv41-fixed-input-fixture';source.mkdir()
        for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
            (source/name).write_bytes(Path(config[key]).read_bytes())
        source_cell='core-2xdep4-i8000-c1-cold-o2048'
        (source/'run-config.json').write_text(json.dumps({'cell_id':source_cell,'layout':'2xdep4','image':config['image'],'task_family':'fixed','repetition':0}))
        (source/'prompt-result.json').write_text(json.dumps({'status':'passed','cleanup':'passed','task_family':'fixed','cell_id':source_cell,'repetition':0,'tasks':1}))
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,0,0)]
        (source/'fixed-tasks.json').write_text(json.dumps(tasks));runtime.seal_evidence(source);config['tasks_file']=str(source/'fixed-tasks.json')
        return config,source

    def test_preparation_binds_trial_arm_shape_repetition_and_real_input_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config,source=self.fixture(root);prepared=qualified.prepare(config,root)
            self.assertEqual(prepared['tasks'][0]['target_input_tokens'],8000)
            self.assertTrue(prepared['attempt_cell_id'].startswith('fixedqual-'))
            control=copy.deepcopy(config);control['cell_id']=config['cell_id'].replace('challenger','control_before')
            with self.assertRaisesRegex(ValueError,'trial arm'):qualified.prepare(control,root)
            control['profile_id']='control';qualified.prepare(control,root)
            other=next(c['id'] for c in prepared['manifest']['cells'] if c['suite']=='tuning_trial' and c['layout']=='2xdep4' and
                c['parameters']=={'lever':'dspark_static','arm':'challenger','input':128000,'concurrency':1})
            for fields in ({'execution_phase':'scored'},{'repetition':1},{'repetition':True},{'tasks_file':str(source/'invented.json')},
                           {'cell_id':other},{'trial_profile_id':'missing'},{'run_id':'x'*49}):
                self.assertTrue(any(config.get(k)!=v for k,v in fields.items()),'negative fixture must change its input')
                with self.subTest(fields=fields),self.assertRaises(ValueError):qualified.prepare(dict(config,**fields),root)
            path=source/'prompt-result.json';v=json.loads(path.read_text());v['task_family']='pilot';path.write_text(json.dumps(v));runtime.seal_evidence(source)
            with self.assertRaisesRegex(ValueError,'generation'):qualified.prepare(config,root)

    def test_method_does_not_change_with_new_run_deadline_or_unrelated_toolkit_pin(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config,_=self.fixture(root);prepared=qualified.prepare(config,root)
            digest=qualified.method_fingerprint(config,prepared)
            changed=dict(config,run_id='another-run',run_dir=str(root/'dsv41-another-run'),deadline_utc='2026-11-04T00:00:00Z')
            self.assertEqual(digest,qualified.method_fingerprint(changed,prepared))
            other=copy.deepcopy(prepared);other['manifest']['runtime']['toolkit_manifest_sha256']='d'*64
            self.assertEqual(digest,qualified.method_fingerprint(changed,other))
            changed['hypothesis']='A different observed runtime failure changes the recovery hypothesis.'
            self.assertNotEqual(digest,qualified.method_fingerprint(changed,prepared))
            read=Path.read_bytes
            with patch.object(Path,'read_bytes',lambda path:read(path)+(b'changed pinned parser' if path.name=='speculation_evidence.py' else b'')):
                self.assertNotEqual(digest,qualified.method_fingerprint(config,prepared))
            with patch.object(Path,'read_bytes',lambda path:read(path)+(b'unrelated panel edit' if path.name=='gsm8k.py' else b'')):
                self.assertEqual(digest,qualified.method_fingerprint(config,prepared))


if __name__=='__main__':unittest.main()
