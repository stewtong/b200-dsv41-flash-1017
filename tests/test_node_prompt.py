from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import campaign
import node_prompt


class PromptGenerationTests(unittest.TestCase):
    def test_fixed_batch_boundary_retains_raw_population_and_failed_command_output(self):
        import json,subprocess,tempfile,time
        from types import SimpleNamespace
        import fixed_workload,runtime
        from test_vllm_prompt import FakeTokenizer
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',8,0,i) for i in range(8)]
        cell={'input_tokens':8000,'output_tokens':2048,'suite':'core','concurrency':8,'cache':'cold'}
        session=SimpleNamespace(deadline_ns=time.monotonic_ns()+900_000_000_000,ensure_healthy=lambda:None)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=json.dumps(tasks)+'\n'
            with patch.object(node_prompt.runtime,'command',return_value=out) as invoke:
                self.assertEqual(node_prompt.fixed_batch(session,'a'*64,cell,0,None,600,root),tasks)
            self.assertEqual(invoke.call_count,1);self.assertIn('--batch',invoke.call_args.args[0])
            self.assertEqual((root/'generation-batch-stdout.txt').read_text(),out)
            bad=root/'bad';bad.mkdir()
            with patch.object(node_prompt.runtime,'command',return_value=json.dumps(tasks[:-1])):
                with self.assertRaisesRegex(ValueError,'population'):node_prompt.fixed_batch(session,'a'*64,cell,0,None,600,bad)
            self.assertTrue((bad/'generation-batch-stdout.txt').exists());self.assertFalse((bad/'generation-batch-result.json').exists())
            failed=root/'failed';failed.mkdir()
            error=subprocess.CalledProcessError(1,['fixture'],output='retained partial synthetic output',stderr='fixture failure')
            with patch.object(node_prompt.runtime,'command',side_effect=error),self.assertRaises(subprocess.CalledProcessError):
                node_prompt.fixed_batch(session,'a'*64,cell,0,None,600,failed)
            self.assertEqual((failed/'generation-batch-error-stdout.txt').read_text(),'retained partial synthetic output')
            self.assertTrue((failed/'generation-batch-failure.json').exists())
            expired=root/'expired';expired.mkdir()
            with patch.object(node_prompt.runtime,'command',side_effect=subprocess.TimeoutExpired(['fixture'],600,output=b'partial bytes')),self.assertRaises(subprocess.TimeoutExpired):
                node_prompt.fixed_batch(session,'a'*64,cell,0,None,600,expired)
            self.assertEqual((expired/'generation-batch-error-stdout.txt').read_bytes(),b'partial bytes')

    def test_owned_fixed_batch_persists_every_task_from_one_command(self):
        import json,tempfile,time,fixed_workload,runtime
        from test_vllm_prompt import FakeTokenizer
        tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',8,2,i) for i in range(8)]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-fixed-batch'
            config={'matrix':str(campaign.HERE/'matrix.json'),'cell_id':'core-2xtp4-i8000-c8-cold-o2048','layout':'2xtp4',
                'task_family':'fixed','repetition':2,'run_dir':str(root),'run_id':'fixed-batch','model_dir':d,'image':'fixture',
                'prompt_timeout_seconds':10,'batch_fixed_generation':True,'prompt_batch_timeout_seconds':600}
            for key in ('manifest','checkpoint_receipt','image_qualification'):
                path=Path(d)/(key+'.json');path.write_text('{}');config[key]=str(path)
            class Session:
                def __init__(self,*args,**kwargs):self.deadline_ns=time.monotonic_ns()+1200_000_000_000
                def __enter__(self):root.mkdir();(root/'watchdog-heartbeat.json').write_text('{}');return self
                def __exit__(self,*args):pass
                def ensure_healthy(self):pass
                def preflight(self,ports):pass
                def create_start(self,argv):return 'a'*64
            with patch.object(node_prompt.node_smoke,'validate',return_value=({'image_id':'fixture'}, {},1200)), \
                 patch.object(node_prompt.supervisor,'Supervisor',Session), \
                 patch.object(node_prompt.runtime,'verify_retirement',return_value={'passed':True}), \
                 patch.object(node_prompt.runtime,'inspect_owned',return_value={'Image':'fixture','State':{'Running':True}}), \
                 patch.object(node_prompt.runtime,'command',return_value=json.dumps(tasks)) as invoke:
                node_prompt.run(config)
            self.assertEqual(invoke.call_count,1);self.assertIn('--batch',invoke.call_args.args[0])
            self.assertEqual(json.loads((root/'fixed-tasks.json').read_text()),tasks)
            self.assertEqual(len(list(root.glob('task-*.json'))),8);runtime.verify_evidence_manifest(root)
            self.assertEqual(json.loads((root/'prompt-result.json').read_text())['tasks'],8)

    def test_reference_source_checks_immutable_location_and_full_hash(self):
        import tempfile,json,hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'dsv41-toolkit-20261102T1424Z';(p/'reproduce').mkdir(parents=True)
            (p/'reproduce/workload.py').write_text('fixture')
            (p/'MANIFEST.json').write_text(json.dumps({'reproduce/workload.py':hashlib.sha256(b'fixture').hexdigest()}))
            sha=hashlib.sha256((p/'MANIFEST.json').read_bytes()).hexdigest();cfg={'calibration_reference':{'manifest':str(p/'MANIFEST.json'),'sha256':sha}}
            self.assertEqual(node_prompt.reference_source(cfg,'pilot',{'concurrency':1},allowed_root=d),p/'reproduce')
            with self.assertRaises(ValueError):node_prompt.reference_source(cfg,'quality',{'concurrency':1},allowed_root=d)
            with self.assertRaises(ValueError):node_prompt.reference_source(cfg,'pilot',{'concurrency':8},allowed_root=d)
            (p/'reproduce/workload.py').write_text('changed')
            with self.assertRaises(ValueError):node_prompt.reference_source(cfg,'pilot',{'concurrency':1},allowed_root=d)

    def test_reference_generator_parity_is_retained_and_mismatch_never_passes(self):
        import copy,hashlib,json,tempfile,runtime,vllm_prompt
        from test_vllm_prompt import FakeTokenizer
        task=vllm_prompt.build(FakeTokenizer(),'/model',8000,128,'pilot_smoke',1,0,0)
        for mismatch in (False,True):
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as d:
                base=Path(d);root=base/'dsv41-parity';reference=base/'reference/reproduce';reference.mkdir(parents=True)
                for name in ('vllm_prompt.py','workload.py'):(reference/name).write_text('fixture')
                (reference.parent/'MANIFEST.json').write_text('{}');commands=[]
                cfg={'matrix':str(campaign.HERE/'matrix.json'),'cell_id':'pilot_smoke-2xtp4-i8000-c1-cold-o128','layout':'2xtp4',
                     'run_id':'parity','run_dir':str(root),'model_dir':d,'image':'fixture','prompt_timeout_seconds':10,
                     'calibration_reference':{'manifest':str(reference.parent/'MANIFEST.json'),'sha256':hashlib.sha256(b'{}').hexdigest()}}
                for key in ('manifest','checkpoint_receipt','image_qualification'):
                    f=base/(key+'.json');f.write_text('{}');cfg[key]=str(f)
                class Session:
                    def __init__(self,*a,**k):pass
                    def __enter__(self):root.mkdir();(root/'watchdog-heartbeat.json').write_text('{}');return self
                    def __exit__(self,*a):pass
                    def ensure_healthy(self):pass
                    def preflight(self,ports):pass
                    def create_start(self,argv):commands.append(argv);return 'a'*64
                def command(argv,**kwargs):
                    commands.append(argv);row=copy.deepcopy(task)
                    if mismatch and '/reference/vllm_prompt.py' in argv:row['input_tokens']+=1
                    return json.dumps(row)
                with patch.object(node_prompt.node_smoke,'validate',return_value=({'image_id':'image'}, {},600)), \
                     patch.object(node_prompt.supervisor,'Supervisor',Session),patch.object(node_prompt,'reference_source',return_value=reference), \
                     patch.object(node_prompt.runtime,'verify_retirement',return_value={'passed':True}), \
                     patch.object(node_prompt.runtime,'inspect_owned',return_value={'Image':'image','State':{'Running':True}}), \
                     patch.object(node_prompt.runtime,'command',side_effect=command):
                    if mismatch:
                        with self.assertRaisesRegex(ValueError,'reference generator task differs'):node_prompt.run(cfg)
                    else:node_prompt.run(cfg)
                self.assertTrue(any('/reference/vllm_prompt.py' in argv for argv in commands))
                self.assertTrue(any('dst=/reference,readonly' in value for value in commands[0]))
                self.assertEqual(json.loads((root/'generation-parity-000.json').read_text())['status'],'mismatch' if mismatch else 'matched')
                self.assertTrue((root/'generation-candidate-000.json').exists());self.assertTrue((root/'generation-reference-000.json').exists())
                self.assertEqual((root/'prompt-result.json').exists(),not mismatch)
                if not mismatch:runtime.verify_evidence_manifest(root)

    def test_rejects_unknown_cell_before_start(self):
        config = {'matrix':str(campaign.HERE/'matrix.json'), 'cell_id':'missing'}
        with patch.object(node_prompt.node_smoke, 'validate', return_value=({}, {}, 600)), \
             patch.object(node_prompt.supervisor, 'Supervisor') as supervisor:
            with self.assertRaisesRegex(ValueError, 'unknown cell'):
                node_prompt.run(config)
            supervisor.assert_not_called()

    def test_rejects_warm_without_live_cache_evidence(self):
        config = {'matrix':str(campaign.HERE/'matrix.json'),
                  'cell_id':'pilot_warm-2xtp4-i128000-c1-warm_prefix-o512', 'layout':'2xtp4'}
        with patch.object(node_prompt.node_smoke, 'validate', return_value=({}, {}, 600)), \
             patch.object(node_prompt.supervisor, 'Supervisor') as supervisor:
            with self.assertRaisesRegex(ValueError, 'live cache block size evidence'):
                node_prompt.run(config)
            supervisor.assert_not_called()

    def test_warm_quality_requires_the_same_live_cache_provenance(self):
        cell=next(c for c in campaign.ledger(campaign.HERE/'matrix.json')['cells']
                  if c['suite']=='quality' and c['cache']=='warm_prefix')
        config={'matrix':str(campaign.HERE/'matrix.json'),'cell_id':cell['id'],
                'layout':cell['layout'],'task_family':'quality'}
        with patch.object(node_prompt.node_smoke,'validate',return_value=({}, {},600)), \
             patch.object(node_prompt.supervisor,'Supervisor') as session:
            with self.assertRaisesRegex(ValueError,'live cache block size evidence'):node_prompt.run(config)
            session.assert_not_called()

    def test_writes_valid_tasks_and_seals_evidence(self):
        import hashlib
        import json
        import tempfile
        import workload
        import runtime
        seed = workload.seed_for('pilot_smoke', 8000, 1, 0, 0)
        marker = hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        task = {'id':'s0', 'body':workload.request_body('/model', [{'role':'user','content':marker}], 8000, 128, seed, fixed_output=False),
                'input_tokens':8000, 'output_tokens':128, 'fixed_output':False, 'prompt_sha256':'a'*64,
                'expected_answer':marker, 'seed':seed, 'suite':'pilot_smoke', 'session':0}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'dsv41-prompt'
            config = {'matrix':str(campaign.HERE/'matrix.json'), 'cell_id':'pilot_smoke-2xtp4-i8000-c1-cold-o128',
                      'layout':'2xtp4', 'run_dir':str(root), 'run_id':'prompt', 'model_dir':d,
                      'image':'fixture', 'prompt_timeout_seconds':10}
            for key in ('manifest','checkpoint_receipt','image_qualification'):
                path=Path(d)/(key+'.json');path.write_text('{}');config[key]=str(path)
            class Session:
                def __init__(self, *args, **kwargs): pass
                def __enter__(self):
                    root.mkdir(); (root/'watchdog-heartbeat.json').write_text('{}')
                    return self
                def __exit__(self, *args): pass
                def ensure_healthy(self): pass
                def preflight(self, ports): self.ports = ports
                def create_start(self, argv): self.argv = argv; return 'a'*64
            with patch.object(node_prompt.node_smoke, 'validate', return_value=({'image_id':'image-id'}, {}, 600)), \
                 patch.object(node_prompt.supervisor, 'Supervisor', Session), \
                 patch.object(node_prompt.runtime, 'verify_retirement', return_value={'passed':True}), \
                 patch.object(node_prompt.runtime, 'inspect_owned', return_value={'Image':'image-id','State':{'Running':True}}), \
                 patch.object(node_prompt.runtime, 'command', return_value=json.dumps(task)):
                node_prompt.run(config)
            self.assertEqual(json.loads((root/'pilot-tasks.json').read_text()), [task])
            self.assertEqual(json.loads((root/'prompt-result.json').read_text())['status'], 'passed')
            runtime.verify_evidence_manifest(root)

    def test_quality_generation_uses_same_owned_path_and_preserves_family(self):
        import json,tempfile,runtime,quality_tasks
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return '|'.join(x['role']+':'+x['content'] for x in messages)
            def encode(self,text,add_special_tokens=False):return list(text)
        task=quality_tasks.build(Tokenizer(),8000,512,1,2,0)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-quality-prompt';commands=[]
            config={'matrix':str(campaign.HERE/'matrix.json'),'cell_id':next(c['id'] for c in campaign.ledger(campaign.HERE/'matrix.json')['cells'] if c['suite']=='quality' and c['layout']=='2xtp4' and c['input_tokens']==8000 and c['concurrency']==1 and c['cache']=='cold'),
                    'layout':'2xtp4','task_family':'quality','repetition':2,'run_dir':str(root),
                    'run_id':'quality-prompt','model_dir':d,'image':'fixture','prompt_timeout_seconds':10}
            for key in ('manifest','checkpoint_receipt','image_qualification'):
                f=Path(d)/(key+'.json');f.write_text('{}');config[key]=str(f)
            class Session:
                def __init__(self,*args,**kwargs):pass
                def __enter__(self):
                    root.mkdir();(root/'watchdog-heartbeat.json').write_text('{}');return self
                def __exit__(self,*args):pass
                def ensure_healthy(self):pass
                def preflight(self,ports):pass
                def create_start(self,argv):return 'a'*64
            def command(argv,**kwargs):commands.append(argv);return json.dumps(task)
            with patch.object(node_prompt.node_smoke,'validate',return_value=({'image_id':'fixture-image'}, {},600)), \
                 patch.object(node_prompt.supervisor,'Supervisor',Session), \
                 patch.object(node_prompt.runtime,'verify_retirement',return_value={'passed':True}), \
                 patch.object(node_prompt.runtime,'inspect_owned',return_value={'Image':'fixture-image','State':{'Running':True}}), \
                 patch.object(node_prompt.runtime,'command',side_effect=command):
                node_prompt.run(config)
            self.assertIn('/campaign/quality_tasks.py',commands[0])
            self.assertEqual(commands[0][-6:],['/model','8000','512','1','2','0'])
            self.assertEqual(json.loads((root/'quality-tasks.json').read_text()),[task])
            self.assertFalse((root/'pilot-tasks.json').exists())
            result=json.loads((root/'prompt-result.json').read_text())
            self.assertEqual(result['task_family'],'quality');self.assertEqual(result['repetition'],2)
            runtime.verify_evidence_manifest(root)

    def test_fixed_grid_preparation_preserves_cell_repetition_and_separate_artifact(self):
        import json,tempfile,runtime,fixed_workload
        from test_vllm_prompt import FakeTokenizer
        task=fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,2,0)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-fixed-prompt';commands=[]
            config={'matrix':str(campaign.HERE/'matrix.json'),'cell_id':'core-2xtp4-i8000-c1-cold-o2048','layout':'2xtp4',
                'task_family':'fixed','repetition':2,'run_dir':str(root),'run_id':'fixed-prompt','model_dir':d,'image':'fixture','prompt_timeout_seconds':10}
            for key in ('manifest','checkpoint_receipt','image_qualification'):
                path=Path(d)/(key+'.json');path.write_text('{}');config[key]=str(path)
            class Session:
                def __init__(self,*a,**k):pass
                def __enter__(self):root.mkdir();(root/'watchdog-heartbeat.json').write_text('{}');return self
                def __exit__(self,*a):pass
                def ensure_healthy(self):pass
                def preflight(self,ports):pass
                def create_start(self,argv):return 'a'*64
            def command(argv,**kwargs):commands.append(argv);return json.dumps(task)
            with patch.object(node_prompt.node_smoke,'validate',return_value=({'image_id':'fixture-image'}, {},600)), \
                 patch.object(node_prompt.supervisor,'Supervisor',Session),patch.object(node_prompt.runtime,'verify_retirement',return_value={'passed':True}), \
                 patch.object(node_prompt.runtime,'inspect_owned',return_value={'Image':'fixture-image','State':{'Running':True}}),patch.object(node_prompt.runtime,'command',side_effect=command):
                node_prompt.run(config)
            self.assertIn('/campaign/fixed_workload.py',commands[0]);self.assertEqual(commands[0][-8:],['8000','2048','core','1','2','0','--cache-state','cold'])
            self.assertEqual(json.loads((root/'fixed-tasks.json').read_text()),[task]);self.assertFalse((root/'quality-tasks.json').exists())
            self.assertEqual(json.loads((root/'prompt-result.json').read_text())['task_family'],'fixed');runtime.verify_evidence_manifest(root)
