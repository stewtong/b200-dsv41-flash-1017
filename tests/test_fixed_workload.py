from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_workload,verify_fixed_task
from test_vllm_prompt import FakeTokenizer


class FixedWorkloadTests(unittest.TestCase):
    def test_actual_batch_cli_is_byte_equivalent_and_loads_tokenizer_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);module=root/'vllm/tokenizers';module.mkdir(parents=True)
            (root/'vllm/__init__.py').write_text('');(module/'__init__.py').write_text('')
            (module/'deepseek_v41.py').write_text('''import os
class DeepseekV41Tokenizer:
 @classmethod
 def from_pretrained(cls,*args,**kwargs):
  with open(os.environ['FIXTURE_LOAD_LOG'],'a') as f:f.write('loaded\\n')
  return cls()
 def apply_chat_template(self,messages,**kwargs):return '|'.join(x['role']+':'+x['content'] for x in messages)
 def encode(self,text,add_special_tokens=False):return list(text)
''')
            counter=root/'loads.txt';env=dict(os.environ,PYTHONPATH=str(root),FIXTURE_LOAD_LOG=str(counter))
            base=[sys.executable,'-B',fixed_workload.__file__,'/fixture','10000','2048','core','8','1']
            for cache in ('cold','warm_prefix'):
                options=['--cache-state',cache]+(['--block-size','128'] if cache=='warm_prefix' else [])
                counter.write_text('');batch=subprocess.run(base+['0',*options,'--batch'],env=env,capture_output=True,text=True,timeout=10)
                self.assertEqual(batch.returncode,0,batch.stderr);self.assertEqual(counter.read_text().splitlines(),['loaded'])
                tasks=json.loads(batch.stdout);self.assertEqual(len(tasks),8)
                for i,task in enumerate(tasks):
                    serial=subprocess.run(base+[str(i),*options],env=env,capture_output=True,text=True,timeout=10)
                    self.assertEqual(serial.returncode,0,serial.stderr);self.assertEqual(task,json.loads(serial.stdout))
            counter.write_text('')
            bad=subprocess.run(base+['1','--batch'],env=env,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(bad.returncode,0);self.assertFalse(bad.stdout.strip());self.assertEqual(counter.read_text(),'')

    def test_instruction_repair_is_versioned_and_legacy_tasks_still_verify(self):
        tok=FakeTokenizer()
        for cache in ('cold','warm_prefix'):
            args=(tok,'/model',10000,2048,'core',1,0,0,cache,128 if cache=='warm_prefix' else None)
            old=fixed_workload.build(*args,prompt_version='fixed-cache-v2')
            new=fixed_workload.build(*args)
            previous=fixed_workload.build(*args,prompt_version='fixed-cache-v2-ascii-instruction-v1')
            for task in (old,previous,new):verify_fixed_task.verify(task,tok)
            self.assertEqual(old['seed'],new['seed']);self.assertEqual(old['output_tokens'],new['output_tokens'])
            self.assertNotEqual(old['session_prefix'],new['session_prefix'])
            self.assertIn('code exactly, with no label or introduction.',new['body']['messages'][-1]['content'])
            self.assertIn('ASCII only: straight quotes and hyphens.',new['body']['messages'][-1]['content'])
            self.assertLessEqual(len(fixed_workload.ASCII_ENDING),175)
            self.assertNotIn('Use ASCII only:',old['body']['messages'][-1]['content'])
            self.assertEqual(new['body']['min_tokens'],new['body']['max_tokens'])
            wrong=copy.deepcopy(new);wrong['body']['messages'][-1]['content']=wrong['body']['messages'][-1]['content'].replace('ASCII only:', 'Unicode is allowed:')
            with self.assertRaises(ValueError):verify_fixed_task.verify(wrong,tok)

    def test_greedy_anchor_changes_only_sampling_in_the_request(self):
        tok=FakeTokenizer()
        for cache in ('cold','warm_prefix'):
            a=fixed_workload.build(tok,'/model',10000,2048,'anchor',8,1,2,cache,128 if cache=='warm_prefix' else None)
            b=fixed_workload.build(tok,'/model',10000,2048,'anchor_greedy',8,1,2,cache,128 if cache=='warm_prefix' else None)
            self.assertEqual(a['seed'],b['seed']);self.assertEqual(a['prompt_sha256'],b['prompt_sha256'])
            self.assertEqual(a['body'],dict(b['body'],temperature=1.0))
            self.assertEqual(b['reuse_requires'],'fresh engine or independently verified prefix-cache reset')
            for task in (a,b):verify_fixed_task.verify(task,tok)

    def test_output_caps_have_distinct_early_prefixes_and_legacy_input_remains_auditable(self):
        import hashlib
        tok=FakeTokenizer()
        for cache in ('cold','warm_prefix'):
            tasks=[fixed_workload.build(tok,'/model',10000,cap,'output_sensitivity',1,0,0,cache,128 if cache=='warm_prefix' else None) for cap in (512,8192)]
            self.assertEqual(tasks[0]['seed'],tasks[1]['seed'])
            self.assertNotEqual(tasks[0]['body']['messages'][0],tasks[1]['body']['messages'][0])
            for t in tasks:self.assertEqual(verify_fixed_task.verify(t,tok)['status'],'tokenizer_input_verified')
        task=fixed_workload.build(tok,'/model',10000,2048,'core',1,0,0,prompt_version='fixed-cache-v2')
        task['prompt_version']='fixed-cache-v1'
        task['session_prefix']='session_'+hashlib.sha256(json.dumps(['fixed-cache-v1','core',10000,1,0,0,'cold'],separators=(',',':')).encode()).hexdigest()[:20]
        task['body']['messages'][0]['content']='Follow the user instructions precisely. Reply in English using only ASCII characters. '+task['session_prefix']+'.'
        rendered=tok.apply_chat_template(task['body']['messages']);task['prompt_sha256']=hashlib.sha256(rendered.encode()).hexdigest();task['input_tokens']=len(tok.encode(rendered))
        self.assertEqual(verify_fixed_task.verify(task,tok)['status'],'tokenizer_input_verified')

    def test_standalone_input_proof_rejects_resealed_contract_and_prefix_changes(self):
        tok=FakeTokenizer()
        for suite,cap in (('core',2048),('output_sensitivity',8192),('anchor_greedy',2048)):
            with self.subTest(suite=suite):
                cold=fixed_workload.build(tok,'/model',10000,cap,suite,8,1,2)
                warm=fixed_workload.build(tok,'/model',10000,cap,suite,8,1,2,'warm_prefix',128)
                self.assertEqual(cold['seed'],warm['seed']);self.assertNotEqual(cold['session_prefix'],warm['session_prefix'])
                self.assertEqual(warm,fixed_workload.build(tok,'/model',10000,cap,suite,8,1,2,'warm_prefix',128))
                for task in (cold,warm):self.assertEqual(verify_fixed_task.verify(task,tok)['status'],'tokenizer_input_verified')
                self.assertEqual(verify_fixed_task.verify(warm)['status'],'input_contract_only')
                for field,value in (('prefix_token_sha256','0'*64),('seed',0),('concurrency',4)):
                    wrong=copy.deepcopy(warm);wrong[field]=value
                    with self.assertRaises(ValueError):verify_fixed_task.verify(wrong,tok)
                wrong=copy.deepcopy(cold);wrong['body']['ignore_eos']=False
                with self.assertRaises(ValueError):verify_fixed_task.verify(wrong,tok)
                for field,value in (('temperature',False),('stream_options',{'include_usage':1}),('chat_template_kwargs',{'thinking':0,'reasoning_effort':50.0})):
                    wrong=copy.deepcopy(cold);wrong['body'][field]=value
                    with self.assertRaises(ValueError):verify_fixed_task.verify(wrong,tok)
                wrong=copy.deepcopy(warm);wrong['body']['messages'][2]['content']='wrong'
                rendered=tok.apply_chat_template(wrong['body']['messages'])
                import hashlib
                wrong['input_tokens']=len(tok.encode(rendered));wrong['prompt_sha256']=hashlib.sha256(rendered.encode()).hexdigest()
                with self.assertRaises(ValueError):verify_fixed_task.verify(wrong,tok)

    def test_actual_cli_uses_tokenizer_and_validates_before_returning_task(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);module=root/'vllm/tokenizers';module.mkdir(parents=True)
            (root/'vllm/__init__.py').write_text('');(module/'__init__.py').write_text('')
            (module/'deepseek_v41.py').write_text('''class DeepseekV41Tokenizer:
 @classmethod
 def from_pretrained(cls,*args,**kwargs):return cls()
 def apply_chat_template(self,messages,**kwargs):return '|'.join(x['role']+':'+x['content'] for x in messages)
 def encode(self,text,add_special_tokens=False):return list(text)
''')
            env=dict(os.environ,PYTHONPATH=str(root))
            argv=[sys.executable,'-B',str(Path(fixed_workload.__file__)),'/fixture','10000','2048','core','1','0','0','--cache-state','warm_prefix','--block-size','128']
            result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr);task=json.loads(result.stdout)
            self.assertEqual(verify_fixed_task.verify(task,FakeTokenizer())['status'],'tokenizer_input_verified')
            bad=subprocess.run(argv[:-1]+['0'],env=env,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(bad.returncode,0);self.assertFalse(bad.stdout.strip())
