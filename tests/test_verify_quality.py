import json
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import quality_tasks
import verify_quality


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return '|'.join(x['role']+':'+x['content'] for x in messages)
    def encode(self,text,add_special_tokens=False):return list(text)


class QualityVerificationTests(unittest.TestCase):
    def test_scoring_includes_failures_and_budget_exhaustion(self):
        for rep in range(3):
            task=quality_tasks.build(Tokenizer(),1000,512,1,rep,0)
            answer=json.dumps(task['expected_answer']) if isinstance(task['expected_answer'],list) else task['expected_answer']
            raw={'complete':True,'http_status':200,'request_id':'s0','events':[{'kind':'event','data':{'choices':[{'delta':{'content':answer},'finish_reason':'stop'}]}},
                 {'kind':'event','data':{'choices':[],'usage':{'prompt_tokens':task['input_tokens'],'completion_tokens':1}}},{'kind':'done','data':None}]}
            self.assertTrue(verify_quality.evaluate(task,raw,rep,1,1000,Tokenizer())['correct'])
            self.assertEqual(verify_quality.evaluate(task,raw,rep,1,1000,Tokenizer())['depth_verification'],'provided_tokenizer_recomputed')
            raw['events'][0]['data']['choices'][0]['finish_reason']='length'
            self.assertEqual(verify_quality.evaluate(task,raw,rep,1,1000)['reason'],'budget_exhausted')
            raw['complete']=False
            self.assertEqual(verify_quality.evaluate(task,raw,rep,1,1000)['reason'],'request_failed')

    def test_rejects_forged_depth_with_tokenizer(self):
        task=quality_tasks.build(Tokenizer(),1000,512,1,0,0);task['observed_depths'][0]+=.001
        with self.assertRaisesRegex(ValueError,'observed depth mismatch'):
            verify_quality.evaluate(task,{},0,1,1000,Tokenizer())

    def test_rejects_changed_context_record(self):
        task=quality_tasks.build(Tokenizer(),1000,512,1,0,0)
        task['body']['messages'][1]['content']=task['body']['messages'][1]['content'].replace(task['expected_answer'],'value_deadbeef')
        with self.assertRaisesRegex(ValueError,'target occurrence'):verify_quality.evaluate(task,{},0,1,1000)

    def test_rejects_tampered_expected_answer(self):
        task=quality_tasks.build(Tokenizer(),1000,512,1,0,0);task['expected_answer']='forged'
        with self.assertRaisesRegex(ValueError,'contract mismatch'):verify_quality.evaluate(task,{},0,1,1000)
