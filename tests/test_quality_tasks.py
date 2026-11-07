from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import quality_tasks


class FakeTokenizer:
    def apply_chat_template(self,messages,**kwargs):return '|'.join(x['role']+':'+x['content'] for x in messages)
    def encode(self,text,add_special_tokens=False):return list(text)


class QualityTests(unittest.TestCase):
    def test_synthetic_tasks_are_deterministic_and_scored(self):
        for session in range(9):
            a=quality_tasks.build(FakeTokenizer(),10000,512,9,0,session)
            self.assertEqual(a,quality_tasks.build(FakeTokenizer(),10000,512,9,0,session))
            self.assertTrue(all(abs(x-y)<=.005 for x,y in zip(a['target_depths'],a['observed_depths'])))
            import json
            answer=json.dumps(a['expected_answer']) if isinstance(a['expected_answer'],list) else a['expected_answer']
            self.assertTrue(quality_tasks.score(a,answer)['correct'])
            self.assertFalse(quality_tasks.score(a,'wrong')['correct'])

    def test_single_session_rotates_task_kind_across_repetitions(self):
        kinds=[quality_tasks.build(FakeTokenizer(),1000,512,1,r,0)['task_kind'] for r in range(3)]
        self.assertEqual(kinds,['identifier','multi_key','arithmetic'])

    def test_session_prefix_is_unique_and_early(self):
        a=quality_tasks.build(FakeTokenizer(),1000,512,2,0,0)
        b=quality_tasks.build(FakeTokenizer(),1000,512,2,0,1)
        self.assertNotEqual(a['session_prefix'],b['session_prefix'])
        self.assertTrue(a['body']['messages'][1]['content'].startswith(a['session_prefix']))
        self.assertTrue(b['body']['messages'][1]['content'].startswith(b['session_prefix']))

    def test_multikey_extraction_failure_is_distinct(self):
        task={'task_kind':'multi_key','expected_answer':['a','b']}
        self.assertTrue(quality_tasks.score(task,'not json')['extraction_failed'])
        self.assertFalse(quality_tasks.score(task,'["b","a"]')['extraction_failed'])

    def test_warm_prefix_conditions_without_the_answer_and_is_disjoint_from_cold(self):
        import json,hashlib,verify_quality
        tokenizer=FakeTokenizer()
        for repetition in range(3):
            cold=quality_tasks.build(tokenizer,8000,512,1,repetition,0)
            warm=quality_tasks.build(tokenizer,8000,512,1,repetition,0,'warm_prefix',128)
            self.assertEqual(warm,quality_tasks.build(tokenizer,8000,512,1,repetition,0,'warm_prefix',128))
            self.assertNotEqual(cold['session_prefix'],warm['session_prefix'])
            self.assertEqual(cold['expected_answer'],warm['expected_answer'])
            self.assertNotEqual(warm['expected_conditioning_answer'],warm['expected_answer'])
            prefix=tokenizer.encode(tokenizer.apply_chat_template(warm['conditioning_body']['messages']))
            measured=tokenizer.encode(tokenizer.apply_chat_template(warm['body']['messages']))
            self.assertEqual(len(prefix),((8000-256)//128)*128)
            self.assertEqual(measured[:len(prefix)],prefix)
            self.assertEqual(hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),warm['prefix_token_sha256'])
            result=verify_quality.evaluate(warm,{'request_id':'s0','complete':False},repetition,1,8000,tokenizer)
            self.assertEqual(result['reason'],'request_failed')
            self.assertEqual(result['depth_verification'],'provided_tokenizer_recomputed')

    def test_warm_quality_rejects_changed_history_prefix_or_conditioning_contract(self):
        import copy,verify_quality
        task=quality_tasks.build(FakeTokenizer(),8000,512,1,0,0,'warm_prefix',128)
        for key in ('prefix_token_sha256','prefix_tokens','expected_conditioning_answer'):
            changed=copy.deepcopy(task);changed[key]='forged'
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_quality.evaluate(changed,{'request_id':'s0','complete':False},0,1,8000,FakeTokenizer())
        task['body']['messages'][2]['content']=task['expected_answer']
        with self.assertRaisesRegex(ValueError,'conditioning contract'):
            verify_quality.evaluate(task,{'request_id':'s0','complete':False},0,1,8000,FakeTokenizer())
