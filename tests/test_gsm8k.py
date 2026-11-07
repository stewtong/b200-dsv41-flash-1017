import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import gsm8k


class GSM8KTests(unittest.TestCase):
    def test_exact_numeric_and_ambiguous_extraction(self):
        self.assertTrue(gsm8k.score('work\n#### 1000','calculation\n#### 1,000.0')['correct'])
        for output in ['1000','#### 1,00','#### 1000\nmore','#### 1000\n#### 1000']:
            result=gsm8k.score('#### 1000',output)
            self.assertFalse(result['correct']);self.assertTrue(result['extraction_failed'])

    def test_failed_and_truncated_requests_stay_in_denominator(self):
        for kwargs in [{'complete':False},{'http_status':500},{'finish_reason':'length'}]:
            result=gsm8k.score('#### 3','#### 3',**kwargs)
            self.assertFalse(result['correct']);self.assertEqual(result['denominator'],1)

    def test_calibration_excludes_scored_and_repeated_questions(self):
        rows=[{'question':'q'+str(i),'answer':'#### 1'} for i in range(60)]
        chosen=gsm8k.calibration(rows+rows,rows[:5],50)
        self.assertEqual(len({x['question'] for x in chosen}),50)
        self.assertTrue({x['question'] for x in chosen}.isdisjoint(x['question'] for x in rows[:5]))
        self.assertEqual(chosen,gsm8k.calibration(rows+rows,rows[:5],50))

    def test_source_schema_and_count_are_required(self):
        raw=b'{"question":"q","answer":"#### 1"}\n'
        self.assertEqual(len(gsm8k.records(raw,1)),1)
        with self.assertRaises(ValueError):gsm8k.records(raw,2)
        with self.assertRaises(ValueError):gsm8k.records(raw.replace(b'#### 1',b'unknown'),1)

    def test_task_uses_measured_template_and_keeps_oracle_out_of_prompt(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return json.dumps([messages,kwargs],sort_keys=True)
            def encode(self,text,**kwargs):return list(text)
        row={'question':'How many marbles remain?','answer':'Private oracle explanation.\n#### 1037'}
        first=gsm8k.task(Tokenizer(),row,7,True,4096)
        second=gsm8k.task(Tokenizer(),row,7,True,4096)
        self.assertEqual(first,second)
        self.assertNotIn('Private oracle',json.dumps(first['body']))
        self.assertEqual(first['body']['chat_template_kwargs'],{'thinking':True,'reasoning_effort':50})
        self.assertNotIn('ignore_eos',first['body'])
