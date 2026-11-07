"""The warm measured request asks for prose instead of filler continuation."""
import copy,hashlib,json
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_pilot,warm_pilot_prompt,workload


class SyntheticWordTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return '<bos> '+''.join('<'+m['role']+'> '+m['content']+' </turn> ' for m in messages)+'<assistant> '
    def encode(self,text,**kwargs):return text.split()


class WarmPilotReportTests(unittest.TestCase):
    def test_actual_builder_preserves_exact_prefix_and_explicit_instruction(self):
        tokenizer=SyntheticWordTokenizer();n=128000;seed=workload.seed_for('pilot_warm',n,1,0,0);marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
        task=warm_pilot_prompt.build(tokenizer,'/model',n,512,1,0,0,128,marker)
        self.assertEqual(task['prompt_version'],'pilot-warm-report-v2')
        self.assertTrue(task['body']['messages'][-1]['content'].endswith(warm_pilot_prompt.INSTRUCTION))
        self.assertEqual(task['input_tokens'],n);self.assertEqual(task['prefix_tokens'],127744)
        render=lambda messages:tokenizer.encode(tokenizer.apply_chat_template(messages))
        prefix=render(task['conditioning_body']['messages']);measured=render(task['body']['messages'])
        self.assertEqual(prefix,measured[:len(prefix)])
        matrix=Path(node_pilot.__file__).with_name('matrix.json');cell='pilot_warm-2xtp4-i128000-c1-warm_prefix-o512'
        node_pilot.prepare(matrix,cell,[task])
        for altered in (dict(task,prompt_version='legacy'),dict(task,fixed_report_instruction='Continue.')):
            with self.assertRaisesRegex(ValueError,'versioned long-form'):node_pilot.prepare(matrix,cell,[altered])
        altered=copy.deepcopy(task);altered['body']['messages'][-1]['content']='More context: the the the. Continue.'
        with self.assertRaisesRegex(ValueError,'versioned long-form'):node_pilot.prepare(matrix,cell,[altered])


if __name__=='__main__':unittest.main()
