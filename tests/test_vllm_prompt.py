from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import vllm_prompt

class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '|'.join(x['role']+':'+x['content'] for x in messages)
    def encode(self, text, add_special_tokens=False): return list(text)

class PromptTests(unittest.TestCase):
    def test_shape_and_seed_are_layout_independent(self):
        one = vllm_prompt.build(FakeTokenizer(), '/model', 10000, 128, 'pilot_smoke', 1, 0, 0)
        two = vllm_prompt.build(FakeTokenizer(), '/model', 10000, 128, 'pilot_smoke', 1, 0, 0)
        self.assertEqual(one, two)
        self.assertLessEqual(abs(one['input_tokens']-10000), 25)
        self.assertNotIn('ignore_eos', one['body'])
        fit = vllm_prompt.build(FakeTokenizer(), '/model', 10000, 512, 'pilot_fit', 1, 0, 0)
        self.assertTrue(fit['body']['ignore_eos'])

    def test_wrong_conditioning_answer_fails(self):
        with self.assertRaises(ValueError):
            vllm_prompt.build_warm(FakeTokenizer(), '/model', 10000, 512, 1, 0, 0, 64, 'wrong')
