import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import backend_evidence


class BackendEvidenceTests(unittest.TestCase):
    def text(self):
        config = {'attention_config': {'backend': '<AttentionBackendEnum.MEGA: \'example.Backend\'>',
                                       'indexer_kv_dtype': 'mxfp4'},
                  'tensor_parallel_size': 1, 'data_parallel_size': 2, 'data_parallel_size_local': 2,
                  'enable_expert_parallel': True, 'kernel_config': {'moe_backend': 'deep_gemm_mega_moe'}}
        return ('Launching Rust frontend: server --args-json ' + json.dumps(config) + '\nworld_size=2 rank=0 local_rank=0\n' +
                "Using DeepSeek's fp8_ds_mla KV cache format.\n" +
                '(Worker_DP0_EP0 pid=12) Setting kv cache block size to 128 for MEGA/SWA backend(s).\n' +
                '(Worker_DP1_EP1 pid=13) Setting kv cache block size to 128 for MEGA/SWA backend(s).\n')

    def test_installed_config_and_worker_selections(self):
        result = backend_evidence.verify(self.text(), 'MEGA', 1, 2, 'fp8_ds_mla', True)
        self.assertEqual(result['status'], 'configured_backend_confirmed')
        self.assertFalse(result['kernel_execution_profiled'])

    def test_dtype_missing_worker_and_conflicting_config_fail(self):
        for text in [self.text().replace('fp8_ds_mla','nvfp4_ds_mla'),
                     self.text().replace('Worker_DP1_EP1','Worker_DP0_EP0'),
                     self.text().replace('AttentionBackendEnum.MEGA','AttentionBackendEnum.OTHER')]:
            with self.assertRaises(ValueError): backend_evidence.verify(text, 'MEGA', 1, 2, 'fp8_ds_mla', True)

    def test_expected_kv_precision_requires_installed_configuration(self):
        text=self.text().replace('"tensor_parallel_size": 1', '"kv_cache_dtype": "fp8", "tensor_parallel_size": 1')
        proof=backend_evidence.verify(text,'MEGA',1,2,'fp8_ds_mla',True,kv_dtype='fp8')
        self.assertEqual(proof['configured_kv_dtype'],'fp8')
        for wrong in [self.text(),text.replace('"fp8"','"auto"'),text.replace('"fp8"','"nvfp4_ds_mla"')]:
            with self.assertRaisesRegex(ValueError,'KV dtype'):
                backend_evidence.verify(wrong,'MEGA',1,2,'fp8_ds_mla',True,kv_dtype='fp8')

    def test_pilot_rejects_unexpected_speculative_default(self):
        text=self.text().replace('"tensor_parallel_size": 1', '"speculative_config": {"num_speculative_tokens": 3}, "tensor_parallel_size": 1')
        with self.assertRaisesRegex(ValueError,'unexpectedly enables speculation'):
            backend_evidence.verify(text,'MEGA',1,2,'fp8_ds_mla',True)
