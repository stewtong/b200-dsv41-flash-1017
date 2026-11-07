from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import cache_evidence


class CacheConfigTests(unittest.TestCase):
    def test_startup_log_fallback_requires_distinct_workers_and_enabled_prefix(self):
        prefix="non-default args: {'enable_prefix_caching': True}\n"
        worker=lambda i,size:f'(Worker_DP{i}_EP{i} pid={100+i}) INFO Setting kv cache block size to {size} for backend(s).\n'
        good=prefix+''.join(worker(i,128) for i in range(8))
        self.assertEqual(cache_evidence.cache_block_size_from_logs({'a':good},8)['block_size'],128)
        with self.assertRaises(ValueError):cache_evidence.cache_block_size_from_logs({'a':prefix+worker(0,128)*8},8)
        with self.assertRaises(ValueError):cache_evidence.cache_block_size_from_logs({'a':good.replace('True','False')},8)
        with self.assertRaises(ValueError):cache_evidence.cache_block_size_from_logs({'a':good+worker(0,64)},8)
    def test_live_config_requires_consistent_enabled_block_size(self):
        good='vllm:cache_config_info{block_size="64",enable_prefix_caching="True"} 1.0'
        self.assertEqual(cache_evidence.cache_block_size({'a':good,'b':good})['block_size'],64)
        for data in ({'a':''},{'a':good,'b':good.replace('64','32')},
                     {'a':good.replace('True','False')},{'a':good.replace('64','0')}):
            with self.assertRaises(ValueError): cache_evidence.cache_block_size(data)

    def test_present_invalid_metric_never_qualifies_as_absent(self):
        self.assertFalse(cache_evidence.has_cache_metric({'a':'vllm:other 1\n','b':''}))
        self.assertTrue(cache_evidence.has_cache_metric({'a':'# HELP vllm:cache_config_info help only'}))
        for present in ('vllm:cache_config_info{block_size="64",enable_prefix_caching="False"} 1',
                        'vllm:cache_config_info invalid',
                        'vllm:cache_config_info{block_size="32",enable_prefix_caching="True"} 1'):
            data={'a':present,'b':'vllm:cache_config_info{block_size="64",enable_prefix_caching="True"} 1'}
            self.assertTrue(cache_evidence.has_cache_metric(data))
            with self.assertRaises(ValueError):cache_evidence.cache_block_size(data)
