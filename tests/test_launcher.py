import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[1] / 'reproduce/launcher.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
IMAGE = 'example/image@sha256:' + 'a' * 64

class LauncherTests(unittest.TestCase):
    def test_layouts_cover_eight_gpus_and_bind_loopback(self):
        for name in launcher.LAYOUTS:
            with self.subTest(layout=name):
                result = launcher.render(name, IMAGE, 'fixture-1', '/models/checkpoint', '/cache/vllm')
                self.assertEqual(result['status'], 'candidate_unqualified')
                self.assertEqual(sorted(g for c in result['commands'] for g in c['gpu_ids']), list(range(8)))
                self.assertIn('--disable-retries', result['router_argv'])
                self.assertEqual(result['router_argv'][result['router_argv'].index('--host') + 1], '127.0.0.1')
                self.assertEqual(result['router_argv'][result['router_argv'].index('--intra-node-data-parallel-size') + 1], str(launcher.LAYOUTS[name][1]))
                for c in result['commands']:
                    argv = c['argv']
                    self.assertIn('127.0.0.1:%d:8000' % c['host_port'], argv)
                    self.assertEqual(argv[argv.index('--restart') + 1], 'no')
                    self.assertEqual(argv[argv.index('--gpus') + 1], '"device=' + ','.join(map(str, c['gpu_ids'])) + '"')
                    self.assertIn('run_id=fixture-1', argv)
                    self.assertNotIn('--speculative-config', argv)
                    self.assertIn('--language-model-only', argv)
                    self.assertEqual(argv[argv.index('--max-model-len')+1], '1048576')
                    self.assertEqual(argv[argv.index('--kv-cache-dtype')+1],
                                     'fp8_ds_mla' if name=='dep8' else 'fp8')
                    self.assertEqual(json.loads(argv[argv.index('--attention-config')+1])['indexer_kv_dtype'], 'mxfp4')
                    if name in ('4xdep2', '2xdep4', 'dep8'):
                        self.assertEqual(json.loads(argv[argv.index('--kernel-config') + 1])['moe_backend'], 'deep_gemm_mega_moe')
                        self.assertTrue(json.loads(argv[argv.index('--engram-config') + 1])['embedding_across_dp'])
    def test_unpinned_or_unsafe_inputs_fail(self):
        for args in [('2xtp4', 'image:latest', 'ok', '/m', '/c'),
                     ('2xtp4', IMAGE, 'BAD', '/m', '/c'),
                     ('2xtp4', IMAGE, 'ok', '/m,readonly', '/c'),
                     ('2xtp4', IMAGE, 'ok', '/m/../x', '/c')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                launcher.render(*args)

    def test_container_cpu_partition_is_applied(self):
        result = launcher.render('dep8', IMAGE, 'fixture', '/m', '/c', '8,9,10')
        argv = result['commands'][0]['argv']
        self.assertEqual(argv[argv.index('--cpuset-cpus') + 1], '8,9,10')
        with self.assertRaises(ValueError):
            launcher.render('dep8', IMAGE, 'fixture', '/m', '/c', '1-3;bad')

if __name__ == '__main__':
    unittest.main()
