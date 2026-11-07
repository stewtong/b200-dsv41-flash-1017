from pathlib import Path
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_smoke

class FakeSession:
    def __init__(self, lock, root, run_id, deadline_ns, **kwargs):
        self.root=Path(root); self.count=0;self.run_id=run_id
        self.sampler_stats={'count':10,'total_seconds':2.0,'max_seconds':.3,'max_interval_seconds':1.0}
    def __enter__(self):
        self.root.mkdir(); (self.root/'watchdog-heartbeat.json').write_text('{}')
        (self.root/'execution-state.json').write_text(json.dumps({'run_id':self.run_id,'container_ids':[]}));return self
    def __exit__(self, *args): return False
    def ensure_healthy(self):
        if (self.root/'nvml-measurement-mode.json').exists():
            path=self.root/'nvml-live.json';sample=json.loads(path.read_text())
            sample.update(measurement_mode=True,sample_started_ns=time.monotonic_ns())
            path.write_text(json.dumps(sample))
    def preflight(self, ports): pass
    def start_nvml_collector(self):
        (self.root/'nvml-live.json').write_text(json.dumps({'sample_started_ns':time.monotonic_ns()+10**12,
            'devices':[{'uuid':'GPU-'+str(i),'index':i} for i in range(8)]}));return {'pid':1}
    def activate_nvml_safety(self, identity): pass
    def create_start(self, argv):
        self.count+=1;cid=str(self.count)*64
        path=self.root/'execution-state.json';s=json.loads(path.read_text());s['container_ids'].append(cid);path.write_text(json.dumps(s));return cid
    def start_router(self, argv): pass
    def start_sampler(self, urls, names, ranks=None, server_cpus=None, nvml_identity=None): (self.root/'engine-heartbeat.json').write_text('{}')
    def mark_ready(self, evidence): self.evidence=evidence

class NodeSmokeTests(unittest.TestCase):
    def setUp(self):
        affinity = {0, 1, 2, 3}
        def get_affinity(_pid): return set(affinity)
        def set_affinity(_pid, cpus):
            affinity.clear(); affinity.update(cpus)
        for target, value in [('sched_getaffinity', get_affinity), ('sched_setaffinity', set_affinity)]:
            p = patch.object(node_smoke.os, target, value, create=True)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(node_smoke.cpu_partition, 'discover', return_value={'policy':'fixture', 'client_cpus':[0,1], 'server_cpus':[2,3]})
        p.start(); self.addCleanup(p.stop)
        p = patch.object(node_smoke.nvml_safety_probe,'validate_live')
        p.start(); self.addCleanup(p.stop)

    def test_orchestrates_without_claiming_rank_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            config={'layout':'2xtp4','image':'example/image@sha256:'+'a'*64,'run_id':'test',
                    'model_dir':'/model','cache_dir':'/cache','run_dir':str(root), 'required_metric_names':['active']}
            for key in ('manifest','image_qualification','router_qualification','checkpoint_receipt'):
                path=Path(d)/(key+'.json'); path.write_text('{}'); config[key]=str(path)
            Path(config['manifest']).write_text(json.dumps({'runtime':{'starting_profiles':{'2xtp4':node_smoke.launcher.profile_sha256('2xtp4')}}}))
            image={'image_id':'fixture'}; router={'router_executable':'/tmp/vllm-router'}
            answer={'passed':True,'raw':{'complete':True,'http_status':200}}
            snapshot={'processes':{str(100+i):{'namespace_pid':10+i%4,'start_ticks':200+i} for i in range(8)},
                      'gpu_processes':[{'pid':100+i,'gpu_uuid':'GPU-'+str(i)} for i in range(8)]}
            def capture(cid,run_id,path,**kwargs):
                installed={'kv_cache_dtype':'fp8','tensor_parallel_size':4,'attention_config':{'backend':"<AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41: 'fixture'>",'indexer_kv_dtype':'mxfp4'}}
                text='Launching Rust frontend: fixture --args-json '+json.dumps(installed)+'\nworld_size=4 rank=0\n'
                text+='(EngineCore pid=9) GPU KV cache size: 2,097,152 tokens, Maximum concurrency for 1,048,576 tokens per request: 2.00x\n'
                for i in range(4):text+=f'(Worker_TP{i} pid={10+i}) Setting kv cache block size to 128 for FLASHINFER_MLA_SPARSE_DSV41/SWA backend(s).\n'
                path.write_text(text)
            def command(argv,**kwargs):
                self.assertEqual(argv[:4],['sudo','-n','docker','top'])
                offset=(int(argv[4][0])-1)*4
                return 'PID\n'+'\n'.join(str(100+offset+i) for i in range(4))
            with patch.object(node_smoke.process_metrics,'capture',return_value=snapshot), \
                 patch.object(node_smoke.runtime,'inspect_owned'), \
                 patch.object(node_smoke.runtime,'capture_owned_logs',side_effect=capture), \
                 patch.object(node_smoke.runtime,'command',side_effect=command), \
                 patch.object(node_smoke.runtime,'process_identity',side_effect=lambda pid:{'start_ticks':pid+100}), \
                 patch.object(node_smoke,'validate',return_value=(image,router,3600)), \
                 patch.object(node_smoke.supervisor,'Supervisor',FakeSession), \
                 patch.object(node_smoke.runtime,'verify_placement',return_value={'fixture':True}), \
                 patch.object(node_smoke.runtime,'verify_retirement',return_value={'fixture':True}), \
                 patch.object(node_smoke,'await_health'), \
                 patch.object(node_smoke.telemetry,'scrape',return_value='vllm:cache_config_info{block_size="64",enable_prefix_caching="True"} 1.0'), \
                 patch.object(node_smoke.readiness,'known_answer',side_effect=lambda *a,**k:dict(answer)):
                node_smoke.run(config)
            result=json.loads((root/'smoke-result.json').read_text())
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['rank_coverage'],'unverified')
            placement=json.loads((root/'rank-placement.json').read_text())
            self.assertEqual(placement['status'],'passed')
            self.assertEqual([len(g['workers']) for g in placement['containers']],[4,4])
            qualification=json.loads((root/'telemetry-qualification.json').read_text())
            self.assertEqual(qualification['count'],10)
            self.assertIn('not established',qualification['scope'])
            timing=json.loads((root/'startup-timing.json').read_text())
            self.assertGreaterEqual(timing['semantic_ready_ns'],timing['launch_started_ns'])
            self.assertEqual(len(timing['backend_health_observed_seconds']),2)
            budget=json.loads((root/'kv-budget.json').read_text())
            self.assertEqual([r['engines'][0]['equivalent_full_context_tokens'] for r in budget['replicas']],[2097152,2097152])
            node_smoke.runtime.verify_evidence_manifest(root)

    def test_failure_cannot_overwrite_another_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-test'; root.mkdir()
            config={'run_dir':str(root),'run_id':'test'}
            (root/'execution-state.json').write_text(json.dumps({'run_id':'other','driver_pid':os.getpid(),'driver_identity':{'pid':1}}))
            node_smoke.record_failure(config, ValueError('bad'), d, lambda _: {'pid':1})
            self.assertFalse((root/'smoke-failure.json').exists())

    def test_semantic_failure_cannot_create_success_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run'
            config={'layout':'dep8','image':'example/image@sha256:'+'a'*64,'run_id':'test',
                    'model_dir':'/model','cache_dir':'/cache','run_dir':str(root), 'required_metric_names':['active']}
            for key in ('manifest','image_qualification','router_qualification','checkpoint_receipt'):
                path=Path(d)/(key+'.json'); path.write_text('{}'); config[key]=str(path)
            answer={'passed':False,'raw':{'complete':True,'http_status':200}}
            with patch.object(node_smoke,'validate',return_value=({'image_id':'fixture'},{'router_executable':'/tmp/vllm-router'},3600)), \
                 patch.object(node_smoke.supervisor,'Supervisor',FakeSession), \
                 patch.object(node_smoke.runtime,'verify_placement',return_value={'fixture':True}), \
                 patch.object(node_smoke.runtime,'verify_retirement',return_value={'fixture':True}), \
                 patch.object(node_smoke,'await_health'), \
                 patch.object(node_smoke.telemetry,'scrape',return_value='vllm:cache_config_info{block_size="64",enable_prefix_caching="True"} 1.0'), \
                 patch.object(node_smoke.readiness,'known_answer',side_effect=lambda *a,**k:dict(answer)):
                with self.assertRaises(node_smoke.runtime.RuntimeErrorClosed): node_smoke.run(config)
            self.assertFalse((root/'smoke-result.json').exists())
            self.assertFalse(json.loads((root/'readiness-0.json').read_text())['passed'])

class ManifestBindingTests(unittest.TestCase):
    def test_rejects_changed_plan_before_work(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = root/'plan.md'; plan.write_text('original')
            manifest = {'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),
                        'matrix_sha256':node_smoke.sha(Path(node_smoke.__file__).with_name('matrix.json')),
                        'runtime':{'image_digest':'fixture', 'toolkit_manifest_sha256':'c'*64, 'starting_profiles':{'dep8':node_smoke.launcher.profile_sha256('dep8')}}}
            source = root/'manifest.json'; source.write_text(json.dumps(manifest))
            config = {'plan':str(plan), 'manifest':str(source), 'image':'fixture', 'layout':'dep8', 'toolkit_manifest':str(Path(node_smoke.__file__).resolve().parents[1]/'MANIFEST.json')}
            with patch.object(node_smoke.campaign, 'validate_phase', return_value=True), \
                 patch.object(node_smoke.toolkit_integrity, 'verify', return_value='c'*64):
                self.assertEqual(node_smoke.verify_manifest(config), manifest)
                manifest['runtime']['starting_profiles']['dep8'] = '0'*64
                source.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'starting profile'):
                    node_smoke.verify_manifest(config)
                manifest['runtime']['starting_profiles']['dep8'] = node_smoke.launcher.profile_sha256('dep8')
                source.write_text(json.dumps(manifest))
                plan.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'plan content hash mismatch'):
                    node_smoke.verify_manifest(config)
