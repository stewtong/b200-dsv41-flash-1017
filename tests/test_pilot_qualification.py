import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import verify_pilot


class PilotQualificationTests(unittest.TestCase):
    def fixture(self, root):
        def put(name,data): (root/name).write_text(json.dumps(data))
        ids=[format(i+1,'064x') for i in range(4)]
        put('execution-state.json',{'container_ids':ids})
        devices=[{'uuid':'GPU-'+str(i),'index':i} for i in range(8)]
        put('nvml-live.json',{'devices':devices})
        snapshot={'processes':{},'gpu_processes':[]}; groups=[]
        for index,cid in enumerate(ids):
            config={'kv_cache_dtype':'fp8','attention_config':{'backend':"<AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41: 'fixture.Backend'>",'indexer_kv_dtype':'mxfp4'},
                    'kernel_config':{'moe_backend':'deep_gemm_mega_moe'},'enable_expert_parallel':True,'data_parallel_size':2}
            text='Launching Rust frontend: fixture --args-json '+json.dumps(config)+'\nworld_size=2 rank=0\n'
            rows=[]; pids=[]
            for rank in range(2):
                gpu=index*2+rank;host=100+gpu;namespace=10+rank
                text+=f'(Worker_DP{rank}_EP{rank} pid={namespace}) Setting kv cache block size to 128 for FLASHINFER_MLA_SPARSE_DSV41/SWA backend(s).\n'
                pids.append(host);snapshot['processes'][str(host)]={'namespace_pid':namespace,'start_ticks':7+gpu}
                snapshot['gpu_processes'].append({'pid':host,'gpu_uuid':'GPU-'+str(gpu)})
                rows.append({'dp_rank':rank,'namespace_pid':namespace,'host_pid':host,'start_ticks':7+gpu,'gpu_uuid':'GPU-'+str(gpu),'gpu_index':gpu})
            (root/('rank-top-'+cid+'.txt')).write_text('PID\n'+'\n'.join(map(str,pids)))
            (root/('rank-startup-'+cid+'.log')).write_text(text)
            groups.append({'container_id':cid,'workers':rows})
        put('rank-process-snapshot.json',snapshot)
        put('rank-placement.json',{'status':'passed','containers':groups})
        put('backend-selection.json',{'status':'configured_backend_confirmed','containers':[{'container_id':c} for c in ids]})
        return {'layout':'4xdep2','expected_ranks':{'http://fixture/metrics':[0,1]}},{'attribution':{'selected_rank':{'endpoint':'http://fixture/metrics','rank':'0'}}}

    def test_independent_raw_join_and_tampered_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config,result=self.fixture(root)
            verify_pilot.verify_single_request_placement(root,config,result)
            p=root/'rank-placement.json';x=json.loads(p.read_text());x['containers'][0]['workers'][0]['host_pid']=999;p.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'raw join'):
                verify_pilot.verify_single_request_placement(root,config,result)

    def test_missing_gpu_record_cannot_be_inferred_from_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config,result=self.fixture(root)
            p=root/'rank-process-snapshot.json';x=json.loads(p.read_text());x['gpu_processes'].pop();p.write_text(json.dumps(x))
            with self.assertRaisesRegex(ValueError,'physical GPU'):
                verify_pilot.verify_single_request_placement(root,config,result)

    def test_tp_full_shard_join_and_wrong_geometry_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config,result=self.fixture(root);config['layout']='4xtp2'
            for p in root.glob('rank-startup-*.log'):
                text=p.read_text().replace('Worker_DP0_EP0','Worker_TP0').replace('Worker_DP1_EP1','Worker_TP1')
                line,*rest=text.splitlines();installed=json.loads(line.split('--args-json ',1)[1])
                installed.pop('enable_expert_parallel');installed.pop('kernel_config');installed.pop('data_parallel_size')
                installed['tensor_parallel_size']=2
                p.write_text('Launching Rust frontend: fixture --args-json '+json.dumps(installed)+'\n'+'\n'.join(rest))
            p=root/'rank-placement.json';x=json.loads(p.read_text())
            for group in x['containers']:
                for row in group['workers']:row['tp_rank']=row.pop('dp_rank')
            p.write_text(json.dumps(x))
            verify_pilot.verify_single_request_placement(root,config,result)
            first=next(root.glob('rank-startup-*.log'));first.write_text(first.read_text().replace('world_size=2','world_size=4'))
            with self.assertRaisesRegex(ValueError,'world size'):
                verify_pilot.verify_single_request_placement(root,config,result)

    def test_independent_verifier_requires_actual_kv_precision(self):
        for wrong in ('auto','nvfp4_ds_mla',None):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as d:
                root=Path(d);config,result=self.fixture(root)
                p=next(root.glob('rank-startup-*.log'));line,*rest=p.read_text().splitlines()
                installed=json.loads(line.split('--args-json ',1)[1])
                if wrong is None:installed.pop('kv_cache_dtype')
                else:installed['kv_cache_dtype']=wrong
                p.write_text('Launching Rust frontend: fixture --args-json '+json.dumps(installed)+'\n'+'\n'.join(rest))
                with self.assertRaisesRegex(ValueError,'KV precision'):
                    verify_pilot.verify_single_request_placement(root,config,result)
