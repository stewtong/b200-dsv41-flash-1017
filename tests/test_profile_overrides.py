import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import launcher,profile_overrides

IMAGE='example/image@sha256:'+'a'*64
BASE_HASHES={
 '2xtp4':'d020daff21fc85db044138972ca3d4b0d61eddeb6bcbc81ca41b22ec726fe8ed',
 '4xdep2':'5b3ac97cae671f50fbd9d0298702e6363ec4ddd68616e9de3103334d57742409',
 '2xdep4':'da53e95d2fa2fda5e54a568e28cefe63ec0f44078d72e074f424514339d5e959',
 'dep8':'25840b0ddf9cea4351088cd72329b59fc95d8cd94b1599bd9faa87163fa416b0',
 'tp8':'9c933150c172d8260fd58eaa35f57ffea65aa00b49c0e7afc63cc44660cf479e',
 '4xtp2':'eda7070d1e172f9afa4f85b99ce4485607765f4651827300d6beb3fd5a328ac3'}


class ProfileCandidateTests(unittest.TestCase):
    def test_hisparse_uses_the_installed_connector_contract_without_changing_resources(self):
        for layout in ('4xdep2','2xdep4'):
            base=launcher.render(layout,IMAGE,'fixture','/model-source','/cache-source')
            candidate=launcher.render(layout,IMAGE,'fixture','/model-source','/cache-source',overrides={'hisparse_host_pool_gib':32})
            for before,after in zip(base['commands'],candidate['commands']):
                self.assertEqual(before['argv'][:before['argv'].index(IMAGE)+1],after['argv'][:after['argv'].index(IMAGE)+1])
                argv=after['argv'];transfer=json.loads(argv[argv.index('--kv-transfer-config')+1])
                self.assertEqual(transfer,{'kv_connector':'HiSparseConnector','kv_role':'kv_both','kv_connector_extra_config':{'host_pool_gib':32}})
                self.assertEqual(json.loads(argv[argv.index('--attention-config')+1])['hisparse_config'],{})
                self.assertNotIn('--kv-offloading-size',argv)
            self.assertEqual(candidate['status'],'candidate_unqualified')
        for bad in (0,True,64):
            with self.assertRaises(ValueError):launcher.profile_sha256('4xdep2',{'hisparse_host_pool_gib':bad})
        with self.assertRaisesRegex(ValueError,'only to DEP2 and DEP4'):launcher.profile_sha256('tp8',{'hisparse_host_pool_gib':32})

    def test_starting_hashes_and_noop_candidates_preserve_bound_manifests(self):
        for layout,digest in BASE_HASHES.items():
            self.assertEqual(launcher.profile_sha256(layout),digest)
            self.assertEqual(launcher.profile_sha256(layout,{}),digest)
            self.assertEqual(launcher.profile_sha256(layout,{'enable_prefix_caching':True}),digest)
            base=launcher.render(layout,IMAGE,'fixture','/model-source','/cache-source')
            disabled=launcher.render(layout,IMAGE,'fixture','/model-source','/cache-source',overrides={'enable_prefix_caching':False})
            for before,after in zip(base['commands'],disabled['commands']):
                expected=list(before['argv']);expected[expected.index('--enable-prefix-caching')]='--disable-prefix-caching'
                self.assertEqual(after['argv'],expected)
            with self.assertRaises(ValueError):launcher.profile_sha256(layout,{'enable_prefix_caching':0})
            self.assertEqual(launcher.profile_sha256(layout,{'gpu_memory_utilization':.9,
                'max_num_batched_tokens':8192,'prefill_schedule_interval':1,
                'per_request_spec_decode_metrics':'none'}),digest)

    def test_real_block_rejection_adaptive_candidate_preserves_all_resource_arguments(self):
        delta={'speculative_config':{'method':'dspark','num_speculative_tokens':4,
                'rejection_sample_method':'block','enable_adaptive_verification':True},
               'per_request_spec_decode_metrics':'detailed','prefill_schedule_interval':2}
        base=launcher.render('4xdep2',IMAGE,'fixture','/model-source','/cache-source')
        changed=launcher.render('4xdep2',IMAGE,'fixture','/model-source','/cache-source',overrides=delta)
        self.assertEqual(changed['status'],'candidate_unqualified')
        for old,new in zip(base['commands'],changed['commands']):
            self.assertEqual(old['gpu_ids'],new['gpu_ids']);self.assertEqual(old['host_port'],new['host_port'])
            self.assertEqual(old['argv'][:old['argv'].index(IMAGE)+1],new['argv'][:new['argv'].index(IMAGE)+1])
            argv=new['argv'];self.assertEqual(json.loads(argv[argv.index('--speculative-config')+1]),delta['speculative_config'])
        self.assertNotEqual(launcher.profile_sha256('4xdep2',delta),BASE_HASHES['4xdep2'])
        disabled=profile_overrides.apply(changed,{'speculative_config':None})
        self.assertTrue(all('--speculative-config' not in c['argv'] for c in disabled['commands']))

    def test_synthetic_extra_checkpoint_and_ownership_overrides_fail_without_mutation(self):
        base=launcher.render('dep8',IMAGE,'fixture','/model-source','/cache-source');saved=copy.deepcopy(base)
        spec={'method':'dspark','num_speculative_tokens':4,'rejection_sample_method':'block'}
        for bad in [{'model_dir':'/other'}, {'gpu_ids':[0]}, {'max_num_seqs':129},
                    {'speculative_config':dict(spec,rejection_sample_method='synthetic')},
                    {'speculative_config':dict(spec,synthetic_acceptance_length=3.51)},
                    {'speculative_config':dict(spec,model='other/checkpoint')},
                    {'speculative_config':dict(spec,num_speculative_tokens_per_batch_size=[[1,8,2],[8,32,4]])}]:
            with self.assertRaises(ValueError):profile_overrides.apply(base,bad)
            self.assertEqual(base,saved)

    def test_megattention_switch_keeps_explicit_fp8_and_cli_renders_declared_knobs(self):
        delta={'attention_backend':'FLASHMLA_MEGA_ATTN_DSV41',
               'compilation_config':{'cudagraph_capture_sizes':[1,2,4,8],'max_cudagraph_capture_size':8},
               'engram_config':{'use_thp':True},'router_timeout_seconds':20000}
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'candidate.json';f.write_text(json.dumps(delta))
            out=subprocess.run([sys.executable,'-B',str(Path(launcher.__file__)),'--layout','2xdep4',
                '--image',IMAGE,'--run-id','fixture','--model-dir','/model-source','--cache-dir','/cache-source',
                '--overrides-file',str(f)],capture_output=True,text=True,check=True)
            plan=json.loads(out.stdout)
            campaign=subprocess.run([sys.executable,'-B',str(Path(launcher.__file__).with_name('campaign.py')),'render-launch','--layout','2xdep4',
                '--image',IMAGE,'--run-id','fixture','--model-dir','/model-source','--cache-dir','/cache-source',
                '--overrides-file',str(f)],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(campaign.stdout),plan)
        for cmd in plan['commands']:
            argv=cmd['argv'];self.assertEqual(argv[argv.index('--kv-cache-dtype')+1],'fp8_ds_mla')
            self.assertEqual(json.loads(argv[argv.index('--engram-config')+1])['use_thp'],True)
        self.assertEqual(plan['router_argv'][plan['router_argv'].index('--request-timeout-secs')+1],'20000')
