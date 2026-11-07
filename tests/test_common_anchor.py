"""Common anchors reuse the fixed capture path; these are input/registry fixtures."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,fixed_workload,launcher,node_fixed_qualification,profile_registry,runtime
from test_vllm_prompt import FakeTokenizer

class CommonAnchorTests(unittest.TestCase):
 def test_all_six_starting_profiles_accept_only_their_exact_anchor_workload(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);manifest=campaign.ledger(Path(campaign.__file__).with_name('matrix.json'));manifest['plan_sha256']='a'*64;manifest['runtime']['image_digest']='example/image@sha256:'+'b'*64
   for layout in ('2xtp4','4xdep2','2xdep4','dep8','tp8','4xtp2'):
    with self.subTest(layout=layout):
     cells=[c for c in manifest['cells'] if c['layout']==layout and c['suite'] in ('anchor','anchor_greedy')]
     args=dict(hypothesis='Observe the declared common starting settings without an optimization or capacity claim.',deciding_cells=[c['id'] for c in cells],protected_tail='p95_request_seconds')
     record=profile_registry.register('common-'+layout,layout,{},manifest,**args)
     self.assertEqual(record['profile_sha256'],launcher.profile_sha256(layout))
     with self.assertRaisesRegex(ValueError,'unchanged NoSpec'):profile_registry.register('bad-'+layout,layout,{'max_num_seqs':8},manifest,**args)
     bound=copy.deepcopy(manifest);bound['runtime']['qualification_profiles']={record['id']:record}
     for cache in ('cold','warm_prefix'):
      cell=next(c for c in cells if c['suite']=='anchor' and c['input_tokens']==8000 and c['concurrency']==1 and c['cache']==cache)
      root=base/('dsv41-'+layout+'-'+cache);root.mkdir();source=root/'dsv41-input';source.mkdir()
      cfg={'run_id':'anchor-fixture','execution_phase':'qualification','layout':layout,'profile_id':record['id'],'cell_id':cell['id'],'repetition':2,'request_timeout_seconds':120,'matrix':str(Path(campaign.__file__).with_name('matrix.json')),'image':bound['runtime']['image_digest'],'tasks_file':str(source/'fixed-tasks.json')}
      tp,dp,_=launcher.LAYOUTS[layout];cfg['expected_ranks']={f'http://127.0.0.1:{31000+i}/metrics':list(range(dp)) for i in range(8//(tp*dp))}
      for key,name,data in (('manifest','manifest.json',bound),('checkpoint_receipt','checkpoint-receipt.json',{'synthetic':True}),('image_qualification','image-qualification.json',{'synthetic':True})):
       path=root/name;path.write_text(json.dumps(data));cfg[key]=str(path);(source/name).write_bytes(path.read_bytes())
      origin={'cell_id':cell['id'],'layout':layout,'image':cfg['image'],'task_family':'fixed','repetition':2,'fixed_prompt_version':'fixed-performance-v1'}
      (source/'run-config.json').write_text(json.dumps(origin));(source/'prompt-result.json').write_text(json.dumps({'status':'passed','cleanup':'passed','task_family':'fixed','cell_id':cell['id'],'repetition':2,'tasks':1}))
      task=fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'anchor',1,2,0,cache,128 if cache=='warm_prefix' else None,prompt_version='fixed-performance-v1');(source/'fixed-tasks.json').write_text(json.dumps([task]));runtime.seal_evidence(source)
      prepared=node_fixed_qualification.prepare(cfg,base);self.assertEqual(prepared['binding']['profile_sha256'],record['profile_sha256']);self.assertEqual(prepared['tasks'],[task]);self.assertIn('common-settings anchor',node_fixed_qualification.result_scope(prepared))
      self.assertEqual(node_fixed_qualification.capture_budget(cfg,prepared),280 if cache=='warm_prefix' else 140)
      changed=copy.deepcopy(task);changed['suite']='core';(source/'fixed-tasks.json').write_text(json.dumps([changed]));runtime.seal_evidence(source)
      with self.assertRaises(ValueError):node_fixed_qualification.prepare(cfg,base)
