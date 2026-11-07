"""Synthetic generator-lineage fixtures; no installed encoder or model claim."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,fixed_input_reuse,fixed_input_source,fixed_workload,runtime
from test_vllm_prompt import FakeTokenizer


def toolkits(base):
 result=[]
 for stamp in ('20261103T0001Z','20261103T0002Z'):
  root=base/('dsv41-toolkit-'+stamp);(root/'reproduce').mkdir(parents=True)
  for name in fixed_input_reuse.GENERATORS:(root/'reproduce'/name).write_bytes((Path(fixed_workload.__file__).parent/name).read_bytes())
  (root/'runtime-version.txt').write_text(stamp)
  manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
  path=root/'MANIFEST.json';path.write_text(json.dumps(manifest,sort_keys=True));result.append(path)
 return result

class FixedInputReuseTests(unittest.TestCase):
 def fixture(self,base,cache='cold'):
  old,new=toolkits(base);source=base/'dsv41-fixed-origin';source.mkdir()
  manifest=campaign.ledger(Path(campaign.__file__).with_name('matrix.json'));manifest['plan_sha256']='a'*64
  manifest['runtime'].update(image_digest='example/image@sha256:'+'b'*64,sampling_source={'synthetic':'fixture'},temperature=1.0,top_p=1.0,toolkit_manifest_sha256=hashlib.sha256(old.read_bytes()).hexdigest())
  target=copy.deepcopy(manifest);target['runtime']['toolkit_manifest_sha256']=hashlib.sha256(new.read_bytes()).hexdigest()
  cfg={'layout':'dep8','cell_id':f'anchor-dep8-i8000-c1-{cache}-o2048','image':manifest['runtime']['image_digest'],'tasks_file':str(source/'fixed-tasks.json'),'toolkit_manifest':str(new),'fixed_input_reuse':{'source_layout':'2xtp4','toolkit_manifest':str(old),'toolkit_sha256':hashlib.sha256(old.read_bytes()).hexdigest()}}
  for key,name,data in (('manifest','manifest.json',target),('checkpoint_receipt','checkpoint-receipt.json',{'synthetic':True}),('image_qualification','image-qualification.json',{'synthetic':True})):
   p=base/name;p.write_text(json.dumps(data));cfg[key]=str(p);(source/name).write_bytes(p.read_bytes())
  (source/'manifest.json').write_text(json.dumps(manifest));ident=f'anchor-2xtp4-i8000-c1-{cache}-o2048'
  (source/'run-config.json').write_text(json.dumps({'layout':'2xtp4','cell_id':ident,'image':cfg['image'],'task_family':'fixed','fixed_prompt_version':'fixed-performance-v1','repetition':0}))
  (source/'prompt-result.json').write_text(json.dumps({'status':'passed','cleanup':'passed','task_family':'fixed','cell_id':ident,'repetition':0,'tasks':1}))
  task=fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'anchor',1,0,0,cache,128 if cache=='warm_prefix' else None,prompt_version='fixed-performance-v1');(source/'fixed-tasks.json').write_text(json.dumps([task]));runtime.seal_evidence(source)
  return cfg,source,target,new

 def test_exact_bytes_cross_layout_and_runtime_update_keep_retained_proof(self):
  for cache in ('cold','warm_prefix'):
   with self.subTest(cache=cache),tempfile.TemporaryDirectory() as d:
    base=Path(d);cfg,source,target,new=self.fixture(base,cache)
    loaded=fixed_input_source.load(cfg,8000,1,0,base,suite='anchor',cache=cache)
    self.assertEqual(loaded['task_bytes'],(source/'fixed-tasks.json').read_bytes())
    proof,files=loaded['input_lineage'];dest=base/'proof';dest.mkdir();runtime.atomic_json(dest/'lineage.json',proof)
    for name,data in files.items():(dest/name).write_bytes(data)
    self.assertEqual(fixed_input_reuse.verify(cfg,source,target,dest),proof)
    (dest/'fixed_workload.py').write_bytes((dest/'fixed_workload.py').read_bytes()+b'\n')
    with self.assertRaisesRegex(ValueError,'generator bytes'):fixed_input_reuse.verify(cfg,source,target,dest)

 def test_missing_reuse_or_changed_plan_encoder_sampling_and_checkpoint_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);cfg,source,target,new=self.fixture(base)
   missing=copy.deepcopy(cfg);missing.pop('fixed_input_reuse')
   with self.assertRaisesRegex(ValueError,'generation differs'):fixed_input_source.load(missing,8000,1,0,base,suite='anchor')
   for key,value in (('plan_sha256','c'*64),('tokenizer_revision','changed'),('temperature',0.0)):
    wrong=copy.deepcopy(target)
    if key=='plan_sha256':wrong[key]=value
    else:wrong['runtime'][key]=value
    Path(cfg['manifest']).write_text(json.dumps(wrong))
    with self.subTest(key=key),self.assertRaisesRegex(ValueError,'dependencies'):fixed_input_source.load(cfg,8000,1,0,base,suite='anchor')
   Path(cfg['manifest']).write_text(json.dumps(target));Path(cfg['checkpoint_receipt']).write_text('{"synthetic":"changed"}')
   with self.assertRaisesRegex(ValueError,'dependency changed'):fixed_input_source.load(cfg,8000,1,0,base,suite='anchor')

 def test_changed_generator_and_legacy_task_cannot_be_relabelled(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);cfg,source,target,new=self.fixture(base);data=json.loads(new.read_text());old_bytes=(new.parent/'reproduce/fixed_workload.py').read_bytes();(new.parent/'reproduce/fixed_workload.py').write_bytes(old_bytes+b'\n');data['reproduce/fixed_workload.py']=hashlib.sha256(old_bytes+b'\n').hexdigest();new.write_text(json.dumps(data,sort_keys=True));target['runtime']['toolkit_manifest_sha256']=hashlib.sha256(new.read_bytes()).hexdigest();Path(cfg['manifest']).write_text(json.dumps(target))
   with self.assertRaisesRegex(ValueError,'generator source'):fixed_input_source.load(cfg,8000,1,0,base,suite='anchor')
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);cfg,source,target,new=self.fixture(base)
   old=fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'anchor',1,0,0);(source/'fixed-tasks.json').write_text(json.dumps([old]));runtime.seal_evidence(source)
   with self.assertRaisesRegex(ValueError,'legacy workloads'):fixed_input_source.load(cfg,8000,1,0,base,suite='anchor')
