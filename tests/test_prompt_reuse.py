"""Direct, byte-preserving pilot reuse with independently checked provenance."""
import copy
import hashlib
import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,node_pilot,node_pilot_block,prompt_reuse,runtime,verify_pilot,workload


class PromptReuseTests(unittest.TestCase):
    def test_exact_priority_amendment_preserves_inputs_and_requires_explicit_proof(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,source,target,tasks=self.fixture(d)
            old=json.loads((source/'manifest.json').read_text());new=json.loads((target/'manifest.json').read_text())
            old['plan_sha256'],new['plan_sha256']=prompt_reuse.PRIORITY_PLAN_PAIR
            (source/'manifest.json').write_text(json.dumps(old));runtime.seal_evidence(source)
            (target/'manifest.json').write_text(json.dumps(new));prior=(source/'SHA256SUMS').read_bytes();load=prompt_reuse.load
            with patch.object(prompt_reuse,'load',side_effect=lambda c,r:load(c,r,allowed_cache=d)):
                self.assertEqual(node_pilot.verified_tasks(cfg,d),tasks);node_pilot.copy_prompt_provenance(cfg,target,tasks)
            verify_pilot.verify_prompt_provenance(target,tasks,cfg,cfg['cell_id'])
            proof=target/'prompt-provenance/prompt-reuse.json';original=proof.read_bytes();value=json.loads(original)
            self.assertEqual(value['plan_supersession']['authority'],'owner-priority-amendment-20261102')
            value.pop('plan_supersession');proof.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'reuse binding differs'):
                verify_pilot.verify_prompt_provenance(target,tasks,cfg,cfg['cell_id'])
            proof.write_bytes(original)
            for old_hash,new_hash in (prompt_reuse.PRIORITY_PLAN_PAIR[::-1],('f'*64,prompt_reuse.PRIORITY_PLAN_PAIR[1])):
                changed_old=copy.deepcopy(old);changed_new=copy.deepcopy(new)
                changed_old['plan_sha256']=old_hash;changed_new['plan_sha256']=new_hash
                with self.assertRaisesRegex(ValueError,'dependencies'):
                    prompt_reuse.compare(cfg,json.loads((source/'run-config.json').read_text()),changed_old,changed_new,
                        json.loads(Path(cfg['prompt_reuse']['toolkit_manifest']).read_text()),json.loads(Path(cfg['toolkit_manifest']).read_text()))
            self.assertEqual((source/'SHA256SUMS').read_bytes(),prior);self.assertEqual((source/'pilot-tasks.json').read_bytes(),tasks)

    def fixture(self,base):
        base=Path(base);toolkits=[]
        for stamp in ('20261102T1506Z','20261102T1600Z'):
            root=base/('dsv41-toolkit-'+stamp);(root/'reproduce').mkdir(parents=True)
            for name in ('vllm_prompt.py','workload.py'):(root/'reproduce'/name).write_text('# synthetic public generator fixture\n')
            files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
            (root/'MANIFEST.json').write_text(json.dumps(files));toolkits.append(root)
        cells=campaign.ledger(Path(campaign.__file__).with_name('matrix.json'))['cells']
        original=next(c for c in cells if c['id']=='pilot_fit-2xtp4-i128000-c1-cold-o512')
        target=next(c for c in cells if c['id']=='pilot_fit-2xdep4-i128000-c1-cold-o512')
        common={'plan_sha256':'a'*64,'matrix_sha256':'b'*64,'cells':[original,target],
                'runtime':{'image_digest':'fixture/image@sha256:'+'c'*64,'model_revision':'synthetic-checkpoint',
                    'tokenizer_revision':'synthetic-checkpoint','sampling_source':{'status':'synthetic'},
                    'temperature':1.0,'top_p':1.0,'starting_profiles':{'2xtp4':'d'*64,'2xdep4':'e'*64}}}
        manifests=[]
        for tk in toolkits:
            value=copy.deepcopy(common);value['runtime']['toolkit_manifest_sha256']=hashlib.sha256((tk/'MANIFEST.json').read_bytes()).hexdigest();manifests.append(value)
        source=base/'dsv41-prompt-original';source.mkdir();task_bytes=b'[{"id":"s0"}]'
        old_config={'layout':'2xtp4','cell_id':original['id'],'image':common['runtime']['image_digest'],'task_family':'pilot'}
        for name,data in {'run-config.json':old_config,'manifest.json':manifests[0],
                          'prompt-result.json':{'status':'passed','cleanup':'passed','cell_id':original['id'],'tasks':1,'task_family':'pilot'},
                          'checkpoint-receipt.json':{'status':'synthetic'},'image-qualification.json':{'status':'synthetic'}}.items():
            (source/name).write_text(json.dumps(data))
        (source/'pilot-tasks.json').write_bytes(task_bytes);runtime.seal_evidence(source)
        target_root=base/'cell';target_root.mkdir();(target_root/'manifest.json').write_text(json.dumps(manifests[1]))
        for name in ('checkpoint-receipt.json','image-qualification.json'):(target_root/name).write_bytes((source/name).read_bytes())
        config={'cell_id':target['id'],'layout':'2xdep4','image':old_config['image'],
                'manifest':str(target_root/'manifest.json'),'checkpoint_receipt':str(target_root/'checkpoint-receipt.json'),
                'image_qualification':str(target_root/'image-qualification.json'),'tasks_file':str(source/'pilot-tasks.json'),
                'toolkit_manifest':str(toolkits[1]/'MANIFEST.json'),
                'prompt_reuse':{'source_layout':'2xtp4','toolkit_manifest':str(toolkits[0]/'MANIFEST.json'),
                    'toolkit_sha256':manifests[0]['runtime']['toolkit_manifest_sha256']}}
        return config,source,target_root,task_bytes

    def test_actual_input_and_provenance_apis_keep_original_bytes_and_record(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,source,target,tasks=self.fixture(d);prior=(source/'SHA256SUMS').read_bytes();load=prompt_reuse.load
            with patch.object(prompt_reuse,'load',side_effect=lambda c,r:load(c,r,allowed_cache=d)):
                self.assertEqual(node_pilot.verified_tasks(cfg,d),tasks)
                node_pilot.copy_prompt_provenance(cfg,target,tasks)
            verify_pilot.verify_prompt_provenance(target,tasks,cfg,cfg['cell_id'])
            self.assertEqual((source/'SHA256SUMS').read_bytes(),prior)
            self.assertEqual((source/'pilot-tasks.json').read_bytes(),tasks)
            self.assertEqual(json.loads((source/'run-config.json').read_text())['layout'],'2xtp4')
            self.assertFalse((source/'prompt-reuse.json').exists())
            runtime.verify_evidence_manifest(source)
            # Provenance is still checked independently if a caller reseals an
            # altered target profile, shape, phase or source-code declaration.
            original=(target/'manifest.json').read_bytes()
            for mutation in ('profile','shape','plan'):
                changed=json.loads(original)
                if mutation=='profile':changed['runtime']['starting_profiles']['2xdep4']='f'*64
                if mutation=='shape':changed['cells'][1]['concurrency']=8
                if mutation=='plan':changed['plan_sha256']='f'*64
                (target/'manifest.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):verify_pilot.verify_prompt_provenance(target,tasks,cfg,cfg['cell_id'])
            (target/'manifest.json').write_bytes(original)
            for altered in (dict(cfg,execution_phase='qualification'),dict(cfg,prompt_reuse={**cfg['prompt_reuse'],'source_layout':'2xdep4'})):
                with self.assertRaises(ValueError):verify_pilot.verify_prompt_provenance(target,tasks,altered,cfg['cell_id'])
            # A resealed source-code change must fail independently too.
            proof=target/'prompt-provenance/target-toolkit-MANIFEST.json';source_files=json.loads(proof.read_text());source_files['reproduce/workload.py']='f'*64;proof.write_text(json.dumps(source_files))
            updated=json.loads(original);updated['runtime']['toolkit_manifest_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest();(target/'manifest.json').write_text(json.dumps(updated))
            with self.assertRaisesRegex(ValueError,'generator code differs'):verify_pilot.verify_prompt_provenance(target,tasks,cfg,cfg['cell_id'])

    def test_changed_generator_or_encoding_pin_is_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,source,target,tasks=self.fixture(d);value=json.loads((target/'manifest.json').read_text())
            for key in ('model_revision','tokenizer_revision','image_digest','top_p','temperature','sampling_source'):
                changed=copy.deepcopy(value);changed['runtime'][key]='changed synthetic pin';(target/'manifest.json').write_text(json.dumps(changed))
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):prompt_reuse.load(cfg,source,allowed_cache=d)
            (target/'manifest.json').write_text(json.dumps(value))
            tk=Path(cfg['toolkit_manifest']);data=json.loads(tk.read_text());file=tk.parent/'reproduce/workload.py';file.write_text('# changed fixture generator\n');data['reproduce/workload.py']=hashlib.sha256(file.read_bytes()).hexdigest();tk.write_text(json.dumps(data))
            value['runtime']['toolkit_manifest_sha256']=hashlib.sha256(tk.read_bytes()).hexdigest();(target/'manifest.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'generator source differs'):prompt_reuse.load(cfg,source,allowed_cache=d)

    def test_undeclared_alias_and_changed_original_bytes_remain_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,source,target,tasks=self.fixture(d);no_reuse=dict(cfg);no_reuse.pop('prompt_reuse')
            with self.assertRaisesRegex(ValueError,'does not match pilot'):node_pilot.verified_tasks(no_reuse,d)
            (source/'pilot-tasks.json').write_bytes(b'[]')
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'checksum mismatch'):node_pilot.verified_tasks(cfg,d)

    def test_shared_and_new_inputs_can_share_one_explicit_block(self):
        entries=[];seen=[]
        reuse={'source_layout':'2xtp4','toolkit_manifest':'/data/hf-cache/dsv41-toolkit-20261102T1506Z/MANIFEST.json','toolkit_sha256':'a'*64}
        for n in (128000,512000):
            row={'cell_id':f'pilot_fit-2xdep4-i{n}-c1-cold-o512','tasks_file':'/data/fixture/pilot-tasks.json','request_timeout_seconds':1800}
            if n==128000:row['prompt_reuse']=reuse
            entries.append(row)
        def prepared(config):
            seen.append(config);n=128000 if config['cell_id']==entries[0]['cell_id'] else 512000
            seed=workload.seed_for('pilot_fit',n,1,0,0);marker=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
            return json.dumps([{'id':'s0','session':0,'suite':'pilot_fit','seed':seed,'expected_answer':marker,
                'input_tokens':n,'output_tokens':512,'fixed_output':True,'prompt_sha256':'b'*64,
                'body':workload.request_body('/model',[{'role':'user','content':marker}],n,512,seed,fixed_output=True)}]).encode()
        config={'matrix':str(Path(campaign.__file__).with_name('matrix.json')),'layout':'2xdep4','cells':entries,
                'required_metric_names':list(node_pilot.PILOT_METRICS),
                'expected_ranks':{f'http://127.0.0.1:{31000+i}/metrics':[0,1,2,3] for i in range(2)}}
        with patch.object(node_pilot,'verified_tasks',side_effect=prepared):
            rows,_,budget=node_pilot_block.prepare(config)
        self.assertEqual(len(rows),2);self.assertEqual(budget,3600)
        self.assertEqual(seen[0]['prompt_reuse'],reuse);self.assertNotIn('prompt_reuse',seen[1])


if __name__=='__main__':unittest.main()
