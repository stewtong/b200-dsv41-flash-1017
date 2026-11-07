"""Qualification glue controls; physical, HTTP and cadence helpers test separately."""
from contextlib import redirect_stdout
import copy,hashlib,io,json,shutil,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_workload
from test_vllm_prompt import FakeTokenizer
import campaign,node_fixed_qualification as producer,profile_registry,routing_map,runtime,verify_fixed_qualification as verifier
from test_engine_counters import export
import test_node_fixed_qualification as preparation_fixture


class FixedQualificationVerifierTests(unittest.TestCase):
    def test_zero_variance_is_a_cell_population_rule(self):
        def report(accepted):
            hist=[0]*4;hist[accepted]=2
            return verifier.speculation_evidence.detailed_payload({'num_spec_tokens':3,'num_spec_steps':2,
                'num_accepted_draft_tokens':accepted*2,'num_draft_tokens':6,'mean_acceptance_length':1+accepted,
                'draft_acceptance_rate':accepted/3,'acceptance_histogram':hist,'per_step_accepted':[accepted]*2,'per_step_drafted':[3]*2},3)
        rows=[{'raw':{}},{'raw':{}}]
        with patch.object(verifier.speculation_evidence,'from_stream',side_effect=[report(0),report(3)]):
            self.assertEqual(len(verifier.speculative_population(rows,3)),2)
        with patch.object(verifier.speculation_evidence,'from_stream',side_effect=[report(0),report(0)]):
            with self.assertRaisesRegex(ValueError,'variable acceptance'):verifier.speculative_population(rows,3)

    def test_cli_glue_keeps_adoption_closed_and_rejects_resealed_provenance_and_placement(self):
        self.glue()

    def test_common_anchor_cli_glue_preserves_independent_provenance_and_placement(self):
        self.glue(anchor=True)

    def test_fixed_input_runtime_reuse_at_actual_verification_cli(self):
        self.glue(reuse=True)

    def test_frozen_performance_cli_keeps_scored_repetition_separate_from_full_cell_acceptance(self):
        self.glue(scored=True)

    def glue(self,anchor=False,reuse=False,scored=False):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);config,source=preparation_fixture.FixedQualificationTests().fixture(base)
            config['cell_id']=config['cell_id'].replace('challenger','control_before');config['profile_id']='control'
            if scored:
                from test_fixed_profiles import manifest_fixture
                manifest=manifest_fixture()
                manifest['runtime']['image_digest']=config['image']
                import fixed_profiles
                record=fixed_profiles.register('2xdep4',manifest,
                    manifest['runtime']['fixed_profiles']['2xdep4']['qualification_evidence'])
                manifest['runtime']['fixed_profiles']['2xdep4']=record
                Path(config['manifest']).write_text(json.dumps(manifest))
                (source/'manifest.json').write_bytes(Path(config['manifest']).read_bytes())
                config.update(execution_phase='fixed-performance',profile_id=record['id'],
                    cell_id='core-2xdep4-i8000-c1-cold-o2048',matrix=str(campaign.HERE/'comparison-matrix.json'))
                config.pop('trial_profile_id')
                origin=json.loads((source/'run-config.json').read_text())
                origin['fixed_prompt_version']='fixed-performance-v1'
                (source/'run-config.json').write_text(json.dumps(origin))
                tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,0,0,prompt_version='fixed-performance-v1')]
                (source/'fixed-tasks.json').write_text(json.dumps(tasks));runtime.seal_evidence(source)
            if anchor:
                manifest=json.loads(Path(config['manifest']).read_text())
                cells=[c['id'] for c in manifest['cells'] if c['layout']=='2xdep4' and c['suite'] in ('anchor','anchor_greedy')]
                record=profile_registry.register('common-nospec','2xdep4',{},manifest,hypothesis='Measure the declared common starting settings without optimization selection.',deciding_cells=cells,protected_tail='p95_request_seconds')
                manifest['runtime']['qualification_profiles']['common-nospec']=record
                Path(config['manifest']).write_text(json.dumps(manifest));(source/'manifest.json').write_bytes(Path(config['manifest']).read_bytes())
                config.update(cell_id='anchor-2xdep4-i8000-c1-cold-o2048',profile_id='common-nospec')
                for name in ('run-config.json','prompt-result.json'):
                    path=source/name;row=json.loads(path.read_text());row['cell_id']=config['cell_id']
                    if name=='run-config.json':row['fixed_prompt_version']='fixed-performance-v1'
                    path.write_text(json.dumps(row))
                tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'anchor',1,0,0,prompt_version='fixed-performance-v1')]
                (source/'fixed-tasks.json').write_text(json.dumps(tasks));runtime.seal_evidence(source)
            if reuse:
                from test_fixed_input_reuse import toolkits
                old_toolkit,new_toolkit=toolkits(base)
                old_manifest=json.loads(Path(config['manifest']).read_text())
                old_manifest['runtime'].update(toolkit_manifest_sha256=hashlib.sha256(old_toolkit.read_bytes()).hexdigest(),sampling_source={'synthetic':'fixture'},temperature=1.0,top_p=1.0)
                target_manifest=copy.deepcopy(old_manifest);target_manifest['runtime']['toolkit_manifest_sha256']=hashlib.sha256(new_toolkit.read_bytes()).hexdigest()
                (source/'manifest.json').write_text(json.dumps(old_manifest));Path(config['manifest']).write_text(json.dumps(target_manifest))
                config.update(toolkit_manifest=str(new_toolkit),fixed_input_reuse={'source_layout':config['layout'],'toolkit_manifest':str(old_toolkit),'toolkit_sha256':hashlib.sha256(old_toolkit.read_bytes()).hexdigest()})
                origin=json.loads((source/'run-config.json').read_text());origin['fixed_prompt_version']='fixed-performance-v1';(source/'run-config.json').write_text(json.dumps(origin))
                tasks=[fixed_workload.build(FakeTokenizer(),'/model',8000,2048,'core',1,0,0,prompt_version='fixed-performance-v1')]
                (source/'fixed-tasks.json').write_text(json.dumps(tasks));runtime.seal_evidence(source)
            prepared=producer.prepare(config,base);root=Path(config['run_dir']);root.mkdir()
            def put(path,data):
                path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
            for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
                (root/name).write_bytes(Path(config[key]).read_bytes())
            put(Path('run-config.json'),config);(root/'fixed-matrix.json').write_bytes(Path(config['matrix']).read_bytes())
            put(Path('profile-binding.json'),profile_registry.resolve(config,prepared['manifest']))
            put(Path('router-qualification.json'),{'lock_sha256':'synthetic-lock','router_executable_sha256':'synthetic-executable'})
            if reuse:
                lineage,files=prepared['input_lineage'];proof=root/'fixed-input-reuse-proof';proof.mkdir();(proof/'lineage.json').write_text(json.dumps(lineage))
                for name,data in files.items():(proof/name).write_bytes(data)
            shutil.copytree(source,root/'fixed-input-source');(root/'routing-map-source').mkdir()
            put(Path('routing-map-source/fixture.json'),{'synthetic':'routing helper is isolated in this glue test'})
            runtime.seal_evidence(root/'routing-map-source')
            records=[]
            for i in range(128):
                unit=i%8
                for repeat in (0,1):records.append({'session_id':'grid'+str(i),'request_id':'grid'+str(i)+'-r'+str(repeat),
                    'backend_port':31000+unit//4,'rank_headers':[str(unit%4)]})
            mapping=routing_map.build(records,[31000,31001],4);placement=routing_map.select(mapping,1,0);selected=placement[0]
            ident=producer.identity(config['cell_id'],prepared['binding']['profile_sha256'],0)
            plan={'cell_id':config['cell_id'],'attempt_cell_id':ident,'repetition':0,'profile_sha256':prepared['binding']['profile_sha256'],
                'source_manifest_sha256':prepared['source_manifest_sha256'],'placement':placement,'mapping_sha256':mapping['mapping_sha256']}
            put(Path('fixed-qualification-plan.json'),plan)
            put(Path('fixed-attribution/requests/measured/fixed-inputs.json'),{'tasks':prepared['tasks'],'session_ids':{'s0':selected['session_header']}})
            put(Path('fixed-attribution/requests/fixed-session-contract.json'),{'schema':'fixed-session-capture/v1'})
            put(Path('fixed-attribution/attribution-contract.json'),{'schema':'fixed-attribution-capture/v1',
                'scope':'isolated parent glue fixture; HTTP child verifier is mocked here'})
            put(Path('fixed-attribution/requests/measured/s0.json'),{'start_ns':2_000_000_000,'end_ns':5_000_000_000,'completion_tokens':2048})
            outcome={'status':'passed','cell_id':ident,'qualification_cell_id':config['cell_id'],'repetition':0,
                'profile_sha256':prepared['binding']['profile_sha256'],'performance_adoption_eligible':False,
                'scope':('one common-settings anchor repetition; full anchor coverage, greedy control and final scored profiles remain pending' if anchor else 'one unscored registered tuning repetition; profile selection, full telemetry/overhead and scored acceptance remain pending')}
            if scored:outcome.update(scope=producer.result_scope(prepared),scored_performance_evidence=True)
            put(Path('fixed-qualification-result.json'),outcome)
            metrics={url:'\n'.join(export().replace('engine="0"','engine="'+str(r)+'"') for r in ranks) for url,ranks in config['expected_ranks'].items()}
            # TYPE declarations repeat identically across rank exports, as permitted.
            samples=[{'monotonic_ns':3_000_000_000,'engine_metrics':metrics}]
            attributed={'status':'raw_fixed_attribution_verified','session_routes':{selected['session_header']:[selected['endpoint'],selected['rank']]},'populations':[]}
            runtime.seal_evidence(root)
            with patch.object(verifier.verify_engine_profile,'verify',return_value={'status':'registered_startup_verified','profile_sha256':prepared['binding']['profile_sha256']}),\
                 patch.object(verifier.routing_map,'load_verified',return_value=mapping),\
                 patch.object(verifier.verify_fixed_attribution,'verify',return_value=attributed),\
                 patch('verify_kernel_warmup.verify',return_value={'status':'isolated glue fixture; warmup has separate raw and ordering controls'}),\
                 patch.object(verifier.verify_telemetry,'verify_window',return_value={'status':'declared synthetic cadence control'}),\
                 patch.object(verifier.verify_telemetry,'iter_lines',side_effect=lambda *args:iter(samples)):
                if anchor:
                    with patch.object(verifier.launcher,'profile_sha256',return_value='0'*64):
                        with self.assertRaisesRegex(ValueError,'canonical starting profile'):verifier.verify(root)
                checked=verifier.verify(root);self.assertEqual(checked['valid_tokens'],2048);self.assertFalse(checked['performance_adoption_eligible'])
                self.assertEqual(checked['verified_speculative_token_count_reason'],'not_applicable_no_speculation')
                if reuse:
                    path=root/'fixed-input-reuse-proof/fixed_workload.py';old_bytes=path.read_bytes();path.write_bytes(old_bytes+b'\n');runtime.seal_evidence(root)
                    with self.assertRaisesRegex(ValueError,'generator bytes'):verifier.verify(root)
                    path.write_bytes(old_bytes);runtime.seal_evidence(root)
                output=io.StringIO()
                with patch.object(sys,'argv',['campaign.py','verify-fixed-qualification','--run-dir',str(root)]),redirect_stdout(output):campaign.main()
                self.assertFalse(json.loads(output.getvalue())['performance_adoption_eligible'])
                wrong=copy.deepcopy(plan);wrong['placement'][0]['rank']='3';put(Path('fixed-qualification-plan.json'),wrong);runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'frozen placement'):verifier.verify(root)
                put(Path('fixed-qualification-plan.json'),plan)
                original=(root/'fixed-input-source/checkpoint-receipt.json').read_bytes();put(Path('fixed-input-source/checkpoint-receipt.json'),{'synthetic':'changed'})
                runtime.seal_evidence(root/'fixed-input-source');wrong=copy.deepcopy(plan);wrong['source_manifest_sha256']=hashlib.sha256((root/'fixed-input-source/SHA256SUMS').read_bytes()).hexdigest();put(Path('fixed-qualification-plan.json'),wrong);runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'input dependency'):verifier.verify(root)
                (root/'fixed-input-source/checkpoint-receipt.json').write_bytes(original);runtime.seal_evidence(root/'fixed-input-source');put(Path('fixed-qualification-plan.json'),plan)
                wrong=dict(outcome,performance_adoption_eligible=True);put(Path('fixed-qualification-result.json'),wrong);runtime.seal_evidence(root)
                with self.assertRaisesRegex(ValueError,'completion record'):verifier.verify(root)


if __name__=='__main__':unittest.main()
