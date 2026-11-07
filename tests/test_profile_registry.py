import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import backend_evidence
import campaign
import launcher
import profile_registry


class ProfileRegistryTests(unittest.TestCase):
    def manifest(self):
        result=campaign.ledger(Path(campaign.__file__).with_name('matrix.json'))
        result['plan_sha256']='a'*64;result['runtime']['image_digest']='example/image@sha256:'+'b'*64
        return result

    def record(self,m,delta=None):
        ids=[c['id'] for c in m['cells'] if c['suite']=='tuning_trial' and c['layout']=='2xdep4' and
             c['parameters']['lever']=='dspark_static' and c['parameters']['input']==128000 and c['parameters']['concurrency']==8]
        return profile_registry.register('dep4-static-k5','2xdep4',delta or {'speculative_config':{
            'method':'dspark','num_speculative_tokens':5,'rejection_sample_method':'block','enable_adaptive_verification':False},
            'per_request_spec_decode_metrics':'detailed'},m,hypothesis='Static K5 may improve valid burst throughput under matched load.',
            deciding_cells=ids,protected_tail='p95_ttft_seconds')

    def test_preregistered_runtime_selection_is_exact_and_never_a_pilot_override(self):
        m=self.manifest();record=self.record(m);profile_registry.validate_record(record,m)
        m['runtime']['qualification_profiles']={record['id']:record}
        config={'layout':'2xdep4','execution_phase':'qualification','profile_id':record['id']}
        binding=profile_registry.resolve(config,m)
        self.assertEqual(binding['profile_sha256'],launcher.profile_sha256('2xdep4',record['profile_delta']))
        self.assertEqual(profile_registry.expected_runtime(binding,'2xdep4')['speculative_config']['rejection_sample_method'],'block')
        for wrong in [dict(config,execution_phase='pilot'),dict(config,profile_id='not-registered'),dict(config,profile_overrides={}),dict(config,layout='dep8'),dict(config,execution_phase='scored')]:
            with self.assertRaises(ValueError):profile_registry.resolve(wrong,m)
        for key,value in [('profile_sha256','0'*64),('image_digest','example/image@sha256:'+'c'*64),('hypothesis','changed but not rebound')]:
            altered=copy.deepcopy(record);altered[key]=value
            with self.assertRaises(ValueError):profile_registry.validate_record(altered,m)
        altered=copy.deepcopy(record);altered['profile_delta']['speculative_config']['num_speculative_tokens']=4
        with self.assertRaises(ValueError):profile_registry.validate_record(altered,m)
        # A self-referential candidate is not a matched control.
        altered=copy.deepcopy(record);altered['control_profile_sha256']=altered['profile_sha256']
        altered['registration_sha256']=profile_registry.record_digest(altered)
        m['runtime']['qualification_profiles'][record['id']]=altered
        with self.assertRaisesRegex(ValueError,'control is not registered'):profile_registry.resolve(config,m)

    def test_actual_registration_cli_retains_definition_and_refuses_overwrite(self):
        m=self.manifest();record=self.record(m)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,value in [('manifest',m),('delta',record['profile_delta']),('cells',record['deciding_cell_ids'])]:(root/(name+'.json')).write_text(json.dumps(value))
            argv=['campaign.py','profile-register','--manifest',str(root/'manifest.json'),'--profile-id',record['id'],
                  '--layout',record['layout'],'--delta',str(root/'delta.json'),'--hypothesis',record['hypothesis'],
                  '--deciding-cells',str(root/'cells.json'),'--protected-tail',record['protected_tail_metric'],'--output',str(root/'profile.json')]
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):campaign.main()
            persisted=json.loads((root/'profile.json').read_text())
            self.assertEqual(persisted,record)
            profile_registry.validate_record(persisted,m)
            config={'layout':'2xdep4','execution_phase':'qualification','profile_id':record['id']}
            m['runtime']['qualification_profiles']={record['id']:persisted}
            self.assertEqual(profile_registry.resolve(config,m)['profile_sha256'],record['profile_sha256'])
            with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):campaign.main()

    def test_reordered_nested_candidate_keys_preserve_exact_commands_and_hashes(self):
        manifest=self.manifest();record=self.record(manifest)
        reordered=json.loads(json.dumps(record,sort_keys=True))
        profile_registry.validate_record(reordered,manifest)
        image=manifest['runtime']['image_digest']
        self.assertEqual(launcher.render('2xdep4',image,'fixture','/model','/cache',
            overrides=record['profile_delta']),launcher.render('2xdep4',image,'fixture','/model','/cache',
            overrides=reordered['profile_delta']))
        self.assertEqual(launcher.profile_sha256('2xdep4',record['profile_delta']),
                         launcher.profile_sha256('2xdep4',reordered['profile_delta']))

    def test_installed_speculation_must_match_declared_real_block_configuration(self):
        from test_backend_evidence import BackendEvidenceTests
        expected={'method':'dspark','num_speculative_tokens':5,'rejection_sample_method':'block','enable_adaptive_verification':False}
        original=BackendEvidenceTests().text()
        text=original.replace('"tensor_parallel_size": 1','"speculative_config": '+json.dumps(expected)+', "tensor_parallel_size": 1')
        backend_evidence.verify(text,'MEGA',1,2,'fp8_ds_mla',True,expected_spec=expected)
        for altered in [text.replace('"block"','"synthetic"'),text.replace('"num_speculative_tokens": 5','"num_speculative_tokens": 4'),
                        text.replace('"enable_adaptive_verification": false','"enable_adaptive_verification": 0')]:
            with self.assertRaisesRegex(ValueError,'real-block'):backend_evidence.verify(altered,'MEGA',1,2,'fp8_ds_mla',True,expected_spec=expected)
        with self.assertRaisesRegex(ValueError,'unexpectedly enables'):backend_evidence.verify(text,'MEGA',1,2,'fp8_ds_mla',True)


if __name__=='__main__':unittest.main()
