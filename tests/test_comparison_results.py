"""Descriptive metrics and actual partial-coverage CLI; no synthetic measurement credit."""
import copy,json,hashlib,tarfile
from pathlib import Path
import subprocess,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign,comparison_results,fixed_profiles,runtime
from test_fixed_profiles import manifest_fixture


class ComparisonResultTests(unittest.TestCase):
    def test_output_metadata_whitespace_and_chunk_decode_boundaries_are_distinct(self):
        raw={'events':[{'kind':'event','offset_ns':5,'data':{'choices':[{'delta':{'role':'assistant'}}]}},
            {'kind':'event','offset_ns':10,'data':{'choices':[{'delta':{'content':' \n'}}]}},
            {'kind':'event','offset_ns':20,'data':{'choices':[{'delta':{'content':'answer'}}]}},
            {'kind':'event','offset_ns':40,'data':{'choices':[{'delta':{'content':' text'}}]}}]}
        self.assertEqual(comparison_results.output_offsets(raw),(10,20,40))
        self.assertNotIn('p95_nearest_rank',comparison_results.distribution([1,2,3]))
        self.assertEqual(comparison_results.distribution(list(range(1,21)))['p95_nearest_rank'],19)

    def test_actual_cli_keeps336_incomplete_and_preserves_prior_output_on_bad_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=root/'plan.md';plan.write_text('Synthetic scoped result fixture.')
            manifest=manifest_fixture();manifest['plan_sha256']=campaign.digest(plan)
            for layout,record in list(manifest['runtime']['fixed_profiles'].items()):
                manifest['runtime']['fixed_profiles'][layout]=fixed_profiles.register(layout,manifest,record['qualification_evidence'])
            # Historical derivation does not reopen an expired operational window.
            manifest['runtime']['operational_deadline']='2020-01-01T00:00:00+00:00'
            selected=[c['id'] for c in manifest['cells']]
            with self.assertRaisesRegex(ValueError,'deadline expired'):
                campaign.validate_phase(manifest,'fixed-performance',cell_ids=selected)
            original=copy.deepcopy(manifest)
            self.assertTrue(campaign.validate_phase(manifest,'fixed-performance',cell_ids=selected,evidence=True))
            self.assertEqual(original,manifest)
            source=root/'manifest.json';source.write_text(json.dumps(manifest));output=root/'results.json'
            argv=[sys.executable,'-B',str(campaign.HERE/'campaign.py'),'derive-comparison',
                '--manifest',str(source),'--plan',str(plan),'--output',str(output)]
            result=subprocess.run(argv,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            value=json.loads(output.read_text());self.assertEqual(len(value['cells']),336)
            self.assertEqual((value['required_cold_cells'],value['required_warm_cells']),(288,48))
            self.assertEqual(value['accepted_cells'],0);self.assertFalse(value['complete'])
            self.assertTrue(all(c['missing_repetitions']==[0,1,2] for c in value['cells']))
            before=output.read_bytes()
            missing=subprocess.run(argv+['--run-dir',str(root/'missing')],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(missing.returncode,0);self.assertEqual(output.read_bytes(),before)
            # Byte-complete archives still require actual semantic capture proof.
            sealed=root/'dsv41-empty-capture';sealed.mkdir();(sealed/'fixture.txt').write_text('Synthetic non-capture.')
            runtime.seal_evidence(sealed);archive=root/'capture.tar.gz'
            with tarfile.open(archive,'w:gz') as bundle:bundle.add(sealed,arcname=sealed.name)
            index=root/'archive-index.json';index.write_text(json.dumps([{'archive':str(archive),
                'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                'seal_sha256':hashlib.sha256((sealed/'SHA256SUMS').read_bytes()).hexdigest(),'root_name':sealed.name}]))
            bad_archive=subprocess.run(argv+['--archive-index',str(index)],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(bad_archive.returncode,0);self.assertEqual(output.read_bytes(),before)
            plan.write_text('Changed authority.')
            changed=subprocess.run(argv,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(changed.returncode,0);self.assertEqual(output.read_bytes(),before)


if __name__=='__main__':unittest.main()
