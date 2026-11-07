from datetime import datetime,timedelta,timezone
from pathlib import Path
import copy,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import sampling_source,campaign

CARD=Path(__file__).with_name('fixtures')/'pinned-model-card.md'

class SamplingSourceTests(unittest.TestCase):
    def test_actual_pinned_card_hash_blob_and_recommendation(self):
        proof=sampling_source.verify(CARD)
        self.assertEqual(proof['parameters'],{'temperature':1.0,'top_p':1.0})
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'card.md';f.write_bytes(CARD.read_bytes().replace(b'`top_p` | 1.0',b'`top_p` | 0.95'))
            with self.assertRaisesRegex(ValueError,'pinned checkpoint'):sampling_source.verify(f)
            f.unlink();f.symlink_to(CARD)
            with self.assertRaisesRegex(ValueError,'unsafe'):sampling_source.verify(f)

    def test_scored_validator_requires_matching_source_and_sampling(self):
        proof=sampling_source.verify(CARD);rev=proof['revision']
        # Declarative validator fixture, not a qualified profile or GPU result.
        m={'complete_universe':True,'plan_sha256':'d'*64,'runtime':{
            'image_digest':'example/image@sha256:'+'a'*64,
            'profile_hashes':dict.fromkeys(('2xtp4','4xdep2','2xdep4','dep8'),'b'*64),
            'operational_deadline':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
            'model_revision':rev,'tokenizer_revision':rev,'router_pin':'c'*64,'retirement_gate':'passed',
            'safety_controls':dict.fromkeys(('ownership','watchdog','cancellation','cleanup','persistence'),True),
            'capture_schema':'http-body-chunks/v1','temperature':1.0,'top_p':1.0,'sampling_source':proof}}
        self.assertTrue(campaign.validate_phase(m,'scored'))
        for mutate in [lambda r:r.pop('sampling_source'),lambda r:r.update(top_p=.95),lambda r:r.update(temperature=.6),
                       lambda r:r['sampling_source'].update(sha256='0'*64),
                       lambda r:r['sampling_source']['parameters'].update(temperature=.6)]:
            bad=copy.deepcopy(m);mutate(bad['runtime'])
            with self.assertRaisesRegex(ValueError,'sampling_source|sampling source'):
                campaign.validate_phase(bad,'scored')
