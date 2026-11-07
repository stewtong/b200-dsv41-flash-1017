import sys,unittest,copy,json,hashlib,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import speculation_evidence as s

class SpeculationEvidenceTests(unittest.TestCase):
    def text(self,drafts=4,drafted=12,accepted=6,vector=(3,2,1)):
        lines=[]
        for name,value in zip(s.SCALARS,(drafts,drafted,accepted)):
            lines.extend([f'# TYPE {s.PREFIX+name} counter',f'{s.PREFIX+name}_total{{model_name="/model",engine="0"}} {value}'])
        lines.append(f'# TYPE {s.POSITION.removesuffix("_total")} counter')
        lines.extend(f'{s.POSITION}{{model_name="/model",engine="0",position="{i}"}} {v}' for i,v in enumerate(vector))
        return '\n'.join(lines)
    def check(self,after,before=None):
        return s.verify_delta({'ep':before or self.text(0,0,0,(0,0,0))},{'ep':after},{'ep':[0]},3)
    def test_known_prefix_survival_counts_recover_distribution_and_bonus_mean(self):
        report=self.check(self.text())
        self.assertEqual(report['accepted_count_distribution'],[1,1,1,1])
        self.assertEqual(report['accepted_count_variance'],1.25)
        self.assertEqual(report['ranks'][0]['mean_acceptance_length_including_bonus'],2.5)
        self.assertFalse(report['zero_variance_requires_investigation'])
        self.assertIsNone(report['verified_token_count'])
        self.assertEqual(report['status'],'partial_missing_verified_token_counter')
    def test_synthetic_zero_variance_and_unobserved_are_distinct(self):
        report=self.check(self.text(4,12,8,(4,4,0)))
        self.assertTrue(report['zero_variance_requires_investigation'])
        self.assertEqual(report['accepted_count_distribution'],[0,0,4,0])
        empty=self.check(self.text(0,0,0,(0,0,0)))
        self.assertIsNone(empty['zero_variance_requires_investigation'])
        self.assertIsNone(empty['accepted_count_variance'])
    def test_missing_rank_position_bad_cdf_and_counter_reset_fail(self):
        for bad in [self.text().replace('engine="0"','engine="1"'),
                    self.text().replace('position="1"','position="9"'),
                    self.text(4,12,6,(2,3,1)),self.text(4,12,7,(3,2,1)),
                    self.text().replace(' counter',' gauge')]:
            with self.assertRaises(ValueError):self.check(bad)
        with self.assertRaisesRegex(ValueError,'reset'):
            self.check(self.text(0,0,0,(0,0,0)),self.text())

    def test_foreign_missing_extra_and_unparsed_labels_cannot_enter_pinned_population(self):
        original=self.text()
        for text in (original.replace('model_name="/model"','model_name="/foreign"'),
                     original.replace('model_name="/model",',''),
                     original.replace('engine="0"','engine="0",extra="foreign"'),
                     original.replace('engine="0"','engine="0",garbage'),
                     original.replace('engine="0"','engine="0",engine="0"')):
            self.assertNotEqual(text,original)
            with self.subTest(text=text),self.assertRaises(ValueError):self.check(text)

    def payload(self):
        return {'num_spec_tokens':3,'num_spec_steps':4,'num_accepted_draft_tokens':6,'num_draft_tokens':12,
                'mean_acceptance_length':2.5,'draft_acceptance_rate':.5,'acceptance_histogram':[1,1,1,1],
                'per_step_accepted':[0,1,2,3],'per_step_drafted':[3,3,3,3]}
    def test_detailed_steps_verify_summary_and_missing_arrays_fail(self):
        result=s.detailed_payload(self.payload(),3)
        self.assertEqual(result['verification_steps'],4)
        self.assertEqual(result['accepted_count_variance'],1.25)
        self.assertFalse(result['zero_variance_requires_investigation'])
        self.assertIsNone(result['verified_token_count'])
        for change in [lambda x:x.pop('per_step_accepted'),lambda x:x.update(num_spec_steps=True),
                       lambda x:x.update(acceptance_histogram=[0,0,4,0]),lambda x:x['per_step_drafted'].__setitem__(3,2)]:
            bad=copy.deepcopy(self.payload());change(bad)
            with self.assertRaises(ValueError):s.detailed_payload(bad,3)
    def test_detailed_metrics_are_replayed_from_final_usage_wire(self):
        event={'choices':[],'usage':{'prompt_tokens':8000,'completion_tokens':10},
               'metrics':{'speculative_decoding':self.payload()}}
        data=('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode()
        raw={'complete':True,'http_status':200,'error':None,'start_ns':0,'terminal_ns':2,
             'wire_chunks':[{'offset_ns':1,'base64':base64.b64encode(data).decode(),'eof':False},
                            {'offset_ns':2,'base64':'','eof':True}],
             'wire_byte_count':len(data),'wire_bytes_observed':len(data),'wire_limit_bytes':65536,
             'wire_truncated':False,'wire_response_kind':'sse','wire_sha256':hashlib.sha256(data).hexdigest(),
             'wire_expected_body_bytes':len(data),'events':[{'kind':'event','data':event,'offset_ns':1},
                                                          {'kind':'done','data':None,'offset_ns':1}]}
        self.assertEqual(s.from_stream(raw,3)['verification_steps'],4)
        raw['events'][0]['data']['metrics']['speculative_decoding']['num_spec_steps']=5
        with self.assertRaisesRegex(ValueError,'retained wire chunks'):s.from_stream(raw,3)
