import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import traffic


class TrafficTests(unittest.TestCase):
    def rep(self,count=20):
        rows=[{'id':str(i),'scheduled_seconds':60+i,'deadline_seconds':400,
               'terminal_seconds':62+i,'first_visible_seconds':61+i,
               'decode_seconds_per_token':.01,'input_tokens':8000,
               'designated_cache':'cold','valid':True} for i in range(count)]
        return {'start':60,'end':360,'drain_end':400,'requests':rows,
                'queue_samples':[{'seconds':t,'layers':{'client':[],'router':[],'engine':[]}} for t in range(60,360)]}

    def test_seeded_arrivals_preserve_workload_order_and_window(self):
        a=traffic.offered_schedule(.4,7);b=traffic.offered_schedule(.8,7)
        self.assertEqual(a,traffic.offered_schedule(.4,7))
        self.assertNotEqual(a,traffic.offered_schedule(.4,8))
        shape=lambda row:(row['input_tokens'],row['output_tokens'],row['session'])
        self.assertEqual(list(map(shape,a)),list(map(shape,b[:len(a)])))
        self.assertTrue(all(0<r['scheduled_seconds']<360 for r in a))
        self.assertTrue(all(r['phase']==('ramp' if r['scheduled_seconds']<60 else 'measured') for r in a))

    def test_passing_population_and_undersampling(self):
        good=traffic.assess_rate([self.rep()],[(8000,'cold')])
        self.assertEqual(good['verdict'],'passed')
        self.assertEqual(good['buckets'][0]['p95_ttft_seconds'],1)
        self.assertEqual(traffic.assess_rate([self.rep(19)],[(8000,'cold')])['verdict'],'inconclusive')
        self.assertEqual(traffic.assess_rate([self.rep()],[(8000,'cold'),(1000000,'cold')])['verdict'],'inconclusive')

    def test_drain_completion_does_not_inflate_window_rate(self):
        r=self.rep()
        for row in r['requests']:row['terminal_seconds']=370
        x=traffic.assess_rate([r],[(8000,'cold')])
        self.assertEqual(x['verdict'],'failed');self.assertEqual(x['repetitions'][0]['cohort_valid_fraction'],1)
        self.assertEqual(x['repetitions'][0]['in_window_rate_fraction'],0)

    def test_failed_request_and_repetition_cannot_disappear_by_pooling(self):
        bad=self.rep();bad['requests'][0].pop('terminal_seconds');bad['requests'][0]['valid']=False
        x=traffic.assess_rate([self.rep(200),bad],[(8000,'cold')])
        self.assertEqual(x['verdict'],'failed')
        self.assertEqual(x['repetitions'][1]['offered'],20)
        self.assertEqual(x['repetitions'][1]['valid_cohort'],19)

    def test_queue_layers_deduplicated_and_missing_sample_inconclusive(self):
        r=self.rep()
        for sample in r['queue_samples']:
            sample['layers']={'client':['same'],'router':['same'],'engine':['same']}
        self.assertEqual(traffic.queue_evidence(r['queue_samples'],60,360)['fitted_increase'],0)
        r['queue_samples'].pop()
        self.assertEqual(traffic.assess_rate([r],[(8000,'cold')])['verdict'],'inconclusive')

    def test_queue_growth_and_visible_latency_fail(self):
        r=self.rep()
        for s in r['queue_samples']:
            if s['seconds']>=300:s['layers']['client']=['one','two','three']
        self.assertEqual(traffic.assess_rate([r],[(8000,'cold')])['verdict'],'failed')
        r=self.rep()
        for row in r['requests']:row['first_visible_seconds']=row['scheduled_seconds']+11;row['terminal_seconds']=row['scheduled_seconds']+12
        self.assertEqual(traffic.assess_rate([r],[(8000,'cold')])['verdict'],'failed')
