import copy
from collections import Counter
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import routing_map


class RoutingMapTests(unittest.TestCase):
    def records(self):
        return [{'session_id':'grid'+str(i),'request_id':'grid'+str(i)+'-r'+str(r),
                 'backend_port':31000+(i%8)//2,'rank_headers':[str(i%2)]}
                for i in range(128) for r in (0,1)]

    def test_balanced_high_demand_and_rotating_low_demand(self):
        m=routing_map.build(self.records(),list(range(31000,31004)),2)
        for c in (1,2,4,8,16,32,64,128):
            rows=routing_map.select(m,c,0)
            counts=Counter((r['endpoint'],r['rank']) for r in rows)
            self.assertEqual(len(rows),c);self.assertEqual(len({r['session_header'] for r in rows}),c)
            if c>=8:self.assertEqual(set(counts.values()),{c//8})
        selected=[routing_map.select(m,1,r)[0] for r in range(3)]
        self.assertEqual([(r['endpoint'],r['rank']) for r in selected],
                         [('http://127.0.0.1:31000/metrics','0'),('http://127.0.0.1:31000/metrics','1'),('http://127.0.0.1:31001/metrics','0')])

    def test_missing_stability_duplicate_and_changed_rank_rejected(self):
        original=self.records()
        for mutate in (lambda a:a.pop(),lambda a:a.append(a[0]),lambda a:a[1].update(rank_headers=['1'])):
            a=copy.deepcopy(original);mutate(a)
            with self.assertRaises(ValueError):routing_map.build(a,list(range(31000,31004)),2)
        m=routing_map.build(original,list(range(31000,31004)),2);m['sessions']['0'][0]='changed'
        with self.assertRaisesRegex(ValueError,'fingerprint'):routing_map.select(m,1,0)
