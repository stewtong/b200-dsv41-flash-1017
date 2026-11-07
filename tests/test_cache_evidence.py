from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import cache_evidence as c

def metrics(q0,h0,q1,h1):
    return '\n'.join(f'vllm:prefix_cache_{name}_total{{model_name="m",engine="{rank}"}} {value}'
                     for rank,q,h in [(0,q0,h0),(1,q1,h1)] for name,value in [('queries',q),('hits',h)])

class CacheTests(unittest.TestCase):
    def test_single_rank_hit_and_missing_rank(self):
        before={'u':metrics(10,5,20,10)}; after={'u':metrics(110,85,20,10)}
        result=c.difference(before,after,{'u':[0,1]})
        self.assertEqual(result['selected']['rank'],'0')
        self.assertEqual(result['hit_fraction'],0.8)
        with self.assertRaises(ValueError): c.difference(before,after,{'u':[0,1,2]})
        with self.assertRaises(ValueError): c.difference(before,{'u':metrics(110,85,21,10)},{'u':[0,1]})
