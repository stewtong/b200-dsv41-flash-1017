import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import routing_evidence
import verify_pilot


class RoutingEvidenceTests(unittest.TestCase):
    def fixture(self):
        text="Consistent hash routing: key='header:x-session-id:s0' -> worker='http://127.0.0.1:31000@0' (index=0)\nConsistent hash routing: key='header:x-session-id:s1' -> worker='http://127.0.0.1:31000@1' (index=1)\n"
        tasks=[{'id':'s0','input_tokens':8000},{'id':'s1','input_tokens':32000}]
        ranks={'http://127.0.0.1:31000/metrics':[0,1]}
        delta=[{'endpoint':url,'rank':str(rank),'queries':7997 if rank==0 else 31997} for url,rs in ranks.items() for rank in rs]
        return text,tasks,ranks,delta

    def test_concurrent_route_receipt_recomputes_independently(self):
        text,tasks,ranks,delta=self.fixture();proof=routing_evidence.verify(text,tasks,ranks,delta)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'pilot-routing.log').write_text(text);(root/'router.log').write_text(text+'shutdown\n')
            (root/'routing-evidence.json').write_text(json.dumps(proof))
            self.assertEqual(verify_pilot.verify_routes(root,tasks,{'expected_ranks':ranks},delta),proof['requests'])
            proof['requests'][0]['rank']='1';(root/'routing-evidence.json').write_text(json.dumps(proof))
            with self.assertRaisesRegex(ValueError,'receipt'):verify_pilot.verify_routes(root,tasks,{'expected_ranks':ranks},delta)

    def test_missing_duplicate_unowned_and_counter_disagreement_rejected(self):
        original=self.fixture()
        for mutate in (lambda a:a.__setitem__(0,a[0].splitlines()[0]),
                       lambda a:a.__setitem__(0,a[0]+a[0]),
                       lambda a:a.__setitem__(0,a[0].replace('s1','unknown')),
                       lambda a:a.__setitem__(0,a[0].replace('@1','@9')),
                       lambda a:a[3][0].update(queries=100)):
            a=copy.deepcopy(list(original));mutate(a)
            with self.assertRaises(ValueError):routing_evidence.verify(*a)

    def test_warm_conditioning_requires_same_route_twice(self):
        text,tasks,ranks,delta=self.fixture();text=text.splitlines()[0]+'\n';tasks=tasks[:1];delta[1]['queries']=0
        with self.assertRaises(ValueError):routing_evidence.verify(text,tasks,ranks,delta,True)
        self.assertEqual(len(routing_evidence.verify(text*2,tasks,ranks,delta,True)['requests']),1)

    def test_shared_log_span_excludes_previous_cells_and_rejects_shift(self):
        text,tasks,ranks,delta=self.fixture();proof=routing_evidence.verify(text,tasks,ranks,delta)
        prefix=text+'earlier cell complete\n';raw=text.encode()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'router.log').write_bytes(prefix.encode()+raw+b'cleanup\n')
            (root/'pilot-routing.log').write_bytes(raw)
            (root/'routing-evidence.json').write_text(json.dumps(proof))
            span={'start_byte':len(prefix.encode()),'end_byte':len(prefix.encode())+len(raw),
                  'sha256':hashlib.sha256(raw).hexdigest()}
            (root/'routing-log-span.json').write_text(json.dumps(span))
            self.assertEqual(verify_pilot.verify_routes(root,tasks,{'expected_ranks':ranks},delta),proof['requests'])
            span['start_byte']+=1;(root/'routing-log-span.json').write_text(json.dumps(span))
            with self.assertRaisesRegex(ValueError,'log span'):
                verify_pilot.verify_routes(root,tasks,{'expected_ranks':ranks},delta)
