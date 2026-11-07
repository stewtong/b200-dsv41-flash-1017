"""Bound long fixed blocks without changing per-request or cleanup deadlines."""
import copy,sys,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_smoke,node_fixed_qualification

class QualificationBudgetTests(unittest.TestCase):
    def fixture(self,hours=6):
        now=datetime(2026,11,3,tzinfo=timezone.utc)
        c={'execution_phase':'qualification','deadline_utc':(now+timedelta(hours=hours)).isoformat(),
           'cells':[{'cell_id':'anchor-2xtp4-i1000000-c32-warm_prefix-o2048','tasks_file':'/data/fixture.json',
                     'request_timeout_seconds':7200,'repetition':0}]}
        m={'runtime':{'operational_deadline':(now+timedelta(days=1)).isoformat()}}
        return c,m,now
    def test_warm_million_request_and_conditioning_fit_extended_block(self):
        c,m,now=self.fixture();seconds=node_smoke.session_seconds(c,m,budget='qualification-block',now=now)
        budget=node_fixed_qualification.capture_budget(c['cells'][0],{'tasks':[{'cache_designation':'warm_prefix'}]})
        self.assertEqual(budget,14440);self.assertGreater(seconds,budget+4320)
        with self.assertRaises(ValueError):node_smoke.session_seconds(c,m,now=now)
    def test_extended_budget_requires_explicit_bounded_qualification_rows(self):
        c,m,now=self.fixture()
        for mutate in (
            lambda x:x.update(execution_phase='pilot'),lambda x:x.pop('cells'),
            lambda x:x.update(cells=x['cells']*25),lambda x:x['cells'][0].update(cell_id='core-2xtp4-i1000000-c32-warm_prefix-o2048'),
            lambda x:x['cells'][0].update(extra=True)):
            bad=copy.deepcopy(c);mutate(bad)
            with self.subTest(config=bad),self.assertRaises(ValueError):node_smoke.session_seconds(bad,m,budget='qualification-block',now=now)
    def test_total_and_manifest_deadlines_cannot_be_extended(self):
        c,m,now=self.fixture(12);self.assertEqual(node_smoke.session_seconds(c,m,budget='qualification-block',now=now),43200)
        c['deadline_utc']=(now+timedelta(hours=12,seconds=1)).isoformat()
        with self.assertRaises(ValueError):node_smoke.session_seconds(c,m,budget='qualification-block',now=now)
        c,m,now=self.fixture();m['runtime']['operational_deadline']=(now+timedelta(hours=5)).isoformat()
        with self.assertRaises(ValueError):node_smoke.session_seconds(c,m,budget='qualification-block',now=now)
