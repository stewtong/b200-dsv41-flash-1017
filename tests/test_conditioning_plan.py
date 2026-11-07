import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import conditioning_plan


class ConditioningPlanTests(unittest.TestCase):
    def test_expands_warm_requests_and_does_not_overwrite(self):
        data={'plan_sha256':'a'*64,'matrix_sha256':'b'*64,'cells':[
              {'id':'warm','suite':'core','layout':'dep8','input_tokens':8000,'concurrency':2,'cache':'warm_prefix','repetitions':3},
              {'id':'cold','suite':'core','layout':'dep8','input_tokens':8000,'concurrency':2,'cache':'cold','repetitions':3}]}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'ledger.json';source.write_text(json.dumps(data));out=root/'warm.jsonl'
            self.assertEqual(conditioning_plan.write(source,out),6)
            result=[json.loads(x) for x in out.read_text().splitlines()]
            self.assertFalse(result[0]['complete_universe'])
            self.assertEqual(len({x['id'] for x in result[1:]}),6)
            with self.assertRaises(FileExistsError):conditioning_plan.write(source,out)

    def test_requires_plan_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'ledger.json';source.write_text('{"cells":[]}')
            with self.assertRaises(ValueError):conditioning_plan.write(source,root/'out')

    def test_multi_turn_dependencies_do_not_invent_conditioning_requests(self):
        ledger={'cells':[{'id':'grow','suite':'growing','layout':'dep8','parameters':{'input':128000,'concurrency':2},'repetitions':3},
                         {'id':'cold','suite':'cache_control','layout':'dep8','parameters':{'input':32000,'concurrency':1,'control':'cold'},'repetitions':1}]}
        rows=list(conditioning_plan.rows(ledger));self.assertEqual(len(rows),28)
        self.assertTrue(all(r['kind']=='measured_turn_dependency' and not r['separate_conditioning_request'] for r in rows))
        self.assertEqual(len({r['id'] for r in rows}),28)
        for r in rows:
            if r['parent_cell']=='cold' or r['turn']==0:
                self.assertEqual(r['designated_cache'],'cold');self.assertIsNone(r['prefix_source'])
            else:
                self.assertEqual(r['designated_cache'],'warm_prefix')
                self.assertTrue(r['prefix_source'].endswith('-turn'+str(r['turn']-1)))
