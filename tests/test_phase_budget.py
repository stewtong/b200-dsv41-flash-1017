"""Keep extended pilot sessions explicit and bound to new manifest deadlines."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_smoke


class PhaseBudgetTests(unittest.TestCase):
    def test_only_explicit_pilot_blocks_receive_the_extended_window(self):
        now=datetime(2026,11,2,tzinfo=timezone.utc)
        manifest={'runtime':{'operational_deadline':(now+timedelta(hours=20)).isoformat()}}
        config={'deadline_utc':(now+timedelta(hours=10)).isoformat(),
                'cells':[{'cell_id':'pilot_fit-2xtp4-i1000000-c1-cold-o512','tasks_file':'/data/tasks.json','request_timeout_seconds':7200}]}
        self.assertEqual(node_smoke.session_seconds(config,manifest,budget='pilot-block',now=now),36000)
        shared=copy.deepcopy(config);shared['cells'][0]['prompt_reuse']={'source_layout':'2xdep4','toolkit_manifest':'/data/hf-cache/fixture/MANIFEST.json','toolkit_sha256':'a'*64}
        self.assertEqual(node_smoke.session_seconds(shared,manifest,budget='pilot-block',now=now),36000)
        shared['cells'][0]['prompt_reuse']=True
        with self.assertRaises(ValueError):node_smoke.session_seconds(shared,manifest,budget='pilot-block',now=now)
        with self.assertRaises(ValueError):node_smoke.session_seconds(config,manifest,now=now)
        for change in ({'execution_phase':'qualification'},{'cells':[]},{'cells':[{'cell_id':'core-cell'}]},
                       {'deadline_utc':(now+timedelta(hours=12,seconds=1)).isoformat()}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                node_smoke.session_seconds(dict(config,**change),manifest,budget='pilot-block',now=now)
        config['deadline_utc']=(now+timedelta(hours=3)).isoformat()
        self.assertEqual(node_smoke.session_seconds(config,manifest,now=now),10800)

    def test_manifest_deadline_and_expiry_still_apply(self):
        now=datetime(2026,11,2,tzinfo=timezone.utc);manifest={'runtime':{'operational_deadline':(now+timedelta(hours=2)).isoformat()}}
        for delta in (timedelta(hours=3),timedelta(0),timedelta(seconds=-1)):
            with self.assertRaises(ValueError):node_smoke.session_seconds({'deadline_utc':(now+delta).isoformat()},manifest,now=now)
        with self.assertRaises(ValueError):node_smoke.session_seconds({'deadline_utc':'2026-11-02T01:00:00'},manifest,now=now)
