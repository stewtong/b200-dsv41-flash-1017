import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CLI=Path(__file__).resolve().parents[1]/'reproduce/campaign.py'


class TrafficCliTests(unittest.TestCase):
    def test_plan_and_slo_artifacts_at_actual_cli_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=root/'plan.json'
            subprocess.run([sys.executable,'-B',str(CLI),'traffic-plan','--rate','.4','--seed','7','--output',str(plan)],check=True,capture_output=True)
            schedule=json.loads(plan.read_text());self.assertFalse(schedule.get('dispatched',False));self.assertEqual(schedule['measured_seconds'],300)
            self.assertTrue(schedule['requests'])
            source=root/'trace.json';out=root/'verdict.json'
            rep={'start':60,'end':360,'drain_end':400,
                 'requests':[{'id':str(i),'scheduled_seconds':60+i,'terminal_seconds':62+i,'first_visible_seconds':61+i,
                              'deadline_seconds':400,'input_tokens':8000,'designated_cache':'cold','valid':True} for i in range(20)],
                 'queue_samples':[{'seconds':t,'layers':{'client':[],'router':[],'engine':[]}} for t in range(60,360)]}
            source.write_text(json.dumps({'required_buckets':[[8000,'cold']],'repetitions':[rep]}))
            subprocess.run([sys.executable,'-B',str(CLI),'slo-assess','--input',str(source),'--output',str(out)],check=True,capture_output=True)
            result=json.loads(out.read_text());self.assertEqual(result['verdict'],'passed')
            self.assertEqual(result['source_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(result['buckets'][0]['p95_ttft_seconds'],1)
            previous=out.read_bytes();source.write_text('{}')
            bad=subprocess.run([sys.executable,'-B',str(CLI),'slo-assess','--input',str(source),'--output',str(out)],capture_output=True)
            self.assertNotEqual(bad.returncode,0);self.assertEqual(out.read_bytes(),previous)
