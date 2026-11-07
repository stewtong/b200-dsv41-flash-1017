"""Intentional unscored sampler controls retain independent safety enforcement."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import runtime,supervisor,telemetry


class SamplerOverheadControlTests(unittest.TestCase):
    def test_total_instrumentation_contract_is_explicit_and_binds_the_worker_arm(self):
        for arm in ('control_before','challenger','control_after'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);s,cell=self.fixture(root,arm)
                runtime.atomic_json(root/'run-config.json',{'cell_id':cell['id'],'layout':'2xtp4','overhead_contract':'total-worker-and-host/v1'})
                runtime.atomic_json(root/'profile-binding.json',{'delta':{'worker_observation':{'fixture':True}} if arm=='challenger' else {}})
                record=s.begin_sampler_control(cell['id'],s.deadline_ns-121_000_000_000)
                self.assertEqual(record['schema'],'bounded-sampler-overhead-control/v2')
                self.assertEqual(record['measurement_contract'],'total-worker-and-host/v1')
                self.assertTrue(record['safety_watchdog_enabled'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s,cell=self.fixture(root,'challenger')
            runtime.atomic_json(root/'run-config.json',{'cell_id':cell['id'],'layout':'2xtp4','overhead_contract':'total-worker-and-host/v1'})
            runtime.atomic_json(root/'profile-binding.json',{'delta':{}})
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'worker observation arm differs'):
                s.begin_sampler_control(cell['id'],s.deadline_ns-121_000_000_000)
            self.assertIsNone(s.sampler_control);self.assertFalse((root/'sampler-control.json').exists())

    def fixture(self,root,arm='control_before'):
        cell={'id':'overhead-fixture','suite':'sampler_overhead','layout':'2xtp4','repetitions':3,
              'parameters':{'arm':arm,'input':8000,'concurrency':8}}
        runtime.atomic_json(root/'execution-state.json',{'ready':True,'stopping':False})
        runtime.atomic_json(root/'run-config.json',{'cell_id':cell['id'],'layout':'2xtp4'})
        runtime.atomic_json(root/'manifest.json',{'cells':[cell]})
        s=supervisor.Supervisor(root/'lock',root,'fixture',time_limit(),phase='qualification')
        s.sampler=object()
        return s,cell

    def test_only_exact_preregistered_unscored_control_can_change_sampler(self):
        for arm,mode in [('control_before','essential'),('challenger','full'),('control_after','essential')]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);s,cell=self.fixture(root,arm)
                record=s.begin_sampler_control(cell['id'],s.deadline_ns-121_000_000_000)
                self.assertEqual(record['mode'],mode);self.assertTrue(record['safety_watchdog_enabled'])
                self.assertTrue(json.loads((root/'execution-state.json').read_text())['ready'])
                with self.assertRaises(runtime.RuntimeErrorClosed):s.begin_sampler_control(cell['id'],record['until_ns'])
        changes=[{'suite':'core'},{'layout':'dep8'},{'repetitions':1},
                 {'parameters':{'arm':'control_before','input':8000,'concurrency':1}},
                 {'parameters':{'arm':'invented','input':8000,'concurrency':8}}]
        for change in changes:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);s,cell=self.fixture(root);cell.update(change)
                runtime.atomic_json(root/'manifest.json',{'cells':[cell]})
                with self.assertRaises(runtime.RuntimeErrorClosed):s.begin_sampler_control('overhead-fixture',s.deadline_ns-121_000_000_000)
                self.assertIsNone(s.sampler_control);self.assertFalse((root/'sampler-control.json').exists())

    def test_essential_engine_poll_has_one_total_budget(self):
        clock=[0];calls=[]
        def scrape(url,names,timeout):
            calls.append(timeout);clock[0]+=7_000_000_000 if len(calls)==1 else 2_000_000_000
            return 'declared fixture'
        with patch.object(telemetry,'time',SimpleNamespace(monotonic_ns=lambda:clock[0])), \
             patch.object(telemetry,'scrape',side_effect=scrape):
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'essential engine cycle deadline'):
                telemetry.essential_engine_sample(['a','b'],['required'],{'a':[0],'b':[0]})
        self.assertEqual(calls,[2,1])

    def test_essential_arm_uses_separate_raw_records_and_keeps_engine_heartbeat(self):
        self.run_sampler(expired=False)

    def test_expired_control_fails_closed_instead_of_continuing_unmeasured(self):
        self.run_sampler(expired=True)

    def run_sampler(self,expired):
        clock=[10.0];rows={}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime.atomic_json(root/'execution-state.json',{})
            s=supervisor.Supervisor(root/'lock',root,'fixture',10**15,phase='qualification')
            s.sampler_control={'mode':'essential','cell_id':'overhead-fixture',
                               'until_ns':9_000_000_000 if expired else 60_000_000_000}
            class Stop:
                def is_set(self):return len(rows.get('essential-engine',[]))>=3
                def wait(self,seconds):clock[0]+=seconds
            class Thread:
                def __init__(self,target,daemon):self.target=target
                def start(self):self.target()
            def sample(*args):
                clock[0]+=.1
                return {'monotonic_ns':round(clock[0]*1e9),'engine_metrics':{'fixture':'declared metrics'}}
            def append(root,prefix,index,size,row):rows.setdefault(prefix,[]).append(row);return index,size
            s.sampler_stop=Stop()
            with patch.object(s,'ensure_healthy'),patch.object(supervisor.threading,'Thread',Thread), \
                 patch.object(supervisor.time,'monotonic',side_effect=lambda:clock[0]), \
                 patch.object(supervisor.time,'monotonic_ns',side_effect=lambda:round(clock[0]*1e9)), \
                 patch.object(supervisor.telemetry,'sample') as full, \
                 patch.object(supervisor.telemetry,'essential_engine_sample',side_effect=sample), \
                 patch.object(supervisor.runtime,'append_sample',side_effect=append), \
                 patch.object(supervisor.watchdog,'cleanup') as cleanup:
                s.start_sampler(['fixture'],['required'],{'fixture':[0]})
            full.assert_not_called()
            if expired:
                self.assertTrue(s.failed.is_set());cleanup.assert_called_once()
                self.assertIn('control deadline expired',json.loads((root/'sampler-failure.json').read_text())['message'])
                self.assertFalse((root/'engine-heartbeat.json').exists())
            else:
                self.assertFalse(s.failed.is_set());cleanup.assert_not_called()
                self.assertEqual(len(rows['essential-engine']),3);self.assertNotIn('telemetry',rows)
                self.assertEqual(s.sampler_stats['count'],0)
                self.assertEqual(json.loads((root/'engine-heartbeat.json').read_text())['mode'],'essential')
                self.assertEqual(json.loads((root/'sampler-control-applied.json').read_text())['cell_id'],'overhead-fixture')


def time_limit():
    import time
    return time.monotonic_ns()+400_000_000_000


if __name__=='__main__':unittest.main()
