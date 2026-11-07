"""Actual owned subprocess sampling under parent interpreter pressure."""
from pathlib import Path
import json,os,subprocess,sys,tempfile,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import isolated_sampler,runtime,supervisor

class IsolatedSamplerTests(unittest.TestCase):
    def test_parent_rejects_dead_sampler_and_retains_first_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s=supervisor.Supervisor(root/'lock',root,'fixture',time.monotonic_ns()+10**10)
            class Dead:
                def is_alive(self):return False
            s.sampler_process=Dead()
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'exited'):s.ensure_healthy()
            row={'message':'first raw error','type':'ValueError'};runtime.atomic_json(root/'sampler-failure.json',row)
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'original record retained'):s.ensure_healthy()
            self.assertEqual(json.loads((root/'sampler-failure.json').read_text()),row)

    @unittest.skipUnless(sys.platform.startswith('linux'),'actual /proc ownership requires Linux')
    def test_owned_sampler_survives_parent_gil_pressure_and_stops(self):self.process_case(False)

    @unittest.skipUnless(sys.platform.startswith('linux'),'actual /proc ownership requires Linux')
    def test_collection_failure_retained_without_self_cleanup(self):self.process_case(True)

    def process_case(self,fail):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);identity=runtime.process_identity(os.getpid())
            runtime.atomic_json(root/'execution-state.json',{'run_id':'fixture','driver_identity':identity,
                'owned_processes':[],'deadline_ns':time.monotonic_ns()+30_000_000_000,'safety_mount':d,'stopping':False})
            shim=root/'fixture.py';shim.write_text('import sys,time\nfrom pathlib import Path\n'
                +'sys.path.insert(0,'+repr(str(Path(supervisor.__file__).parent))+')\n'
                +'import supervisor,isolated_sampler\n'
                +'def sample(*a,**k):\n'
                +(" raise ValueError('injected acquisition error')\n" if fail else
                  " time.sleep(.005)\n return {'monotonic_ns':time.monotonic_ns(),'uncorrected_ecc':{},'collection_seconds':{'fixture':.005}}\n")
                +'supervisor.telemetry.sample=sample\nraise SystemExit(isolated_sampler.main(sys.argv[-1]))\n')
            actual=subprocess.Popen
            def fixture_popen(argv,**kwargs):
                self.assertEqual(Path(argv[2]).name,'isolated_sampler.py')
                return actual([argv[0],'-B',str(shim),*argv[3:]],**kwargs)
            owner=supervisor.Supervisor(root/'unused.lock',root,'fixture',time.monotonic_ns()+30_000_000_000)
            with patch.object(isolated_sampler.subprocess,'Popen',side_effect=fixture_popen):owner.start_sampler(['fixture'],['fixture'],isolated=True)
            child=owner.sampler_process
            try:
                until=time.monotonic()+10
                while not (root/('sampler-failure.json' if fail else 'isolated-sampler-stats.json')).exists():
                    self.assertTrue(child.is_alive());self.assertLess(time.monotonic(),until);time.sleep(.02)
                state=json.loads((root/'execution-state.json').read_text());self.assertIn(child.identity,state['owned_processes'])
                self.assertEqual(state['measurement_sampler_identity'],child.identity)
                if fail:
                    child.join(5);self.assertFalse(child.is_alive())
                    self.assertEqual(json.loads((root/'sampler-failure.json').read_text())['message'],'injected acquisition error')
                    self.assertFalse((root/'cleanup-result.json').exists())
                    with self.assertRaises(runtime.RuntimeErrorClosed):owner.ensure_healthy()
                else:
                    time.sleep(.65);old=sys.getswitchinterval();sys.setswitchinterval(5)
                    try:
                        until=time.monotonic()+2.3
                        while time.monotonic()<until:pass
                    finally:sys.setswitchinterval(old)
                    time.sleep(.6);owner.ensure_healthy();self.assertGreaterEqual(owner.sampler_stats['count'],6)
                    state['stopping']=True;runtime.atomic_json(root/'execution-state.json',state)
                    child.join(5);self.assertFalse(child.is_alive());self.assertEqual(child.child.returncode,0)
                    rows=[json.loads(x) for p in sorted(root.glob('telemetry-*.jsonl')) for x in p.read_text().splitlines()]
                    self.assertGreaterEqual(len(rows),6)
                    self.assertLessEqual(max(b['monotonic_ns']-a['monotonic_ns'] for a,b in zip(rows,rows[1:])),1_100_000_000)
                    self.assertFalse((root/'sampler-failure.json').exists())
            finally:
                state=json.loads((root/'execution-state.json').read_text());state['stopping']=True;runtime.atomic_json(root/'execution-state.json',state)
                child.join(10)
                if child.is_alive():child.child.terminate();child.child.wait(5)
