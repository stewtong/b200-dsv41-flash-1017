import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import attempts
import pilot_recover
import pilot_registry
import runtime


def fixture_config(base,run_dir):
    manifest=Path(base)/'manifest.json'
    manifest.write_text(json.dumps({'plan_sha256':'a'*64,'runtime':{'toolkit_manifest_sha256':'b'*64,'starting_profiles':{'dep8':'c'*64}}}))
    return {'cell_id':'cell','run_dir':str(run_dir),'manifest':str(manifest),'image':'fixture','source_revision':'revision','layout':'dep8'}


class PilotRecoveryTests(unittest.TestCase):
    def test_prelaunch_crash_is_reconciled_without_replaying(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); registry=base/'attempts'; config=fixture_config(base,base/'dsv41-run')
            attempt=pilot_registry.begin(config,'a'*64,registry,lambda pid:{'pid':pid,'start_ticks':1,'boot_id':'fixture'})
            calls=[]
            def preflight(run,ports,invoke):calls.append((run,ports));return {}
            result=pilot_recover.recover(attempt,registry,base,base/'lock',lambda *a,**k:'',lambda _:False,lambda *_:True,preflight)
            self.assertEqual(result['status'],'interrupted_reconciled');self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((attempt/'attempt.json').read_text())['status'],'interrupted')
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin(registry,'cell')

    def test_watchdog_cleaned_crash_reconciles_exact_container(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);registry=base/'attempts';run=base/'dsv41-run';run.mkdir();cid='a'*64
            config=fixture_config(base,run);identity={'pid':7,'start_ticks':1,'boot_id':'fixture'}
            attempt=pilot_registry.begin(config,'a'*64,registry,lambda pid:identity)
            state={'run_id':'run','watchdog_stop':False,'driver_identity':identity,'watchdog_identity':{'pid':8},
                   'owned_processes':[],'container_ids':[cid]}
            (run/'execution-state.json').write_text(json.dumps(state));(run/'watchdog-failure.json').write_text('{"reason":"driver died"}')
            (run/'cleanup-result.json').write_text(json.dumps({'success':True,'outcomes':[{'container_id':cid,'status':'stopped','log_status':'captured'}]}))
            (run/'gpu-idle-after-cleanup.json').write_text('{"idle":true}');(run/('container-'+cid+'-1.log')).write_text('fixture')
            present=[True]
            def invoke(argv,timeout=30):
                if argv[0]=='ps':return ''
                if 'inspect' in argv:return json.dumps([{'Id':cid,'Config':{'Labels':{'campaign':runtime.CAMPAIGN,'run_id':'run'}},'State':{'Running':False}}])
                if 'rm' in argv:present[0]=False;return cid
                if 'ps' in argv:return cid if present[0] else ''
                raise AssertionError(argv)
            result=pilot_recover.recover(attempt,registry,base,base/'lock',invoke,lambda _:False,lambda *_:True,lambda *a:{})
            self.assertFalse(present[0]);self.assertEqual(result['status'],'interrupted_reconciled')
            runtime.verify_evidence_manifest(run)
            self.assertEqual(json.loads((attempt/'attempt.json').read_text())['status'],'interrupted')

    def test_live_driver_blocks_reconciliation(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);registry=base/'attempts';config=fixture_config(base,base/'dsv41-run')
            attempt=pilot_registry.begin(config,'a'*64,registry,lambda pid:{'pid':pid})
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'driver remains alive'):
                pilot_recover.recover(attempt,registry,base,base/'lock',lambda *a,**k:'',lambda _:True)
            self.assertEqual(json.loads((attempt/'attempt.json').read_text())['status'],'running')

    def test_explicit_qualification_result_recovery_uses_its_verifier_without_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);registry=base/'attempts';run=base/'dsv41-run';run.mkdir();config=fixture_config(base,run)
            owner={'pid':7,'start_ticks':1,'boot_id':'fixture'}
            attempt=pilot_registry.begin(config,'a'*64,registry,lambda pid:owner)
            (run/'execution-state.json').write_text(json.dumps({'run_id':'run','driver_identity':owner,'watchdog_identity':{'pid':8},'container_ids':[],'owned_processes':[]}))
            (run/'fixed-qualification-result.json').write_text(json.dumps({'status':'passed','cell_id':'cell'}));runtime.seal_evidence(run)
            checked=[]
            def verify(path):
                checked.append(path);return {'status':'passed','cell_id':'cell','scope':'synthetic verifier dispatch control'}
            result=pilot_recover.recover(attempt,registry,base,base/'lock',lambda *a,**k:'',lambda _:False,lambda *_:True,lambda *a:{},
                result_name='fixed-qualification-result.json',verifier=verify)
            self.assertEqual(checked,[run]);self.assertEqual(result['status'],'terminal_recovered')
            self.assertEqual(json.loads((attempt/'attempt.json').read_text())['status'],'passed')
