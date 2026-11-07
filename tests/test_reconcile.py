from pathlib import Path
import json
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import reconcile
import runtime

class ReconcileTests(unittest.TestCase):
    def test_exact_owned_cleanup_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-run1'; root.mkdir(); cid='a'*64
            state={'run_id':'run1','watchdog_stop':False,'driver_identity':{'pid':1},'watchdog_identity':{'pid':2},'container_ids':[cid]}
            (root/'execution-state.json').write_text(json.dumps(state))
            (root/'watchdog-failure.json').write_text('{"reason":"fixture driver lost"}')
            (root/'cleanup-result.json').write_text(json.dumps({'success':True,'outcomes':[{'container_id':cid,'status':'stopped','log_status':'captured'}]}))
            (root/'gpu-idle-after-cleanup.json').write_text('{"idle":true}')
            (root/('container-'+cid+'-1.log')).write_text('fixture')
            runtime.seal_evidence(root)
            present=[True]; calls=[]; interrupt=[True]
            def fake(argv, timeout=30):
                calls.append(argv)
                if 'inspect' in argv:
                    return json.dumps([{'Id':cid,'Config':{'Labels':{'campaign':runtime.CAMPAIGN,'run_id':'run1'}},'State':{'Running':False}}])
                if 'rm' in argv:
                    if interrupt[0]:
                        interrupt[0]=False
                        raise runtime.RuntimeErrorClosed('fixture interruption')
                    present[0]=False; return cid
                if 'ps' in argv: return cid if present[0] else ''
                raise AssertionError(argv)
            args=(root, str(Path(d)/'lock'), d, fake, lambda _:False, lambda _:True)
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed, 'fixture interruption'):
                reconcile.reconcile(*args)
            self.assertTrue(present[0])
            reconcile.reconcile(*args)
            self.assertFalse(present[0])
            runtime.verify_evidence_manifest(root)
            self.assertTrue((root/'removal-preview.json').exists())
            reconcile.reconcile(*args)
            self.assertEqual(sum('rm' in argv for argv in calls),2)
            runtime.verify_evidence_manifest(root)

    def test_never_removes_with_live_driver(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-run1'; root.mkdir()
            (root/'execution-state.json').write_text(json.dumps({'run_id':'run1','watchdog_stop':True,'driver_identity':{},'watchdog_identity':{}}))
            with self.assertRaises(runtime.RuntimeErrorClosed):
                reconcile.reconcile(root, str(Path(d)/'lock'), d, alive=lambda _:True)

    def test_failed_cleanup_recovery_preserves_failure_and_checks_stopped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-run1';root.mkdir();cid='b'*64
            state={'run_id':'run1','watchdog_stop':False,'driver_identity':{'pid':1},'watchdog_identity':{'pid':2},'container_ids':[cid]}
            (root/'execution-state.json').write_text(json.dumps(state))
            original=json.dumps({'success':False,'outcomes':[{'container_id':cid,'status':'failed','log_status':'captured'}]})
            (root/'cleanup-result.json').write_text(original)
            (root/('container-'+cid+'-1.log')).write_text('fixture')
            present=[True];running=[True]
            def fake(argv,timeout=30):
                if 'inspect' in argv:return json.dumps([{'Id':cid,'Config':{'Labels':{'campaign':runtime.CAMPAIGN,'run_id':'run1'}},'State':{'Running':running[0]}}])
                if 'ps' in argv:return cid if present[0] else ''
                if 'rm' in argv:present[0]=False;return cid
                raise AssertionError(argv)
            args=(root,str(Path(d)/'lock'),d,fake,lambda _:False,lambda _:True)
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'verified stopped'):
                reconcile.recover_failed_cleanup(*args)
            self.assertFalse((root/'cleanup-recovery.json').exists())
            running[0]=False
            reconcile.recover_failed_cleanup(*args)
            reconcile.recover_failed_cleanup(*args)
            self.assertEqual((root/'cleanup-result.json').read_text(),original)
            recovery=root/'cleanup-recovery.json';good=recovery.read_text()
            recovery.write_text(good.replace('independently_verified_stopped','unverified'))
            with self.assertRaises(runtime.RuntimeErrorClosed): reconcile.reconcile(*args)
            self.assertTrue(present[0])
            recovery.write_text(good)
            reconcile.reconcile(*args)
            self.assertFalse(present[0])

    def test_recover_when_stop_succeeded_but_idle_check_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-run1';root.mkdir();cid='c'*64
            state={'run_id':'run1','watchdog_stop':False,'driver_identity':{'pid':1},'watchdog_identity':{'pid':2},'container_ids':[cid]}
            (root/'execution-state.json').write_text(json.dumps(state))
            (root/'driver-failure.json').write_text('{}')
            original=json.dumps({'success':True,'outcomes':[{'container_id':cid,'status':'stopped','log_status':'captured'}]})
            (root/'cleanup-result.json').write_text(original)
            (root/('container-'+cid+'-1.log')).write_text('fixture')
            present=[True]
            def fake(argv,timeout=30):
                if 'inspect' in argv:return json.dumps([{'Id':cid,'Config':{'Labels':{'campaign':runtime.CAMPAIGN,'run_id':'run1'}},'State':{'Running':False}}])
                if 'ps' in argv:return cid if present[0] else ''
                if 'rm' in argv:present[0]=False;return cid
                raise AssertionError(argv)
            args=(root,str(Path(d)/'lock'),d,fake,lambda _:False,lambda _:True)
            reconcile.recover_failed_cleanup(*args)
            reconcile.recover_failed_cleanup(*args)
            self.assertEqual((root/'cleanup-result.json').read_text(),original)
            reconcile.reconcile(*args)
            self.assertFalse(present[0])
