"""Ordinary startup transitions serialize with real short ownership inspection."""
from contextlib import contextmanager
from pathlib import Path
import json,sys,tempfile,threading,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import runtime,supervisor


class StartupOwnershipLockTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state={'container_ids':[],'owned_processes':[],'stopping':False,'ready':False}
            runtime.atomic_json(root/'execution-state.json',state)
            runtime.atomic_json(root/'engine-heartbeat.json',{'monotonic_ns':time.monotonic_ns()})
            readiness=root/'readiness.json';runtime.atomic_json(readiness,{'passed':True,'raw':{'complete':True,'http_status':200},'identity':'router'})
            session=supervisor.Supervisor(root/'node.lock',root,'fixture',time.monotonic_ns()+30_000_000_000);session.preflight_done=True
            yield session,state,readiness

    @contextmanager
    def contention(self,session,change=lambda:None):
        held=threading.Event();release=threading.Event()
        def hold():
            with runtime.NodeLock(session.root/'cleanup.lock'):
                held.set();release.wait(3);change()
        thread=threading.Thread(target=hold);thread.start();self.assertTrue(held.wait(1));timer=threading.Timer(.2,release.set);timer.start()
        try:yield
        finally:release.set();thread.join();timer.join()

    def test_readiness_waits_then_records_exact_success(self):
        with self.fixture() as (session,state,readiness):
            with self.contention(session):session.mark_ready([readiness])
            actual=json.loads((session.root/'execution-state.json').read_text());self.assertTrue(actual['ready']);self.assertFalse(actual['stopping'])
            self.assertEqual(actual['readiness_evidence'],[str(readiness)])

    def test_readiness_never_reopens_a_stopping_owner(self):
        with self.fixture() as (session,state,readiness):
            def stopping():runtime.atomic_json(session.root/'execution-state.json',{**state,'stopping':True})
            with self.contention(session,stopping):
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'cleanup has already started'):session.mark_ready([readiness])
            self.assertEqual(json.loads((session.root/'execution-state.json').read_text()),{**state,'stopping':True})

    def test_health_failure_during_wait_prevents_ready(self):
        with self.fixture() as (session,state,readiness):
            with self.contention(session,session.failed.set):
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'safety condition'):session.mark_ready([readiness])
            self.assertEqual(json.loads((session.root/'execution-state.json').read_text()),state)

    def test_collector_registration_waits_before_exact_gate(self):
        with self.fixture() as (session,state,readiness):
            identity={'boot_id':'fixture','pid':42,'start_ticks':1}
            with self.contention(session),patch.object(supervisor.subprocess,'Popen') as popen,patch.object(runtime,'process_identity',return_value=identity):
                popen.return_value.pid=42
                self.assertEqual(session.start_nvml_collector(),identity)
            actual=json.loads((session.root/'execution-state.json').read_text());self.assertEqual(actual['owned_processes'],[identity])
            self.assertEqual(json.loads((session.root/'nvml-gate/router-start.ok').read_text())['pid'],42)

    def test_stopping_during_wait_never_creates_collector_or_gate(self):
        with self.fixture() as (session,state,readiness):
            def stopping():runtime.atomic_json(session.root/'execution-state.json',{**state,'stopping':True})
            with self.contention(session,stopping),patch.object(supervisor.subprocess,'Popen') as popen:
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'cleanup has already started'):session.start_nvml_collector()
                popen.assert_not_called()
            self.assertFalse((session.root/'nvml-gate').exists())

    def test_persistent_contention_keeps_readiness_closed(self):
        with self.fixture() as (session,state,readiness):
            with runtime.NodeLock(session.root/'cleanup.lock'):
                started=time.monotonic()
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'campaign lock held'):session.mark_ready([readiness])
                self.assertGreaterEqual(time.monotonic()-started,5)
            self.assertEqual(json.loads((session.root/'execution-state.json').read_text()),state)


if __name__=='__main__':unittest.main()
