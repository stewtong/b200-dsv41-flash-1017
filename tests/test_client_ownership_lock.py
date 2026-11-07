"""Real lock contention must preserve client ownership and fail closed."""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,threading,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import model_clients,runtime


class ClientOwnershipLockTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);child=root/'client';child.mkdir()
            config={'schema':'fixed-model-client-worker/v1','owner_root':str(root)}
            runtime.atomic_json(child/'config.json',config)
            digest=hashlib.sha256((child/'config.json').read_bytes()).hexdigest()
            script="from pathlib import Path;import sys,time;p=Path(sys.argv[1]);\nwhile not (p/'exit.json').exists():time.sleep(.01)\n"
            process=subprocess.Popen([sys.executable,'-B','-c',script,str(child)])
            try:
                # Linux uses actual /proc incarnation joins. On macOS this
                # fixture substitutes only that unavailable /proc adapter;
                # flock contention, child exit, gates and persisted bytes are real.
                if Path('/proc').is_dir():
                    identity=runtime.process_identity(process.pid)
                    adapter=__import__('contextlib').nullcontext()
                else:
                    identity={'boot_id':'macos-fixture','pid':process.pid,'start_ticks':0}
                    original_alive=runtime.process_alive
                    adapter=patch.object(runtime,'process_alive',side_effect=lambda value:
                        process.poll() is None if value==identity else original_alive(value))
                with adapter:
                    runtime.atomic_json(child/'owned-identity.json',identity)
                    runtime.atomic_json(child/'worker-result.json',{'status':'passed','requests':1})
                    state={'run_id':'fixture','stopping':False,'owned_processes':[identity]}
                    runtime.atomic_json(root/'execution-state.json',state)
                    class Session:
                        failed=False
                        def ensure_healthy(self):
                            if self.failed:raise runtime.RuntimeErrorClosed('fixture safety failure')
                    session=Session();session.root=root;session.run_id='fixture'
                    yield session,[(process,child,digest)],state
            finally:
                if process.poll() is None:process.terminate()
                process.wait(timeout=5)

    def holder(self,root,release,held,mutate=None):
        with runtime.NodeLock(root/'cleanup.lock'):
            held.set();release.wait(3)
            if mutate is not None:mutate()

    def test_retirement_waits_for_actual_short_lock_contention(self):
        with self.fixture() as (session,children,before):
            release=threading.Event();held=threading.Event()
            thread=threading.Thread(target=self.holder,args=(session.root,release,held));thread.start();self.assertTrue(held.wait(1))
            timer=threading.Timer(.2,release.set);timer.start()
            try:model_clients.retire(session,children)
            finally:release.set();thread.join();timer.join()
            state=json.loads((session.root/'execution-state.json').read_text())
            self.assertEqual(state['owned_processes'],[])
            self.assertEqual(state['completed_processes'][0]['identity'],before['owned_processes'][0])
            self.assertEqual(children[0][0].returncode,0)
            self.assertEqual(json.loads((children[0][1]/'release-receipt.json').read_text())['exit_code'],0)

    def test_stopping_after_lock_wait_preserves_owned_client(self):
        with self.fixture() as (session,children,before):
            release=threading.Event();held=threading.Event()
            def stopping():runtime.atomic_json(session.root/'execution-state.json',{**before,'stopping':True})
            thread=threading.Thread(target=self.holder,args=(session.root,release,held,stopping));thread.start();self.assertTrue(held.wait(1))
            timer=threading.Timer(.2,release.set);timer.start()
            try:
                with self.assertRaisesRegex(RuntimeError,'outside active owner'):model_clients.retire(session,children)
            finally:release.set();thread.join();timer.join()
            self.assertFalse((children[0][1]/'exit.json').exists())
            self.assertTrue(runtime.process_alive(before['owned_processes'][0]))
            self.assertEqual(json.loads((session.root/'execution-state.json').read_text()),{**before,'stopping':True})

    def test_safety_failure_after_wait_prevents_retirement(self):
        with self.fixture() as (session,children,before):
            release=threading.Event();held=threading.Event()
            def failure():session.failed=True
            thread=threading.Thread(target=self.holder,args=(session.root,release,held,failure));thread.start();self.assertTrue(held.wait(1))
            timer=threading.Timer(.2,release.set);timer.start()
            try:
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'fixture safety failure'):model_clients.retire(session,children)
            finally:release.set();thread.join();timer.join()
            self.assertFalse((children[0][1]/'exit.json').exists())
            self.assertEqual(json.loads((session.root/'execution-state.json').read_text()),before)

    def test_persistent_contention_fails_without_state_or_gate_write(self):
        with self.fixture() as (session,children,before):
            with runtime.NodeLock(session.root/'cleanup.lock'):
                started=time.monotonic()
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'campaign lock held'):model_clients.retire(session,children)
                elapsed=time.monotonic()-started
            self.assertGreaterEqual(elapsed,5)
            self.assertLess(elapsed,6.5)
            self.assertFalse((children[0][1]/'exit.json').exists())
            self.assertEqual(json.loads((session.root/'execution-state.json').read_text()),before)
            self.assertTrue(runtime.process_alive(before['owned_processes'][0]))


if __name__=='__main__':unittest.main()
