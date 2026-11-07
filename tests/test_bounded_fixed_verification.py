"""Read-only audit concurrency preserves order, bounded work and failures."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import verify_fixed_block


class BoundedFixedVerificationTests(unittest.TestCase):
    def test_four_thread_ceiling_for_larger_population(self):
        gate=threading.Event();started=threading.Event();lock=threading.Lock()
        active=0;peak=0;calls=[]
        def audit(cell):
            nonlocal active,peak
            with lock:
                active+=1;peak=max(peak,active);calls.append(cell)
                if active==4:started.set()
            try:
                if not gate.wait(5):raise RuntimeError('fixture gate was not released')
                return {'cell':cell}
            finally:
                with lock:active-=1
        with patch.object(verify_fixed_block.verify_fixed_qualification,'verify',side_effect=audit):
            with ThreadPoolExecutor(max_workers=1) as caller:
                pending=caller.submit(verify_fixed_block.verify_children,list(range(12)))
                try:self.assertTrue(started.wait(5),'four audits did not start')
                finally:gate.set()
                self.assertEqual(pending.result(timeout=5),[{'cell':i} for i in range(12)])
        self.assertEqual(peak,4)
        self.assertCountEqual(calls,range(12))

    def test_reverse_completion_keeps_preregistered_result_order(self):
        ready=threading.Barrier(5);gates=[threading.Event() for _ in range(4)]
        done=[threading.Event() for _ in range(4)];finished=[]
        def audit(cell):
            ready.wait(timeout=5)
            if not gates[cell].wait(5):raise RuntimeError('fixture gate was not released')
            finished.append(cell);done[cell].set();return {'cell':cell}
        with patch.object(verify_fixed_block.verify_fixed_qualification,'verify',side_effect=audit):
            with ThreadPoolExecutor(max_workers=1) as caller:
                pending=caller.submit(verify_fixed_block.verify_children,list(range(4)))
                try:
                    ready.wait(timeout=5)
                    for cell in reversed(range(4)):
                        gates[cell].set();self.assertTrue(done[cell].wait(5))
                finally:
                    for gate in gates:gate.set()
                self.assertEqual(pending.result(timeout=5),[{'cell':i} for i in range(4)])
        self.assertEqual(finished,[3,2,1,0])

    def test_child_failure_propagates_instead_of_returning_partial_success(self):
        def audit(cell):
            if cell==2:raise ValueError('synthetic invalid raw record')
            return {'cell':cell}
        with patch.object(verify_fixed_block.verify_fixed_qualification,'verify',side_effect=audit):
            with self.assertRaisesRegex(ValueError,'invalid raw record'):
                verify_fixed_block.verify_children(range(6))

    def test_serial_path_and_invalid_work_limits(self):
        with patch.object(verify_fixed_block.verify_fixed_qualification,'verify',side_effect=lambda c:{'cell':c}) as audit:
            self.assertEqual(verify_fixed_block.verify_children(range(3),workers=1),[{'cell':i} for i in range(3)])
            self.assertEqual(audit.call_count,3)
            for directories,workers in [([],4),(range(25),4),([1],0),([1],5),([1],True)]:
                with self.subTest(count=len(directories),workers=workers),self.assertRaises(ValueError):
                    verify_fixed_block.verify_children(directories,workers=workers)
            self.assertEqual(audit.call_count,3)


if __name__=='__main__':unittest.main()
