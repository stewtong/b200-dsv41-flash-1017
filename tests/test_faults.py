from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import faults
import runtime


class FaultTests(unittest.TestCase):
    def test_detects_owned_gpu_faults_and_ignores_general_text(self):
        cid='a'*64
        faults.check([cid],lambda *a,**k:'healthy; memory free')
        for line in ('torch.cuda.OutOfMemoryError: test','CUDA out of memory',
                     'CUDA error: an illegal memory access was encountered','CUDA_ERROR_ILLEGAL_ADDRESS'):
            with self.assertRaises(runtime.RuntimeErrorClosed): faults.check([cid],lambda *a,**k:line)
    def test_log_reader_fails_on_excess_output_and_command_failure(self):
        self.assertEqual(faults.read_logs([sys.executable,'-c','print("ok")']).strip(),'ok')
        with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'size limit'):
            faults.read_logs([sys.executable,'-c','print("x"*10000)'],limit=100)
        with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'failed'):
            faults.read_logs([sys.executable,'-c','raise SystemExit(2)'])

    def test_rejects_invalid_or_duplicate_ids(self):
        with self.assertRaises(runtime.RuntimeErrorClosed):faults.check(['short'])
        with self.assertRaises(runtime.RuntimeErrorClosed):faults.check(['a'*64,'a'*64])
