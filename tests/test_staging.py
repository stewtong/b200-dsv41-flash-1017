from pathlib import Path
import json
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import runtime
import stage_checkpoint
import stage_watchdog
import stage_worker

class StagingTests(unittest.TestCase):
    def test_headroom_includes_next_file_and_reserve(self):
        stage_checkpoint.check_headroom(1_000_000_000_000, 400_000_000_000, 100_000_000_000)
        with self.assertRaises(runtime.RuntimeErrorClosed):
            stage_checkpoint.check_headroom(1_000_000_000_000, 250_000_000_000, 100_000_000_000)

    def test_nested_mount_rejected(self):
        with self.assertRaises(runtime.RuntimeErrorClosed):
            stage_checkpoint.check_no_mounts('/data/model', lambda *a, **k:'/\n/data/model/other\n')

    def test_previous_unrecorded_worker_blocks_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'dsv41-stage-old'; root.mkdir()
            (root/'stage-state.json').write_text(json.dumps({'driver_identity':None,'worker_identity':None,'watchdog_identity':None}))
            marker=Path(d)/'owner.json'; marker.write_text(json.dumps({'run_dir':str(root)}))
            listing='123 /usr/bin/python /code/stage_worker.py '+str(root.resolve())+' /model\n'
            with self.assertRaises(runtime.RuntimeErrorClosed):
                stage_checkpoint.reconcile_previous(marker, lambda *a, **k:listing, lambda _:False, d)
            with self.assertRaises(runtime.RuntimeErrorClosed):
                stage_checkpoint.reconcile_previous(marker, lambda *a, **k:listing.replace('/usr/bin/python ', '/usr/bin/python -B '), lambda _:False, d)

    def test_new_gpu_work_stops_only_recorded_download(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); identity={'pid':42}
            runtime.atomic_json(root/'stage-state.json',{'driver_identity':{'pid':1},'worker_identity':identity,
                                'deadline_ns':time.monotonic_ns()+60_000_000_000})
            runtime.atomic_json(root/'stage-driver-heartbeat.json',{'monotonic_ns':time.monotonic_ns()})
            sample={'host':{'free_disk_fraction':.5},'uncorrected_ecc':{}}
            with patch.object(stage_watchdog.telemetry,'safety_sample',return_value=sample), \
                 patch.object(stage_watchdog.runtime,'verify_gpus_idle',side_effect=runtime.RuntimeErrorClosed('GPU busy')), \
                 patch.object(stage_watchdog.runtime,'stop_owned_process') as stop:
                self.assertEqual(stage_watchdog.main(root),1)
                stop.assert_called_once_with(identity,timeout=20)

    def test_staging_deadline_stops_only_recorded_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); now=time.monotonic_ns(); identity={'pid':42}
            runtime.atomic_json(root/'stage-state.json',{'driver_identity':{'pid':1},'worker_identity':identity,'deadline_ns':now-1})
            runtime.atomic_json(root/'stage-driver-heartbeat.json',{'monotonic_ns':now})
            sample={'host':{'free_disk_fraction':0.5},'uncorrected_ecc':{}}
            stopped=[]
            with patch.object(stage_watchdog.telemetry,'safety_sample',return_value=sample), \
                 patch.object(stage_watchdog.runtime,'verify_gpus_idle',return_value=True), \
                 patch.object(stage_watchdog.runtime,'process_alive',return_value=True), \
                 patch.object(stage_watchdog.runtime,'stop_owned_process',side_effect=lambda p,timeout=20:stopped.append(p)):
                self.assertEqual(stage_watchdog.main(root),1)
            self.assertEqual(stopped,[identity])
            self.assertTrue((root/'stage-watchdog-failure.json').exists())

    def test_worker_uses_public_download_without_implicit_token(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'go').write_text('{}'); calls=[]
            fake=types.SimpleNamespace(hf_hub_download=lambda *a,**k:calls.append((a,k)))
            with patch.dict(sys.modules,{'huggingface_hub':fake}), patch.dict('os.environ',{},clear=False):
                self.assertEqual(stage_worker.main(d,d,'revision','file.bin','go'),0)
                import os
                self.assertEqual(os.environ['HF_HUB_DISABLE_XET'], '0')
                self.assertEqual(os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'], '0')
                self.assertEqual(os.environ['HF_XET_LOG_DEST'], '/dev/null')
            self.assertIs(calls[0][1]['token'],False)
            self.assertEqual(calls[0][1]['revision'],'revision')

    def test_worker_error_omits_exception_text(self):
        import io
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'go').write_text('{}')
            def fail(*args, **kwargs): raise ValueError('secret fixture text')
            fake=types.SimpleNamespace(hf_hub_download=fail); output=io.StringIO()
            with patch.dict(sys.modules,{'huggingface_hub':fake}), patch.dict('os.environ',{},clear=False), patch('sys.stderr',output):
                self.assertEqual(stage_worker.main(d,d,'revision','file.bin','go'),1)
            self.assertEqual(output.getvalue().strip(),'DOWNLOAD_FAILED ValueError')
