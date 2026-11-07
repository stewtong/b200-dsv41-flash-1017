import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import bootstrap_watchdog
import node_bootstrap
import runtime


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_refuses_without_tmux(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'requires tmux'):
                node_bootstrap.run({})

    def test_image_pull_uses_digest_and_empty_client_config(self):
        image='example/image@sha256:'+'a'*64
        cmd=node_bootstrap.image_pull_command(image,Path('/data/hf-cache/dsv41-bootstrap-image'))
        self.assertEqual(cmd[:3],['sudo','-n','env'])
        self.assertEqual(cmd[3],'DOCKER_CONFIG=/data/hf-cache/dsv41-bootstrap-image/docker-empty-config')
        self.assertEqual(cmd[-2:],['pull',image])
        with self.assertRaises(ValueError):node_bootstrap.image_pull_command('example/image:latest',Path('/tmp'))

    def test_image_headroom_reserves_extraction_and_free_space(self):
        node_bootstrap.check_image_headroom(1_000_000_000_000, 500_000_000_000)
        with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'headroom'):
            node_bootstrap.check_image_headroom(1_000_000_000_000, 400_000_000_000)

    def test_process_group_identity_mismatch_does_not_signal(self):
        with patch.object(runtime,'process_identity',return_value={'pid':44,'start_ticks':2}), \
             patch.object(runtime.os,'killpg') as kill:
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'identity mismatch'):
                runtime.stop_owned_process_group({'pid':44,'start_ticks':1})
            kill.assert_not_called()

    def test_watchdog_stops_only_recorded_group_after_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); identity={'pid':44,'start_ticks':1,'boot_id':'fixture'}
            runtime.atomic_json(root/'bootstrap-state.json',{'watchdog_stop':False,'driver_identity':{'pid':7},
                                'worker_identity':identity,'worker_privileged':True,'deadline_ns':time.monotonic_ns()-1})
            runtime.atomic_json(root/'driver-heartbeat.json',{'monotonic_ns':time.monotonic_ns()})
            sample={'host':{'free_disk_fraction':.9},'uncorrected_ecc':{}}
            with patch.object(bootstrap_watchdog.telemetry,'safety_sample',return_value=sample), \
                 patch.object(runtime,'verify_gpus_idle',return_value=True), \
                 patch.object(runtime,'process_alive',return_value=True), \
                 patch.object(runtime,'stop_owned_process_group') as stop:
                self.assertEqual(bootstrap_watchdog.main(root),1)
                stop.assert_called_once_with(identity,privileged=True)
            self.assertTrue((root/'watchdog-failure.json').exists())
