import json
from pathlib import Path
import tempfile
import threading
import time
import subprocess
import types
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import runtime as r
import watchdog as w

class RuntimeTests(unittest.TestCase):
    def test_lock_excludes_second_owner(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'lock'
            with r.NodeLock(p):
                with self.assertRaises(r.RuntimeErrorClosed):
                    with r.NodeLock(p): pass
            with r.NodeLock(p): pass

    def test_lock_rejects_symlink_and_hardlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'target';target.write_text('preserve')
            symlink=root/'symlink';symlink.symlink_to(target)
            with self.assertRaises((OSError,r.RuntimeErrorClosed)):
                with r.NodeLock(symlink): pass
            hardlink=root/'hardlink';hardlink.hardlink_to(target)
            with self.assertRaisesRegex(r.RuntimeErrorClosed,'singly linked'):
                with r.NodeLock(hardlink): pass
            self.assertEqual(target.read_text(),'preserve')

    def test_watchdog_continues_during_bounded_cleanup_lock_contention(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); state={'run_id':'mine','container_ids':[],'owned_processes':[]}
            r.atomic_json(root/'execution-state.json',state)
            with r.NodeLock(root/'cleanup.lock'):
                _, since=w.inspect_ownership(root,state,None,100)
                self.assertEqual(since,100)
                _, since=w.inspect_ownership(root,state,since,129)
                with self.assertRaisesRegex(r.RuntimeErrorClosed,'over 30 seconds'):
                    w.inspect_ownership(root,state,since,131)
            with patch.object(r,'inspect_running_owned') as inspect:
                _, since=w.inspect_ownership(root,state,since,132)
                self.assertIsNone(since)
                inspect.assert_called_once_with([], 'mine', timeout=5)

    def test_stop_checks_exact_ownership_before_mutation(self):
        cid = 'a' * 64
        calls = []
        def fake(argv, timeout=30):
            calls.append(argv)
            if 'inspect' in argv:
                return json.dumps([{'Id': cid, 'Config': {'Labels': {'campaign': r.CAMPAIGN, 'run_id': 'other'}}, 'State': {'Running': True}}])
            raise AssertionError('unexpected mutation')
        with self.assertRaises(r.RuntimeErrorClosed): r.stop_owned(cid, 'mine', fake)
        self.assertEqual(len(calls), 1)

    def test_stop_requires_proof_of_stopped_state(self):
        cid = 'b' * 64
        calls = []
        def fake(argv, timeout=30):
            calls.append(argv)
            if 'inspect' in argv:
                return json.dumps([{'Id': cid, 'Config': {'Labels': {'campaign': r.CAMPAIGN, 'run_id': 'mine'}}, 'State': {'Running': True}}])
            return ''
        with self.assertRaises(r.RuntimeErrorClosed): r.stop_owned(cid, 'mine', fake)
        self.assertEqual(len(calls), 3)

    def test_cleanup_records_exact_timeout_stage(self):
        cid='b'*64
        def fake(argv,timeout=30):
            if 'inspect' in argv:
                return json.dumps([{'Id':cid,'Config':{'Labels':{'campaign':r.CAMPAIGN,'run_id':'mine'}},'State':{'Running':True}}])
            raise subprocess.TimeoutExpired(argv,timeout)
        with self.assertRaises(r.CleanupStepTimeout) as caught:
            r.stop_owned(cid,'mine',fake,timeout=90)
        self.assertEqual(caught.exception.stage,'stop')

    def test_single_container_cleanup_uses_more_than_thirty_seconds_when_available(self):
        with tempfile.TemporaryDirectory() as d:
            cid='d'*64;seen=[]
            result=w.cleanup(d,{'run_id':'mine','container_ids':[cid]},
                stop=lambda _cid,_run,timeout:seen.append(timeout),capture=lambda *a,**k:None)
            self.assertTrue(result['success'])
            self.assertGreater(seen[0],30)
            self.assertLessEqual(seen[0],90)

    def test_cleanup_stops_clients_before_container_and_collector_after(self):
        with tempfile.TemporaryDirectory() as d:
            cid='d'*64;client={'pid':1};collector={'pid':2};seen=[]
            state={'run_id':'mine','container_ids':[cid],'owned_processes':[client,collector],'nvml_identity':collector}
            def idle(_invoke):
                seen.append('idle')
                if seen.count('idle')==1: raise r.RuntimeErrorClosed('GPU compute processes exist')
                if seen.count('idle')==2: raise r.RuntimeErrorClosed('GPU memory not near idle')
            result=w.cleanup(d,state,stop=lambda *_a,**_k:seen.append('container'),
                stop_process=lambda identity,**_k:seen.append('collector' if identity==collector else 'client'),
                capture=lambda *_a,**_k:None,idle=idle)
            self.assertTrue(result['success'])
            self.assertEqual(seen,['client','container','idle','idle','idle','collector'])

    def test_cleanup_rejects_malformed_idle_data_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            collector={'pid':2};calls=[]
            state={'run_id':'mine','container_ids':['d'*64],'owned_processes':[collector],'nvml_identity':collector}
            def idle(_invoke):
                calls.append('idle');raise r.RuntimeErrorClosed('unparseable GPU memory')
            result=w.cleanup(d,state,stop=lambda *_a,**_k:None,stop_process=lambda *_a,**_k:None,
                             capture=lambda *_a,**_k:None,idle=idle)
            self.assertFalse(result['success'])
            self.assertEqual(calls,['idle'])
            self.assertEqual(result['outcomes'][-1]['error_type'],'SafetyCollectorRetained')

    def test_cleanup_timeout_is_unknown_until_fresh_idle_proof(self):
        with tempfile.TemporaryDirectory() as d:
            collector={'pid':2};calls=[]
            state={'run_id':'mine','container_ids':['d'*64],'owned_processes':[collector],'nvml_identity':collector}
            def idle(_invoke):
                calls.append('probe')
                if len(calls)==1: raise subprocess.TimeoutExpired(['nvidia-smi'],3)
            with patch.object(w.time,'sleep'):
                result=w.cleanup(d,state,stop=lambda *_a,**_k:None,stop_process=lambda *_a,**_k:calls.append('stop'),capture=lambda *_a,**_k:None,idle=idle)
            self.assertTrue(result['success'])
            self.assertEqual(calls,['probe','probe','stop'])
            self.assertEqual([x['status'] for x in result['idle_probe_attempts']],['unverified','idle'])

    def test_cleanup_repeated_idle_timeout_exhausts_deadline_and_retains_safety(self):
        with tempfile.TemporaryDirectory() as d:
            collector={'pid':2};stopped=[];clock=[0.0]
            state={'run_id':'mine','container_ids':['d'*64],'owned_processes':[collector],'nvml_identity':collector}
            def idle(_invoke):
                clock[0]+=3
                raise subprocess.TimeoutExpired(['nvidia-smi'],3)
            with patch.object(w.time,'monotonic',side_effect=lambda:clock[0]),patch.object(w.time,'sleep'):
                result=w.cleanup(d,state,stop=lambda *_a,**_k:None,stop_process=lambda *_a,**_k:stopped.append(True),capture=lambda *_a,**_k:None,idle=idle)
            self.assertFalse(result['success'])
            self.assertFalse(stopped)
            self.assertEqual(result['outcomes'][-1]['error_type'],'SafetyCollectorRetained')
            self.assertTrue(all(x['status']=='unverified' for x in result['idle_probe_attempts']))

    def test_cleanup_keeps_collector_when_container_stop_fails(self):
        with tempfile.TemporaryDirectory() as d:
            collector={'pid':2};seen=[]
            state={'run_id':'mine','container_ids':['d'*64],'owned_processes':[collector],'nvml_identity':collector}
            def fail(*_args,**_kwargs): raise r.RuntimeErrorClosed('fixture stop failed')
            result=w.cleanup(d,state,stop=fail,stop_process=lambda *_a,**_k:seen.append('collector'),
                             capture=lambda *_a,**_k:None,idle=lambda _invoke:None)
            self.assertFalse(result['success'])
            self.assertEqual(seen,[])
            self.assertEqual(result['outcomes'][-1]['error_type'],'SafetyCollectorRetained')

    def test_four_stops_share_deadline_concurrently_before_collector_release(self):
        with tempfile.TemporaryDirectory() as d:
            import threading
            ids=[format(i+1,'064x') for i in range(4)];collector={'pid':2}
            barrier=threading.Barrier(4);stopped=[];idle_seen=[];collector_stopped=[]
            state={'run_id':'mine','container_ids':ids,'owned_processes':[collector],'nvml_identity':collector}
            def stop(cid,run,timeout):
                self.assertGreater(timeout,70)
                self.assertEqual(run,'mine');barrier.wait(timeout=1);stopped.append(cid)
            def idle(_invoke):self.assertEqual(set(stopped),set(ids));idle_seen.append(True)
            def stop_process(identity,timeout):
                self.assertTrue(idle_seen);self.assertEqual(identity,collector);collector_stopped.append(True)
            result=w.cleanup(d,state,stop=stop,stop_process=stop_process,capture=lambda *_a,**_k:None,idle=idle)
            self.assertTrue(result['success']);self.assertTrue(collector_stopped)
            self.assertEqual([o['container_id'] for o in result['outcomes'] if 'container_id' in o],ids)

    def test_duplicate_cleanup_ids_fail_before_stopping_clients(self):
        with tempfile.TemporaryDirectory() as d:
            state={'run_id':'mine','container_ids':['a'*64]*2,'owned_processes':[{'pid':2}]};stopped=[]
            with self.assertRaisesRegex(r.RuntimeErrorClosed,'identities'):
                w.cleanup(d,state,stop_process=lambda *_a,**_k:stopped.append(True))
            self.assertFalse(stopped);self.assertNotIn('stopping',state)

    def test_atomic_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'state.json'
            r.atomic_json(p, {'phase': 'pilot'})
            self.assertEqual(json.loads(p.read_text()), {'phase': 'pilot'})
            self.assertEqual(list(Path(d).iterdir()), [p])

    def test_stale_and_future_heartbeats_fail(self):
        self.assertTrue(r.heartbeat_fresh({'monotonic_ns': 100}, 110, 20))
        self.assertFalse(r.heartbeat_fresh({'monotonic_ns': 100}, 121, 20))
        self.assertFalse(r.heartbeat_fresh({'monotonic_ns': 111}, 110, 20))
        self.assertFalse(r.heartbeat_fresh({}, 110, 20))

    def test_watchdog_cycle_reserves_heartbeat_freshness_margin(self):
        self.assertEqual(w.cycle_deadline(100,None),112)
        self.assertEqual(w.cycle_deadline(105,100),113)
        self.assertEqual(w.cycle_deadline(112,100),113)

    def test_shutdown_without_collector_requires_stopped_owned_containers_and_idle_gpus(self):
        cid='f'*64;state={'run_id':'mine','container_ids':[cid],'stopping':True}
        row={'Id':cid,'Config':{'Labels':{'campaign':r.CAMPAIGN,'run_id':'mine'}},'State':{'Running':False}}
        def fake(argv,timeout=30):
            if 'inspect' in argv:return json.dumps([row])
            if '--query-compute-apps=pid' in argv:return ''
            if '--query-gpu=memory.used' in argv:return '\n'.join(['0']*8)
            raise AssertionError(argv)
        w.verify_shutdown_gpu_idle(state,time.monotonic()+10,fake)
        row['State']['Running']=True
        with self.assertRaisesRegex(r.RuntimeErrorClosed,'container runs'):
            w.verify_shutdown_gpu_idle(state,time.monotonic()+10,fake)
        state['stopping']=False
        with self.assertRaisesRegex(r.RuntimeErrorClosed,'unavailable'):
            w.verify_shutdown_gpu_idle(state,time.monotonic()+10,fake)

    def test_shutdown_lock_deferral_uses_recorded_cleanup_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            state={'stopping':True,'cleanup_deadline_ns':200_000_000_000}
            r.atomic_json(Path(d)/'execution-state.json',state)
            with r.NodeLock(Path(d)/'cleanup.lock'):
                current,blocked=w.inspect_ownership(d,state,100,140)
                self.assertTrue(current['stopping'])
                self.assertEqual(blocked,100)
                with self.assertRaisesRegex(r.RuntimeErrorClosed,'cleanup deadline'):
                    w.inspect_ownership(d,state,100,201)

    def test_surviving_container_blocks_new_launch(self):
        with self.assertRaises(r.RuntimeErrorClosed): r.reconcile_orphans('mine', ['survivor'])
        r.reconcile_orphans('mine', [])

    def test_watchdog_stale_engine_and_cleanup_failure(self):
        now = 20_000_000_000
        reasons = w.check({'monotonic_ns':now}, {'monotonic_ns':now}, {}, now, True, now + 1)
        self.assertEqual(reasons, ['engine sample stale'])
        with tempfile.TemporaryDirectory() as d:
            def fail(cid, run_id, timeout=30): raise RuntimeError('fixture')
            result = w.cleanup(d, {'container_ids':['a' * 64], 'run_id':'mine'}, fail, capture=lambda *args, **kwargs: None)
            self.assertFalse(result['success'])
            self.assertEqual(json.loads((Path(d)/'cleanup-result.json').read_text())['outcomes'][0]['status'], 'failed')

    def test_monitor_samples_before_comparison_and_stops_at_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.monotonic_ns()
            r.atomic_json(Path(d)/'execution-state.json', {'run_id':'mine', 'container_ids':[], 'safety_mount':'/', 'deadline_ns':now-1})
            r.atomic_json(Path(d)/'driver-heartbeat.json', {'monotonic_ns':now})
            with patch.object(w.telemetry, 'safety_sample', side_effect=lambda *a, **k: {'host':{'monotonic_ns':time.monotonic_ns()},'uncorrected_ecc':{},'probe_seconds':{}}), patch.object(w, 'cleanup') as cleanup:
                self.assertEqual(w.monitor(d, interval=0), 1)
                cleanup.assert_called_once()
            failure = json.loads((Path(d)/'watchdog-failure.json').read_text())
            self.assertEqual(failure['reasons'], ['operational deadline'])

    def test_create_persists_id_before_inspection(self):
        cid = 'c' * 64
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'execution-state.json'
            r.atomic_json(path, {'run_id':'mine', 'container_ids':[]})
            def fake(argv, timeout=30):
                if 'create' in argv: return cid + '\n'
                self.assertEqual(json.loads(path.read_text())['container_ids'], [cid])
                return json.dumps([{'Id':cid, 'Config':{'Labels':{'campaign':r.CAMPAIGN,'run_id':'mine'}}, 'State':{'Running':False}}])
            argv = ['docker','run','-d','--label','campaign='+r.CAMPAIGN,'--label','run_id=mine','image']
            self.assertEqual(r.create_owned(argv, 'mine', path, fake), cid)

    def test_pid_reuse_never_signalled(self):
        calls = []
        old = {'pid':42, 'start_ticks':10, 'boot_id':'boot'}
        with self.assertRaises(r.RuntimeErrorClosed):
            r.signal_owned_process(old, identity_fn=lambda _: dict(old, start_ticks=11), kill=lambda *x:calls.append(x))
        self.assertEqual(calls, [])

    def test_preflight_rejects_another_gpu_process(self):
        def fake(argv, timeout=30):
            if 'docker' in argv: return ''
            return '1234\n'
        with self.assertRaises(r.RuntimeErrorClosed): r.live_preflight('mine', [], fake)

    def test_cleanup_rereads_ids_after_waiting_for_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); state = {'run_id':'mine', 'container_ids':[]}; seen = []
            r.atomic_json(root/'execution-state.json', state)
            with r.NodeLock(root/'cleanup.lock'):
                thread = threading.Thread(target=lambda: w.cleanup(root, state, lambda cid, run, timeout=30: seen.append((cid, run)), capture=lambda *args, **kwargs: None))
                thread.start(); time.sleep(0.1)
                r.atomic_json(root/'execution-state.json', {'run_id':'mine', 'container_ids':['d'*64]})
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(seen, [('d'*64, 'mine')])

    def test_log_capture_keeps_stdout_and_stderr(self):
        cid = 'e'*64
        def fake_inspect(argv, timeout=30):
            return json.dumps([{'Id':cid, 'Config':{'Labels':{'campaign':r.CAMPAIGN,'run_id':'mine'}}}])
        real_popen = subprocess.Popen
        def fake_popen(argv, **kwargs):
            return real_popen([sys.executable, '-c', 'import sys; print("out"); print("err", file=sys.stderr)'], **kwargs)
        with tempfile.TemporaryDirectory() as d, patch.object(r.subprocess, 'Popen', side_effect=fake_popen):
            p = Path(d)/'container.log'
            r.capture_owned_logs(cid, 'mine', p, fake_inspect)
            self.assertIn('out', p.read_text()); self.assertIn('err', p.read_text())

    def test_placement_rejects_wrong_port_or_gpu(self):
        cid='f'*64
        row={'Id':cid,'Image':'image-id','Config':{'Labels':{'campaign':r.CAMPAIGN,'run_id':'mine'}},
             'State':{'Running':True},'HostConfig':{'DeviceRequests':[{'DeviceIDs':['0','1']}]},
             'NetworkSettings':{'Ports':{'8000/tcp':[{'HostIp':'127.0.0.1','HostPort':'31000'}]}}}
        def fake(argv, timeout=30): return json.dumps([row])
        self.assertEqual(r.verify_placement(cid,'mine',[0,1],31000,'image-id',fake)['gpu_ids'], [0,1])
        with self.assertRaises(r.RuntimeErrorClosed): r.verify_placement(cid,'mine',[0,2],31000,'image-id',fake)
        row['NetworkSettings']['Ports']['8000/tcp'][0]['HostIp']='0.0.0.0'
        with self.assertRaises(r.RuntimeErrorClosed): r.verify_placement(cid,'mine',[0,1],31000,'image-id',fake)

    def test_placement_rejects_effective_cpu_overlap(self):
        cid='f'*64
        row={'Id':cid,'Image':'image-id','Config':{'Labels':{'campaign':r.CAMPAIGN,'run_id':'mine'}},
             'State':{'Running':True,'Pid':123},'HostConfig':{'CpusetCpus':'2,3','DeviceRequests':[{'DeviceIDs':['0']}]},
             'NetworkSettings':{'Ports':{'8000/tcp':[{'HostIp':'127.0.0.1','HostPort':'31000'}]}}}
        def fake(argv, timeout=30): return json.dumps([row])
        with patch.object(r.os, 'sched_getaffinity', return_value={2,3}, create=True):
            self.assertEqual(r.verify_placement(cid,'mine',[0],31000,'image-id',fake,expected_cpus='2,3')['effective_cpus'],[2,3])
        with patch.object(r.os, 'sched_getaffinity', return_value={1,2}, create=True):
            with self.assertRaisesRegex(r.RuntimeErrorClosed, 'effective CPU'):
                r.verify_placement(cid,'mine',[0],31000,'image-id',fake,expected_cpus='2,3')

    def test_watchdog_detects_stopped_engine(self):
        cid='a'*64
        row={'Id':cid, 'Config':{'Labels':{'campaign':r.CAMPAIGN,'run_id':'mine'}}, 'State':{'Running':False}}
        with self.assertRaises(r.RuntimeErrorClosed):
            r.inspect_running_owned([cid], 'mine', lambda *a, **k: json.dumps([row]))

    def test_cleanup_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            seen=[]; state={'run_id':'mine','container_ids':['a'*64]}
            def stop(cid, run, timeout=30): seen.append(cid)
            def capture(*args, **kwargs): pass
            self.assertTrue(w.cleanup(d,state,stop,capture=capture)['success'])
            self.assertTrue(w.cleanup(d,state,stop,capture=capture)['success'])
            self.assertEqual(seen, ['a'*64])

    def test_retirement_requires_all_four_stopped_and_timer_inactive(self):
        with tempfile.TemporaryDirectory() as d:
            names=['a','b','c','d']
            config={'excluded_weight_path':str(Path(d)/'old'), 'retired_container_names':names, 'retired_timer':'collector.timer'}
            rows=[{'Name':'/'+n,'Id':n*64,'State':{'Running':False}} for n in names]
            invoke=lambda *a,**k:json.dumps(rows)
            inactive=lambda *a,**k:types.SimpleNamespace(returncode=3,stdout='inactive\n')
            self.assertTrue(r.verify_retirement(config,invoke,inactive)['previous_weights_absent'])
            rows[0]['State']['Running']=True
            with self.assertRaises(r.RuntimeErrorClosed): r.verify_retirement(config,invoke,inactive)

if __name__ == '__main__': unittest.main()
