"""Native ABI, no-CLI composition and independent raw projection checks."""
import copy,ctypes,json,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import nvml_safety_probe as n
import telemetry,verify_telemetry
import test_verify_telemetry as window_fixture


class NativeMeasurementTests(unittest.TestCase):
    def collector(self):
        c=n.Collector.__new__(n.Collector);c.handles=[(i,i,'GPU-'+format(i,'08x')) for i in range(8)]
        def util(handle,p):p._obj.gpu=75;p._obj.memory=60;return 0
        def clock(handle,domain,p):self.assertEqual(domain.value,1);p._obj.value=1500;return 0
        def power(handle,p):p._obj.value=500123;return 0
        def processes(handle,count,rows):
            rows[0].pid=100+handle;rows[0].usedGpuMemory=4*1024**2;rows[0].gpuInstanceId=2**32-1;rows[0].computeInstanceId=2**32-1;count._obj.value=1;return 0
        c.lib=SimpleNamespace(nvmlDeviceGetUtilizationRates=util,nvmlDeviceGetClockInfo=clock,
                              nvmlDeviceGetPowerUsage=power,nvmlDeviceGetComputeRunningProcesses_v3=processes)
        return c

    def test_installed_process_struct_and_native_performance_units(self):
        self.assertEqual(ctypes.sizeof(n.ProcessInfo),24);self.assertEqual(n.ProcessInfo.usedGpuMemory.offset,8)
        self.assertEqual(n.ProcessInfo.gpuInstanceId.offset,16);self.assertEqual(ctypes.sizeof(n.Utilization),8)
        c=self.collector();base={'devices':[{'index':i,'uuid':u,'total_bytes':100*1024**2} for i,_,u in c.handles],'call_timings':[]}
        with patch.object(c,'read',return_value=base):result=c.read_measurement()
        self.assertEqual(result['measurement_schema'],'isolated-nvml-process-and-performance/v1')
        self.assertEqual(result['devices'][7]['compute_processes'][0]['pid'],107)
        self.assertEqual(result['devices'][0]['performance']['power_milliwatts'],500123)
        self.assertEqual(len(result['call_timings']),32)

    def test_process_table_growth_is_bounded_and_missing_support_fails(self):
        c=self.collector();calls=[]
        def grow(handle,count,rows):calls.append(count._obj.value);count._obj.value=65;return 7
        c.lib.nvmlDeviceGetComputeRunningProcesses_v3=grow
        with self.assertRaises(RuntimeError):c.process_rows(0,[])
        self.assertEqual(calls,[64,65])
        c.lib.nvmlDeviceGetComputeRunningProcesses_v3=lambda *a:2
        with self.assertRaises(RuntimeError):c.process_rows(0,[])

    def test_measurement_error_keeps_original_safety_feed_and_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'nvml-measurement-mode.json').write_text('{}');(root/'execution-state.json').write_text('{"stopping":false}')
            class C:
                setup_timings=[]
                def read_measurement(self):raise RuntimeError('unsupported native metric')
                def read(self,include_processes=False):return {'devices':[]}
                def close(self):pass
            with patch.object(n,'Collector',C),patch.object(n.time,'sleep',side_effect=StopIteration):
                with self.assertRaises(StopIteration):n.loop(root)
            self.assertTrue((root/'nvml-measurement-failure.json').exists())
            self.assertFalse(json.loads((root/'nvml-live.json').read_text())['measurement_mode'])

    def test_actual_sample_composition_never_calls_gpu_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c=self.collector();start=time.monotonic_ns()
            base={'devices':[{'index':i,'uuid':u,'total_bytes':100*1024**2,'free_bytes':90*1024**2,'used_bytes':10*1024**2,'reserved_bytes':0,'volatile_uncorrected_ecc':0} for i,_,u in c.handles],'call_timings':[]}
            with patch.object(c,'read',return_value=base):raw=c.read_measurement()
            raw.update(pid=42,sample_started_ns=start,monotonic_ns=time.monotonic_ns(),sample_seconds=.01,sample_gap_seconds=.5,measurement_mode=True)
            (root/'nvml-live.json').write_text(json.dumps(raw))
            def invoke(argv,timeout=30):
                self.assertEqual(argv[:3],['sudo','-n','journalctl']);return ''
            metrics='active{engine="0"} 0\n';captured=[]
            def processes(*args,**kwargs):captured.extend(kwargs['gpu_processes']);return {'processes':{},'gpu_processes':kwargs['gpu_processes']}
            with patch.object(n.runtime,'process_alive',return_value=True),patch.object(telemetry.runtime,'safety_sample',return_value={'monotonic_ns':time.monotonic_ns()}),patch.object(telemetry.process_metrics,'capture',side_effect=processes),patch.object(telemetry,'scrape',return_value=metrics):
                result=telemetry.sample(['http://127.0.0.1:31000/metrics'],['active'],mount='/',invoke=invoke,nvml_identity={'pid':42},run_dir=root)
            self.assertEqual(len(captured),8)
            self.assertIn('500.123',result['gpu_performance_csv'])
            self.assertEqual(result['gpu_performance_source'],raw['measurement_schema'])

    def test_independent_verifier_recomputes_native_projection_and_requires_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture=window_fixture.TelemetryWindowTests();ranks,rows,timings=fixture.fixture(root)
            schema='isolated-nvml-process-and-performance/v1';(root/'manifest.json').write_text(json.dumps({'runtime':{'measurement_source_schema':schema}}))
            for row in rows:
                row['gpu_performance_source']=schema;raw=row['nvml_safety'];raw['measurement_schema']=schema;raw['sample_started_ns']=raw['monotonic_ns']-10_000_000
                for device in raw['devices']:
                    i=device['index'];device['total_bytes']=100*1024**2
                    device['performance']={'gpu_utilization_percent':0,'memory_utilization_percent':0,'sm_clock_mhz':1000,'power_milliwatts':100000,'sample_started_ns':raw['sample_started_ns'],'sample_completed_ns':raw['monotonic_ns']}
                    device['compute_processes']=[{'pid':100+i,'used_gpu_memory_bytes':4*1024**2}]
                row['gpu_performance_csv']='\n'.join(str(i)+', 0, 0, 1000, 100.0' for i in range(8))
                for process in row['compute_processes']['gpu_processes']:process['gpu_memory_mib']=4
            fixture.save(root,rows,timings)
            self.assertEqual(verify_telemetry.verify_window(root,11_200_000_000,13_600_000_000,ranks)['status'],'pilot_window_coverage_verified')
            rows[2]['gpu_performance_csv']=rows[2]['gpu_performance_csv'].replace('100.0','99.0');fixture.save(root,rows,timings)
            with self.assertRaisesRegex(ValueError,'projection differs'):verify_telemetry.verify_window(root,11_200_000_000,13_600_000_000,ranks)
            rows[2].pop('gpu_performance_source');fixture.save(root,rows,timings)
            with self.assertRaisesRegex(ValueError,'source missing'):verify_telemetry.verify_window(root,11_200_000_000,13_600_000_000,ranks)
