import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import verify_telemetry as v


class TelemetryWindowTests(unittest.TestCase):
    def fixture(self,root):
        ranks={'http://127.0.0.1:31000/metrics':list(range(8))};workers=[]
        for i in range(8):workers.append({'host_pid':100+i,'namespace_pid':10+i,'start_ticks':20+i,'gpu_index':i,'gpu_uuid':'GPU-'+str(i)})
        (root/'rank-placement.json').write_text(json.dumps({'status':'passed','containers':[{'workers':workers}]}))
        rows=[];timings=[]
        metric=''.join('# TYPE '+name+' '+kind+'\n'+''.join(name+'{engine="'+str(i)+'"} 0\n' for i in range(8)) for name,kind in v.PILOT_TYPES.items())
        for second in range(5):
            ns=(second+10)*10**9
            rows.append({'monotonic_ns':ns,'host':{'monotonic_ns':ns-10**7,'available_memory_fraction':.5,'free_disk_fraction':.5},
                'uncorrected_ecc':{str(i):0 for i in range(8)},'xid_lines':0,
                'nvml_safety':{'devices':[{'index':i,'uuid':'GPU-'+str(i)} for i in range(8)],'monotonic_ns':ns-10**7,'sample_seconds':.01,'sample_gap_seconds':.5},
                'gpu_performance_csv':'\n'.join(str(i)+', 0, 0, 1000, 100' for i in range(8)),
                'compute_processes':{'processes':{str(w['host_pid']):{'namespace_pid':w['namespace_pid'],'start_ticks':w['start_ticks']} for w in workers},
                                    'gpu_processes':[{'pid':w['host_pid'],'gpu_uuid':w['gpu_uuid']} for w in workers]},
                'engine_metrics':{next(iter(ranks)):metric}})
            timings.append({'monotonic_ns':ns+10**7,'elapsed_seconds':.1,'start_interval_seconds':1 if second else 0,'thread_cpu_seconds':.01})
        self.save(root,rows,timings);return ranks,rows,timings
    def save(self,root,rows,timings):
        for name,data in (('telemetry',rows),('sampler-timing',timings)):
            (root/(name+'-0000.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in data))
    def test_measured_window_counts_exclude_pre_request_samples(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ranks,rows,t=self.fixture(root)
            result=v.verify_window(root,11_200_000_000,13_600_000_000,ranks)
            self.assertEqual(result['samples_in_window'],2);self.assertEqual(result['checked_samples'],4)
    def test_missing_type_worker_or_cadence_never_passes(self):
        for change in ('type','worker','gap','stale','cost'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);ranks,rows,t=self.fixture(root)
                if change=='type':rows[2]['engine_metrics'][next(iter(ranks))]=rows[2]['engine_metrics'][next(iter(ranks))].replace('prefix_cache_queries_total counter','prefix_cache_queries_total gauge')
                if change=='worker':rows[2]['compute_processes']['processes'].pop('100')
                if change=='gap':rows.pop(2);t.pop(2)
                if change=='stale':rows[2]['nvml_safety']['monotonic_ns']-=3_000_000_000
                if change=='cost':t[2]['elapsed_seconds']=2
                self.save(root,rows,t)
                with self.assertRaises(ValueError):v.verify_window(root,11_200_000_000,13_600_000_000,ranks)

    def test_subsecond_request_without_direct_sample_remains_inconclusive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ranks,rows,t=self.fixture(root)
            result=v.verify_window(root,11_200_000_000,11_300_000_000,ranks)
            self.assertEqual(result['samples_in_window'],0)
            self.assertEqual(result['status'],'inconclusive_no_in_window_sample')

    def test_on_time_starts_do_not_hide_completed_sample_gap(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ranks,rows,t=self.fixture(root)
            rows[2]['monotonic_ns']+=230_000_000;t[2]['monotonic_ns']+=230_000_000;t[2]['elapsed_seconds']+=.23
            self.save(root,rows,t)
            with self.assertRaisesRegex(ValueError,'sample gap exceeds'):
                v.verify_window(root,10_500_000_000,13_600_000_000,ranks)

    def test_faster_variable_cost_sampling_and_actual_acquisition_clocks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ranks,rows,t=self.fixture(root)
            for i,cost in enumerate((.27,.55,.28,.48,.29)):
                began=10_000_000_000+i*500_000_000;ns=began+round(cost*1e9)
                rows[i]['monotonic_ns']=ns;rows[i]['host']['monotonic_ns']=ns-10_000_000;rows[i]['nvml_safety']['monotonic_ns']=ns-10_000_000
                t[i].update(monotonic_ns=ns+10_000_000,elapsed_seconds=cost+.01,
                            start_interval_seconds=.5 if i else 0,collection_started_ns=began,
                            sample_completed_ns=ns,target_period_seconds=.5)
            self.save(root,rows,t)
            self.assertEqual(v.verify_window(root,10_300_000_000,12_200_000_000,ranks)['status'],'pilot_window_coverage_verified')
            t[2]['collection_started_ns']=rows[2]['monotonic_ns']+1;self.save(root,rows,t)
            with self.assertRaisesRegex(ValueError,'acquisition clocks'):
                v.verify_window(root,10_300_000_000,12_200_000_000,ranks)
