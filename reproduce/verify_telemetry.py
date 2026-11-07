"""Independently check retained pilot-window cadence, GPU and worker coverage.

Collection costs describe the instrument. Workload impact needs matched controls.
"""
import json
import math
from pathlib import Path
import re
import operator

PILOT_TYPES={'vllm:num_requests_running':'gauge','vllm:num_requests_waiting':'gauge',
             'vllm:kv_cache_usage_perc':'gauge','vllm:prefix_cache_queries_total':'counter',
             'vllm:prefix_cache_hits_total':'counter','vllm:num_preemptions_total':'counter'}
FINISH_LABELS = ('stop','length','abort','error','repetition')


def iter_lines(root,prefix):
    files=sorted(Path(root).glob(prefix+'-*.jsonl'))
    if not files:raise ValueError('missing '+prefix+' records')
    for file in files:
        with file.open() as stream:
            for physical in stream:
                # Preserve str.splitlines(), including Unicode separators and
                # empty records, without retaining an entire rotated file.
                for line in physical.splitlines():
                    yield json.loads(line)


def lines(root,prefix):
    return list(iter_lines(root,prefix))


class IndexedLines:
    """Snapshot record locations; decode only the requested record at a time.

    Iteration still parses every record, including records outside a requested
    measurement window. Existing complete-evidence checks bracket the audit.
    """
    def __init__(self,root,prefix):
        files=sorted(Path(root).glob(prefix+'-*.jsonl'))
        if not files:raise ValueError('missing '+prefix+' records')
        self.files=[];self.records=[]
        for file in files:
            file_index=len(self.files);physical_records=[]
            with file.open() as stream:
                while True:
                    offset=stream.tell();physical=stream.readline()
                    if not physical:break
                    count=len(physical.splitlines())
                    physical_records.append((offset,count))
                    self.records.extend((file_index,offset,i,count) for i in range(count))
            self.files.append((file,physical_records))

    def __len__(self):
        return len(self.records)

    @staticmethod
    def fragments(stream,offset,count):
        if stream.tell()!=offset:stream.seek(offset)
        parts=stream.readline().splitlines()
        if len(parts)!=count:raise ValueError('indexed JSONL record boundaries changed')
        return parts

    def __getitem__(self,index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        file_index,offset,part,count=self.records[operator.index(index)]
        with self.files[file_index][0].open() as stream:
            return json.loads(self.fragments(stream,offset,count)[part])

    def __iter__(self):
        for file,physical_records in self.files:
            with file.open() as stream:
                for offset,count in physical_records:
                    for part in self.fragments(stream,offset,count):
                        yield json.loads(part)


def verify_metrics(snapshot,ranks,types):
    if set(snapshot)!=set(ranks):raise ValueError('metric endpoint coverage differs')
    for url,text in snapshot.items():
        declared={};observed=set()
        for line in text.splitlines():
            match=re.fullmatch(r'# TYPE ([^ ]+) ([^ ]+)',line)
            if match:declared[match[1]]=match[2]
            match=re.match(r'^([^ {]+)\{([^}]*)\}\s+([^ ]+)$',line)
            if not match or match[1] not in types:continue
            rank=re.search(r'(?:^|,)\s*engine="([^"]+)"(?:,|$)',match[2])
            if rank is None:raise ValueError('metric rank missing')
            finish=None
            if match[1]=='vllm:request_success_total':
                labels=re.findall(r'(?:^|,)\s*finished_reason="([^"\\]*)"(?=,|$)',match[2])
                if len(labels)!=1 or labels[0] not in FINISH_LABELS:
                    raise ValueError('invalid finished-reason metric label')
                finish=labels[0]
            key=(match[1],rank[1],finish);value=float(match[3])
            if key in observed or not math.isfinite(value) or value<0:raise ValueError('invalid or duplicate rank metric')
            observed.add(key)
        expected={(name,str(rank),finish) for name in types for rank in ranks[url]
                  for finish in (FINISH_LABELS if name=='vllm:request_success_total' else (None,))}
        if observed!=expected:raise ValueError('incomplete measured rank metrics')
        for name,kind in types.items():
            candidates=[declared.get(name)]
            if name.endswith('_total'):candidates.append(declared.get(name[:-6]))
            if kind not in candidates:raise ValueError('metric type differs from required semantics')


def verify_window(root,start_ns,end_ns,ranks,types=PILOT_TYPES):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    native_required=manifest.get('runtime',{}).get('measurement_source_schema')=='isolated-nvml-process-and-performance/v1'
    if type(start_ns) is not int or type(end_ns) is not int or end_ns<=start_ns:raise ValueError('invalid measurement window')
    samples=IndexedLines(root,'telemetry');timings=lines(root,'sampler-timing')
    if len(samples)!=len(timings):raise ValueError('unpaired sampler records')
    placement=json.loads((root/'rank-placement.json').read_text())
    if placement.get('status')!='passed':raise ValueError('physical worker placement unavailable')
    workers=[w for c in placement['containers'] for w in c['workers']]
    if len(workers)!=8 or len({w['gpu_index'] for w in workers})!=8:raise ValueError('physical worker coverage incomplete')
    stamps=[s['monotonic_ns'] for s in samples]
    if any(type(t) is not int for t in stamps) or any(b<=a for a,b in zip(stamps,stamps[1:])):raise ValueError('nonmonotonic telemetry')
    # Include the nearest sample preceding the request window, then the first
    # following sample when available. Observations stay distinct from continuity.
    indices=[i for i,t in enumerate(stamps) if start_ns<=t<=end_ns]
    prior=[i for i,t in enumerate(stamps) if t<start_ns]
    after=[i for i,t in enumerate(stamps) if t>end_ns]
    if prior:indices.insert(0,prior[-1])
    if after:indices.append(after[0])
    if not indices:raise ValueError('no samples near measured work')
    window_points=[start_ns]+[stamps[i] for i in indices if start_ns<=stamps[i]<=end_ns]+[end_ns]
    if stamps[indices[0]]>start_ns or start_ns-stamps[indices[0]]>1_100_000_000:
        raise ValueError('measured window lacks fresh opening sample')
    if not after and end_ns-stamps[indices[-1]]>1_100_000_000:raise ValueError('measured window closing coverage missing')
    if max(b-a for a,b in zip(window_points,window_points[1:]))>1_100_000_000:raise ValueError('measured window sample gap exceeds1.1seconds')
    inferred=[t['monotonic_ns']-t['elapsed_seconds']*1e9 for t in timings]
    for i in indices:
        if i and (not 0<inferred[i]-inferred[i-1]<=1_120_000_000 or
                  abs((inferred[i]-inferred[i-1])/1e9-timings[i]['start_interval_seconds'])>.02):
            raise ValueError('raw sampler clocks disagree with1Hz cadence')
    costs=[];cpu=[];actual=[];ecc=None
    for i in indices:
        row=samples[i];timing=timings[i]
        if 'collection_started_ns' in timing:
            began=timing['collection_started_ns'];completed=timing.get('sample_completed_ns')
            if (type(began) is not int or type(completed) is not int or
                completed!=row['monotonic_ns'] or not began<=completed<=timing['monotonic_ns'] or
                completed-began>1_000_000_000 or timing.get('target_period_seconds')!=.5):
                raise ValueError('retained sample acquisition clocks invalid')
        elapsed=timing['elapsed_seconds'];interval=timing['start_interval_seconds']
        if (not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or not 0<=elapsed<=1 or
            not isinstance(interval,(int,float)) or not math.isfinite(interval) or not 0<=interval<=1.1 or
            timing['monotonic_ns']<row['monotonic_ns'] or timing['monotonic_ns']-row['monotonic_ns']>1_000_000_000):
            raise ValueError('measured sampler timing invalid')
        if not isinstance(timing['thread_cpu_seconds'],(int,float)) or not math.isfinite(timing['thread_cpu_seconds']) or timing['thread_cpu_seconds']<0:raise ValueError('sampler CPU evidence invalid')
        host=row['host']
        if (not 0<=row['monotonic_ns']-host['monotonic_ns']<=1_500_000_000 or
            not .1<=host['available_memory_fraction']<=1 or not .15<=host['free_disk_fraction']<=1):raise ValueError('host safety coverage missing or stale')
        counts=row['uncorrected_ecc']
        if set(map(str,counts))!=set(map(str,range(8))) or row['xid_lines']!=0:raise ValueError('GPU safety disposition invalid')
        if ecc is None:ecc=counts
        if counts!=ecc:raise ValueError('ECC counters changed in measured coverage')
        verify_metrics(row['engine_metrics'],ranks,types)
        nvml=row['nvml_safety'];devices=nvml['devices']
        if (len(devices)!=8 or {d['index'] for d in devices}!=set(range(8)) or
            {d['uuid'] for d in devices}!={w['gpu_uuid'] for w in workers} or
            not 0<=nvml['sample_seconds']<=1 or not 0<=nvml['sample_gap_seconds']<=1.1 or
            row['monotonic_ns']-nvml['monotonic_ns']<0 or row['monotonic_ns']-nvml['monotonic_ns']>1_500_000_000):
            raise ValueError('GPU safety sample missing or stale')
        performance=[]
        for line in row['gpu_performance_csv'].splitlines():
            values=[v.strip() for v in line.split(',')]
            if len(values)!=5 or not values[0].isdigit() or any(not math.isfinite(float(v)) or float(v)<0 for v in values[1:]):raise ValueError('GPU performance record malformed')
            performance.append(int(values[0]))
        if len(performance)!=8 or set(performance)!=set(range(8)):raise ValueError('GPU performance coverage incomplete')
        if native_required and row.get('gpu_performance_source')!='isolated-nvml-process-and-performance/v1':
            raise ValueError('required native GPU measurement source missing')
        if row.get('gpu_performance_source')=='isolated-nvml-process-and-performance/v1':
            if nvml.get('measurement_schema')!=row['gpu_performance_source']:raise ValueError('GPU measurement source differs')
            expected=[];process_rows=[]
            for device in devices:
                p=device.get('performance',{});procs=device.get('compute_processes')
                if (any(type(p.get(k)) is not int or p[k]<0 for k in ('gpu_utilization_percent','memory_utilization_percent','sm_clock_mhz','power_milliwatts','sample_started_ns','sample_completed_ns')) or
                    p['gpu_utilization_percent']>100 or p['memory_utilization_percent']>100 or
                    not nvml['sample_started_ns']<=p['sample_started_ns']<=p['sample_completed_ns']<=nvml['monotonic_ns'] or
                    not isinstance(procs,list) or len({x.get('pid') for x in procs})!=len(procs) or
                    any(type(x.get('pid')) is not int or x['pid']<=0 or type(x.get('used_gpu_memory_bytes')) is not int or not 0<=x['used_gpu_memory_bytes']<=device['total_bytes'] for x in procs)):
                    raise ValueError('raw native GPU measurement invalid')
                expected.append(', '.join(str(v) for v in (device['index'],p['gpu_utilization_percent'],p['memory_utilization_percent'],p['sm_clock_mhz'],p['power_milliwatts']/1000)))
                process_rows.extend({'gpu_uuid':device['uuid'],'pid':x['pid'],'gpu_memory_mib':x['used_gpu_memory_bytes']//1024**2} for x in procs)
            if row['gpu_performance_csv']!='\n'.join(expected) or row['compute_processes']['gpu_processes']!=process_rows:
                raise ValueError('native GPU projection differs from raw collector')
        for worker in workers:
            pid=worker['host_pid'];record=row['compute_processes']['processes'].get(str(pid))
            matches=[p for p in row['compute_processes']['gpu_processes'] if p['pid']==pid]
            if (not record or record.get('start_ticks')!=worker['start_ticks'] or
                record.get('namespace_pid')!=worker['namespace_pid'] or len(matches)!=1 or matches[0]['gpu_uuid']!=worker['gpu_uuid']):
                raise ValueError('measured process does not match physical worker incarnation')
        costs.append(elapsed);cpu.append(timing['thread_cpu_seconds'])
        if start_ns<=row['monotonic_ns']<=end_ns:actual.append(row['monotonic_ns'])
    result={'status':'pilot_window_coverage_verified' if actual else 'inconclusive_no_in_window_sample','samples_in_window':len(actual),'checked_samples':len(indices),
            'window_seconds':(end_ns-start_ns)/1e9,'max_collection_seconds':max(costs),
            'sample_thread_cpu_seconds':sum(cpu),
            'scope':'retained pilot metric types/ranks, physical workers, eight GPUs and bounded sampling; excludes subprocess CPU and matched workload-overhead acceptance'}
    binding=json.loads((root/'profile-binding.json').read_text()) if (root/'profile-binding.json').exists() else {}
    if (binding.get('delta') or {}).get('worker_observation') is not None:
        import verify_worker_observations
        result['worker_observation_window']=verify_worker_observations.verify_window(root,start_ns,end_ns,[samples[i] for i in indices])
    return result


def main():
    import sys
    import runtime
    root=Path(sys.argv[1]);runtime.verify_evidence_manifest(root)
    config=json.loads((root/'run-config.json').read_text())
    rows=[json.loads(p.read_text()) for p in (root/'pilot-attempt').glob('s[0-9]*.json')]
    if not rows:raise ValueError('pilot request records missing')
    result=verify_window(root,min(r['start_ns'] for r in rows),max(r['end_ns'] for r in rows),config['expected_ranks'])
    runtime.verify_evidence_manifest(root);print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
