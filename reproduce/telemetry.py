"""Collect raw loopback engine metrics and GPU/host safety samples."""
import re
import subprocess
import time
import urllib.request
from urllib.parse import urlsplit
import runtime
import process_metrics
import json
from pathlib import Path
import nvml_safety_probe

NAME = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^\n]*\})?\s+')
LABELED = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}\s+')
ENGINE = re.compile(r'(?:^|,)\s*engine="([^"]+)"(?:,|$)')
# Pinned engine exports every FinishReason, including abort/error, under this
# historically named counter. These are distinct series, not duplicates.
FINISHED_REASONS = ('stop','length','abort','error','repetition')
FINISHED_COUNTER = 'vllm:request_success_total'

def scrape(url, required_names, timeout=3, capture=None):
    p = urlsplit(url)
    if p.scheme != 'http' or p.hostname != '127.0.0.1' or not p.port or p.path != '/metrics':
        raise ValueError('metrics endpoint must be loopback /metrics')
    with urllib.request.urlopen(url, timeout=timeout) as r:
        if r.status != 200: raise ValueError('metrics endpoint did not return 200')
        text = r.read(8 * 1024 * 1024 + 1).decode('utf-8')
    if len(text.encode()) > 8 * 1024 * 1024: raise ValueError('metrics response exceeds limit')
    if capture is not None: capture(text)
    names = {m.group(1) for line in text.splitlines() if (m := NAME.match(line))}
    missing = sorted(set(required_names) - names)
    if missing: raise ValueError('missing required metric names: ' + ', '.join(missing))
    if not names: raise ValueError('empty metrics')
    return text


def essential_engine_sample(endpoints,required_names,expected_ranks):
    """Bounded engine freshness for an explicit unscored overhead control.

    The independent safety watchdog/collector remain enabled. This excludes
    measured process/GPU detail and cannot support a one-hertz coverage claim.
    """
    if not expected_ranks or set(endpoints)!=set(expected_ranks):
        raise ValueError('explicit essential engine population required')
    started=time.monotonic_ns();deadline=started+8_000_000_000;metrics={}
    for endpoint in endpoints:
        remaining=(deadline-time.monotonic_ns())/1e9
        if remaining<=0:raise runtime.RuntimeErrorClosed('essential engine cycle deadline exceeded')
        metrics[endpoint]=scrape(endpoint,required_names,timeout=min(2,remaining))
        if time.monotonic_ns()>=deadline:raise runtime.RuntimeErrorClosed('essential engine cycle deadline exceeded')
    check_rank_coverage(metrics,required_names,expected_ranks)
    return {'started_ns':started,'monotonic_ns':time.monotonic_ns(),'deadline_ns':deadline,
            'engine_metrics':metrics,'coverage':'essential engine freshness only; no1Hz/process/GPU-performance claim'}

def safety_sample(mount='/data', invoke=runtime.command, ecc_baseline=None, gpu_timeout=5, journal_window=6, deadline=None, nvml_data=None):
    if gpu_timeout not in (5,8) or journal_window not in (6,14): raise ValueError('invalid safety sampling bounds')
    def budget(limit):
        remaining = limit if deadline is None else min(limit, deadline-time.monotonic())
        if remaining <= 0: raise runtime.RuntimeErrorClosed('safety sampling cycle deadline exceeded')
        return remaining
    started=time.monotonic()
    host = runtime.safety_sample(mount)
    host_done=time.monotonic()
    gpu=None
    if nvml_data is None:
        gpu = invoke(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,utilization.memory,clocks.sm,power.draw,ecc.errors.uncorrected.volatile.total',
                      '--format=csv,noheader,nounits'], timeout=budget(gpu_timeout))
        lines = gpu.strip().splitlines()
        if len(lines) != 8: raise ValueError('expected eight GPU samples')
        ecc = {}
        for line in lines:
            fields = [x.strip() for x in line.split(',')]
            if len(fields) != 7: raise ValueError('invalid GPU sample')
            index, count = int(fields[0]), int(fields[6])
            if index in ecc or count < 0: raise ValueError('invalid GPU or ECC sample')
            ecc[index] = count
    else:
        ecc={row['index']:row['volatile_uncorrected_ecc'] for row in nvml_data['devices']}
    gpu_done=time.monotonic()
    if set(ecc) != set(range(8)): raise ValueError('incomplete GPU indices')
    if ecc_baseline is not None:
        if set(ecc_baseline) != set(ecc) or any(ecc[k] > ecc_baseline[k] for k in ecc):
            raise ValueError('uncorrected GPU ECC count increased or baseline incomplete')
    kernel = invoke(['sudo','-n','journalctl','-k','--since',str(journal_window)+' seconds ago','--no-pager','-o','cat'], timeout=budget(5))
    journal_done=time.monotonic()
    if deadline is not None and journal_done >= deadline: raise runtime.RuntimeErrorClosed('safety sampling cycle deadline exceeded')
    xid = [line for line in kernel.splitlines() if 'xid' in line.lower()]
    if xid: raise ValueError('GPU Xid observed: ' + '\n'.join(xid))
    return {'monotonic_ns':time.monotonic_ns(), 'host':host, 'gpu_csv':gpu,
            'nvml_safety':nvml_data, 'gpu_source':'nvml' if nvml_data is not None else 'nvidia-smi-aggregate', 'uncorrected_ecc':ecc,
            'xid_lines':0, 'journal_window_seconds':journal_window,
            'probe_seconds':{'host':host_done-started,'gpu':gpu_done-host_done,'journal':journal_done-gpu_done}}

def check_rank_coverage(metrics, required_names, expected_ranks):
    if set(metrics) != set(expected_ranks): raise ValueError('engine endpoint coverage mismatch')
    for url,text in metrics.items():
        seen = set()
        for line in text.splitlines():
            match = LABELED.match(line)
            if not match or match.group(1) not in required_names: continue
            rank = ENGINE.search(match.group(2))
            if rank is None: raise ValueError('required metric missing engine label')
            reason = None
            if match.group(1) == FINISHED_COUNTER:
                reasons = re.findall(r'(?:^|,)\s*finished_reason="([^"\\]*)"(?=,|$)',match.group(2))
                if len(reasons)!=1 or reasons[0] not in FINISHED_REASONS:
                    raise ValueError('missing, duplicate or unknown finished reason')
                reason = reasons[0]
            key = (match.group(1),rank.group(1),reason)
            if key in seen: raise ValueError('duplicate required metric series')
            seen.add(key)
        expected = {(name,str(rank),reason) for name in required_names for rank in expected_ranks[url]
                    for reason in (FINISHED_REASONS if name==FINISHED_COUNTER else (None,))}
        if seen != expected: raise ValueError('required per-rank metrics missing or unexpected')


def shutdown_pending_sample(mount,deadline,last_gpu_sample_ns,invoke=runtime.command):
    """Host/kernel evidence while an already-owned cleanup awaits GPU proof.

    This is not a fresh GPU sample. Dispatch is already disabled, and the
    separate fixed cleanup deadline still requires a fresh final idle proof.
    """
    started=time.monotonic();host=runtime.safety_sample(mount);host_done=time.monotonic()
    remaining=min(5,deadline-time.monotonic())
    if remaining<=0:raise runtime.RuntimeErrorClosed('shutdown host safety deadline exceeded')
    kernel=invoke(['sudo','-n','journalctl','-k','--since','14 seconds ago','--no-pager','-o','cat'],timeout=remaining)
    finished=time.monotonic()
    if finished>=deadline:raise runtime.RuntimeErrorClosed('shutdown host safety deadline exceeded')
    if any('xid' in line.lower() for line in kernel.splitlines()):raise ValueError('GPU Xid observed during owned cleanup')
    return {'monotonic_ns':time.monotonic_ns(),'host':host,'gpu_csv':None,'nvml_safety':None,
        'uncorrected_ecc':None,'gpu_safety_status':'unknown_pending_owned_cleanup',
        'last_gpu_sample_ns':last_gpu_sample_ns,'xid_lines':0,'journal_window_seconds':14,
        'probe_seconds':{'host':host_done-started,'gpu':0.0,'journal':finished-host_done},
        'scope':'host/kernel coverage only; no fresh GPU claim and no permission to resume dispatch'}

def sample(endpoints, required_names, mount='/data', invoke=runtime.command, ecc_baseline=None, expected_ranks=None, server_cpus=None, nvml_identity=None, run_dir=None):
    if not endpoints or not required_names or len(set(endpoints)) != len(endpoints):
        raise ValueError('unique endpoint and metric contracts required')
    started=time.monotonic()
    nvml=None
    if nvml_identity is not None:
        if run_dir is None: raise ValueError('NVML run directory required')
        if any((Path(run_dir)/name).exists() for name in ('nvml-cadence-failure.json','nvml-measurement-failure.json')):
            raise runtime.RuntimeErrorClosed('measured NVML cadence failure was recorded')
        nvml=nvml_safety_probe.validate_live(json.loads((Path(run_dir)/'nvml-live.json').read_text()),nvml_identity,time.monotonic_ns(),max_age_ns=1_500_000_000,measured=True)
        if nvml.get('measurement_schema')!='isolated-nvml-process-and-performance/v1':
            raise ValueError('native NVML measurement fields unavailable')
        # Independently typed raw values are retained in nvml_safety. The CSV is
        # a compatibility projection, not another CLI collection.
        for row in nvml['devices']:
            perf=row.get('performance',{});processes=row.get('compute_processes')
            if (any(type(perf.get(k)) is not int or perf[k]<0 for k in ('gpu_utilization_percent','memory_utilization_percent','sm_clock_mhz','power_milliwatts','sample_started_ns','sample_completed_ns')) or
                perf['gpu_utilization_percent']>100 or perf['memory_utilization_percent']>100 or
                not nvml['sample_started_ns']<=perf['sample_started_ns']<=perf['sample_completed_ns']<=nvml['monotonic_ns'] or
                not isinstance(processes,list) or len({p.get('pid') for p in processes})!=len(processes) or
                any(type(p.get('pid')) is not int or p['pid']<=0 or type(p.get('used_gpu_memory_bytes')) is not int or not 0<=p['used_gpu_memory_bytes']<=row['total_bytes'] for p in processes)):
                raise ValueError('native NVML measurement record malformed')
    data = safety_sample(mount, invoke, ecc_baseline, nvml_data=nvml)
    safety_done=time.monotonic()
    processes=None if nvml is None else [{'gpu_uuid':r['uuid'],'pid':p['pid'],'gpu_memory_mib':p['used_gpu_memory_bytes']//1024**2}
        for r in nvml['devices'] for p in r['compute_processes']]
    data['compute_processes'] = process_metrics.capture(invoke, server_cpus=server_cpus,gpu_processes=processes)
    process_done=time.monotonic()
    if nvml is not None:
        data['gpu_performance_csv']='\n'.join(', '.join(str(v) for v in (r['index'],r['performance']['gpu_utilization_percent'],r['performance']['memory_utilization_percent'],r['performance']['sm_clock_mhz'],r['performance']['power_milliwatts']/1000)) for r in nvml['devices'])
        data['gpu_performance_source']='isolated-nvml-process-and-performance/v1'
    gpu_detail_done=time.monotonic()
    data['engine_metrics'] = {url:scrape(url, required_names) for url in endpoints}
    if run_dir is not None and (Path(run_dir)/'worker-observation-binding.json').exists():
        if (Path(run_dir)/'worker-observation-active.json').exists():
            import measurement_launch
            data['worker_observations']=measurement_launch.collect(run_dir,time.monotonic_ns())
        else:
            data['worker_observations']={'status':'awaiting_physical_binding'}
    metrics_done=time.monotonic()
    data['collection_seconds']={'safety':safety_done-started,'processes':process_done-safety_done,'gpu_performance':gpu_detail_done-process_done,'engine_metrics':metrics_done-gpu_detail_done,'total':metrics_done-started}
    if expected_ranks is not None:
        check_rank_coverage(data['engine_metrics'],required_names,expected_ranks)
    data['monotonic_ns'] = time.monotonic_ns()
    data['coverage'] = 'raw endpoint metrics; per-rank semantics require qualification'
    return data
