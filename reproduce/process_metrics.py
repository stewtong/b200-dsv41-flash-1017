"""Capture raw GPU-process CPU and RSS evidence without assuming rank identity."""
from pathlib import Path
import re
import runtime


def parse_cpu_list(value):
    result = set()
    for part in value.split(','):
        bounds = part.split('-')
        if len(bounds) > 2 or any(not x.isdigit() for x in bounds): raise ValueError('invalid CPU list')
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first or last > 1_000_000: raise ValueError('invalid CPU range')
        result.update(range(first, last+1))
    return result


def capture(invoke=runtime.command, proc_root='/proc', server_cpus=None, gpu_processes=None):
    if gpu_processes is not None:
        if not isinstance(gpu_processes,list) or any(set(r)!={'gpu_uuid','pid','gpu_memory_mib'} or type(r['pid']) is not int or type(r['gpu_memory_mib']) is not int for r in gpu_processes):
            raise ValueError('invalid retained GPU process table')
        raw='\n'.join(str(r['gpu_uuid'])+', '+str(r['pid'])+', '+str(r['gpu_memory_mib']) for r in gpu_processes)
    else:
        raw = invoke(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                      '--format=csv,noheader,nounits'], timeout=5)
    rows = []
    for line in raw.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) != 3 or not re.fullmatch(r'GPU-[0-9a-fA-F-]+', parts[0]) or not parts[1].isdigit() or not parts[2].isdigit():
            raise ValueError('invalid GPU compute-process sample')
        rows.append({'gpu_uuid':parts[0], 'pid':int(parts[1]), 'gpu_memory_mib':int(parts[2])})
    if not rows:
        raise ValueError('no GPU compute processes in serving sample')
    processes = {}
    for pid in sorted({row['pid'] for row in rows}):
        base = Path(proc_root)/str(pid)
        stat = (base/'stat').read_text()
        fields = stat[stat.rfind(')')+2:].split()
        if len(fields) < 20: raise ValueError('invalid process stat')
        status = dict(line.split(':',1) for line in (base/'status').read_text().splitlines() if ':' in line)
        rss = status.get('VmRSS','').strip().split()
        affinity = status.get('Cpus_allowed_list','').strip()
        if len(rss) != 2 or not rss[0].isdigit() or rss[1] != 'kB' or not re.fullmatch(r'[0-9,-]+',affinity):
            raise ValueError('missing process RSS or CPU affinity')
        if server_cpus is not None and not parse_cpu_list(affinity).issubset(set(server_cpus)):
            raise ValueError('GPU process overlaps reserved client CPUs')
        after = (base/'stat').read_text()
        after_fields = after[after.rfind(')')+2:].split()
        if len(after_fields) < 20 or fields[19] != after_fields[19]: raise ValueError('process identity changed during sample')
        processes[str(pid)] = {'start_ticks':int(fields[19]), 'cpu_ticks':int(fields[11])+int(fields[12]),
                               'rss_kib':int(rss[0]), 'cpu_affinity':affinity}
        namespace = status.get('NSpid','').split()
        if namespace:
            if not all(x.isdigit() for x in namespace) or int(namespace[0]) != pid:
                raise ValueError('invalid namespace process identity')
            processes[str(pid)]['namespace_pid'] = int(namespace[-1])
    return {'gpu_processes':rows, 'processes':processes,
            'scope':'raw process samples; GPU rank and container ownership are not established'}
