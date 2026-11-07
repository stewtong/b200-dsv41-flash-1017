"""Join retained single-axis worker logs to exact PIDs and physical GPU UUIDs."""
import re


def _map_workers(log_text, container_host_pids, process_records, gpu_processes,
                 gpu_devices, expected_ranks, axis):
    """Only accept the installed single-axis worker naming for the chosen layout."""
    expected = set(expected_ranks)
    if not expected or any(type(x) is not int or x < 0 for x in expected):
        raise ValueError('explicit worker ranks required')
    host_pids = set(container_host_pids)
    if len(host_pids) != len(container_host_pids):
        raise ValueError('duplicate container process identity')
    workers = {}
    pattern = (r'\(Worker_DP(\d+)_EP\d+\s+pid=(\d+)\)' if axis == 'dp' else
               r'\(Worker_TP(\d+)\s+pid=(\d+)\)')
    for rank, pid in re.findall(pattern, log_text):
        rank, pid = int(rank), int(pid)
        if rank in workers and workers[rank] != pid:
            raise ValueError('worker incarnation changed in retained logs')
        workers[rank] = pid
    if set(workers) != expected or len(set(workers.values())) != len(workers):
        raise ValueError('worker coverage incomplete or ambiguous')
    devices = {d['uuid']: d['index'] for d in gpu_devices}
    if len(devices) != len(gpu_devices) or len(set(devices.values())) != len(devices):
        raise ValueError('GPU device identities ambiguous')
    result = []
    for rank, namespace_pid in sorted(workers.items()):
        matches = [int(host) for host, p in process_records.items()
                   if int(host) in host_pids and p.get('namespace_pid') == namespace_pid]
        if len(matches) != 1:
            raise ValueError('worker namespace PID does not join to exactly one owned host PID')
        host = matches[0]
        rows = [r for r in gpu_processes if r['pid'] == host]
        if len(rows) != 1 or rows[0]['gpu_uuid'] not in devices:
            raise ValueError('worker does not join to exactly one physical GPU')
        identity = process_records[str(host)]
        if type(identity.get('start_ticks')) is not int or identity['start_ticks'] <= 0:
            raise ValueError('host process incarnation missing')
        result.append({axis+'_rank': rank, 'namespace_pid': namespace_pid,
                       'host_pid': host, 'start_ticks': identity['start_ticks'],
                       'gpu_uuid': rows[0]['gpu_uuid'], 'gpu_index': devices[rows[0]['gpu_uuid']]})
    if len({r['gpu_uuid'] for r in result}) != len(result):
        raise ValueError('workers unexpectedly share a GPU')
    return {'workers': result,
            'scope': axis.upper()+' log-to-container-PID-to-GPU join; request engine routing requires separate evidence'}


def map_dp_workers(log_text, container_host_pids, process_records, gpu_processes, gpu_devices, expected_dp_ranks):
    return _map_workers(log_text, container_host_pids, process_records, gpu_processes, gpu_devices, expected_dp_ranks, 'dp')


def map_tp_workers(log_text, container_host_pids, process_records, gpu_processes, gpu_devices, expected_tp_ranks):
    return _map_workers(log_text, container_host_pids, process_records, gpu_processes, gpu_devices, expected_tp_ranks, 'tp')
