"""Reserve whole physical cores for the client on the qualified two-NUMA node."""
import os
import subprocess


def partition(topology, allowed, cores_per_node=4):
    groups = {}
    for line in topology.splitlines():
        if not line or line.startswith('#'):
            continue
        cpu, core, socket, node = map(int, line.split(','))
        groups.setdefault((node, socket, core), set()).add(cpu)
    allowed = set(allowed)
    if set().union(*groups.values()) != allowed:
        raise ValueError('CPU topology does not exactly match allowed affinity')
    nodes = sorted({key[0] for key in groups})
    if len(nodes) != 2 or any(node < 0 for node in nodes):
        raise ValueError('expected two NUMA nodes')
    client = set()
    for node in nodes:
        cores = sorted(key for key in groups if key[0] == node)
        if len(cores) <= cores_per_node or any(len(groups[k]) != 2 for k in cores):
            raise ValueError('expected complete two-thread cores with spare capacity')
        for key in cores[:cores_per_node]:
            client.update(groups[key])
    server = allowed - client
    return {'policy': 'two-numa-four-physical-cores-per-node-v1',
            'client_cpus': sorted(client), 'server_cpus': sorted(server)}


def discover():
    topology = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET,NODE'], text=True, timeout=15)
    return partition(topology, os.sched_getaffinity(0))


def cpu_list(cpus):
    return ','.join(map(str, sorted(cpus)))
