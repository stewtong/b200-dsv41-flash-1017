"""Node-side ownership and container primitives for the benchmark runner."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import socket
import selectors
import subprocess
import time

CAMPAIGN = 'dsv41-1017'
DEFAULT_NODE_LOCK = '/data/hf-cache/dsv41-campaign.lock'

class RuntimeErrorClosed(RuntimeError):
    pass

class CleanupStepTimeout(RuntimeErrorClosed):
    def __init__(self, stage):
        self.stage = stage
        super().__init__('cleanup '+stage+' timed out')

class NodeLock:
    def __init__(self, path, wait_seconds=0):
        self.path = Path(path)
        self.wait_seconds = wait_seconds
        self.fd = None

    def __enter__(self):
        flags=os.O_RDWR | os.O_CREAT | getattr(os,'O_NOFOLLOW',0) | getattr(os,'O_CLOEXEC',0)
        descriptor=os.open(self.path,flags,0o600)
        try:
            info=os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
                raise RuntimeErrorClosed('lock must be a singly linked regular file')
            self.fd=os.fdopen(descriptor,'a+')
        except BaseException:
            os.close(descriptor)
            raise
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.fd.close()
                    self.fd = None
                    raise RuntimeErrorClosed('campaign lock held')
                time.sleep(0.05)
        return self

    def __exit__(self, *_):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()

def command(args, timeout=30):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout).stdout

def command_combined(args, timeout=30):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout).stdout

def inspect_owned(container_id, run_id, invoke=command, timeout=30):
    if not isinstance(container_id, str) or len(container_id) != 64 or any(c not in '0123456789abcdef' for c in container_id):
        raise RuntimeErrorClosed('full container ID required')
    data = json.loads(invoke(['sudo', '-n', 'docker', 'inspect', container_id], timeout=timeout))
    if len(data) != 1:
        raise RuntimeErrorClosed('inspect did not return exactly one container')
    row = data[0]
    labels = row.get('Config', {}).get('Labels') or {}
    if row.get('Id') != container_id or labels.get('campaign') != CAMPAIGN or labels.get('run_id') != run_id:
        raise RuntimeErrorClosed('container identity or ownership mismatch')
    return row

def inspect_running_owned(container_ids, run_id, invoke=command, timeout=5):
    if not container_ids: return
    if len(set(container_ids)) != len(container_ids) or any(not re_full_id(x) for x in container_ids):
        raise RuntimeErrorClosed('invalid recorded container identities')
    data = json.loads(invoke(['sudo','-n','docker','inspect'] + container_ids, timeout=timeout))
    if len(data) != len(container_ids) or {x.get('Id') for x in data} != set(container_ids):
        raise RuntimeErrorClosed('missing owned container')
    for row in data:
        labels = row.get('Config', {}).get('Labels') or {}
        if labels.get('campaign') != CAMPAIGN or labels.get('run_id') != run_id or row.get('State', {}).get('Running') is not True:
            raise RuntimeErrorClosed('owned container stopped or identity changed')

def re_full_id(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def stop_owned(container_id, run_id, invoke=command, timeout=30):
    deadline = time.monotonic() + timeout
    try: inspect_owned(container_id, run_id, invoke, timeout=min(15, timeout/4))
    except subprocess.TimeoutExpired as exc: raise CleanupStepTimeout('initial_inspect') from exc
    remaining = deadline - time.monotonic()
    if remaining <= 1: raise RuntimeErrorClosed('no remaining cleanup time')
    try:
        invoke(['sudo', '-n', 'docker', 'stop', '--time', str(max(0, min(10,int(remaining)-3))), container_id], timeout=max(0.5, remaining-0.5))
    except subprocess.TimeoutExpired as exc: raise CleanupStepTimeout('stop') from exc
    remaining = deadline - time.monotonic()
    if remaining <= 0: raise RuntimeErrorClosed('no time to verify stop')
    try: state = inspect_owned(container_id, run_id, invoke, timeout=remaining).get('State', {})
    except subprocess.TimeoutExpired as exc: raise CleanupStepTimeout('final_inspect') from exc
    if state.get('Running') is not False:
        raise RuntimeErrorClosed('container remains running or state is missing')

def capture_owned_logs(container_id, run_id, output, invoke=command, timeout=30):
    deadline = time.monotonic() + timeout
    inspect_owned(container_id, run_id, invoke, timeout=min(5, timeout/4))
    path = Path(output)
    args = ['sudo', '-n', 'docker', 'logs', '--timestamps', container_id]
    with path.open('xb') as f:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        total = 0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise RuntimeErrorClosed('log capture deadline exceeded')
                events = selector.select(min(remaining, 1))
                if not events: continue
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk: break
                total += len(chunk)
                if total > 128 * 1024 * 1024: raise RuntimeErrorClosed('log capture exceeds evidence limit')
                f.write(chunk)
            if proc.wait(timeout=max(0.1, deadline-time.monotonic())) != 0:
                raise RuntimeErrorClosed('docker log capture failed')
            f.flush(); os.fsync(f.fileno())
        except BaseException:
            if proc.poll() is None:
                proc.kill(); proc.wait(timeout=5)
            raise
        finally:
            selector.close()
            proc.stdout.close()

def remove_stopped_owned(container_id, run_id, invoke=command):
    row = inspect_owned(container_id, run_id, invoke)
    if row.get('State', {}).get('Running') is not False:
        raise RuntimeErrorClosed('refusing to remove a running container')
    invoke(['sudo','-n','docker','rm',container_id], timeout=30)
    remaining = invoke(['sudo','-n','docker','ps','-aq','--no-trunc','--filter','id='+container_id], timeout=30).split()
    if remaining:
        raise RuntimeErrorClosed('container still exists after removal')

def create_owned(run_argv, run_id, state_path, invoke=command):
    """Create without starting, persist full ID, then confirm exact labels.

    If the process dies after create but before persistence, later acquisition
    must discover the surviving campaign label and refuse a new launch.
    """
    if run_argv[:3] != ['docker', 'run', '-d']:
        raise RuntimeErrorClosed('unexpected launch command shape')
    required = ['campaign=' + CAMPAIGN, 'run_id=' + run_id]
    if any(x not in run_argv for x in required):
        raise RuntimeErrorClosed('missing campaign ownership labels')
    create_argv = ['sudo', '-n', 'docker', 'create'] + run_argv[3:]
    cid = invoke(create_argv, timeout=60).strip()
    if len(cid) != 64 or any(c not in '0123456789abcdef' for c in cid):
        raise RuntimeErrorClosed('docker create did not return a full ID')
    path = Path(state_path)
    state = json.loads(path.read_text())
    if state.get('run_id') != run_id or cid in state.get('container_ids', []):
        raise RuntimeErrorClosed('state identity mismatch or duplicate container')
    state.setdefault('container_ids', []).append(cid)
    atomic_json(path, state)
    row = inspect_owned(cid, run_id, invoke)
    if row.get('State', {}).get('Running') is not False:
        raise RuntimeErrorClosed('created container state not verified')
    return cid

def start_owned(container_id, run_id, invoke=command):
    before = inspect_owned(container_id, run_id, invoke)
    if before.get('State', {}).get('Running') is not False:
        raise RuntimeErrorClosed('container not verified stopped before start')
    invoke(['sudo', '-n', 'docker', 'start', container_id], timeout=60)
    if inspect_owned(container_id, run_id, invoke).get('State', {}).get('Running') is not True:
        raise RuntimeErrorClosed('container not running after start')

def verify_placement(container_id, run_id, expected_gpu_ids, host_port, image_id, invoke=command, expected_cpus=None):
    row = inspect_owned(container_id, run_id, invoke)
    if row.get('Image') != image_id or row.get('State', {}).get('Running') is not True:
        raise RuntimeErrorClosed('running image identity mismatch')
    devices = row.get('HostConfig', {}).get('DeviceRequests') or []
    ids = [x for request in devices for x in (request.get('DeviceIDs') or [])]
    if sorted(ids) != sorted(str(x) for x in expected_gpu_ids):
        raise RuntimeErrorClosed('container GPU placement mismatch')
    bindings = (row.get('NetworkSettings', {}).get('Ports') or {}).get('8000/tcp') or []
    if bindings != [{'HostIp':'127.0.0.1', 'HostPort':str(host_port)}]:
        raise RuntimeErrorClosed('container port binding mismatch')
    if expected_cpus is not None and row.get('HostConfig', {}).get('CpusetCpus') != expected_cpus:
        raise RuntimeErrorClosed('container CPU placement mismatch')
    effective_cpus = None
    if expected_cpus is not None:
        pid = row.get('State', {}).get('Pid')
        if not isinstance(pid, int) or pid <= 0:
            raise RuntimeErrorClosed('container process identity missing')
        effective_cpus = sorted(os.sched_getaffinity(pid))
        if not effective_cpus or not set(effective_cpus).issubset({int(x) for x in expected_cpus.split(',')}):
            raise RuntimeErrorClosed('container effective CPU affinity mismatch')
    return {'container_id':container_id, 'gpu_ids':expected_gpu_ids, 'host_port':host_port, 'image_id':image_id, 'cpuset_cpus':expected_cpus, 'effective_cpus':effective_cpus}

def create_and_start(run_argv, run_id, run_dir, invoke=command):
    """Serialize create/start against watchdog cleanup and honor stop signal."""
    root = Path(run_dir)
    with NodeLock(root/'cleanup.lock', wait_seconds=10):
        state = json.loads((root/'execution-state.json').read_text())
        if (root/'watchdog-failure.json').exists() or (root/'cleanup-result.json').exists() or state.get('stopping'):
            raise RuntimeErrorClosed('watchdog has stopped dispatch')
        cid = create_owned(run_argv, run_id, root/'execution-state.json', invoke)
        if (root/'watchdog-failure.json').exists():
            raise RuntimeErrorClosed('watchdog stopped dispatch before start')
        start_owned(cid, run_id, invoke)
        return cid

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def append_sample(root, prefix, index, size, sample, limit=50_000_000):
    line = (json.dumps(sample, separators=(',', ':'))+'\n').encode()
    if len(line) > limit: raise RuntimeErrorClosed('single telemetry sample exceeds limit')
    if size + len(line) > limit: index, size = index + 1, 0
    with (Path(root)/(prefix+'-%04d.jsonl'%index)).open('ab') as f:
        f.write(line); f.flush(); os.fsync(f.fileno())
    return index, size + len(line)

def seal_evidence(root):
    root = Path(root)
    lines = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise RuntimeErrorClosed('symlink in evidence tree')
        if not path.is_file() or path == root/'SHA256SUMS': continue
        if '.tmp.' in path.name: raise RuntimeErrorClosed('temporary evidence remains')
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
        lines.append(h.hexdigest() + '  ' + path.relative_to(root).as_posix() + '\n')
    if not lines: raise RuntimeErrorClosed('empty evidence tree')
    atomic_bytes(root/'SHA256SUMS',''.join(lines).encode())

def verify_evidence_manifest(root, allowed_unlisted=()):
    """Verify all sealed files before a controlled evidence extension."""
    root = Path(root)
    listed = {}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, sep, name = line.partition('  ')
        rel = Path(name)
        if (not sep or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or rel.is_absolute() or '..' in rel.parts or name in listed):
            raise RuntimeErrorClosed('invalid evidence manifest entry')
        listed[name] = digest
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink(): raise RuntimeErrorClosed('symlink in evidence tree')
        if path.is_file() and path != root/'SHA256SUMS':
            actual.add(path.relative_to(root).as_posix())
    if not listed or not set(listed).issubset(actual) or actual - set(listed) - set(allowed_unlisted):
        raise RuntimeErrorClosed('evidence file coverage mismatch')
    for name, expected in listed.items():
        h = hashlib.sha256()
        with (root/name).open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
        if h.hexdigest() != expected: raise RuntimeErrorClosed('evidence checksum mismatch')

def safety_sample(mount='/data'):
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            key, _, value = line.partition(':')
            if key in ('MemTotal', 'MemAvailable'):
                info[key] = int(value.split()[0])
    if set(info) != {'MemTotal', 'MemAvailable'} or info['MemTotal'] <= 0:
        raise RuntimeErrorClosed('missing memory safety telemetry')
    st = os.statvfs(mount)
    if st.f_blocks <= 0:
        raise RuntimeErrorClosed('missing disk safety telemetry')
    memory = info['MemAvailable'] / info['MemTotal']
    disk = st.f_bavail / st.f_blocks
    if memory < 0.10 or disk < 0.15:
        raise RuntimeErrorClosed('memory or disk safety threshold crossed')
    return {'monotonic_ns': time.monotonic_ns(), 'available_memory_fraction': memory,
            'free_disk_fraction': disk}

def heartbeat_fresh(record, now_ns, max_age_ns=15_000_000_000):
    stamp = record.get('monotonic_ns')
    return isinstance(stamp, int) and 0 <= now_ns - stamp <= max_age_ns

def reconcile_orphans(expected_run, running_containers):
    """Reject any pre-existing campaign container, including the current run's.

    Recovery must inspect recorded IDs and outcomes explicitly before another launch.
    """
    if running_containers:
        raise RuntimeErrorClosed('campaign containers exist; explicit recovery required')
    if not expected_run:
        raise RuntimeErrorClosed('run identity missing')

def live_preflight(run_id, ports, invoke=command):
    """Require no campaign containers, no GPU compute processes and free ports."""
    labeled = invoke(['sudo', '-n', 'docker', 'ps', '-aq', '--no-trunc',
                      '--filter', 'label=campaign=' + CAMPAIGN]).split()
    reconcile_orphans(run_id, labeled)
    verify_gpus_idle(invoke)
    for port in ports:
        if not isinstance(port, int) or not 31000 <= port <= 31199:
            raise RuntimeErrorClosed('invalid campaign port')
        try:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
        except OSError as exc:
            raise RuntimeErrorClosed('campaign port is occupied') from exc
    return safety_sample()

def verify_gpus_idle(invoke=command):
    processes = invoke(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip()
    if processes:
        raise RuntimeErrorClosed('GPU compute processes exist')
    used = invoke(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).splitlines()
    if len(used) != 8:
        raise RuntimeErrorClosed('expected eight GPU memory readings')
    try:
        if any(int(x.strip()) > 2048 for x in used):
            raise RuntimeErrorClosed('GPU memory not near idle')
    except ValueError as exc:
        raise RuntimeErrorClosed('unparseable GPU memory') from exc
    return True

def verify_retirement(config, invoke=command, run=subprocess.run):
    old = Path(config['excluded_weight_path'])
    names = config['retired_container_names']
    timer = config['retired_timer']
    if (not old.is_absolute() or not isinstance(names,list) or len(names)!=4 or len(set(names))!=4 or
        not all(re.fullmatch(r'[a-z0-9][a-z0-9_.-]*',n) for n in names) or
        not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*\.timer',timer)):
        raise ValueError('invalid retired-service preflight configuration')
    if os.path.lexists(old): raise RuntimeErrorClosed('previous weights still exist')
    containers=json.loads(invoke(['sudo','-n','docker','inspect']+names,timeout=10))
    if (len(containers)!=4 or {x.get('Name','').lstrip('/') for x in containers}!=set(names) or
        any(x.get('State',{}).get('Running') is not False for x in containers)):
        raise RuntimeErrorClosed('previous container stop state unverified')
    status=run(['sudo','-n','systemctl','is-active',timer],capture_output=True,text=True,timeout=10)
    if status.returncode != 3 or status.stdout.strip() != 'inactive':
        raise RuntimeErrorClosed('previous collector timer not verified inactive')
    return {'containers_stopped':True,'timer_inactive':True,'previous_weights_absent':True,
            'container_ids':{x['Name'].lstrip('/'):x['Id'] for x in containers}}

def process_identity(pid, proc_root='/proc'):
    if not isinstance(pid, int) or pid <= 1:
        raise RuntimeErrorClosed('invalid PID')
    root = Path(proc_root)
    # comm may contain spaces or parentheses, so split after its final ')'.
    text = (root/str(pid)/'stat').read_text()
    tail = text[text.rfind(')') + 2:].split()
    if len(tail) < 20: raise RuntimeErrorClosed('invalid process stat')
    return {'pid': pid, 'start_ticks': int(tail[19]),
            'boot_id': (root/'sys/kernel/random/boot_id').read_text().strip()}

def signal_owned_process(identity, sig=signal.SIGTERM, identity_fn=process_identity, kill=os.kill):
    if identity_fn(identity['pid']) != identity:
        raise RuntimeErrorClosed('process identity changed; refusing signal')
    kill(identity['pid'], sig)

def process_alive(identity):
    try:
        if process_identity(identity['pid']) != identity: return False
        stat = Path('/proc', str(identity['pid']), 'stat').read_text()
        return stat[stat.rfind(')') + 2:].split()[0] != 'Z'
    except FileNotFoundError:
        return False

def stop_owned_process(identity, timeout=20):
    try:
        signal_owned_process(identity)
    except (FileNotFoundError, ProcessLookupError):
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_alive(identity): return
        time.sleep(0.1)
    try:
        signal_owned_process(identity, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        return
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not process_alive(identity): return
        time.sleep(0.05)
    raise RuntimeErrorClosed('process remained after SIGKILL')

def stop_owned_process_group(identity, timeout=20, privileged=False):
    """Stop only a recorded session leader and its dedicated process group."""
    pid = identity['pid']
    if process_identity(pid) != identity or os.getpgid(pid) != pid or os.getsid(pid) != pid:
        raise RuntimeErrorClosed('process group leader identity mismatch')
    if privileged:
        subprocess.run(['sudo','-n','/bin/kill','-TERM','--','-'+str(pid)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=5)
    else:
        os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    raise RuntimeErrorClosed('owned process group did not exit')
