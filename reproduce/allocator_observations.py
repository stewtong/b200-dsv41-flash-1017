"""Read pinned process-local CUDA allocator statistics without resetting them.

Not active until explicitly bound to the worker observer and live-qualified.
These peaks cover the allocator's own last-reset interval, not a request window.
"""
import hashlib
import inspect
import time
from pathlib import Path

SOURCE_PINS={
    'cuda/memory.py':'ed3e6068d73ef9d2165f7276476041ee5bfe8f19db863e3e11172d840812173b',
    'version.py':'b31eaf43d0d1176473e4d63a41a46b9fcf85afd285c3609375865f78f56c1a65'}
FIELDS={
    'allocated_bytes':'allocated_bytes.all.current',
    'reserved_bytes':'reserved_bytes.all.current',
    'allocated_peak_bytes_since_engine_reset':'allocated_bytes.all.peak',
    'reserved_peak_bytes_since_engine_reset':'reserved_bytes.all.peak',
    'allocation_retries':'num_alloc_retries',
    'allocator_ooms':'num_ooms'}
PEAK_SCOPE='process-local allocator peaks since the engine last reset them; includes startup; not cell-local peaks or total device memory'


def pinned_reader(torch_module,device_index):
    if type(device_index) is not int or device_index<0:
        raise ValueError('allocator observation requires an explicit device index')
    root=Path(torch_module.__file__).parent
    for name,digest in SOURCE_PINS.items():
        p=root/name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('installed Torch allocator source differs')
    memory=torch_module.cuda.memory
    for name in ('memory_stats','get_allocator_backend'):
        if Path(inspect.getsourcefile(getattr(memory,name)))!=root/'cuda/memory.py':
            raise ValueError('imported Torch allocator function source differs')
    backend=memory.get_allocator_backend()
    if backend!='native':
        # Pinned source warns that cudaMallocAsync returns meaningless zero stats.
        raise ValueError('allocator statistics require the qualified native backend')
    return lambda:read(memory.memory_stats,device_index,backend)


def read(stats_function,device_index,backend):
    started=time.monotonic_ns()
    stats=stats_function(device_index)
    values={}
    for name,key in FIELDS.items():
        value=stats.get(key)
        if type(value) is not int or value<0:
            raise ValueError('required allocator statistic missing or malformed: '+key)
        values[name]=value
    if (backend!='native' or
            values['allocated_bytes']>values['allocated_peak_bytes_since_engine_reset'] or
            values['reserved_bytes']>values['reserved_peak_bytes_since_engine_reset']):
        raise ValueError('allocator backend or peak bounds differ')
    return {'schema':'process-cuda-allocator-observation/v1',
        'device_index':device_index,'backend':backend,'started_ns':started,
        'completed_ns':time.monotonic_ns(),'values':values,
        'source_pins':dict(SOURCE_PINS),'peak_scope':PEAK_SCOPE}
