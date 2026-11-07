"""Read NVML memory and volatile uncorrected ECC in an isolated process."""
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import runtime

PERIOD_SECONDS = 0.5

def validate_live(data, identity, now_ns, max_age_ns=10_000_000_000, measured=False):
    if not runtime.process_alive(identity) or data.get('pid')!=identity['pid']:
        raise runtime.RuntimeErrorClosed('NVML collector process dead or identity changed')
    return validate_sample(data,now_ns,max_age_ns,measured)


def validate_sample(data, now_ns, max_age_ns=10_000_000_000, measured=False):
    rows=data.get('devices',[]);stamp=data.get('monotonic_ns');start=data.get('sample_started_ns')
    if type(stamp) is not int or type(start) is not int or start>stamp or not 0<=now_ns-stamp<=max_age_ns or not 0<=now_ns-start<=max_age_ns:
        raise runtime.RuntimeErrorClosed('NVML collector sample stale')
    seconds=data.get('sample_seconds');gap=data.get('sample_gap_seconds')
    if (not isinstance(seconds,(int,float)) or isinstance(seconds,bool) or not math.isfinite(seconds) or seconds<0 or
        not isinstance(gap,(int,float)) or isinstance(gap,bool) or not math.isfinite(gap) or gap<0):
        raise runtime.RuntimeErrorClosed('NVML collector timing invalid')
    if measured and (data.get('measurement_mode') is not True or seconds>1 or gap>1.1):
        raise runtime.RuntimeErrorClosed('NVML collector measured cadence failed')
    if (len(rows)!=8 or {r.get('index') for r in rows}!=set(range(8)) or len({r.get('uuid') for r in rows})!=8 or
        any(not r.get('uuid') or type(r.get('volatile_uncorrected_ecc')) is not int or r['volatile_uncorrected_ecc']<0 or
            type(r.get('used_bytes')) is not int or r['used_bytes']<0 or type(r.get('total_bytes')) is not int or r['total_bytes']<=0 or
            type(r.get('free_bytes')) is not int or r['free_bytes']<0 or type(r.get('reserved_bytes')) is not int or r['reserved_bytes']<0
            for r in rows)):
        raise runtime.RuntimeErrorClosed('NVML collector device coverage incomplete')
    return data

class MemoryV2(ctypes.Structure):
    _fields_=[('version',ctypes.c_uint),('total',ctypes.c_ulonglong),('reserved',ctypes.c_ulonglong),
              ('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]


class Utilization(ctypes.Structure):
    _fields_=[('gpu',ctypes.c_uint),('memory',ctypes.c_uint)]


class ProcessInfo(ctypes.Structure):
    # Installed nvmlProcessInfo_t aliases the v2 structure for the v3 call.
    _fields_=[('pid',ctypes.c_uint),('usedGpuMemory',ctypes.c_ulonglong),
              ('gpuInstanceId',ctypes.c_uint),('computeInstanceId',ctypes.c_uint)]


class Collector:
    def __init__(self):
        self.lib=ctypes.CDLL('libnvidia-ml.so.1');self.setup_timings=[];self.handles=[]
        self.call(self.setup_timings,'nvmlInit_v2')
        count=ctypes.c_uint();self.call(self.setup_timings,'nvmlDeviceGetCount_v2',ctypes.byref(count))
        if count.value!=8: raise ValueError('expected eight NVML devices')
        for index in range(count.value):
            handle=ctypes.c_void_p();self.call(self.setup_timings,'nvmlDeviceGetHandleByIndex_v2',ctypes.c_uint(index),ctypes.byref(handle))
            uuid=ctypes.create_string_buffer(96);self.call(self.setup_timings,'nvmlDeviceGetUUID',handle,uuid,ctypes.c_uint(len(uuid)))
            self.handles.append((index,handle,uuid.value.decode('ascii')))

    def call(self,timings,name,*args):
        started=time.monotonic_ns();rc=getattr(self.lib,name)(*args);elapsed=(time.monotonic_ns()-started)/1e9
        timings.append({'call':name,'seconds':elapsed,'status':int(rc)})
        if rc!=0: raise RuntimeError(name+' returned NVML status '+str(rc))

    def process_count(self, handle, timings):
        # The installed NVML header explicitly permits a NULL infos pointer when
        # infoCount starts at zero. This avoids depending on a process-struct ABI.
        count=ctypes.c_uint(0);started=time.monotonic_ns()
        rc=self.lib.nvmlDeviceGetComputeRunningProcesses_v3(handle,ctypes.byref(count),None)
        timings.append({'call':'nvmlDeviceGetComputeRunningProcesses_v3',
                        'seconds':(time.monotonic_ns()-started)/1e9,'status':int(rc)})
        if not ((rc==0 and count.value==0) or (rc==7 and count.value>0)):
            raise RuntimeError('NVML compute-process count unavailable or inconsistent')
        return count.value

    def read(self, include_processes=False):
        rows=[];timings=[]
        for index,handle,uuid in self.handles:
            memory=MemoryV2();memory.version=ctypes.sizeof(MemoryV2) | (2 << 24)
            self.call(timings,'nvmlDeviceGetMemoryInfo_v2',handle,ctypes.byref(memory))
            ecc=ctypes.c_ulonglong();self.call(timings,'nvmlDeviceGetTotalEccErrors',handle,ctypes.c_int(1),ctypes.c_int(0),ctypes.byref(ecc))
            rows.append({'index':index,'uuid':uuid,'total_bytes':memory.total,
                         'reserved_bytes':memory.reserved,'free_bytes':memory.free,'used_bytes':memory.used,'volatile_uncorrected_ecc':ecc.value})
            if include_processes: rows[-1]['compute_process_count']=self.process_count(handle,timings)
        return {'devices':rows,'call_timings':timings,'scope':'NVML memory, ECC and optional shutdown compute-process count; not utilization, clocks or power'}

    def process_rows(self,handle,timings):
        capacity=64
        for attempt in range(2):
            count=ctypes.c_uint(capacity);rows=(ProcessInfo*capacity)();started=time.monotonic_ns()
            rc=self.lib.nvmlDeviceGetComputeRunningProcesses_v3(handle,ctypes.byref(count),rows)
            timings.append({'call':'nvmlDeviceGetComputeRunningProcesses_v3','seconds':(time.monotonic_ns()-started)/1e9,'status':int(rc)})
            if rc==7 and attempt==0 and capacity<count.value<=4096:
                capacity=count.value;continue
            if rc!=0 or count.value>capacity:raise RuntimeError('bounded NVML process table unavailable')
            result=[{'pid':r.pid,'used_gpu_memory_bytes':r.usedGpuMemory,
                     'gpu_instance_id':r.gpuInstanceId,'compute_instance_id':r.computeInstanceId} for r in rows[:count.value]]
            if len({r['pid'] for r in result})!=len(result) or any(r['pid']<=0 or r['used_gpu_memory_bytes']==2**64-1 for r in result):
                raise ValueError('NVML process identity or memory unavailable')
            return result
        raise RuntimeError('NVML process table changed beyond bounded retry')

    def read_measurement(self):
        result=self.read();timings=result['call_timings']
        for row,(_,handle,_) in zip(result['devices'],self.handles):
            began=time.monotonic_ns();util=Utilization();clock=ctypes.c_uint();power=ctypes.c_uint()
            self.call(timings,'nvmlDeviceGetUtilizationRates',handle,ctypes.byref(util))
            self.call(timings,'nvmlDeviceGetClockInfo',handle,ctypes.c_int(1),ctypes.byref(clock))
            self.call(timings,'nvmlDeviceGetPowerUsage',handle,ctypes.byref(power))
            processes=self.process_rows(handle,timings)
            if util.gpu>100 or util.memory>100 or any(p['used_gpu_memory_bytes']>row['total_bytes'] for p in processes):
                raise ValueError('NVML measured values outside device bounds')
            row['performance']={'gpu_utilization_percent':util.gpu,'memory_utilization_percent':util.memory,
                                'sm_clock_mhz':clock.value,'power_milliwatts':power.value,
                                'sample_started_ns':began,'sample_completed_ns':time.monotonic_ns()}
            row['compute_processes']=processes
        result.update(measurement_schema='isolated-nvml-process-and-performance/v1',
                      scope='native NVML memory/ECC/processes and sampled utilization/SM clock/power; hardware utilization periods vary and power is a one-second average on newer GPUs')
        return result

    def close(self): self.lib.nvmlShutdown()


def sample():
    collector=Collector()
    try:
        result=collector.read();result['setup_timings']=collector.setup_timings;return result
    finally:
        collector.close()


def measurement_active(root):
    if not (root/'nvml-measurement-mode.json').exists(): return False
    return not json.loads((root/'execution-state.json').read_text())['stopping']


def drain_active(root):
    return ((root/'nvml-drain-request.json').exists() and
            json.loads((root/'execution-state.json').read_text()).get('stopping') is True)


def loop(root):
    root=Path(root);index=size=0;collector=Collector();previous=None;previous_mode=False;measurement_failed=False
    try:
        while True:
            measured_started=measurement_active(root)
            started=time.monotonic();started_ns=time.monotonic_ns();cpu_started=time.process_time()
            if measured_started and not measurement_failed:
                try:result=collector.read_measurement()
                except Exception as exc:
                    # A measurement error invalidates work, while the original
                    # memory/ECC feed remains available for bounded cleanup.
                    measurement_failed=True
                    runtime.atomic_json(root/'nvml-measurement-failure.json',{'type':type(exc).__name__,'monotonic_ns':time.monotonic_ns()})
                    result=collector.read(include_processes=drain_active(root))
            else:result=collector.read(include_processes=True) if drain_active(root) else collector.read()
            finished=time.monotonic()
            measured=measured_started and measurement_active(root) and not measurement_failed
            if measured and not previous_mode: previous=None
            gap=finished-previous if previous is not None else 0.0
            result.update(pid=os.getpid(),monotonic_ns=time.monotonic_ns(),sample_started_ns=started_ns,sample_seconds=finished-started,
                          sample_gap_seconds=gap,process_cpu_seconds=time.process_time()-cpu_started,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          measurement_mode=measured,setup_timings=collector.setup_timings if previous is None and not measured else [])
            if measured and (result['sample_seconds']>1 or gap>1.1):
                # Preserve the first invalid measured sample, but keep the independent
                # safety feed alive while the measured consumer aborts the run.
                failure=root/'nvml-cadence-failure.json'
                if not failure.exists(): runtime.atomic_json(failure,result)
            index,size=runtime.append_sample(root,'nvml-samples',index,size,result)
            runtime.atomic_json(root/'nvml-live.json',result);previous=finished;previous_mode=measured
            time.sleep(max(0,PERIOD_SECONDS-(time.monotonic()-started)))
    finally: collector.close()


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--loop': loop(sys.argv[2])
    elif len(sys.argv)==1: print(json.dumps(sample(),sort_keys=True))
    else: raise SystemExit('usage: nvml_safety_probe.py [--loop RUN_DIR]')
