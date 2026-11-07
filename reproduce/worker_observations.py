"""Observe successful target sampling and static KV layout without tensor reads.

These records are unqualified until the installed adapter, logical/physical
rank join, measured cadence and matched instrumentation overhead pass.
"""
from datetime import datetime,timezone
import hashlib,json,math,os,threading,time
from pathlib import Path
import runtime


def integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError(name+' must be an explicit bounded integer')
    return value


def scalar(value):
    if value is None or type(value) in (str,bool,int):return value
    if type(value) is float and math.isfinite(value):return value
    raise ValueError('unsupported observation scalar')


def spec_record(spec):
    kind=getattr(spec,'kind',None)
    if kind is not None:kind=getattr(kind,'value',kind)
    cache_role=getattr(spec,'cache_role',None)
    if cache_role is not None:cache_role=getattr(cache_role,'value',cache_role)
    return {'class':type(spec).__name__,'kind':scalar(kind),'cache_role':scalar(cache_role),
        'cache_dtype_str':scalar(getattr(spec,'cache_dtype_str',None)),
        'is_index_group_leader':scalar(getattr(spec,'is_index_group_leader',None)),
        'block_size':integer(spec.block_size,'KV block size',1),
        'page_size_bytes':integer(spec.page_size_bytes,'KV page size',1),
        'dtype':str(spec.dtype) if hasattr(spec,'dtype') else None,
        'dcp_sharded':bool(spec.dcp_sharded),'is_draft':bool(spec.is_draft)}


def kv_layout(config):
    groups=[]
    for index,group in enumerate(config.kv_cache_groups):
        names=list(group.layer_names)
        if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
            raise ValueError('invalid resolved KV group layer population')
        spec=group.kv_cache_spec;per_layer=getattr(spec,'kv_cache_specs',None)
        if per_layer is not None:
            if set(per_layer)!=set(names):raise ValueError('resolved uniform KV layer coverage differs')
            specs={name:spec_record(per_layer[name]) for name in names}
        else:specs={name:spec_record(spec) for name in names}
        role=getattr(group.role,'value',group.role)
        groups.append({'group_id':index,'role':scalar(role),'host_resident':scalar(group.host_resident),
            'layer_specs':specs})
    if not groups:raise ValueError('resolved KV layout is empty')
    tensors=[]
    for tensor in config.kv_cache_tensors:
        tensors.append({'configured_bytes':integer(tensor.size,'KV backing bytes',1),
            'layer_names':list(tensor.layers),'host_resident':scalar(tensor.host_resident)})
    return {'schema':'resolved-worker-kv-layout/v1','device_pool_blocks':integer(config.num_blocks,'device pool blocks',2),
        'groups':groups,'configured_tensors':tensors,'layout':scalar(config.kv_cache_layout),
        'host_pool_blocks':scalar(config.hisparse_host_num_blocks),'host_block_stride':scalar(config.hisparse_host_block_stride),
        'host_pool_shared_across_local_tp':scalar(config.hisparse_shared_host_pool),
        'kv_tp_replicas':integer(config.kv_tp_replicas,'KV TP replicas',1),
        'scope':'resolved configuration and configured backing bytes; not dynamic group occupancy or measured allocation peaks'}


class Recorder:
    def __init__(self,root,identity,deadline_ns,*,period=.5,allocator_reader=None):
        if not .05<=period<=.5:raise ValueError('worker observation period outside qualification range')
        self.root=Path(root);self.root.mkdir(exist_ok=False)
        self.identity=dict(identity);self.deadline_ns=integer(deadline_ns,'deadline',1)
        self.allocator_reader=allocator_reader
        if bool(self.identity.get('allocator_sha256'))!=(allocator_reader is not None):
            raise ValueError('allocator reader differs from registered identity')
        if self.deadline_ns<=time.monotonic_ns():raise ValueError('worker observation deadline already expired')
        self.period=period;self.lock=threading.Lock();self.stop=threading.Event();self.thread=None
        self.sample_calls=0;self.grammar_sample_calls=0;self.verification_steps=0;self.target_verified_positions=0;self.error=None;self.layout_sha256=None
        runtime.atomic_json(self.root/'identity.json',self.identity)

    def observe_sample(self,original,hidden_states,input_batch,grammar_output):
        if self.error is not None:raise RuntimeError('worker observation persistence failed')
        drafted=integer(input_batch.num_draft_tokens,'actual admitted target draft positions')
        if type(input_batch.num_reqs) is not int or input_batch.num_reqs<0 or input_batch.num_reqs==0 and drafted:
            raise ValueError('worker observation global unsharded batch dimensions differ')
        # The source-pinned V2 method computes target logits before rejection.
        # Commit only after it returns successfully. No tensor synchronization,
        # acceptance override, token mutation or inference retry is introduced.
        result=original(hidden_states,input_batch,grammar_output)
        with self.lock:
            self.sample_calls+=1
            if grammar_output is not None:self.grammar_sample_calls+=1
            if drafted:
                self.verification_steps+=1;self.target_verified_positions+=drafted
        return result

    def record_layout(self,config):
        value=kv_layout(config);raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(raw).hexdigest()
        with self.lock:
            if self.layout_sha256 is not None and self.layout_sha256!=digest:raise RuntimeError('resolved KV layout changed')
            if self.layout_sha256 is None:
                runtime.atomic_json(self.root/'kv-layout.json',value);self.layout_sha256=digest
        return value

    def snapshot(self):
        started=time.monotonic_ns()
        with self.lock:
            values={'successful_sample_calls':self.sample_calls,'successful_verification_steps':self.verification_steps,
                'successful_grammar_sample_calls':self.grammar_sample_calls,
                'target_verified_draft_positions':self.target_verified_positions,'kv_layout_canonical_sha256':self.layout_sha256}
        allocator={} if self.allocator_reader is None else {'allocator':self.allocator_reader()}
        return {'schema':'worker-target-observation/v1','identity':self.identity,'started_ns':started,
            'completed_ns':time.monotonic_ns(),'values':values,
            **allocator,
            'scope':'global pre-TP-sharding draft positions on successful V2 sampling; mirrored TP records, aggregate TP0 only; excludes bonus positions, accepted-token counts and GPU completion timing'}

    def start(self):
        if self.thread is not None:raise RuntimeError('worker recorder already started')
        def loop():
            index=0;size=0
            try:
                while not self.stop.is_set():
                    started=time.monotonic()
                    if time.monotonic_ns()>=self.deadline_ns:raise RuntimeError('worker observation deadline expired')
                    row=self.snapshot();index,size=runtime.append_sample(self.root,'samples',index,size,row)
                    runtime.atomic_json(self.root/'live.json',row)
                    self.stop.wait(max(0,self.period-(time.monotonic()-started)))
            except BaseException as exc:
                self.error=type(exc).__name__
                runtime.atomic_json(self.root/'failure.json',{'type':type(exc).__name__,'scope':'observation failed; no metric acceptance'})
        self.thread=threading.Thread(target=loop,name='dsv41-worker-observation',daemon=True);self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=5)
        if self.thread is not None and self.thread.is_alive():raise RuntimeError('worker observation thread did not stop')
        runtime.atomic_json(self.root/'closed.json',{'at':datetime.now(timezone.utc).isoformat(),
            'monotonic_ns':time.monotonic_ns(),'failed':self.error is not None})
