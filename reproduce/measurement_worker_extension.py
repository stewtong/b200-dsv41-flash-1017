"""Opt-in source-pinned worker extension for missing benchmark observations.

Not enabled by the starting launcher. Installed-source parity, rank mapping,
live cadence and matched total overhead must pass before attribution use.
"""
import hashlib,importlib.util,inspect,json,os,re,socket,time,types
from pathlib import Path
import worker_observations
import allocator_observations
from worker_observations import Recorder

SOURCE_PINS={
 'v1/worker/gpu/model_runner.py':'6275324640fd91b9973864c9c6713556c2af6d83ab67c107cc26930ea05532b8',
 'v1/worker/gpu_worker.py':'0832fc191435b57d0797b2b600cd4a27fb97a04a035394cc96b52ca27a79a2d3',
 'v1/worker/worker_base.py':'56bd163cc79f4221e391837aac138b8f4f644cf6d6ec1e585056427ab759e1cc',
 'v1/kv_cache_interface.py':'1e0b5027cc13ed845488ff3f671292e6c57fcae99c5b7130ad92c9d176a38684'}


def verify_sources(root,expected=SOURCE_PINS):
    root=Path(root)
    for relative,digest in expected.items():
        path=root/relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('worker observation source mismatch: '+relative)
    return dict(expected)


def verify_imported_modules(c,root=Path('/campaign-source')):
    modules=[(worker_observations,'worker_observations.py','recorder_sha256'),
            (worker_observations.runtime,'runtime.py','runtime_sha256')]
    if 'allocator_sha256' in c:modules.append((allocator_observations,'allocator_observations.py','allocator_sha256'))
    for module,name,key in modules:
        path=Path(module.__file__)
        if (path!=Path(root)/name or path.is_symlink() or
                hashlib.sha256(path.read_bytes()).hexdigest()!=c[key]):
            raise ValueError('worker observer imported module differs from registered source')


def contract(root,expected_sha,extension_sha):
    root=Path(root);path=root/'contract.json'
    if root.is_symlink() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('worker observation contract content differs')
    c=json.loads(path.read_text())
    required={'schema','run_id','replica_id','boot_id','deadline_ns','extension_sha256','profile_sha256',
        'toolkit_sha256','recorder_sha256','runtime_sha256','tp_size','dp_size','expected_gpu_ids'}
    if 'allocator_sha256' in c:
        required|={'allocator_sha256','allocator_source_pins'}
        if (not re.fullmatch(r'[a-f0-9]{64}',str(c['allocator_sha256'])) or
                c.get('allocator_source_pins')!=allocator_observations.SOURCE_PINS):
            raise ValueError('allocator registered source pins differ')
    if (set(c)!=required or c['schema']!='owned-worker-observation/v1' or
            not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}',str(c['run_id'])) or
            type(c['replica_id']) is not int or not 0<=c['replica_id']<8 or
            type(c['deadline_ns']) is not int or not time.monotonic_ns()<c['deadline_ns']<=time.monotonic_ns()+86_400_000_000_000 or
            c['extension_sha256']!=extension_sha or
            any(not re.fullmatch(r'[a-f0-9]{64}',str(c[k])) for k in ('profile_sha256','toolkit_sha256','recorder_sha256','runtime_sha256')) or
            type(c['tp_size']) is not int or c['tp_size'] not in (1,2,4,8) or
            type(c['dp_size']) is not int or c['dp_size'] not in (1,2,4,8) or
            c['tp_size']*c['dp_size']>8 or not isinstance(c['expected_gpu_ids'],list) or
            len(c['expected_gpu_ids'])!=c['tp_size']*c['dp_size'] or
            len(set(c['expected_gpu_ids']))!=len(c['expected_gpu_ids']) or
            any(type(i) is not int or not 0<=i<8 for i in c['expected_gpu_ids'])):
        raise ValueError('worker observation contract identity, deadline or geometry differs')
    return c


def install_hooks(worker,recorder):
    runner=worker.model_runner
    original_sample=runner.sample;original_init=runner.initialize_kv_cache
    if list(inspect.signature(original_sample).parameters)!=['hidden_states','input_batch','grammar_output']:
        raise ValueError('worker sampling interface differs from pinned V2 source')
    if list(inspect.signature(original_init).parameters)!=['kv_cache_config','is_profiling','kv_cache_allocation_context']:
        raise ValueError('worker KV initialization interface differs from pinned V2 source')
    def observed_sample(self,hidden_states,input_batch,grammar_output):
        return recorder.observe_sample(original_sample,hidden_states,input_batch,grammar_output)
    def observed_init(self,kv_cache_config,is_profiling=False,kv_cache_allocation_context=None):
        if type(is_profiling) is not bool:raise ValueError('KV profiling scope must be explicit')
        result=original_init(kv_cache_config,is_profiling=is_profiling,kv_cache_allocation_context=kv_cache_allocation_context)
        if not is_profiling:recorder.record_layout(self.kv_cache_config)
        return result
    runner.sample=types.MethodType(observed_sample,runner)
    runner.initialize_kv_cache=types.MethodType(observed_init,runner)


class CampaignMeasurementExtension:
    def init_worker_extension(self):
        if hasattr(self,'_dsv41_observation_recorder'):raise RuntimeError('worker measurement extension initialized twice')
        if (getattr(self,'use_v2_model_runner',False) is not True or
                self.parallel_config.pipeline_parallel_size!=1 or
                type(self.model_runner).__module__!='vllm.v1.worker.gpu.model_runner' or
                type(self.model_runner).__name__!='GPUModelRunner'):
            raise ValueError('worker observation requires exact V2 runner and PP1')
        extension_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if os.environ.get('DSV41_MEASUREMENT_EXTENSION_SHA256')!=extension_sha:
            raise ValueError('worker extension source pin differs')
        root=Path(os.environ.get('DSV41_MEASUREMENT_ROOT','/nonexistent'))
        if str(root)!='/campaign-measurement':raise ValueError('worker observation requires its exact owned mount')
        c=contract(root,os.environ.get('DSV41_MEASUREMENT_CONTRACT_SHA256'),extension_sha)
        verify_imported_modules(c)
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if c['boot_id']!=boot:raise ValueError('worker observation host boot differs')
        sources=verify_sources(Path(importlib.util.find_spec('vllm').origin).parent)
        from vllm.distributed import get_tp_group
        tp_rank=get_tp_group().rank_in_group;dp_rank=self.model_runner.dp_rank
        if (self.parallel_config.tensor_parallel_size!=c['tp_size'] or
                self.parallel_config.data_parallel_size!=c['dp_size'] or
                not 0<=tp_rank<c['tp_size'] or not 0<=dp_rank<c['dp_size']):
            raise ValueError('worker observation rank geometry differs')
        identity={'run_id':c['run_id'],'replica_id':c['replica_id'],'boot_id':boot,'namespace_pid':os.getpid(),
            'container_hostname':socket.gethostname(),'tp_rank':tp_rank,'dp_rank':dp_rank,'worker_rank':self.rank,
            'tp_size':c['tp_size'],'dp_size':c['dp_size'],'expected_gpu_ids':c['expected_gpu_ids'],
            'contract_sha256':os.environ['DSV41_MEASUREMENT_CONTRACT_SHA256'],'extension_sha256':extension_sha,
            'recorder_sha256':c['recorder_sha256'],'runtime_sha256':c['runtime_sha256'],
            'profile_sha256':c['profile_sha256'],'toolkit_sha256':c['toolkit_sha256'],'source_pins':sources}
        allocator_reader=None
        if 'allocator_sha256' in c:
            import torch
            device_index=self.device.index
            allocator_reader=allocator_observations.pinned_reader(torch,device_index)
            identity.update(allocator_sha256=c['allocator_sha256'],allocator_source_pins=c['allocator_source_pins'],local_device_index=device_index)
        recorder=Recorder(root/('worker-'+str(os.getpid())),identity,c['deadline_ns'],allocator_reader=allocator_reader)
        install_hooks(self,recorder);self._dsv41_observation_recorder=recorder;recorder.start()

    def destroy_worker_extension(self):
        recorder=getattr(self,'_dsv41_observation_recorder',None)
        if recorder is not None:recorder.close()
