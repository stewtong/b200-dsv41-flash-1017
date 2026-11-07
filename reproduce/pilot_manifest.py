"""Bind a partial pilot ledger to verified node receipts and an operational deadline."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import campaign
import check_fix
import launcher
import node_smoke
import runtime
import stream
import toolkit_integrity
import fixed_workload


def build(matrix,plan,image,router,weights,deadline,retirement,toolkit_sha):
    ledger=campaign.ledger(matrix,plan)
    if not retirement.get('containers_stopped') or not retirement.get('timer_inactive') or not retirement.get('previous_weights_absent'):
        raise ValueError('retirement evidence incomplete')
    if (not re.fullmatch(r'[0-9a-f]{64}',toolkit_sha) or image.get('image') is None or
        not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}',image['image']) or
        not re.fullmatch(r'[0-9a-f]{40}',str(image.get('source_revision')))):
        raise ValueError('image or toolkit evidence incomplete')
    expected_invariants={name for names in check_fix.REQUIRED.values() for name in names}
    invariants=image.get('installed',{}).get('invariants',{})
    if set(invariants)!=expected_invariants or set(invariants.values())!={'invariants-present'}:
        raise ValueError('installed fix evidence incomplete')
    drivers=image.get('driver_versions',[]); installed=image.get('installed',{})
    if (image.get('entrypoint')!=['vllm','serve'] or len(drivers)!=8 or len(set(drivers))!=1 or
        not all(installed.get(k) for k in ('vllm_version','torch_version','cuda_version','nccl_version'))):
        raise ValueError('image dependency or driver evidence incomplete')
    if router.get('lock_sha256')!=node_smoke.ROUTER_LOCK_SHA or not router.get('router_executable_sha256'):
        raise ValueError('router qualification incomplete')
    if (weights.get('revision')!=node_smoke.MODEL_REVISION or weights.get('inventory_sha256')!=node_smoke.INVENTORY_SHA or
        weights.get('files')!=83 or weights.get('bytes')!=510311045295 or weights.get('file_set_verified') is not True or
        not isinstance(weights.get('metadata_bytes'),int) or weights['metadata_bytes']>1_000_000_000):
        raise ValueError('checkpoint receipt incomplete')
    parsed=datetime.fromisoformat(deadline.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed<=datetime.now(timezone.utc): raise ValueError('deadline invalid')
    ledger['runtime'].update(image_digest=image['image'],router_pin=router['lock_sha256'],operational_deadline=deadline,
                             retirement_gate='passed',starting_profiles={name:launcher.profile_sha256(name) for name in launcher.LAYOUTS},
                             safety_controls=dict.fromkeys(('ownership','watchdog','cancellation','cleanup','persistence'),True),
                             toolkit_manifest_sha256=toolkit_sha)
    ledger['runtime']['capture_schema']=stream.CAPTURE_SCHEMA
    ledger['runtime']['kv_budget_schema']='group-aware-kv-capacity/v1'
    ledger['runtime']['cleanup_drain_schema']='remaining-shared-cleanup-deadline/v1'
    ledger['runtime']['shutdown_safety_schema']='owned-cleanup-gpu-proof/v1'
    ledger['runtime']['measurement_source_schema']='isolated-nvml-process-and-performance/v1'
    ledger['runtime']['fixed_input_schema']=(fixed_workload.PERFORMANCE_VERSION
        if ledger.get('matrix_schema')=='dsv41-b200-campaign/v3' else fixed_workload.VERSION)
    ledger['runtime']['pilot_warm_input_schema']='pilot-warm-report-v2'
    ledger['runtime']['qualification_scope']='declared candidate inputs; backend, routing, capacity and GPU behavior remain unqualified'
    campaign.validate_phase(ledger,'pilot')
    return ledger


def run(config):
    if not os.environ.get('TMUX'): raise runtime.RuntimeErrorClosed('pilot manifest preparation requires tmux')
    output=Path(config['output'])
    if output.parent!=Path('/data/hf-cache') or not re.fullmatch(r'dsv41-manifest-[a-z0-9-]+\.json',output.name):
        raise ValueError('invalid manifest output path')
    if os.path.lexists(output): raise ValueError('manifest output already exists')
    toolkit=Path(config['toolkit_manifest'])
    root=toolkit.parent.resolve(strict=True)
    if root.parent!=Path('/data/hf-cache') or toolkit != Path(__file__).resolve().parents[1]/'MANIFEST.json':
        raise ValueError('toolkit manifest must describe running source')
    toolkit_sha=toolkit_integrity.verify(toolkit)
    with runtime.NodeLock(runtime.DEFAULT_NODE_LOCK):
        retirement=runtime.verify_retirement(config); runtime.verify_gpus_idle()
        image_path=Path(config['image_qualification']); router_path=Path(config['router_qualification']); weight_path=Path(config['checkpoint_receipt'])
        runtime.verify_evidence_manifest(image_path.parent.parent)
        runtime.verify_evidence_manifest(router_path.parent.parent)
        runtime.verify_evidence_manifest(weight_path.parent)
        image=json.loads(image_path.read_text());router=json.loads(router_path.read_text());weights=json.loads(weight_path.read_text())
        if image.get('image')!=config['image'] or image.get('source_revision')!=config['source_revision']:
            raise ValueError('image pin mismatch')
        import sampling_source
        sampling=sampling_source.verify(Path(config['model_dir'])/'README.md')
        matrix=Path(config.get('matrix',root/'reproduce/matrix.json'))
        if matrix not in (root/'reproduce/matrix.json',root/'reproduce/comparison-matrix.json'):
            raise ValueError('manifest matrix must be an exact running-source matrix')
        manifest=build(matrix,config['plan'],image,router,weights,config['deadline_utc'],retirement,toolkit_sha)
        manifest['runtime']['sampling_source']=sampling
        manifest['runtime']['temperature']=sampling['parameters']['temperature']
        manifest['runtime']['top_p']=sampling['parameters']['top_p']
        profiles={}
        for path in config.get('profile_records',[]):
            import profile_registry
            record=json.loads(Path(path).read_text());profile_registry.validate_record(record,manifest)
            if record['id'] in profiles:raise ValueError('duplicate preregistered profile')
            profiles[record['id']]=record
        if profiles:manifest['runtime']['qualification_profiles']=profiles
        fixed={}
        for path in config.get('fixed_profile_records',[]):
            import fixed_profiles
            record=fixed_profiles.validate(json.loads(Path(path).read_text()),manifest)
            if record['layout'] in fixed:raise ValueError('duplicate frozen fixed configuration')
            fixed[record['layout']]=record
        if fixed:
            if set(fixed)!={'2xtp4','4xdep2','2xdep4','dep8'}:raise ValueError('four fixed configurations required')
            manifest['runtime']['fixed_profiles']=fixed
            manifest['runtime']['profile_hashes']={layout:r['profile_sha256'] for layout,r in fixed.items()}
        payload=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
        with output.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    return {'manifest':str(output),'sha256':hashlib.sha256(payload).hexdigest(),'complete_universe':manifest['complete_universe']}
