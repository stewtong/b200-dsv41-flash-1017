"""Verify an installed fixed-input source before reusing its exact task bytes."""
import hashlib,json,re
from pathlib import Path
import runtime,verify_fixed_task,node_smoke,fixed_input_reuse


def load(config,n,concurrency,repetition,allowed_root='/data',*,suite='core',cache='cold',output_tokens=2048):
    reuse=config.get('fixed_input_reuse')
    source_layout=reuse.get('source_layout') if isinstance(reuse,dict) else config['layout']
    source_cell=f"{suite}-{source_layout}-i{n}-c{concurrency}-{cache}-o{output_tokens}"
    path=Path(config['tasks_file']);source=path.parent
    if (not path.is_absolute() or path.is_symlink() or path.name!='fixed-tasks.json' or source.is_symlink() or
            not source.resolve(strict=True).is_relative_to(Path(allowed_root).resolve()) or
            not re.fullmatch(r'dsv41-[a-z0-9-]+',source.name)):
        raise ValueError('fixed tuning inputs must be a sealed owned generator artifact')
    runtime.verify_evidence_manifest(source);generation=node_smoke.read(source/'prompt-result.json');cfg=node_smoke.read(source/'run-config.json')
    if cfg.get('generation_method') is not None or generation.get('generation_method') is not None:
        import streamed_input_source
        streamed_input_source.verify(source)
    if (generation.get('status')!='passed' or generation.get('cleanup')!='passed' or generation.get('task_family')!='fixed' or
            generation.get('repetition')!=repetition or generation.get('cell_id')!=source_cell or
            cfg.get('cell_id')!=source_cell or cfg.get('layout')!=source_layout or cfg.get('image')!=config['image'] or
            cfg.get('task_family')!='fixed' or cfg.get('repetition')!=repetition):
        raise ValueError('fixed tuning generation differs from selected shape or repetition')
    lineage=fixed_input_reuse.load(config,source) if reuse is not None else None
    for key,name in [('manifest','manifest.json'),('checkpoint_receipt','checkpoint-receipt.json'),('image_qualification','image-qualification.json')]:
        if name=='manifest.json' and lineage is not None:continue
        if (source/name).read_bytes()!=Path(config[key]).read_bytes():raise ValueError('fixed tuning source dependency changed')
    data=path.read_bytes();tasks=json.loads(data)
    if generation.get('tasks')!=len(tasks) or len(tasks)!=concurrency or {t.get('id') for t in tasks}!={'s'+str(i) for i in range(concurrency)}:
        raise ValueError('fixed tuning input population differs')
    for task in tasks:
        verify_fixed_task.verify(task)
        if reuse is not None and task.get('prompt_version')!='fixed-performance-v1':raise ValueError('fixed reuse cannot relabel legacy workloads')
        if task.get('prompt_version')=='fixed-performance-v1' and cfg.get('fixed_prompt_version')!='fixed-performance-v1':
            raise ValueError('performance workload lacks explicit installed generator version')
        if (task.get('suite')!=suite or task.get('target_input_tokens')!=n or task.get('concurrency')!=concurrency or
                task.get('cache_designation')!=cache or task.get('output_tokens')!=output_tokens or task.get('repetition')!=repetition):
            raise ValueError('fixed tuning input contract differs')
    runtime.verify_evidence_manifest(source)
    return {'tasks':tasks,'task_bytes':data,'source':source,
        'source_manifest_sha256':hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest(),**({'input_lineage':lineage} if lineage is not None else {})}
