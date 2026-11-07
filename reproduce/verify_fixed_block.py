"""Independently bind qualification repetitions to their shared engine session."""
import hashlib,json,re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runtime,verify_fixed_qualification


def verify_children(directories, *, workers=4):
    """Keep ordered results and failures while bounding read-only audit threads."""
    directories=list(directories)
    if not 1<=len(directories)<=24 or type(workers) is not int or not 1<=workers<=4:
        raise ValueError('one to twenty-four cells and one to four audit threads required')
    if workers==1 or len(directories)==1:
        return [verify_fixed_qualification.verify(cell) for cell in directories]
    with ThreadPoolExecutor(max_workers=min(workers,len(directories))) as pool:
        return list(pool.map(verify_fixed_qualification.verify,directories))


def verify(directory, *, workers=4):
    root=Path(directory);runtime.verify_evidence_manifest(root)
    config=json.loads((root/'run-config.json').read_text());result=json.loads((root/'fixed-qualification-block-result.json').read_text())
    state=json.loads((root/'execution-state.json').read_text());entries=config.get('cells')
    scored=config.get('execution_phase')=='fixed-performance'
    if (config.get('execution_phase') not in ('qualification','fixed-performance') or not isinstance(entries,list) or not 1<=len(entries)<=24 or
            result.get('status')!='passed' or state.get('run_id')!=config.get('run_id')):
        raise ValueError('qualification block population or shared owner differs')
    checked=[];ids=[];last_end=0;last_route_end=0;trial=None
    actual_router=(root/'router.log').read_bytes()
    raw_names={p.name for p in root.glob('*.jsonl') if p.name.startswith(('telemetry-','sampler-timing-','nvml-samples-','safety-'))}
    manifest=json.loads((root/'manifest.json').read_text())
    record=manifest['runtime']['fixed_profiles'][config['layout']] if scored else manifest['runtime']['qualification_profiles'][config['profile_id']]
    prepared=[]
    for entry in entries:
        if not isinstance(entry,dict) or set(entry)!={'cell_id','tasks_file','request_timeout_seconds','repetition'}:
            raise ValueError('qualification block entry fields differ')
        declared={k:v for k,v in config.items() if k!='cells'};declared.update(entry)
        # Directory identity is recomputed from the preregistered profile and
        # repetition, without importing the dispatcher or its preparation.
        ident='fixedqual-'+hashlib.sha256(json.dumps([entry['cell_id'],record['profile_sha256'],entry['repetition']],separators=(',',':')).encode()).hexdigest()
        if ident in ids:raise ValueError('duplicate qualification block repetition')
        ids.append(ident);cell=root/'cells'/ident
        if json.loads((cell/'run-config.json').read_text())!=declared:raise ValueError('qualification cell differs from declared block entry')
        prepared.append((entry,ident,cell))
    results=verify_children([cell for _,_,cell in prepared],workers=workers)
    for (entry,ident,cell),verified in zip(prepared,results):
        if verified['cell_id']!=ident or verified['performance_adoption_eligible'] is not False:
            raise ValueError('qualification block changed identity or adoption disposition')
        if bool(verified.get('scored_performance_evidence',False))!=scored:
            raise ValueError('fixed block scoring phase differs')
        named=next(c for c in manifest['cells'] if c['id']==entry['cell_id']);params=named.get('parameters',{})
        key=(('fixed-performance',None,None,verified['profile_sha256']) if scored
             else (named['suite'],params.get('lever'),params.get('arm'),verified['profile_sha256']))
        if trial is None:trial=key
        if key!=trial:raise ValueError('qualification block mixes levers, arms or profiles')
        proof=json.loads((cell/'block-provenance.json').read_text());shared=proof.get('shared_file_sha256',{})
        if (proof.get('schema')!='shared-fixed-qualification-session/v1' or proof.get('run_id')!=config['run_id'] or
                proof.get('driver_identity')!=state['driver_identity'] or
                not raw_names.issubset(shared) or 'shared-run-config.json' not in shared):
            raise ValueError('qualification shared provenance or raw population differs')
        for name,digest in shared.items():
            rel=Path(name)
            if rel.is_absolute() or '..' in rel.parts or not re.fullmatch(r'[a-f0-9]{64}',str(digest)):
                raise ValueError('invalid shared qualification source path or hash')
            source=root/('run-config.json' if name=='shared-run-config.json' else name)
            if hashlib.sha256(source.read_bytes()).hexdigest()!=digest or hashlib.sha256((cell/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('qualification child shared records differ from actual session')
        requests=cell/'fixed-attribution/requests/measured'
        rows=[json.loads(p.read_text()) for p in requests.glob('s[0-9]*.json')]
        begin=min(r['start_ns'] for r in rows);end=max(r['end_ns'] for r in rows)
        if begin<last_end:raise ValueError('qualification request windows overlap or differ from declared order')
        last_end=end
        span=json.loads((cell/'fixed-attribution/routing-span.json').read_text());a=span['start_byte'];z=span['end_byte']
        if (a<last_route_end or not 0<=a<=z<=len(actual_router) or
                actual_router[a:z]!=(cell/'fixed-attribution/routing.log').read_bytes()):
            raise ValueError('qualification router span differs from actual shared log')
        last_route_end=z;checked.append(verified)
    if {p.name for p in (root/'cells').iterdir()}!=set(ids):raise ValueError('qualification block cell directories differ')
    if result!={'status':'passed','cells':checked,
        'scope':('distinct frozen-configuration performance repetitions sharing one startup and cleanup; independent cell and matched comparison acceptance remain separate' if scored else 'distinct unscored registered repetitions sharing one startup and cleanup; no profile adoption or scored acceptance')}:
        raise ValueError('qualification block completion differs from independent results')
    runtime.verify_evidence_manifest(root)
    return {'status':'passed','cells':checked,'scope':('distinct frozen-configuration performance repetitions; independent shared startup and cleanup verified' if scored else 'distinct unscored qualification repetitions; one verified shared startup and cleanup')}


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),sort_keys=True))
