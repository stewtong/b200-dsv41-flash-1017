"""Plan-bound preregistration and selection of unscored tuning candidates."""
import copy
import hashlib
import json
import re

import launcher
import profile_overrides

TOPS={'2xtp4','4xdep2','2xdep4','dep8'}
ACCEPTANCE={'minimum_gain':.10,'maximum_p95_regression':.05,'gsm8k_noninferiority_margin':.01,
            'required_quality':'paired95pct_lower_bound_above_negative_margin; no identifier/tool/stream regression',
            'low_sample_tail':'100-request validation before a tail-dependent adoption'}


def record_digest(record):
    return hashlib.sha256(json.dumps({k:v for k,v in record.items() if k!='registration_sha256'},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def template(layout,delta):
    result=launcher.render(layout,'example/image@sha256:'+'0'*64,'profile','/model-source','/cache-source','0,1',overrides=delta)
    result['cpu_policy']='two-numa-four-physical-cores-per-node-v1'
    return result


def register(ident,layout,delta,manifest,*,hypothesis,deciding_cells,protected_tail,control=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',ident) or layout not in launcher.LAYOUTS:
        raise ValueError('invalid tuning profile identity or layout')
    profile_overrides.validate(delta)
    if (not isinstance(hypothesis,str) or not 12<=len(hypothesis)<=1000 or
        not isinstance(deciding_cells,list) or not deciding_cells or len(set(deciding_cells))!=len(deciding_cells) or
        protected_tail not in ('p95_ttft_seconds','p95_request_seconds','p95_stream_window_seconds_per_token')):
        raise ValueError('preregister hypothesis, deciding cells and protected tail')
    known={c['id']:c for c in manifest['cells']}
    anchor=all(i in known and known[i]['suite'] in ('anchor','anchor_greedy') for i in deciding_cells)
    if anchor and (delta!={} or control is not None):raise ValueError('common anchors require the unchanged NoSpec starting profile')
    allowed_suites=('anchor','anchor_greedy') if anchor else ('tuning_trial','tuning_extra')
    if (not anchor and layout not in TOPS) or any(i not in known or known[i]['layout']!=layout or
           known[i]['suite'] not in allowed_suites for i in deciding_cells):
        raise ValueError('deciding tuning anchors differ from the declared ledger')
    runtime=manifest['runtime']
    if (any(not re.fullmatch(r'[0-9a-f]{64}',str(manifest.get(k))) for k in ('plan_sha256','matrix_sha256')) or
        not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}',str(runtime.get('image_digest'))) or
        runtime.get('model_revision')!='1bae0e1088993b9b82de08f59de5be5a34e42170' or
        runtime.get('tokenizer_revision')!=runtime.get('model_revision')):
        raise ValueError('profile registration requires bound plan, matrix, image and checkpoint')
    if control is not None:
        validate_record(control,manifest)
        if control['layout']!=layout:raise ValueError('control layout differs')
    base=control['profile_sha256'] if control is not None else launcher.profile_sha256(layout)
    payload=template(layout,delta)
    record={'schema':'registered-tuning-profile/v1','id':ident,'layout':layout,'profile_delta':copy.deepcopy(delta),
            'profile_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'template':payload,'control_profile_sha256':base,'hypothesis':hypothesis,
            'deciding_cell_ids':list(deciding_cells),'primary_metric':'geomean_paired_valid_burst_throughput',
            'protected_tail_metric':protected_tail,'acceptance_policy':copy.deepcopy(ACCEPTANCE),
            'plan_sha256':manifest['plan_sha256'],'matrix_sha256':manifest['matrix_sha256'],
            'image_digest':runtime['image_digest'],'model_revision':runtime['model_revision'],
            'tokenizer_revision':runtime['tokenizer_revision'],'status':'preregistered_unqualified',
            'scope':'exact candidate registration; no installed compatibility, performance, quality or adoption claim'}
    record['registration_sha256']=record_digest(record)
    return record


def validate_record(record,manifest):
    if not isinstance(record,dict) or record.get('schema')!='registered-tuning-profile/v1':raise ValueError('invalid profile record')
    expected=register(record['id'],record['layout'],record['profile_delta'],manifest,
        hypothesis=record['hypothesis'],deciding_cells=record['deciding_cell_ids'],protected_tail=record['protected_tail_metric'])
    # A later complete-profile candidate can name an already declared control.
    # The parent hash never changes the candidate's exact command structure.
    parent=record.get('control_profile_sha256')
    if not re.fullmatch(r'[0-9a-f]{64}',str(parent)):raise ValueError('control profile hash missing')
    expected['control_profile_sha256']=parent
    expected['registration_sha256']=record_digest(expected)
    if record!=expected:raise ValueError('profile registration changed or differs from plan/source')
    return record


def resolve(config,manifest):
    phase=config.get('execution_phase','pilot');layout=config['layout']
    if phase=='fixed-performance':
        import fixed_profiles
        return fixed_profiles.resolve(config,manifest)
    if 'profile_overrides' in config:raise ValueError('caller profile overrides are not a registered manifest selection')
    if phase=='pilot':
        if config.get('profile_id') is not None:raise ValueError('pilot must use its no-speculation starting profile')
        expected=manifest['runtime']['starting_profiles'].get(layout)
        if expected!=launcher.profile_sha256(layout):raise ValueError('starting profile does not match launcher')
        return {'phase':'pilot','delta':None,'profile_sha256':expected,'record':None}
    if phase!='qualification':raise ValueError('unknown or unqualified execution phase')
    profiles=manifest['runtime'].get('qualification_profiles',{})
    ident=config.get('profile_id')
    if not isinstance(profiles,dict) or ident not in profiles:raise ValueError('candidate profile is not preregistered in this manifest')
    record=validate_record(profiles[ident],manifest)
    if record['layout']!=layout or record['id']!=ident:raise ValueError('selected candidate layout or identity differs')
    allowed_controls={launcher.profile_sha256(layout)}
    for other_id,other in profiles.items():
        if other_id!=ident and other.get('layout')==layout:
            validate_record(other,manifest);allowed_controls.add(other['profile_sha256'])
    if record['control_profile_sha256'] not in allowed_controls:raise ValueError('candidate control is not registered')
    return {'phase':'qualification','delta':record['profile_delta'],'profile_sha256':record['profile_sha256'],'record':record}


def expected_runtime(binding,layout):
    plan=template(layout,binding['delta'] or {})
    choices=[]
    for command in plan['commands']:
        argv=command['argv'];attention=json.loads(argv[argv.index('--attention-config')+1])
        choices.append({'backend':attention['backend'],'kv_dtype':argv[argv.index('--kv-cache-dtype')+1],
            'speculative_config':json.loads(argv[argv.index('--speculative-config')+1]) if '--speculative-config' in argv else None,
            'moe_backend':json.loads(argv[argv.index('--kernel-config')+1]).get('moe_backend') if '--kernel-config' in argv else None})
    if not choices or any(c!=choices[0] for c in choices):raise ValueError('candidate engine replicas differ')
    return choices[0]


def bound_hash(config,manifest):
    """Read historical identity without recomputing it from today's renderer."""
    phase=config.get('execution_phase','pilot')
    if phase=='pilot':value=manifest['runtime']['starting_profiles'][config['layout']]
    elif phase=='qualification':
        record=manifest['runtime'].get('qualification_profiles',{}).get(config.get('profile_id'),{})
        if record.get('layout')!=config['layout']:raise ValueError('historical candidate layout differs')
        value=record.get('profile_sha256')
    elif phase=='fixed-performance':
        record=manifest['runtime'].get('fixed_profiles',{}).get(config['layout'],{})
        if record.get('layout')!=config['layout'] or record.get('id')!=config.get('profile_id'):
            raise ValueError('historical fixed configuration identity differs')
        value=record.get('profile_sha256')
    else:raise ValueError('unknown historical profile phase')
    if not re.fullmatch(r'[0-9a-f]{64}',str(value)):raise ValueError('historical profile hash invalid')
    return value
