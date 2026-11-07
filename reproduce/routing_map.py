"""Freeze native-router session headers for balanced, rotating controlled grids."""
import hashlib
import json
import re


def placement(row,ports,dp):
    if row['backend_port'] not in ports:raise ValueError('unregistered backend')
    values=row['rank_headers']
    if dp==1:
        if values:raise ValueError('unexpected TP rank header')
        rank=0
    else:
        if len(values)!=1 or not str(values[0]).isdigit() or not 0<=int(values[0])<dp:raise ValueError('missing or invalid DP rank header')
        rank=int(values[0])
    return ports.index(row['backend_port'])*dp+rank


def build(records,ports,dp,required_slots=None,require_recheck=True):
    if not ports or len(set(ports))!=len(ports) or type(dp) is not int or dp<1:raise ValueError('invalid routing geometry')
    units=len(ports)*dp;slots=required_slots if required_slots is not None else 128//units
    if slots<1 or slots*units<128:raise ValueError('map must cover128 controlled requests')
    sessions={};seen=set()
    for row in records:
        session=row['session_id'];request=row['request_id']
        if not re.fullmatch(r'grid[0-9]+',session) or request not in (session+'-r0',session+'-r1') or request in seen:
            raise ValueError('duplicate or unknown calibration request')
        seen.add(request);unit=placement(row,ports,dp)
        if session in sessions and sessions[session]!=unit:raise ValueError('native session placement changed')
        sessions[session]=unit
    chosen={str(i):[] for i in range(units)}
    for session in sorted(sessions,key=lambda s:int(s[4:])):
        values=chosen[str(sessions[session])]
        if len(values)<slots:values.append(session)
    if any(len(values)!=slots for values in chosen.values()):raise ValueError('routing map has incomplete unit coverage')
    for values in chosen.values():
        for session in values:
            if session+'-r0' not in seen or (require_recheck and session+'-r1' not in seen):raise ValueError('chosen session lacks stability recheck')
    result={'schema':'dsv41-native-session-map/v1','ports':ports,'dp':dp,'logical_units':units,
            'slots_per_unit':slots,'sessions':chosen,
            'scope':'synthetic native-router header placement only; actual GPU placement and counters require live verification'}
    result['mapping_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def select(mapping,concurrency,repetition,block=0):
    if (type(concurrency) is not int or not 1<=concurrency<=128 or type(repetition) is not int or repetition<0 or type(block) is not int or block<0):
        raise ValueError('invalid controlled placement request')
    signed={k:v for k,v in mapping.items() if k!='mapping_sha256'}
    if hashlib.sha256(json.dumps(signed,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=mapping.get('mapping_sha256'):
        raise ValueError('routing map fingerprint changed')
    units=mapping['logical_units'];sessions=mapping['sessions'];rows=[]
    if set(sessions)!=set(map(str,range(units))):raise ValueError('session map unit coverage missing')
    offset=(block*3+repetition*concurrency)%units
    for session in range(concurrency):
        unit=(offset+session)%units;slot=session//units
        if slot>=len(sessions[str(unit)]):raise ValueError('insufficient calibrated session slots')
        rows.append({'task_session':session,'session_header':sessions[str(unit)][slot],
                     'endpoint':'http://127.0.0.1:'+str(mapping['ports'][unit//mapping['dp']])+'/metrics',
                     'rank':str(unit%mapping['dp'])})
    if len({x['session_header'] for x in rows})!=len(rows):raise ValueError('duplicate controlled session headers')
    return rows


def load_verified(root,router_lock_sha256,router_executable_sha256):
    from pathlib import Path
    import runtime
    root=Path(root);runtime.verify_evidence_manifest(root)
    envelope=json.loads((root/'routing-map.json').read_text())
    if (envelope['router_lock_sha256']!=router_lock_sha256 or
        envelope['router_executable_sha256']!=router_executable_sha256):raise ValueError('routing map uses another router build')
    for file,key in (('arrivals.json','arrivals_sha256'),('router-argv.json','router_argv_sha256')):
        if hashlib.sha256((root/file).read_bytes()).hexdigest()!=envelope[key]:raise ValueError('routing map source changed')
    argv=json.loads((root/'router-argv.json').read_text())
    mapping=envelope['mapping']
    try:
        urls=argv[argv.index('--worker-urls')+1:argv.index('--policy')]
        if (urls!=['http://127.0.0.1:'+str(p) for p in mapping['ports']] or
            argv[argv.index('--policy')+1]!='consistent_hash' or '--disable-retries' not in argv or
            argv[argv.index('--intra-node-data-parallel-size')+1]!=str(mapping['dp'])):
            raise ValueError('router map launch geometry differs')
    except (ValueError,IndexError) as exc:raise ValueError('router map launch geometry differs') from exc
    rows=[r for r in json.loads((root/'arrivals.json').read_text()) if r['session_id']!='no-retry']
    mapping=envelope['mapping']
    if build(rows,mapping['ports'],mapping['dp'],mapping['slots_per_unit'])!=mapping:raise ValueError('routing map differs from retained backend headers')
    return mapping
