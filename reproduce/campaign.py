#!/usr/bin/env python3
"""Campaign CLI: planning, fixtures, guarded staging, and pilot components."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import socketserver
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def expand(m):
    schema=m.get('schema')
    if schema=='dsv41-b200-campaign/v3':
        import comparison_scope
        return comparison_scope.expand(m)
    if schema!='dsv41-b200-campaign/v2':
        raise ValueError('unsupported campaign matrix schema')
    cells = []
    def add(suite, tops, ins, cs, caches, outs, reps):
        for top, n, c, cache, out in itertools.product(tops, ins, cs, caches, outs):
            ident = f'{suite}-{top}-i{n}-c{c}-{cache}-o{out}'
            cells.append(dict(id=ident, suite=suite, layout=top, input_tokens=n,
                              concurrency=c, cache=cache, output_tokens=out,
                              repetitions=reps, status='planned'))
    tops = m['topologies']
    add('core', tops, m['input_tokens'], m['node_concurrency'], m['cache_states'], [m['core_output_tokens']], m['repetitions'])
    for name in ('output_sensitivity', 'historical_bridge_extra'):
        x = m[name]
        add(name, tops, x['input_tokens'], x['node_concurrency'], x['cache_states'], x['output_tokens'], x['repetitions'])
    add('quality', tops, m['input_tokens'], m['node_concurrency'], m['cache_states'], [None], m['repetitions'])
    x = m['common_settings_anchors']
    add('anchor', x['topologies'], x['input_tokens'], x['node_concurrency'], x['cache_states'], [2048], x['repetitions'])
    y = x['greedy_sampling_control']
    add('anchor_greedy', x['topologies'], y['input_tokens'], y['node_concurrency'], y['cache_states'], [2048], y['repetitions'])
    add('pilot_smoke', x['topologies'], [8000], [1], ['cold'], [128], 1)
    add('pilot_fit', tops, [128000, 512000, 1000000], [1], ['cold'], [512], 1)
    add('pilot_warm', tops, [128000], [1], ['warm_prefix'], [512], 1)
    add('pilot_concurrency', tops, [128000, 1000000], [8, 32], ['cold'], [512], 1)
    def supplemental(suite, layout, fields, repetitions=3):
        parts = '-'.join(str(fields[k]) for k in sorted(fields))
        ident = f'{suite}-{layout}-{parts}'
        cells.append({'id':ident, 'suite':suite, 'layout':layout, 'parameters':fields,
                      'repetitions':repetitions, 'status':'planned'})
    for layout in tops:
        for n, concurrency in itertools.product([128000, 256000, 512000, 960000], [1,8,16,32]):
            supplemental('growing', layout, {'input':n,'concurrency':concurrency})
        for n, concurrency, control in itertools.product([32000,128000,512000,1000000], [8,32], ['affinity','no_affinity','changed_system','cold']):
            supplemental('cache_control', layout, {'input':n,'concurrency':concurrency,'control':control})
        for concurrency in [8,16,32,64]:
            supplemental('mixed_load', layout, {'concurrency':concurrency})
        for rate, batch in itertools.product(['0.1','0.2','0.4','0.8','1.6','3.2','6.4'], [1,2,3]):
            supplemental('arrival_ladder', layout, {'rate':rate,'batch':batch})
            if batch > 1: cells[-1]['condition'] = 'matched supplemental batch only for undersampled buckets, subject to stop rules'
        for load in ['own_capacity','common_rate']:
            supplemental('burstiness', layout, {'load':load})
        for (n, output), concurrency in itertools.product([(8000,1000),(128000,2000),(512000,2000),(1000000,512)], [1,8,32,64,128]):
            supplemental('continuous', layout, {'input':n,'output':output,'concurrency':concurrency})
        for concurrency, routing in itertools.product([16,32], ['balanced','consistent_hash','collision']):
            supplemental('placement', layout, {'concurrency':concurrency,'routing':routing})
        for n, concurrency in itertools.product(m['input_tokens'], m['node_concurrency']):
            supplemental('admission', layout, {'input':n,'concurrency':concurrency}, 1)
        for load in ['slo_80pct','near_capacity']:
            supplemental('soak', layout, {'load':load}, 1)
        for n, stage in itertools.product([8000,128000,1000000], ['prefill','decode']):
            supplemental('cancel', layout, {'input':n,'stage':stage})
        supplemental('replica_failure', layout, {'concurrency':16}, 1)
        for thinking in ['on','off']:
            supplemental('gsm8k', layout, {'thinking':thinking,'tasks':1319}, 1)
        for concurrency in [1,8,32]:
            supplemental('coding_panel', layout, {'concurrency':concurrency,'tasks':100}, 1)
        for effort, thinking, variant in itertools.product([25,50,75,100], ['on','off'], ['no_spec','static','adaptive']):
            supplemental('reasoning_effort', layout, {'effort':effort,'thinking':thinking,'variant':variant}, 1)
    protocol_cases = ['plain_text','thinking','tools','tool_result_history','developer_system_placement',
                      'json_schema','tool_choice','empty_content','cancellation','output_cap','over_context']
    tuning_levers = ['dspark_static','dspark_adaptive','engram_resident','prefill_batch',
                     'bounded_replay','graph_admission']
    for layout in tops:
        for api, mode, case in itertools.product(['chat_completions','responses','messages'], ['stream','non_stream'], protocol_cases):
            supplemental('api_protocol', layout, {'api':api,'mode':mode,'case':case}, 1)
        for lever, arm, n, concurrency in itertools.product(tuning_levers, ['control_before','challenger','control_after'],
                                                            [8000,128000,512000,1000000], [1,8,32]):
            supplemental('tuning_trial', layout, {'lever':lever,'arm':arm,'input':n,'concurrency':concurrency})
            cells[-1]['condition'] = 'adaptive requires correct static speculation' if lever == 'dspark_adaptive' else 'settings and decision metrics require preregistration'
    supplemental('long_soak', 'selected', {'minutes':840}, 1)
    import suite_controls
    cells.extend(suite_controls.controls())
    for cell in cells:
        cell['request_contract']=suite_controls.contract(cell)
    if len({x['id'] for x in cells}) != len(cells):
        raise ValueError('duplicate cell ID')
    return cells

def ledger(matrix, plan_path=None):
    import launcher
    m = json.loads(Path(matrix).read_text())
    cells = expand(m)
    if m['schema']=='dsv41-b200-campaign/v3':
        return {'schema':'dsv41-local-ledger/v2','scope_id':m['scope_id'],'matrix_schema':m['schema'],
            'matrix_sha256':digest(matrix),'plan_sha256':digest(plan_path) if plan_path else None,
            'complete_universe':True,'omissions':[],
            'evidence_class':'complete required planning universe, not measured coverage',
            'historical_scope':m['historical_scope'],'deferred_suites':m['deferred_suites'],
            'runtime':{'image_digest':None,'profile_hashes':None,'operational_deadline':None,
                'starting_profiles':{name:launcher.profile_sha256(name) for name in launcher.LAYOUTS},
                'model_revision':m['model_revision'],'tokenizer_revision':m['model_revision'],
                'top_p':1.0,'retirement_gate':'unverified'},'cells':cells}
    checks = {'core': (704, 67320), 'output_sensitivity': (320, 40560),
              'historical_bridge_extra': (12, 264), 'quality': (704, 67320),
              'anchor': (144, 5904), 'anchor_greedy': (6, 144)}
    for name, expected in checks.items():
        rows = [x for x in cells if x['suite'] == name]
        actual = (len(rows), sum(x['concurrency'] * x['repetitions'] for x in rows))
        if actual != expected:
            raise ValueError(f'{name}: expected {expected}, got {actual}')
    return {'schema': 'dsv41-local-ledger/v1', 'matrix_sha256': digest(matrix),
            'plan_sha256': digest(plan_path) if plan_path else None,
            'complete_universe': False,
            'omissions': ['tokenized supplemental history lengths and optional warm traffic extension children', 'resolved conditional tuning settings and decisions', 'measured rate bracket/doubling child IDs', 'frozen panel calibration outcomes and executable coding tasks'],
            'runtime': {'image_digest': None, 'profile_hashes': None, 'operational_deadline': None,
                        'starting_profiles': {name: launcher.profile_sha256(name) for name in launcher.LAYOUTS},
                        'model_revision': m['model_revision'], 'tokenizer_revision': m['model_revision'],
                        'top_p': 1.0, 'retirement_gate': 'unverified'}, 'cells': cells}

def validate_phase(m, phase, *, cell_ids=None, evidence=False):
    """Validate only inputs required by the selected phase, never future results."""
    r = m.get('runtime', {})
    if type(evidence) is not bool or evidence and phase!='fixed-performance':
        raise ValueError('historical evidence validation requires explicit fixed-performance scope')
    fields = {'pilot': ('image_digest', 'starting_profiles', 'operational_deadline',
                         'model_revision', 'tokenizer_revision', 'router_pin', 'safety_controls'),
              'scored': ('image_digest', 'profile_hashes', 'operational_deadline',
                         'model_revision', 'tokenizer_revision', 'router_pin', 'safety_controls', 'top_p','temperature','capture_schema','sampling_source')}
    fields['fixed-performance'] = fields['scored']
    if phase not in fields:
        raise ValueError('unknown phase')
    if phase == 'fixed-performance':
        if (not isinstance(cell_ids, list) or not cell_ids or
                any(not isinstance(i, str) or not i for i in cell_ids) or
                len(set(cell_ids)) != len(cell_ids)):
            raise ValueError('fixed-performance requires distinct explicit cell IDs')
        rows = m.get('cells')
        if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get('id'), str) for row in rows):
            raise ValueError('fixed-performance manifest cell declarations missing')
        known = {row['id']: row for row in rows}
        if len(known) != len(rows):
            raise ValueError('duplicate manifest cell declarations')
        matrix_source=HERE / ('comparison-matrix.json' if m.get('scope_id')=='context-concurrency-336-v1' else 'matrix.json')
        if m.get('matrix_sha256') != digest(matrix_source):
            raise ValueError('fixed-performance matrix differs from current source')
        canonical = {row['id']: row for row in ledger(matrix_source)['cells']}
        if not set(canonical).issubset(known):
            raise ValueError('fixed-performance must retain the full base ledger')
        if m.get('scope_id')=='context-concurrency-336-v1' and set(known)!=set(canonical):
            raise ValueError('schema-v3 manifest must contain exactly the336required cells')
        for ident in cell_ids:
            declared = known.get(ident)
            expected = canonical.get(ident)
            if (expected is None or expected['suite'] not in ('core', 'warm_prefix_panel', 'output_sensitivity', 'historical_bridge_extra') or
                    declared is None or
                    {k: v for k, v in declared.items() if k != 'status'} !=
                    {k: v for k, v in expected.items() if k != 'status'}):
                raise ValueError('fixed-performance cell differs from the fixed-output ledger')
    elif cell_ids is not None:
        raise ValueError('explicit cell selection requires fixed-performance phase')
    if not re.fullmatch(r'[0-9a-f]{64}', str(m.get('plan_sha256'))):
        raise ValueError('plan content hash missing or invalid')
    missing = [k for k in fields[phase] if not r.get(k)]
    if r.get('retirement_gate') != 'passed':
        missing.append('retirement_gate')
    if phase in ('scored', 'fixed-performance') and r.get('capture_schema')!='http-body-chunks/v1':
        missing.append('bounded HTTP body capture schema')
    if phase == 'scored' and not m.get('complete_universe'):
        missing.append('complete_universe')
    if missing:
        raise ValueError('unresolved ' + phase + ' requirements: ' + ', '.join(missing))
    if not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}', str(r['image_digest'])):
        raise ValueError('image digest is not immutable')
    if r['model_revision'] != '1bae0e1088993b9b82de08f59de5be5a34e42170' or r['tokenizer_revision'] != r['model_revision']:
        raise ValueError('model or tokenizer revision mismatch')
    if not re.fullmatch(r'[0-9a-f]{64}', str(r['router_pin'])):
        raise ValueError('router pin must be a SHA-256')
    try:
        deadline = datetime.fromisoformat(r['operational_deadline'].replace('Z','+00:00'))
        if deadline.tzinfo is None or not evidence and deadline <= datetime.now(timezone.utc): raise ValueError('deadline expired or timezone missing')
    except (AttributeError, TypeError) as exc:
        raise ValueError('invalid operational deadline') from exc
    if phase in ('scored', 'fixed-performance'):
        source=r['sampling_source']
        if (not isinstance(source,dict) or source.get('status')!='pinned_model_card_verified' or
            source.get('revision')!=r['model_revision'] or
            source.get('sha256')!='b72b796e4f4c739eca5d0ad360a7679b39118ca7eb64891f7151a527b292c42a' or
            source.get('parameters')!={'temperature':1.0,'top_p':1.0} or r['top_p']!=1.0 or r['temperature']!=1.0):
            raise ValueError('scored sampling source or parameters mismatch')
    profiles = r['starting_profiles'] if phase == 'pilot' else r['profile_hashes']
    required_layouts = {'2xtp4','4xdep2','2xdep4','dep8','tp8','4xtp2'} if phase == 'pilot' else {'2xtp4','4xdep2','2xdep4','dep8'}
    if not isinstance(profiles, dict) or set(profiles) != required_layouts or any(not re.fullmatch(r'[0-9a-f]{64}', str(v)) for v in profiles.values()):
        raise ValueError('phase profile hashes incomplete')
    if not isinstance(r['safety_controls'], dict) or any(r['safety_controls'].get(k) is not True for k in ('ownership','watchdog','cancellation','cleanup','persistence')):
        raise ValueError('safety control declarations incomplete')
    return True

def parse_sse(data):
    # Frames, rather than socket reads, define JSON boundaries.
    result, done = [], False
    for frame in data.replace('\r\n', '\n').split('\n\n'):
        lines = [x[5:].lstrip() for x in frame.split('\n') if x.startswith('data:')]
        if not lines:
            continue
        payload = '\n'.join(lines)
        if payload == '[DONE]':
            done = True
        else:
            result.append(json.loads(payload))
    if not done:
        raise ValueError('missing DONE')
    return result

class Fixture(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        chunks = [b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
                  b'data: {"choices":[{"delta":{"content":"bc"},"finish_reason":"stop"}]}\n\n',
                  b'data: {"choices":[],"usage":{"completion_tokens":3}}\n\n', b'data: [DONE]\n\n']
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()
    def log_message(self, *args):
        pass

def fixture_run(directory, crash=False):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'execution-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'attempts': [], 'completed': False}
    if state['completed']:
        return 'already completed'
    # Preserve every incomplete attempt; an interrupted attempt is never overwritten.
    known = {a['file'] for a in state['attempts']}
    for orphan in sorted(out.glob('attempt-[0-9][0-9][0-9][0-9].json')):
        if orphan.name not in known:
            state['attempts'].append({'file': orphan.name, 'sha256': digest(orphan), 'status': 'incomplete'})
    attempt = max([int(a['file'][8:12]) for a in state['attempts']] or [0]) + 1
    raw = out / f'attempt-{attempt:04d}.json'
    server = socketserver.TCPServer(('127.0.0.1', 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions', data=b'{}', headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as response:
            body = response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    events = parse_sse(body)
    usage = [x['usage']['completion_tokens'] for x in events if 'usage' in x]
    if len(usage) != 1:
        raise ValueError('expected exactly one usage record')
    # Synthetic monotonic times are fixture inputs, not observed network timings.
    record = {'kind': 'synthetic-fixture', 'body': body, 'clock': {'start': 10, 'first': 12, 'last': 14, 'end': 15},
              'completion_tokens': usage[0], 'valid': True}
    atomic(raw, record)
    state['attempts'].append({'file': raw.name, 'sha256': digest(raw), 'status': 'incomplete'})
    atomic(state_path, state)
    if crash:
        raise RuntimeError('injected crash after raw persistence')
    t = record['clock']
    summary = {'ttft_seconds': t['first'] - t['start'],
               'burst_valid_tokens_per_second': record['completion_tokens'] / (t['end'] - t['start']),
               'stream_seconds_per_token': (t['last'] - t['first']) / (record['completion_tokens'] - 1)}
    atomic(out / f'attempt-{attempt:04d}-summary.json', summary)
    state['attempts'][-1]['status'] = 'passed'
    state['completed'] = True
    atomic(state_path, state)
    return 'completed'

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan'); a.add_argument('--matrix', default=str(HERE / 'comparison-matrix.json')); a.add_argument('--plan'); a.add_argument('--output', required=True)
    a = sub.add_parser('scope-migration'); a.add_argument('--old-matrix',default=str(HERE/'matrix.json')); a.add_argument('--matrix',default=str(HERE/'comparison-matrix.json')); a.add_argument('--plan',required=True); a.add_argument('--output',required=True)
    a = sub.add_parser('derive-comparison'); a.add_argument('--manifest',required=True); a.add_argument('--plan',required=True); a.add_argument('--run-dir',action='append',default=[]); a.add_argument('--archive-index',action='append',default=[]); a.add_argument('--output',required=True)
    a = sub.add_parser('traffic-plan'); a.add_argument('--rate',type=float,required=True); a.add_argument('--seed',type=int,required=True); a.add_argument('--gamma-shape',type=float,default=1.0); a.add_argument('--output',required=True)
    a = sub.add_parser('slo-assess'); a.add_argument('--input',required=True); a.add_argument('--output',required=True)
    a = sub.add_parser('fixture'); a.add_argument('--output', required=True); a.add_argument('--crash-after-raw', action='store_true')
    a = sub.add_parser('fixture-burst'); a.add_argument('--output', required=True); a.add_argument('--retry', action='store_true'); a.add_argument('--retry-reason')
    a = sub.add_parser('validate-runtime'); a.add_argument('--manifest', required=True); a.add_argument('--plan', required=True); a.add_argument('--phase', choices=['pilot', 'scored', 'fixed-performance'], default='pilot'); a.add_argument('--cell', action='append', dest='cell_ids')
    a = sub.add_parser('verify-checkpoint'); a.add_argument('--inventory', required=True); a.add_argument('--root', required=True); a.add_argument('--receipt')
    a = sub.add_parser('render-launch'); a.add_argument('--layout', required=True); a.add_argument('--image', required=True); a.add_argument('--run-id', required=True); a.add_argument('--model-dir', required=True); a.add_argument('--cache-dir', required=True); a.add_argument('--overrides-file')
    a = sub.add_parser('conditioning-plan'); a.add_argument('--ledger', required=True); a.add_argument('--output', required=True)
    a = sub.add_parser('coding-panel'); a.add_argument('--output', required=True)
    a = sub.add_parser('verify-coding-capture'); a.add_argument('--run-dir', required=True)
    a = sub.add_parser('verify-cancellation'); a.add_argument('--run-dir', required=True)
    a = sub.add_parser('verify-fixed-capture'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-fixed-session'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-fixed-attribution'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-fixed-qualification'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-fixed-block'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-tuning-performance'); a.add_argument('--manifest',required=True); a.add_argument('--profile-id',required=True); a.add_argument('--catalog',required=True); a.add_argument('--output',required=True)
    a = sub.add_parser('verify-sampler-overhead'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-client-fixture'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-node-client-fixture'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('verify-model-client-check'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('prepare-model-client-check'); a.add_argument('--config',required=True)
    a = sub.add_parser('compare-sampler-overhead'); a.add_argument('--before',required=True); a.add_argument('--on',required=True); a.add_argument('--after',required=True)
    a = sub.add_parser('verify-engine-profile'); a.add_argument('--run-dir',required=True)
    a = sub.add_parser('profile-register'); a.add_argument('--manifest',required=True); a.add_argument('--profile-id',required=True); a.add_argument('--layout',required=True); a.add_argument('--delta',required=True); a.add_argument('--hypothesis',required=True); a.add_argument('--deciding-cells',required=True); a.add_argument('--protected-tail',required=True); a.add_argument('--control-file'); a.add_argument('--output',required=True)
    a = sub.add_parser('node-coding-score-control'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-coding-score'); a.add_argument('--config', required=True)
    a = sub.add_parser('verify-coding-score-control'); a.add_argument('--run-dir', required=True)
    a = sub.add_parser('verify-coding-score'); a.add_argument('--run-dir', required=True); a.add_argument('--capture-run', required=True); a.add_argument('--qualification', required=True)
    a = sub.add_parser('node-pilot-recover'); a.add_argument('--attempt', required=True)
    a = sub.add_parser('node-router-probe'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-pilot-manifest'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-qualify-router'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-bootstrap'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-prompt'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-input-panel'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-qualify-image'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-smoke'); a.add_argument('--config', required=True)
    a = sub.add_parser('reconcile'); a.add_argument('--run-dir', required=True)
    a = sub.add_parser('node-pilot'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-pilot-block'); a.add_argument('--config', required=True)
    a = sub.add_parser('node-pilot-block-recover'); a.add_argument('--run-dir', required=True)
    a = sub.add_parser('stage-checkpoint'); a.add_argument('--config', required=True)
    args = p.parse_args()
    if args.command == 'plan':
        result=ledger(args.matrix,args.plan);atomic(args.output,result)
        if result.get('scope_id')=='context-concurrency-336-v1':
            print(json.dumps({'scope_id':result['scope_id'],'cells':len(result['cells']),
                'repetitions':sum(c['repetitions'] for c in result['cells']),
                'measured_requests':sum(c['concurrency']*c['repetitions'] for c in result['cells']),
                'evidence_class':'planning_only_not_measurements'}))
        else:print('Partial planning ledger written; supplemental suites remain unexpanded.')
    elif args.command=='scope-migration':
        import comparison_scope
        old=ledger(args.old_matrix);new=ledger(args.matrix,args.plan)
        result=comparison_scope.migration(old['cells'],new['cells'])
        result.update(old_matrix_sha256=digest(args.old_matrix),matrix_sha256=digest(args.matrix),plan_sha256=digest(args.plan))
        atomic(args.output,result);print(json.dumps({k:result[k] for k in ('status','old_rows','new_required_cells','dispositions','measurement_credit_transferred')}))
    elif args.command=='derive-comparison':
        import comparison_results
        manifest=json.loads(Path(args.manifest).read_text())
        if manifest.get('plan_sha256')!=digest(args.plan):raise ValueError('comparison plan content hash differs')
        captures=list(args.run_dir);indexes=[]
        for path in args.archive_index:
            archive_rows=json.loads(Path(path).read_text())
            if not isinstance(archive_rows,list):raise ValueError('archive index must be an explicit list of pinned captures')
            import capture_archive
            for row in archive_rows:capture_archive.validate(row)
            captures.extend(archive_rows);indexes.append({'path':str(Path(path).resolve()),'sha256':digest(path)})
        result=comparison_results.derive(manifest,captures)
        if indexes:result['archive_indexes']=indexes
        result['manifest_sha256']=digest(args.manifest)
        atomic(args.output,result)
        print(json.dumps({k:result[k] for k in ('accepted_cells','required_cells','accepted_cold_cells','accepted_warm_cells','measured_repetitions','measured_requests','complete')}))
    elif args.command == 'traffic-plan':
        import traffic
        schedule=traffic.offered_schedule(args.rate,args.seed,gamma_shape=args.gamma_shape)
        atomic(args.output,{'schema':'dsv41-offered-traffic/v1','scope':'planned offered schedule; no inference dispatched','rate':args.rate,'seed':args.seed,'gamma_shape':args.gamma_shape,'ramp_seconds':60,'measured_seconds':300,'requests':schedule})
        print(json.dumps({'requests':len(schedule),'dispatched':False}))
    elif args.command == 'slo-assess':
        import traffic
        source=json.loads(Path(args.input).read_text())
        result=traffic.assess_rate(source['repetitions'],source['required_buckets'])
        result['source_sha256']=digest(args.input)
        atomic(args.output,result); print(json.dumps({'slo_verdict':result['verdict'],'source_sha256':result['source_sha256']}))
    elif args.command == 'fixture':
        print(fixture_run(args.output, args.crash_after_raw))
    elif args.command == 'fixture-burst':
        import attempts, burst
        attempt = attempts.begin(args.output, 'fixture-burst', retry=args.retry, retry_reason=args.retry_reason, method_sha256=digest(HERE/'burst.py'))
        server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Fixture)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            result = burst.execute(f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions',
                                   [{'id':'one','body':{},'expected':'abc'},
                                    {'id':'two','body':{},'expected':'wrong'}], attempt, 5, 'fixture')
            attempts.finish(attempt, 'failed' if result['has_failed_request'] else 'passed',
                            ['one.json', 'two.json', 'summary.json'])
            print(json.dumps({'attempt':str(attempt), 'summary':result['summary']}, sort_keys=True))
        finally:
            server.shutdown(); server.server_close(); thread.join()
    elif args.command == 'verify-checkpoint':
        import verify_checkpoint
        receipt = verify_checkpoint.verify(args.inventory, args.root)
        receipt.update(inventory_sha256=digest(args.inventory), root=str(Path(args.root).resolve()))
        if args.receipt: atomic(args.receipt, receipt)
        print(json.dumps(receipt, sort_keys=True))
    elif args.command == 'render-launch':
        import launcher
        overrides=json.loads(Path(args.overrides_file).read_text()) if args.overrides_file else None
        print(json.dumps(launcher.render(args.layout, args.image, args.run_id, args.model_dir, args.cache_dir,overrides=overrides), indent=2))
    elif args.command == 'conditioning-plan':
        import conditioning_plan
        print('Partial conditioning requests written:', conditioning_plan.write(args.ledger, args.output))
    elif args.command == 'coding-panel':
        import coding_panel
        panel=coding_panel.tasks(); result=coding_panel.validate_panel(panel)
        atomic(args.output, {'panel':panel,'validation':result})
        print(json.dumps(result,sort_keys=True))
    elif args.command == 'verify-coding-capture':
        import verify_coding_capture
        print(json.dumps(verify_coding_capture.verify(args.run_dir),sort_keys=True))
    elif args.command == 'verify-cancellation':
        import verify_cancellation
        print(json.dumps(verify_cancellation.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-fixed-capture':
        import verify_fixed_capture
        print(json.dumps(verify_fixed_capture.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-fixed-session':
        import verify_fixed_session
        print(json.dumps(verify_fixed_session.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-fixed-attribution':
        import verify_fixed_attribution
        print(json.dumps(verify_fixed_attribution.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-fixed-qualification':
        import verify_fixed_qualification
        print(json.dumps(verify_fixed_qualification.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-client-fixture':
        import verify_client_fixture
        print(json.dumps(verify_client_fixture.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-node-client-fixture':
        import verify_node_client_fixture
        print(json.dumps(verify_node_client_fixture.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-model-client-check':
        import verify_model_client_check
        print(json.dumps(verify_model_client_check.verify(args.run_dir),sort_keys=True))
    elif args.command=='prepare-model-client-check':
        import node_model_client_check
        prepared,budget=node_model_client_check.prepare(json.loads(Path(args.config).read_text()))
        print(json.dumps({'status':'inputs_verified_no_dispatch','cell_id':prepared['identity'],
            'repetitions':3,'bursts':9,'requests':9*prepared['concurrency'],'request_budget_seconds':budget},sort_keys=True))
    elif args.command=='verify-sampler-overhead':
        import verify_sampler_overhead
        print(json.dumps(verify_sampler_overhead.verify(args.run_dir),sort_keys=True))
    elif args.command=='compare-sampler-overhead':
        import verify_sampler_overhead
        print(json.dumps(verify_sampler_overhead.compare(args.before,args.on,args.after),sort_keys=True))
    elif args.command=='verify-fixed-block':
        import verify_fixed_block
        print(json.dumps(verify_fixed_block.verify(args.run_dir),sort_keys=True))
    elif args.command=='verify-tuning-performance':
        import verify_tuning_performance
        result=verify_tuning_performance.verify(args.manifest,args.profile_id,args.catalog)
        atomic(args.output,result)
        print(json.dumps({k:result[k] for k in ('performance_gate_passed','performance_adoption_eligible','observed_gain','controls_within_five_percent')},sort_keys=True))
    elif args.command=='verify-engine-profile':
        import verify_engine_profile
        print(json.dumps(verify_engine_profile.verify(args.run_dir),sort_keys=True))
    elif args.command == 'profile-register':
        import profile_registry
        record=profile_registry.register(args.profile_id,args.layout,json.loads(Path(args.delta).read_text()),
            json.loads(Path(args.manifest).read_text()),hypothesis=args.hypothesis,
            deciding_cells=json.loads(Path(args.deciding_cells).read_text()),protected_tail=args.protected_tail,
            control=json.loads(Path(args.control_file).read_text()) if args.control_file else None)
        with Path(args.output).open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps({'status':record['status'],'id':record['id'],'profile_sha256':record['profile_sha256']},sort_keys=True))
    elif args.command in ('node-coding-score-control','node-coding-score'):
        import node_coding_score
        config=json.loads(Path(args.config).read_text())
        result=node_coding_score.control(config) if args.command=='node-coding-score-control' else node_coding_score.run(config)
        print(json.dumps(result,sort_keys=True))
    elif args.command in ('verify-coding-score-control','verify-coding-score'):
        import verify_coding_score
        result=verify_coding_score.control(args.run_dir) if args.command=='verify-coding-score-control' else verify_coding_score.verify(args.run_dir,args.capture_run,args.qualification)
        print(json.dumps(result,sort_keys=True))
    elif args.command == 'node-pilot-recover':
        import pilot_recover
        print(json.dumps(pilot_recover.recover(args.attempt),sort_keys=True))
    elif args.command == 'node-router-probe':
        import node_router_probe
        node_router_probe.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-pilot-manifest':
        import pilot_manifest
        print(json.dumps(pilot_manifest.run(json.loads(Path(args.config).read_text())),sort_keys=True))
    elif args.command == 'node-qualify-router':
        import node_qualify_router
        node_qualify_router.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-bootstrap':
        import node_bootstrap
        node_bootstrap.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-prompt':
        import node_prompt
        node_prompt.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-input-panel':
        import node_input_panel
        node_input_panel.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-qualify-image':
        import node_qualify_image
        node_qualify_image.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-smoke':
        import node_smoke
        node_smoke.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'reconcile':
        import reconcile
        reconcile.reconcile(args.run_dir)
        print('owned containers reconciled')
    elif args.command == 'node-pilot':
        import node_pilot
        node_pilot.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-pilot-block':
        import node_pilot_block
        node_pilot_block.run(json.loads(Path(args.config).read_text()))
    elif args.command == 'node-pilot-block-recover':
        import pilot_block_recover
        print(json.dumps(pilot_block_recover.recover(args.run_dir),sort_keys=True))
    elif args.command == 'stage-checkpoint':
        import stage_checkpoint
        stage_checkpoint.run(json.loads(Path(args.config).read_text()))
    else:
        m = json.loads(Path(args.manifest).read_text())
        validate_phase(m, args.phase, cell_ids=args.cell_ids)
        if m['plan_sha256'] != digest(args.plan): raise ValueError('plan content hash mismatch')
        print(args.phase + ' inputs present; this does not qualify their evidence or start a run')

if __name__ == '__main__':
    main()
