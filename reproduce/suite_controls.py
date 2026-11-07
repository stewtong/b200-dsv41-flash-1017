"""Explicit missing control rows and suite contracts from campaign Sections 4-9.

Conditional rows preserve opportunity and dependencies, never claim execution.
"""
import hashlib
import itertools
import json

TOPS=('2xtp4','4xdep2','2xdep4','dep8')
ANCHOR_INPUTS=(8000,128000,512000,1000000)
ANCHOR_CONCURRENCIES=(1,8,32)
EXTRA_LEVERS={
    'host_kv':('4xdep2','2xdep4'),
    'attention_backend':('4xdep2','2xdep4','dep8'),
    'prefill_schedule_interval':TOPS,
    'graph_capture_sizes':TOPS,
    'thp_engram':TOPS,
    'specialized_decode':('2xtp4',),
    'kv_dtype':TOPS,
}
BASE_LEVERS=('dspark_static','dspark_adaptive','engram_resident','prefill_batch','bounded_replay','graph_admission')


def controls():
    rows=[]
    def add(suite,layout,parameters,repetitions=1,condition=None,dependencies=None):
        serialized=json.dumps(parameters,sort_keys=True,separators=(',',':'))
        suffix=hashlib.sha256(serialized.encode()).hexdigest()[:16]
        rows.append({'id':f'{suite}-{layout}-{suffix}','suite':suite,'layout':layout,
                     'parameters':parameters,'repetitions':repetitions,'status':'planned',
                     'condition':condition,'dependent_inputs':dependencies or {}})
    for layout in TOPS:
        for thinking in ('on','off'):
            add('gsm8k_calibration',layout,{'thinking':thinking,'tasks':50,'effort':50,'concurrency':1,
                'initial_output_cap':4096 if thinking=='on' else 2048},
                dependencies={'calibration_source':'gsm8k-source.json pinned training subset excluded from test',
                              'final_cap':'raise uniformly before freeze when truncation exceeds1%'})
        for n,kind in itertools.product((1000,8000,16000,32000,64000,128000,256000,400000,512000,768000,1000000),
                                       ('identifier','multi_key','arithmetic')):
            add('quality_calibration',layout,{'input':n,'task_kind':kind,'concurrency':1,'initial_output_cap':512},
                dependencies={'final_cap':'c1 task calibration and frozen quality-panel version'})
        for n,c in itertools.product((8000,128000),ANCHOR_CONCURRENCIES):
            add('natural_stop_acceptance',layout,{'input':n,'concurrency':c,'stopping':'natural'},3,
                dependencies={'matched_forced_control':'same frozen profile, sampling, seed and input; acceptance difference reported above10%'})
        for c in (8,16,32,64):
            add('mixed_load_short_control',layout,{'concurrency':c,'input':8000,'ramp_seconds':120,'measured_seconds':600},3,
                dependencies={'session_ids':'same short sessions and scheduled times as paired mixed-load run'})
        for probe in ('reject_sequence_limit','accept_exact_output_room'):
            add('context_boundary',layout,{'probe':probe,'sequence_limit':1048576,'output':2048})
        for stage in ('prefill','decode'):
            add('qualification_cancel_recovery',layout,{'input':8000,'concurrency':1,'stage':stage},
                dependencies={'acceptance':'exact request absence or exact owned engine recreation, health and semantic response'})
        add('qualification_interrupted_resume',layout,{'input':8000,'concurrency':1},
            dependencies={'acceptance':'retain orphan raw attempt, reconcile live identities, preserve completed attempt'})
        for control in ('sampler_overhead','client_ceiling'):
            for arm in ('control_before','challenger','control_after'):
                add(control,layout,{'arm':arm,'input':8000,'concurrency':8},3,
                    dependencies={'acceptance':'matched load timings, CPU and1Hz per-rank coverage; no capacity claim before client ceiling established'})
        for bracket in ('midpoint','confirmation'):
            add('arrival_bracket',layout,{'kind':bracket,'batch':1,'ramp_seconds':60,'measured_seconds':300},
                condition='first measured overload boundary exists',
                dependencies={'rate':'midpoint between last lower rung and first overload; confirmation at recorded bracket decision'})
        add('arrival_doubling_extension',layout,{'ramp_seconds':60,'measured_seconds':300,'batch_ceiling':3},3,
            condition='no overload in initial ladder and remaining reservation/safety budget permits',
            dependencies={'children':'append exact doubled-rate IDs and seeded workload before dispatch, bounded by resource projection and operational deadline'})
        for comparison in ('slo_80pct','near_capacity'):
            add('soak_canary',layout,{'parent_load':comparison,'cadence_seconds':60,'thinking':'on','effort':50},
                dependencies={'parents':'one semantic canary per minute of each measured soak, excluded from rate counts'})
        for lever in (*BASE_LEVERS,*[k for k,v in EXTRA_LEVERS.items() if layout in v]):
            for n,c,arm in itertools.product(ANCHOR_INPUTS,ANCHOR_CONCURRENCIES,('control','candidate')):
                add('tuning_tail_validation',layout,{'lever':lever,'input':n,'concurrency':c,'arm':arm,'requests':100},
                    condition='candidate adoption depends on this deciding tail anchor',
                    dependencies={'decision':'preregister named deciding anchors and protected tails before lever'})
            for thinking,arm in itertools.product(('on','off'),('control','candidate')):
                add('tuning_quality',layout,{'lever':lever,'thinking':thinking,'effort':50,'arm':arm,'tasks':1319},
                    condition='candidate requires optimization quality acceptance',
                    dependencies={'acceptance':'paired95% interval lower bound above minus1percentage point; identifier/tool/stream regressions block adoption'})
        for lever,layouts in EXTRA_LEVERS.items():
            if layout not in layouts:continue
            for arm,n,c in itertools.product(('control_before','challenger','control_after'),ANCHOR_INPUTS,ANCHOR_CONCURRENCIES):
                add('tuning_extra',layout,{'lever':lever,'arm':arm,'input':n,'concurrency':c},3,
                    condition='one supported challenger; installed compatibility and fit established',
                    dependencies={'settings':'exact profile delta, primary metric, deciding anchors, protected tail and resource budget preregistered'})
    for layout,tp,dp in (('2xtp4',4,1),('4xdep2',1,2),('2xdep4',1,4),('dep8',1,8),('tp8',8,1),('4xtp2',2,1)):
        add('router_session_map',layout,{'logical_units':8//tp,'controlled_node_cap':128},
            dependencies={'acceptance':'native synthetic backend header receipts on actual serving ports, repeated selected sessions and pinned router; GPU evidence separate'})
        for unit in range(8//tp):
            add('replica_calibration',layout,{'unit':unit,'input':8000,'concurrency':1,'output':128},3,
                dependencies={'placement':'frozen native-router session map plus live physical-rank/cache-counter evidence'})
    add('long_soak_canary','selected',{'minutes':840,'cadence_seconds':60,'thinking':'on','effort':50},
        dependencies={'profile':'production profile selected after comparable coverage','oracle':'seeded independently computable answer'})
    for row in rows:
        row['evidence_class']='control planning; unexecuted'
    return rows


def contract(cell):
    suite=cell['suite']
    base={'api':'chat_completions','stream':True,'final_usage':True,'temperature':1.0,'top_p':1.0,
          'thinking':False,'reasoning_effort':50,'automatic_retry':False,
          'prefix_caching':True,'speculation_rejection':'real block rejection when enabled',
          'seed_policy':'matched suite/shape/repetition/session across layouts'}
    if suite in ('core','warm_prefix_panel','output_sensitivity','historical_bridge_extra','anchor','anchor_greedy'):
        base.update(stopping='fixed',output_tokens=cell['output_tokens'],ignore_eos=True,
                    min_tokens_equals_max_tokens=True,cache=cell['cache'])
        if suite=='anchor_greedy':base['temperature']=0.0
    elif suite=='quality':
        base.update(stopping='natural',initial_output_cap=512,cache=cell['cache'],
                    oracle='identifier, multi-key at5/50/95% depth, arithmetic; independent expected values',
                    denominator='every scheduled task including failure, truncation and reasoning-only output',
                    unresolved={'output_cap':'quality c1 calibration; frozen panel version required'})
    elif suite=='growing':
        base.update(stopping='fixed',output_tokens=512,turns=4,append_user_tokens=8000,
                    cache='first turn cold, subsequent turns warm',affinity=True)
    elif suite=='cache_control':
        base.update(stopping='fixed',output_tokens=256,turns=4,route_and_cache_evidence='each turn',
                    record=['prefix_mismatches','evictions'])
    elif suite in ('mixed_load','arrival_ladder','arrival_bracket','arrival_doubling_extension','burstiness','placement','soak','long_soak'):
        base.update(stopping='fixed',input_mix={'8000':.5,'128000':.25,'512000':.15,'1000000':.1},
                    output_mix={'512':.6,'2048':.3,'8192':.1},cache='new cold session per request',
                    queue_sampling_hz=1,queue_identity='union of client/router/engine request IDs')
        if suite.startswith('arrival_'):base.update(ramp_seconds=60,measured_seconds=300,supplemental_batch_ceiling=3,
                    stop={'oldest_queue_seconds':120,'errors_fraction_60_seconds':.05})
        elif suite in ('mixed_load','burstiness'):base.update(ramp_seconds=120,measured_seconds=600)
        if suite=='burstiness':base.update(gamma_shape=.5,own_capacity_fraction=.7)
        if suite=='soak':base.update(measured_seconds=3600 if cell['parameters']['load']=='slo_80pct' else 900,
                                      slo_load_fraction=.8 if cell['parameters']['load']=='slo_80pct' else None)
        if suite=='long_soak':base['measured_seconds']=50400
    elif suite=='continuous':base.update(stopping='fixed',output_tokens=cell['parameters']['output'],
                    ramp_seconds=120,measured_seconds=600,cache='new cold sessions',replenish=True)
    elif suite=='gsm8k':base.update(stopping='natural',concurrency=8,reasoning_effort=50,
                    thinking=cell['parameters']['thinking']=='on',tasks=1319,scorer='gsm8k-final-number-v1',
                    unresolved={'output_cap':'uniform cap from excluded50-item calibration'})
    elif suite=='coding_panel':base.update(stopping='natural',tasks=100,minimum_turns=4,
                    tools='deterministic inert tools; exact nested arguments and multiple calls',
                    oracle='executable tests plus tool protocol; mid-conversation system and tool-result history')
    fields=cell.get('parameters',{})
    if suite in ('pilot_smoke','pilot_fit','pilot_warm','pilot_concurrency'):
        base.update(stopping='natural' if suite=='pilot_smoke' else 'fixed',output_tokens=cell['output_tokens'],
                    cache=cell['cache'],speculation='disabled')
    elif suite in ('tuning_trial','tuning_extra','tuning_tail_validation','admission','context_boundary','cancel','replica_failure','sampler_overhead','client_ceiling'):
        base.update(stopping='fixed',output_tokens=2048,min_tokens_equals_max_tokens=True,ignore_eos=True)
    elif suite=='natural_stop_acceptance':base.update(stopping='natural',initial_output_cap=2048)
    elif suite=='api_protocol':base.update(api=fields['api'],stream=fields['mode']=='stream',thinking=fields['case']=='thinking',
                                         support='actual installed route and case must be tested; unsupported is explicit')
    elif suite in ('gsm8k_calibration','tuning_quality'):
        base.update(stopping='natural',concurrency=1 if suite=='gsm8k_calibration' else 8,
                    thinking=fields['thinking']=='on',reasoning_effort=50,
                    initial_output_cap=4096 if fields['thinking']=='on' else 2048)
    elif suite=='quality_calibration':base.update(stopping='natural',initial_output_cap=512,concurrency=1)
    elif suite=='replica_calibration':base.update(stopping='natural',output_tokens=128,cache='cold')
    elif suite in ('soak_canary','long_soak_canary'):base.update(stopping='natural',thinking=True,reasoning_effort=50,initial_output_cap=512)
    elif suite=='reasoning_effort':base.update(stopping='natural',thinking=fields['thinking']=='on',reasoning_effort=fields['effort'],
                                            unresolved={'output_cap':'uniform calibration for shared reasoning-effort panel'})
    return base
