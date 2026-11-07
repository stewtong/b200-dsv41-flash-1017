"""The owner-approved fixed 336-cell comparison, separate from legacy planning."""
import itertools
import suite_controls

SCHEMA = 'dsv41-b200-campaign/v3'
SCOPE_ID = 'context-concurrency-336-v1'
LAYOUTS = ('2xtp4', '4xdep2', '2xdep4', 'dep8')
INPUTS = (8000, 16000, 32000, 64000, 128000, 256000, 400000, 512000, 1000000)
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
WARM_INPUTS = (128000, 256000, 512000, 1000000)
WARM_CONCURRENCIES = (1, 8, 32)


def expand(matrix):
    expected_output = {str(n): 512 if n == 1000000 else 2048 for n in INPUTS}
    warm = matrix.get('warm_prefix_panel', {})
    if (matrix.get('schema') != SCHEMA or matrix.get('scope_id') != SCOPE_ID or
            matrix.get('topologies') != list(LAYOUTS) or matrix.get('input_tokens') != list(INPUTS) or
            matrix.get('node_concurrency') != list(CONCURRENCIES) or matrix.get('cache_states') != ['cold'] or
            type(matrix.get('repetitions')) is not int or matrix['repetitions'] != 3 or
            matrix.get('output_tokens_by_input') != expected_output or
            warm.get('input_tokens') != list(WARM_INPUTS) or
            warm.get('node_concurrency') != list(WARM_CONCURRENCIES) or
            warm.get('cache_states') != ['warm_prefix'] or
            type(warm.get('repetitions')) is not int or warm['repetitions'] != 3 or
            warm.get('output_tokens_by_input') != {str(n): expected_output[str(n)] for n in WARM_INPUTS}):
        raise ValueError('schema-v3 axes or context-dependent output budgets differ from approved scope')
    sampling = matrix.get('core_sampling', {})
    if (sampling.get('temperature') != 1.0 or sampling.get('top_p') != 1.0 or
            sampling.get('thinking') is not False or sampling.get('min_tokens_equals_max_tokens') is not True or
            sampling.get('ignore_eos') is not True):
        raise ValueError('schema-v3 performance sampling differs')
    cells = []
    for suite, inputs, concurrencies, cache in (
            ('core', INPUTS, CONCURRENCIES, 'cold'),
            ('warm_prefix_panel', WARM_INPUTS, WARM_CONCURRENCIES, 'warm_prefix')):
        for layout, n, c in itertools.product(LAYOUTS, inputs, concurrencies):
            output = expected_output[str(n)]
            cell = {'id': f'{suite}-{layout}-i{n}-c{c}-{cache}-o{output}', 'suite': suite,
                    'layout': layout, 'input_tokens': n, 'concurrency': c, 'cache': cache,
                    'output_tokens': output, 'repetitions': 3, 'status': 'planned'}
            cell['request_contract'] = suite_controls.contract(cell)
            cells.append(cell)
    expected = {'core': (288, 864, 27540), 'warm_prefix_panel': (48, 144, 1968)}
    for suite, wanted in expected.items():
        rows = [c for c in cells if c['suite'] == suite]
        actual = (len(rows), sum(c['repetitions'] for c in rows),
                  sum(c['repetitions'] * c['concurrency'] for c in rows))
        if actual != wanted:
            raise ValueError('schema-v3 expanded population differs')
    declarations = (matrix.get('core_expected_cells'), matrix.get('core_expected_repetitions'),
                    matrix.get('core_expected_measured_requests'), warm.get('expected_cells'),
                    warm.get('expected_repetitions'), warm.get('expected_measured_requests'),
                    matrix.get('total_fixed_output_cells'), matrix.get('total_measured_repetitions'),
                    matrix.get('total_measured_requests'))
    if declarations != (288, 864, 27540, 48, 144, 1968, 336, 1008, 29508):
        raise ValueError('schema-v3 declared counts differ')
    if len({c['id'] for c in cells}) != 336:
        raise ValueError('schema-v3 cell identity collision')
    return cells


def migration(old_cells, new_cells):
    """Preserve every old row; matching shape never transfers measurement credit."""
    def shape(cell):
        return tuple(cell.get(k) for k in ('layout', 'input_tokens', 'concurrency', 'cache', 'output_tokens'))
    targets = {shape(c): c['id'] for c in new_cells}
    if len(targets) != 336:
        raise ValueError('migration requires the complete new comparison')
    fixed_suites = {'core', 'output_sensitivity', 'historical_bridge_extra'}
    records = []
    for old in old_cells:
        target = targets.get(shape(old))
        if old['suite'] in fixed_suites and target is not None:
            disposition = 'required'
            reason = 'same fixed-output shape remains required; fresh scored repetitions and preserved history'
        elif old['suite'] == 'core':
            disposition = 'superseded_owner_scope'
            reason = 'old context/cache/output Cartesian row replaced by explicit cold and warm ladders'
        else:
            disposition = 'deferred_owner_scope'
            reason = 'original supplemental, qualification, tuning or anchor requirement is outside current completion'
        records.append({'old_cell_id': old['id'], 'old_suite': old['suite'],
                        'original_status': old.get('status'), 'disposition': disposition,
                        'related_required_cell_id': target, 'reason': reason,
                        'measurement_credit_transferred': False, 'attempt_history_action': 'retain_unchanged'})
    return {'scope_id': SCOPE_ID, 'status': 'planning_migration_history_binding_pending',
            'old_rows': len(old_cells), 'new_required_cells': 336,
            'dispositions': {name: sum(r['disposition'] == name for r in records)
                             for name in ('required', 'superseded_owner_scope', 'deferred_owner_scope')},
            'historical_denominators': {'core': 704, 'fixed_output': 1036, 'quality': 704},
            'rows': records, 'measurement_credit_transferred': 0,
            'scope': 'planning relationships only; bind retained live attempt identities separately before dispatch; no retry reset or score credit'}
