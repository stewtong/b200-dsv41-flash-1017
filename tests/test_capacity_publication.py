"""Reject resealed corruption of the capacity-at-latency-target tables and the added measured-arrivals table."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import runtime
import verify_result_package

RESULTS = Path(__file__).resolve().parents[1] / 'results'
CAPACITY = RESULTS / 'capacity-at-latency-target'
SUSTAINED = RESULTS / 'sustained-speculation-layouts'


def change_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def change_table(path, change):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    change(rows)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def first(rows, **match):
    return next(r for r in rows if all(r[k] == v for k, v in match.items()))


def swap_node_row(rows):
    # Replace a failing window-A verdict with a fabricated row for a K5-vs-NoSpec window.
    index = rows.index(first(rows, window='A', workload='W1', rate_label_per_stream='0.289'))
    rows[index] = dict(rows[index], window='A+', rate_label_per_stream='0.15', **{'pass': 'False'})


def count_contradiction(rows):
    # A p95 moved well under the target while the published counts show far fewer than 95% of turns within 30 s.
    row = first(rows, window='A', workload='W2', rate_label_per_stream='0.16', repeat='0', unit='replica 0')
    row.update(completion_p95_s='25.0')


def conversations_changed(rows):
    for row in rows:
        if row['window'] == 'A' and row['workload'] == 'W1' and row['rate_label_per_stream'] == '0.15':
            row.update(measured_conversations='70', measured_arrivals_per_second=str(round(70 / 360, 4)))


def nospec_pass(rows):
    for row in rows:
        if row['window'] == 'A+' and row['workload'] == 'W1' and row['rate_label_per_stream'] == '0.15' and row['speculation'] == 'NoSpec':
            row.update(completion_p95_s='25.0', drain_ok='True', completed_turns=row['scheduled_measured_turns'],
                       turns_within_30s=row['scheduled_measured_turns'], unit_pass='True',
                       offered_output_tokens_per_second_per_gpu=str(round(int(row['output_tokens_completed_turns']) / 360 / 4, 1)))


class CapacityPublicationTests(unittest.TestCase):
    def assert_rejected(self, source, edit, message):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / source.name
            shutil.copytree(source, root)
            edit(root)
            runtime.seal_evidence(root)
            with self.assertRaisesRegex(ValueError, message):
                verify_result_package.verify(root)

    def test_published_package_passes(self):
        result = verify_result_package.verify(CAPACITY)
        self.assertEqual((result['windows'], result['unit_rows'], result['node_verdicts']), (64, 128, 46))

    def test_table_corruption_is_rejected_by_the_intended_check(self):
        cases = {
            'arrivals': ('tables/arms.csv', lambda rows: rows[0].update(measured_arrivals_per_second='0.9'), 'measured arrivals differ'),
            'scheduled_turns': ('tables/arms.csv', conversations_changed, 'scheduled turns differ'),
            'unit_quantile': ('tables/arms.csv', lambda rows: first(rows, window='A', rate_label_per_stream='0.289', unit='replica 0').update(unit_pass='True'),
                              'unit verdict differs'),
            'unit_drain': ('tables/arms.csv', lambda rows: first(rows, window='B', workload='W1', rate_label_per_stream='0.15', repeat='2',
                                                                 unit='stream 0 (shared engine)').update(unit_pass='True'), 'unit verdict differs'),
            'count_contradiction': ('tables/arms.csv', count_contradiction, 'contradicts turns within 30 s'),
            'nospec_pass': ('tables/arms.csv', nospec_pass, 'NoSpec passed at the lowest tested load'),
            'acceptance_identity': ('tables/arms.csv', lambda rows: first(rows, speculation='K5').update(acceptance_length='3.9'),
                                    'acceptance length differs'),
            'nospec_acceptance': ('tables/arms.csv', lambda rows: first(rows, speculation='NoSpec').update(draft_token_acceptance_rate='0.4'),
                                  'NoSpec reports acceptance'),
            'usability': ('tables/arms.csv', lambda rows: rows[0].update(cell_usable='False'), 'unusable window'),
            'output_rate': ('tables/arms.csv', lambda rows: first(rows, window='A', workload='W1', rate_label_per_stream='0.289', unit='replica 0').update(
                offered_output_tokens_per_second_per_gpu='900.0'), 'offered output rate differs'),
            'output_rate_incomplete': ('tables/arms.csv', lambda rows: first(rows, window='A', workload='W1', rate_label_per_stream='0.45', unit='replica 0').update(
                offered_output_tokens_per_second_per_gpu='800.0'), 'offered output rate differs'),
            'output_rate_dep8': ('tables/arms.csv', lambda rows: first(rows, window='B', workload='W1', rate_label_per_stream='0.15').update(
                offered_output_tokens_per_second_per_gpu='540.0'), 'offered output rate differs'),
            'output_tokens': ('tables/arms.csv', lambda rows: first(rows, window='A', workload='W1', rate_label_per_stream='0.289', unit='replica 0').update(
                output_tokens_completed_turns='0'), 'output tokens differ from completed turns'),
            'output_tokens_over_cap': ('tables/arms.csv', lambda rows: first(rows, window='B', workload='W1', rate_label_per_stream='0.15').update(
                output_tokens_completed_turns='99999999'), 'output tokens differ from completed turns'),
            'output_rate_blank': ('tables/arms.csv', lambda rows: first(rows, window='A', workload='W1', rate_label_per_stream='0.289', unit='replica 0').update(
                offered_output_tokens_per_second_per_gpu=''), 'offered output rate differs'),
            'preemptions': ('tables/arms.csv', lambda rows: rows[0].update(preemptions='7'), 'preemptions differ'),
            'replica_verdict': ('tables/node-cells.csv', lambda rows: first(rows, window='A', rate_label_per_stream='0.2935').update(**{'pass': 'True'}),
                                '2xtp4 node verdict differs from replicas'),
            'node_swap': ('tables/node-cells.csv', swap_node_row, 'node coverage differs'),
            'node_relabel': ('tables/node-cells.csv', lambda rows: first(rows, window='A', rate_label_per_stream='0.2935').update(configuration='2xtp4 K5 '),
                             'node configuration differs'),
            'node_input_rate': ('tables/node-cells.csv', lambda rows: rows[0].update(input_tokens_per_second_node='185000.0'),
                                'node input rate differs'),
            'dep8_verdict': ('tables/node-cells.csv', lambda rows: first(rows, window='B', workload='W1', rate_label_per_stream='0.15', repeat='2').update(**{'pass': 'False'}),
                             'dep8 pooled verdict differs'),
            'paired_interval': ('tables/k5-nospec-paired.csv', lambda rows: rows[0].update(completion_time_ratio_geomean='0.9'),
                                'paired interval does not contain its estimate'),
            'paired_counts': ('tables/k5-nospec-paired.csv', lambda rows: rows[0].update(conversations='99'), 'paired counts differ'),
        }
        for kind, (name, change, message) in cases.items():
            with self.subTest(kind=kind):
                self.assert_rejected(CAPACITY, lambda root: change_table(root / name, change), message)

    def test_missing_output_column_is_rejected(self):
        def drop(root):
            path = root / 'tables/arms.csv'
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream); fields = [f for f in reader.fieldnames if f != 'offered_output_tokens_per_second_per_gpu']; rows = list(reader)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
        self.assert_rejected(CAPACITY, drop, 'arms columns differ')

    def test_report_output_figures_must_match_the_table(self):
        def edit(root):
            path = root / 'report.md'
            text = path.read_text()
            self.assertIn('831 and 846 offered output tokens/s per GPU', text)
            path.write_text(text.replace('831 and 846 offered output tokens/s per GPU', '831 and 946 offered output tokens/s per GPU'))
        self.assert_rejected(CAPACITY, edit, 'report output figures differ from the table')

    def test_summary_and_configuration_changes_are_rejected(self):
        cases = {
            'stated_limit': ('summary.json', lambda d: d['highest_all_passing_measured_arrivals_per_node'].update(W1=0.5778),
                             'stated limit differs from verdicts'),
            'nospec_limit': ('summary.json', lambda d: d.update(nospec_limit=0.2), 'NoSpec limit stated'),
            'kv_capacity': ('configuration.json', lambda d: [e.update(gpu_kv_cache_tokens_per_engine_core=1) for name, c in d['configurations'].items()
                                                             if name.startswith('2xtp4') for e in c['engines']],
                            'KV capacity differ'),
        }
        for kind, (name, change, message) in cases.items():
            with self.subTest(kind=kind):
                self.assert_rejected(CAPACITY, lambda root: change_json(root / name, change), message)

    def test_measured_arrivals_must_match_measured_turns(self):
        cases = {
            'conversations': (lambda rows: rows[0].update(measured_conversations='1'), 'measured arrivals differ from measured turns'),
            'arm': (lambda rows: rows[0].update(arm='B' if rows[0]['arm'] != 'B' else 'A'), 'measured arrivals differ from measured turns'),
            'duplicate': (lambda rows: rows.insert(0, dict(rows[0], measured_conversations='1')), 'measured-arrival identities differ'),
        }
        for kind, (change, message) in cases.items():
            with self.subTest(kind=kind):
                self.assert_rejected(SUSTAINED, lambda root: change_table(root / 'tables/measured-arrivals.csv', change), message)


if __name__ == '__main__':
    unittest.main()
