"""Reject resealed provenance, coverage, metric and chart corruption."""
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

SOURCE = Path(__file__).resolve().parents[1] / 'results/sustained-speculation-layouts'


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


class SustainedPublicationTests(unittest.TestCase):
    def test_resealed_corruption_is_rejected(self):
        for kind in ['old_throughput', 'coverage_identity', 'duplicate_variant', 'source_hash', 'chart_value']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / SOURCE.name
                shutil.copytree(SOURCE, root)
                if kind == 'old_throughput':
                    change_json(root / 'summary.json', lambda d: d['cells'][0].update(
                        window_output_tokens_per_second_per_gpu=999999))
                    change_table(root / 'tables/cells.csv', lambda rows: rows[0].update(
                        window_output_tokens_per_second_per_gpu='999999'))
                elif kind == 'coverage_identity':
                    change_json(root / 'coverage.json', lambda d: d['entries'][0].update(arm='Z', disposition='unknown'))
                elif kind == 'duplicate_variant':
                    change_json(root / 'source-retention.json', lambda d: d['producer_variants'].__setitem__(
                        0, dict(d['producer_variants'][1])))
                elif kind == 'source_hash':
                    change_json(root / 'summary.json', lambda d: d['cells'][0]['sources'].update({'replay.py': '0' * 64}))
                    def change(rows):
                        sources = json.loads(rows[0]['source_sha256']); sources['replay.py'] = '0' * 64
                        rows[0]['source_sha256'] = json.dumps(sources)
                    change_table(root / 'tables/capture-provenance.csv', change)
                else:
                    chart = root / 'figures/acceptance.svg'
                    chart.write_text(chart.read_text().replace('data-value="2.8282005078220984"', 'data-value="9.99"'))
                runtime.seal_evidence(root)
                with self.assertRaises(ValueError):
                    verify_result_package.verify(root)


if __name__ == '__main__':
    unittest.main()
