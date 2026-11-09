"""Exercise the published finalist package through its documented verifier."""
from pathlib import Path
import contextlib
import csv
import io
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))
import runtime
import verify_result_package


class FinalistPublicationTests(unittest.TestCase):
    def test_supplied_result_code_is_not_executed(self):
        source = Path(__file__).resolve().parents[1] / 'results/finalist-validation'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'finalist-validation'
            shutil.copytree(source, root)
            (root / 'reproduce/replay_tables.py').write_text(
                "raise RuntimeError('untrusted result module executed')\n")
            runtime.seal_evidence(root)
            result = verify_result_package.verify(root)
            self.assertEqual(result['status'], 'passed')

    def test_resealed_plot_values_and_population_errors_are_rejected(self):
        source = Path(__file__).resolve().parents[1] / 'results/finalist-validation'
        changes = [('quality.csv', 'difference_pp', '90'),
                   ('natural-regimes.csv', 'interval_high', '1000'),
                   ('load-windows.csv', 'completed_per_second', '500'),
                   ('quality.csv', 'remove_fast', ''),
                   ('quality-outcomes.csv', 'duplicate_seed', ''),
                   ('load-windows.csv', 'duplicate_row', '')]
        for name, field, value in changes:
            with self.subTest(table=name, field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'finalist-validation'
                shutil.copytree(source, root)
                table = root / 'data' / name
                with table.open() as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames
                    rows = list(reader)
                if field == 'remove_fast':
                    rows = [r for r in rows if r['profile'] != 'fast']
                elif field == 'duplicate_seed':
                    rows[1] = dict(rows[0])
                elif field == 'duplicate_row':
                    rows.append(dict(rows[0]))
                else:
                    rows[0][field] = value
                with table.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                runtime.seal_evidence(root)
                with self.assertRaises((ValueError, AssertionError)):
                    verify_result_package.verify(root)

    def test_resealed_wrong_task_count_is_rejected(self):
        source = Path(__file__).resolve().parents[1] / 'results/finalist-validation'
        with contextlib.redirect_stdout(io.StringIO()):
            verify_result_package.verify(source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'finalist-validation'
            shutil.copytree(source, root)
            table = root / 'data/load-windows.csv'
            lines = table.read_text().splitlines()
            rows = list(csv.DictReader(lines))
            rows[0]['successful_in_window'] = str(int(rows[0]['successful_in_window']) + 1)
            with table.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            runtime.seal_evidence(root)
            with self.assertRaises((ValueError, AssertionError)):
                verify_result_package.verify(root)


if __name__ == '__main__':
    unittest.main()
