import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import verify_telemetry as v
import test_verify_telemetry as fixtures


class StreamingTelemetryTests(unittest.TestCase):
    def test_reader_values_order_and_slices_match_original_splitlines(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{'index':i,'payload':'record '+str(i)} for i in range(9)]
            separators=['\n','\r\n','\r','\x85','\u2028','\u2029','\v','\f','\x1c']
            for part,subset in enumerate((rows[:4],rows[4:])):
                text=''.join(json.dumps(row)+separators[row['index']] for row in subset)
                (root/f'telemetry-{part:04d}.jsonl').write_text(text)
            original=[json.loads(line) for file in sorted(root.glob('telemetry-*.jsonl'))
                      for line in file.read_text().splitlines()]
            indexed=v.IndexedLines(root,'telemetry')
            self.assertEqual(v.lines(root,'telemetry'),original)
            self.assertEqual(list(v.iter_lines(root,'telemetry')),original)
            self.assertEqual(list(indexed),original)
            self.assertEqual(len(indexed),len(original))
            self.assertEqual(indexed[-1],original[-1])
            self.assertEqual(indexed[1:7:2],original[1:7:2])
            with self.assertRaises(IndexError):indexed[len(original)]

    def test_unescaped_separator_inside_json_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'telemetry-0000.jsonl').write_text('{"payload":"left\u2028right"}\n')
            for reader in (v.lines,v.iter_lines,v.IndexedLines):
                with self.subTest(reader=reader.__name__),self.assertRaises(ValueError):
                    list(reader(root,'telemetry'))

    def test_index_retains_its_record_population_when_writer_appends(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'telemetry-0000.jsonl'
            file.write_text('{"index":0}\n{"index":1}\n')
            indexed=v.IndexedLines(root,'telemetry')
            with file.open('a') as stream:stream.write('{"index":2}\n')
            self.assertEqual(list(indexed),[{'index':0},{'index':1}])
            self.assertEqual(indexed[1],{'index':1})
            self.assertEqual(len(v.lines(root,'telemetry')),3)

    def test_truncated_indexed_record_never_returns_a_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'telemetry-0000.jsonl'
            file.write_text('{"index":0}\n{"index":1}\n')
            indexed=v.IndexedLines(root,'telemetry');file.write_text('{"index":0}\n')
            with self.assertRaisesRegex(ValueError,'record boundaries changed'):indexed[1]

    def test_window_does_not_materialize_the_full_raw_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ranks,_,_=fixtures.TelemetryWindowTests().fixture(root)
            original=v.lines
            def small_records(path,prefix):
                self.assertNotEqual(prefix,'telemetry','full raw stream must use bounded indexed access')
                return original(path,prefix)
            with patch.object(v,'lines',side_effect=small_records):
                result=v.verify_window(root,11_200_000_000,13_600_000_000,ranks)
            self.assertEqual((result['samples_in_window'],result['checked_samples']),(2,4))

    def test_malformed_unselected_record_and_unpaired_timings_never_pass(self):
        for failure in ('malformed','unpaired','order'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                fixture=fixtures.TelemetryWindowTests();ranks,rows,timings=fixture.fixture(root)
                if failure=='malformed':
                    file=root/'telemetry-0000.jsonl';text=file.read_text().splitlines()
                    text[0]='{unselected malformed record';file.write_text('\n'.join(text)+'\n')
                elif failure=='unpaired':
                    fixture.save(root,rows,timings[1:])
                else:
                    rows[0]['monotonic_ns']=12_000_000_000;fixture.save(root,rows,timings)
                with self.assertRaises(ValueError):
                    v.verify_window(root,11_200_000_000,13_600_000_000,ranks)


if __name__=='__main__':unittest.main()
