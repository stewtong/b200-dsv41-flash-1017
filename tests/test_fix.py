import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('check_fix', Path(__file__).resolve().parents[1]/'reproduce/check_fix.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

class FixTests(unittest.TestCase):
    def test_missing_invariant_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for file, names in c.REQUIRED.items():
                p = root/file; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('\n'.join('class '+name+':\n _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH\n def get_varlen_cudagraph_max_query_len(self):\n  return max_decode_query_len(None)\n' for name in names))
            self.assertEqual(len(c.check(root)), 5)
            p = root/'compressor.py'; p.write_text('class CompressorMetadataBuilder: pass\n')
            with self.assertRaises(ValueError): c.check(root)
