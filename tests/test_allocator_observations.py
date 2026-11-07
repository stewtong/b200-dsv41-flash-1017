"""Allocator semantics and missing-source controls; no CUDA or GPU claim."""
import sys
from pathlib import Path
import unittest
import hashlib
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import allocator_observations as allocator


class AllocatorTests(unittest.TestCase):
    def stats(self):
        return dict(zip(allocator.FIELDS.values(),(10,20,30,40,2,0)))

    def test_reads_one_explicit_device_and_preserves_lifetime_peak_semantics(self):
        calls=[];stats=self.stats()
        def read(device):calls.append(device);return stats
        r=allocator.read(read,3,'native')
        self.assertEqual(calls,[3]);self.assertEqual(r['values']['allocated_bytes'],10)
        self.assertEqual(r['values']['reserved_peak_bytes_since_engine_reset'],40)
        self.assertIn('not cell-local',r['peak_scope'])
        self.assertLessEqual(r['started_ns'],r['completed_ns'])
        self.assertEqual(stats,self.stats())

    def test_missing_boolean_negative_async_and_inconsistent_peak_fail(self):
        for key,value in [('allocated_bytes.all.current',None),('num_ooms',True),
                          ('num_alloc_retries',-1),('allocated_bytes.all.peak',9)]:
            stats=self.stats();stats[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                allocator.read(lambda device:stats,0,'native')
        with self.assertRaisesRegex(ValueError,'backend'):
            allocator.read(lambda device:self.stats(),0,'cudaMallocAsync')

    def test_pinned_imported_sources_are_checked_before_reading_allocator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'cuda').mkdir()
            files={'cuda/memory.py':'# public allocator fixture\n','version.py':'# public version fixture\n'}
            pins={}
            for name,text in files.items():
                p=root/name;p.write_text(text);pins[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            calls=[]
            memory=SimpleNamespace(memory_stats=lambda device:self.stats(),
                get_allocator_backend=lambda:calls.append('backend') or 'native')
            torch=SimpleNamespace(__file__=str(root/'__init__.py'),cuda=SimpleNamespace(memory=memory))
            with patch.object(allocator,'SOURCE_PINS',pins),patch.object(allocator.inspect,'getsourcefile',return_value=str(root/'cuda/memory.py')):
                reader=allocator.pinned_reader(torch,1)
                self.assertEqual(reader()['device_index'],1);self.assertEqual(calls,['backend'])
                (root/'version.py').write_text('# different bytes\n')
                with self.assertRaisesRegex(ValueError,'source differs'):allocator.pinned_reader(torch,1)
                self.assertEqual(calls,['backend'])


if __name__=='__main__':unittest.main()
