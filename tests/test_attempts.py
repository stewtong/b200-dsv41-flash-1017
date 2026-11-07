from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import attempts
import runtime
import hashlib
import json

class AttemptTests(unittest.TestCase):
    def test_explicit_mixed_resume_preserves_each_cell_cap_and_original_failure(self):
        from unittest.mock import patch
        import os
        with tempfile.TemporaryDirectory() as d:
            owner={'boot_id':'fixture','pid':os.getpid(),'start_ticks':1}
            prior=attempts.begin(d,'one',method_sha256='a'*64);(prior/'raw.json').write_text('{}');attempts.finish(prior,'invalid',['raw.json'])
            original=(prior/'attempt.json').read_bytes();specs=[{'cell_id':c,'method_sha256':'b'*64} for c in ('one','two')]
            with patch.object(runtime,'process_identity',return_value=owner):
                for kwargs in ({}, {'retry':True,'retry_reason':'changed exact prefix instruction'},
                    {'retry_cells':{'two':'changed exact prefix instruction'}}):
                    with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin_block(d,specs,'bad',owner,**kwargs)
                    self.assertEqual(len(attempts.load_attempts(d)),1)
                with self.assertRaises(runtime.RuntimeErrorClosed):
                    attempts.begin_block(d,[dict(specs[0],method_sha256='a'*64),specs[1]],'bad',owner,retry_cells={'one':'changed exact prefix instruction'})
                paths=attempts.begin_block(d,specs,'mixed',owner,retry_cells={'one':'changed exact prefix instruction'})
                self.assertEqual((prior/'attempt.json').read_bytes(),original)
                reasons={json.loads((p/'attempt.json').read_text())['cell_id']:json.loads((p/'attempt.json').read_text())['retry_reason'] for p in paths}
                self.assertEqual(reasons,{'one':'changed exact prefix instruction','two':None})
                for p in paths:(p/'raw.json').write_text('{}');attempts.finish(p,'failed',['raw.json'])
                with self.assertRaises(runtime.RuntimeErrorClosed):
                    attempts.begin_block(d,[{'cell_id':'one','method_sha256':'c'*64}],'third',owner,retry_cells={'one':'another changed instruction method'})
                with self.assertRaises(ValueError):
                    attempts.begin_block(d,[{'cell_id':'two','method_sha256':'c'*64}],'undeclared',owner,retry_cells={'one':'another changed instruction method'})

    def test_named_startups_allow_only_one_third_changed_method_and_never_a_fourth(self):
        for cell in ('pilot_smoke-4xdep2-i8000-c1-cold-o128','pilot_smoke-4xtp2-i8000-c1-cold-o128'):
            with self.subTest(cell=cell),tempfile.TemporaryDirectory() as d:
                for index,digest in enumerate(('a'*64,'b'*64)):
                    path=attempts.begin(d,cell,retry=bool(index),retry_reason='changed shutdown collector method' if index else None,method_sha256=digest)
                    (path/'raw.json').write_text('{}');attempts.finish(path,'failed',['raw.json'])
                before={p.name:(p/'attempt.json').read_bytes() for p in Path(d).glob('attempt-*')}
                for kwargs in ({'retry':False,'method_sha256':'c'*64},
                               {'retry':True,'retry_reason':'changed shutdown collector method','method_sha256':'a'*64},
                               {'retry':True,'retry_reason':'changed shutdown collector method','method_sha256':'b'*64}):
                    with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin(d,cell,**kwargs)
                third=attempts.begin(d,cell,retry=True,retry_reason='qualified pending-proof shutdown and native idle checks',method_sha256='c'*64)
                self.assertEqual(len(attempts.load_attempts(d)),3)
                for name,data in before.items():self.assertEqual((Path(d)/name/'attempt.json').read_bytes(),data)
                with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin(d,'unrelated',method_sha256='d'*64)
                (third/'raw.json').write_text('{}');attempts.finish(third,'failed',['raw.json'])
                with self.assertRaises(runtime.RuntimeErrorClosed):
                    attempts.begin(d,cell,retry=True,retry_reason='fourth attempt remains forbidden',method_sha256='d'*64)
                self.assertEqual(len(attempts.load_attempts(d)),3)
        with tempfile.TemporaryDirectory() as d:
            cell='pilot_smoke-4xdep2-i8000-c1-cold-o128'
            path=attempts.begin(d,cell,method_sha256='a'*64);(path/'raw.json').write_text('{}');attempts.finish(path,'passed',['raw.json'])
            with self.assertRaises(runtime.RuntimeErrorClosed):
                attempts.begin(d,cell,retry=True,retry_reason='a passed cell cannot use the exception',method_sha256='b'*64)

    def test_block_retry_checks_every_cell_before_reserving_and_never_allows_third(self):
        from unittest.mock import patch
        import os
        with tempfile.TemporaryDirectory() as d:
            owner={'boot_id':'fixture','pid':os.getpid(),'start_ticks':1}
            for cell in ('one','two'):
                p=attempts.begin(d,cell,method_sha256='a'*64);(p/'raw.json').write_text('{}');attempts.finish(p,'invalid',['raw.json'])
            specs=[{'cell_id':c,'method_sha256':'b'*64} for c in ('one','two')]
            with patch.object(runtime,'process_identity',return_value=owner):
                bad=[dict(specs[0]),dict(specs[1],method_sha256='a'*64)]
                with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin_block(d,bad,'retry',owner,retry=True,retry_reason='changed completion sampler cadence')
                self.assertEqual(len(attempts.load_attempts(d)),2)
                paths=attempts.begin_block(d,specs,'retry',owner,retry=True,retry_reason='changed completion sampler cadence')
                for p in paths:(p/'raw.json').write_text('{}');attempts.finish(p,'invalid',['raw.json'])
                with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin_block(d,[dict(c,method_sha256='c'*64) for c in specs],'third',owner,retry=True,retry_reason='another recorded sampler change')

    def test_audit_withdraws_pass_preserves_evidence_and_keeps_retry_limit(self):
        with tempfile.TemporaryDirectory() as d:
            first=attempts.begin(d,'cell-a',method_sha256='a'*64)
            (first/'raw.json').write_text('{"capture":1}')
            attempts.finish(first,'passed',['raw.json'])
            before=(first/'attempt.json').read_bytes();sha=hashlib.sha256(before).hexdigest()
            audit={'status':'invalid','cell_id':'cell-a','ownership_released':True,
                   'reason':'independent raw cadence audit failed','source_manifest_sha256':'c'*64}
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.invalidate_pass(first,'d'*64,audit)
            self.assertEqual((first/'attempt.json').read_bytes(),before)
            attempts.invalidate_pass(first,sha,audit)
            self.assertEqual((first/'pre-audit-passed-attempt.json').read_bytes(),before)
            self.assertEqual((first/'raw.json').read_text(),'{"capture":1}')
            self.assertEqual(json.loads((first/'attempt.json').read_text())['status'],'invalid')
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin(d,'cell-a')
            second=attempts.begin(d,'cell-a',retry=True,retry_reason='changed sampler completion cadence',method_sha256='b'*64)
            (second/'raw.json').write_text('{}');attempts.finish(second,'failed',['raw.json'])
            with self.assertRaises(runtime.RuntimeErrorClosed):
                attempts.begin(d,'cell-a',retry=True,retry_reason='another sampler change',method_sha256='e'*64)

    def test_audit_needs_exact_released_identity_and_unchanged_prior_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=attempts.begin(d,'cell-a',method_sha256='a'*64);(p/'raw.json').write_text('{}')
            attempts.finish(p,'passed',['raw.json']);raw=(p/'attempt.json').read_bytes()
            audit={'status':'invalid','cell_id':'cell-a','ownership_released':False,
                   'reason':'independent raw cadence audit failed','source_manifest_sha256':'c'*64}
            with self.assertRaises(ValueError):attempts.invalidate_pass(p,hashlib.sha256(raw).hexdigest(),audit)
            audit['ownership_released']=True;(p/'raw.json').write_text('{"changed":1}')
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'prior terminal evidence changed'):
                attempts.invalidate_pass(p,hashlib.sha256(raw).hexdigest(),audit)
            self.assertEqual((p/'attempt.json').read_bytes(),raw)

    def test_crash_raw_preserved_and_completed_not_repeated(self):
        with tempfile.TemporaryDirectory() as root:
            first = attempts.begin(root, 'cell-a', method_sha256='a'*64)
            (first/'raw.json').write_text('{"raw":1}')
            with self.assertRaises(runtime.RuntimeErrorClosed): attempts.begin(root, 'cell-a')
            attempts.mark_interrupted(first, 'fixture process exited; no server work')
            with self.assertRaises(runtime.RuntimeErrorClosed): attempts.begin(root, 'cell-a')
            second = attempts.begin(root, 'cell-a', retry=True, retry_reason='changed transport after interruption', method_sha256='b'*64)
            self.assertEqual((first/'raw.json').read_text(), '{"raw":1}')
            (second/'raw.json').write_text('{"raw":2}')
            attempts.finish(second, 'passed', ['raw.json'])
            with self.assertRaises(runtime.RuntimeErrorClosed): attempts.begin(root, 'cell-a')
            self.assertEqual([x['status'] for x in attempts.load_attempts(root)].count('interrupted'), 1)

    def test_unidentified_orphan_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'attempt-orphan').mkdir()
            with self.assertRaises(runtime.RuntimeErrorClosed): attempts.begin(root, 'cell-a')

    def test_malformed_attempt_metadata_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            orphan=Path(root)/'attempt-bad';orphan.mkdir();(orphan/'attempt.json').write_text('{bad')
            with self.assertRaises(runtime.RuntimeErrorClosed):attempts.begin(root,'cell-a')

    def test_failed_attempt_requires_explicit_retry(self):
        with tempfile.TemporaryDirectory() as root:
            first = attempts.begin(root, 'cell-a', method_sha256='a'*64)
            (first/'raw.json').write_text('{}')
            attempts.finish(first, 'failed', ['raw.json'])
            with self.assertRaises(runtime.RuntimeErrorClosed): attempts.begin(root, 'cell-a')
            with self.assertRaises(runtime.RuntimeErrorClosed): attempts.begin(root, 'cell-a', retry=True)
            with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'changed recorded method'):
                attempts.begin(root,'cell-a',retry=True,retry_reason='changed fixture response',method_sha256='a'*64)
            second=attempts.begin(root, 'cell-a', retry=True, retry_reason='changed fixture response', method_sha256='b'*64)
            self.assertNotEqual(second, first)
            (second/'raw.json').write_text('{}');attempts.finish(second,'failed',['raw.json'])
            with self.assertRaises(runtime.RuntimeErrorClosed):
                attempts.begin(root,'cell-a',retry=True,retry_reason='another fixture change', method_sha256='c'*64)

if __name__ == '__main__': unittest.main()
