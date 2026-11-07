"""Sealed archive retention, bounded extraction and protected-path failure controls."""
from pathlib import Path
import hashlib,io,json,os,sys,tarfile,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import capture_archive,runtime

class CaptureArchiveTests(unittest.TestCase):
    def fixture(self,base):
        root=base/'dsv41-public-fixture';root.mkdir();(root/'a.txt').write_text('Synthetic public fixture.\n')
        os.link(root/'a.txt',root/'b.txt');runtime.seal_evidence(root)
        archive=base/'capture.tar.gz'
        with tarfile.open(archive,'w:gz') as bundle:bundle.add(root,arcname=root.name)
        return root,{'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'seal_sha256':hashlib.sha256((root/'SHA256SUMS').read_bytes()).hexdigest(),'root_name':root.name}

    def test_one_materialized_capture_retains_bytes_links_and_cleans_its_owned_tree(self):
        with tempfile.TemporaryDirectory() as d:
            original,source=self.fixture(Path(d));archive_bytes=Path(source['archive']).read_bytes()
            with capture_archive.materialize(source,reserve_bytes=0) as root:
                materialized=root
                self.assertEqual((root/'a.txt').read_bytes(),(original/'a.txt').read_bytes())
                self.assertEqual((root/'a.txt').stat().st_ino,(root/'b.txt').stat().st_ino)
                runtime.verify_evidence_manifest(root)
            self.assertFalse(materialized.exists());self.assertEqual(Path(source['archive']).read_bytes(),archive_bytes)

    def test_wrong_digest_seal_and_storage_budget_fail_without_changing_source(self):
        with tempfile.TemporaryDirectory() as d:
            original,source=self.fixture(Path(d));before=(original/'SHA256SUMS').read_bytes()
            for change,options in [({'archive_sha256':'0'*64},{}),({'seal_sha256':'0'*64},{}),({},dict(max_bytes=1)),({},dict(reserve_bytes=10**30))]:
                with self.subTest(change=change,options=options),self.assertRaises(ValueError):
                    with capture_archive.materialize({**source,**change},reserve_bytes=options.get('reserve_bytes',0),
                        max_bytes=options.get('max_bytes',64*1024**3)):self.fail('unverified capture yielded')
                self.assertEqual((original/'SHA256SUMS').read_bytes(),before)

    def test_traversal_symlink_and_external_hardlink_are_rejected_before_yield(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);protected=base/'protected.txt';protected.write_text('Retain this public fixture.')
            for kind in ('traversal','symlink','hardlink'):
                archive=base/(kind+'.tar')
                with tarfile.open(archive,'w') as bundle:
                    item=tarfile.TarInfo('dsv41-fixture/../../protected.txt' if kind=='traversal' else 'dsv41-fixture/link')
                    if kind=='traversal':item.size=3;bundle.addfile(item,io.BytesIO(b'bad'))
                    else:
                        item.type=tarfile.SYMTYPE if kind=='symlink' else tarfile.LNKTYPE;item.linkname=str(protected);bundle.addfile(item)
                source={'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                    'seal_sha256':'0'*64,'root_name':'dsv41-fixture'}
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    with capture_archive.materialize(source,reserve_bytes=0):self.fail('unsafe capture yielded')
                self.assertEqual(protected.read_text(),'Retain this public fixture.')

if __name__=='__main__':unittest.main()
