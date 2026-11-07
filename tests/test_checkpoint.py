import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'reproduce'))
import verify_checkpoint as v

class CheckpointTests(unittest.TestCase):
    def test_git_and_lfs_content_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root/'model'; p.mkdir(); (p/'small').write_bytes(b'abc'); (p/'big').write_bytes(b'defg')
            manifest = {'revision':'fixture','files':[
                {'path':'small','size':3,'git_blob':hashlib.sha1(b'blob 3\0abc').hexdigest(),'lfs_sha256':None},
                {'path':'big','size':4,'git_blob':None,'lfs_sha256':hashlib.sha256(b'defg').hexdigest()}]}
            inventory=root/'inventory.json'; inventory.write_text(json.dumps(manifest))
            self.assertEqual(v.verify(inventory, p)['files'], 2)
            meta=p/'.cache/huggingface/download';meta.mkdir(parents=True);(meta/'ok.metadata').write_text('fixture')
            self.assertTrue(v.verify(inventory,p)['file_set_verified'])
            (meta/'stray.incomplete').write_text('partial')
            with self.assertRaisesRegex(ValueError,'incomplete'): v.verify(inventory,p)
            (meta/'stray.incomplete').unlink()
            (p/'unexpected').write_text('extra')
            with self.assertRaisesRegex(ValueError,'file-set'): v.verify(inventory,p)
            (p/'unexpected').unlink()
            (p/'alias').symlink_to('small')
            with self.assertRaisesRegex(ValueError,'symlink'): v.verify(inventory,p)
            (p/'alias').unlink()
            (p/'big').write_bytes(b'xxxx')
            with self.assertRaises(ValueError): v.verify(inventory, p)
