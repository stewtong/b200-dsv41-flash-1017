"""Source-proof validation with synthetic bytes and an explicitly isolated pin."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import cache_source_proof as proof,runtime

FIXTURE_SOURCE='# public synthetic cache source fixture\n'
FIXTURE_PIN=hashlib.sha256(FIXTURE_SOURCE.encode()).hexdigest()
IMAGE='sha256:'+'a'*64


def fixture(root,image_id=IMAGE):
    root.mkdir();pid={'boot_id':'synthetic-fixture','pid':2_000_000_000,'start_ticks':1}
    values={'installed-source.json':{'files':[{'relative_path':proof.RELATIVE_PATH,'sha256':FIXTURE_PIN,'source':FIXTURE_SOURCE}],
            'expected_source_pins':{proof.RELATIVE_PATH:FIXTURE_PIN},'source_bytes':len(FIXTURE_SOURCE.encode())},
        'source-result.json':{'status':'captured','image_id':image_id,'files':1,'source_bytes':len(FIXTURE_SOURCE.encode())},
        'execution-state.json':{'phase':'installed-source-capture','stopping':True,'watchdog_stop':True,
            'container_ids':['fixture-container'],'driver_identity':pid,'watchdog_identity':pid,'owned_processes':[]},
        'source-container.json':{'id':'fixture-container','image_id':image_id,'argv':['docker','--read-only','--network','none']},
        'cleanup-result.json':{'success':True},'gpu-idle-after-cleanup.json':{'idle':True}}
    for name,value in values.items():runtime.atomic_json(root/name,value)
    runtime.seal_evidence(root)
    return root


class CacheSourceProofTests(unittest.TestCase):
    def test_exact_source_image_release_and_resealed_mutations(self):
        with tempfile.TemporaryDirectory() as d,patch.object(proof,'PINNED_SHA256',FIXTURE_PIN):
            root=fixture(Path(d)/'source');self.assertEqual(proof.verify(root,IMAGE)['source_sha256'],FIXTURE_PIN)
            with self.assertRaises(ValueError):proof.verify(root,'sha256:'+'b'*64)
            for name,change in [('installed-source.json',lambda x:x['files'][0].update(source=FIXTURE_SOURCE+'changed')),
                ('source-container.json',lambda x:x['argv'].extend(['--gpus','all'])),
                ('cleanup-result.json',lambda x:x.update(success=False)),
                ('execution-state.json',lambda x:x.update(stopping=False))]:
                path=root/name;raw=path.read_bytes();value=json.loads(raw);change(value);runtime.atomic_json(path,value);runtime.seal_evidence(root)
                with self.subTest(file=name),self.assertRaises(ValueError):proof.verify(root,IMAGE)
                path.write_bytes(raw);runtime.seal_evidence(root)
            with patch.object(proof.runtime,'process_alive',return_value=True),self.assertRaises(ValueError):proof.verify(root,IMAGE)
