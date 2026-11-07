"""Retain identical per-cell raw evidence without multiplying finalized streams."""
import hashlib,json,os
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import node_pilot_block,runtime


class SharedRawStorageTests(unittest.TestCase):
    def fixture(self,base):
        root=Path(base);owner={'pid':999999,'start_ticks':1,'boot_id':'synthetic'}
        runtime.atomic_json(root/'run-config.json',{'run_id':'fixture'})
        runtime.atomic_json(root/'execution-state.json',{'driver_identity':owner,'run_id':'fixture','stopping':True,
            'watchdog_stop':True,'watchdog_identity':dict(owner,pid=999998),'owned_processes':[]})
        runtime.atomic_json(root/'smoke-result.json',{'status':'passed'})
        runtime.atomic_json(root/'cleanup-result.json',{'success':True})
        (root/'telemetry-0000.jsonl').write_bytes(b'{"synthetic":true}\n'*60000)
        nested=root/'worker-observation/replica-0/worker-500';nested.mkdir(parents=True)
        (nested/'samples-0000.jsonl').write_bytes(b'{"synthetic_worker":true}\n'*60000)
        runtime.atomic_json(root/'worker-observation/replica-0/contract.json',{'scope':'synthetic projection fixture'})
        source=root/'worker-observer-source';source.mkdir();(source/'runtime.py').write_text('# public fixture\n')
        proof=root/'worker-source-proof';proof.mkdir();runtime.atomic_json(proof/'source-result.json',{'scope':'public fixture'})
        runtime.seal_evidence(proof)
        return root,owner

    def test_exact_raw_bytes_and_independent_metadata_survive_projection(self):
        with tempfile.TemporaryDirectory() as d:
            root,owner=self.fixture(d);cells=[]
            with patch.object(runtime,'process_alive',return_value=False):
                for name in ('one','two'):
                    cell=root/'cells'/name;cell.mkdir(parents=True);node_pilot_block.project_shared(root,cell,owner);cells.append(cell)
            raw=root/'telemetry-0000.jsonl'
            self.assertTrue(all(os.path.samefile(raw,c/raw.name) for c in cells))
            worker_path='worker-observation/replica-0/worker-500/samples-0000.jsonl'
            self.assertTrue(all(os.path.samefile(root/worker_path,c/worker_path) for c in cells))
            self.assertFalse(os.path.samefile(root/'cleanup-result.json',cells[0]/'cleanup-result.json'))
            for cell in cells:
                proof=json.loads((cell/'block-provenance.json').read_text())
                self.assertEqual(proof['shared_file_sha256'][raw.name],hashlib.sha256(raw.read_bytes()).hexdigest())
                self.assertEqual(proof['shared_file_sha256'][worker_path],hashlib.sha256((root/worker_path).read_bytes()).hexdigest())
                self.assertIn('worker-observer-source/runtime.py',proof['shared_file_sha256'])
                runtime.verify_evidence_manifest(cell/'worker-source-proof')
                runtime.seal_evidence(cell);runtime.verify_evidence_manifest(cell)
            runtime.atomic_json(root/'cleanup-result.json',{'success':False})
            self.assertTrue(json.loads((cells[0]/'cleanup-result.json').read_text())['success'])

    def test_possible_live_writers_block_links_before_raw_projection(self):
        with tempfile.TemporaryDirectory() as d:
            root,owner=self.fixture(d);cell=root/'cells/one';cell.mkdir(parents=True)
            with patch.object(runtime,'process_alive',return_value=True):
                with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'possible writer'):node_pilot_block.project_shared(root,cell,owner)
            self.assertFalse((cell/'telemetry-0000.jsonl').exists())
            self.assertFalse((cell/'worker-observation/replica-0/worker-500/samples-0000.jsonl').exists())


if __name__=='__main__':unittest.main()
