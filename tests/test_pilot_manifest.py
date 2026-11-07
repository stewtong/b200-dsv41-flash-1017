from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import check_fix
import node_smoke
import pilot_manifest


class PilotManifestTests(unittest.TestCase):
    def test_verified_receipts_create_partial_pilot_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            plan=Path(d)/'plan.md';plan.write_text('fixture plan')
            matrix=Path(pilot_manifest.__file__).with_name('matrix.json')
            inv={name:'invariants-present' for names in check_fix.REQUIRED.values() for name in names}
            image={'image':'example/image@sha256:'+'a'*64,'source_revision':'b'*40,'entrypoint':['vllm','serve'],'driver_versions':['580.1']*8,
                   'installed':{'invariants':inv,'vllm_version':'fixture','torch_version':'fixture','cuda_version':'fixture','nccl_version':[1,2,3]}}
            router={'lock_sha256':node_smoke.ROUTER_LOCK_SHA,'router_executable_sha256':'c'*64}
            weights={'revision':node_smoke.MODEL_REVISION,'inventory_sha256':node_smoke.INVENTORY_SHA,'files':83,
                     'bytes':510311045295,'file_set_verified':True,'metadata_bytes':0}
            retirement=dict.fromkeys(('containers_stopped','timer_inactive','previous_weights_absent'),True)
            deadline=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
            m=pilot_manifest.build(matrix,plan,image,router,weights,deadline,retirement,'d'*64)
            self.assertFalse(m['complete_universe'])
            self.assertEqual(len(m['runtime']['starting_profiles']),6)
            self.assertEqual(m['runtime']['fixed_input_schema'],'fixed-cache-v2-ascii-instruction-v2')
            comparison=pilot_manifest.build(matrix.with_name('comparison-matrix.json'),plan,image,router,weights,deadline,retirement,'d'*64)
            self.assertTrue(comparison['complete_universe'])
            self.assertEqual(comparison['runtime']['fixed_input_schema'],'fixed-performance-v1')
            weights['file_set_verified']=False
            with self.assertRaisesRegex(ValueError,'checkpoint receipt'):pilot_manifest.build(matrix,plan,image,router,weights,deadline,retirement,'d'*64)
