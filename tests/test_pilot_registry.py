import json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import pilot_registry
import runtime


def fixture_config(base,run_dir):
    manifest=Path(base)/'manifest.json'
    manifest.write_text(json.dumps({'plan_sha256':'a'*64,'runtime':{'toolkit_manifest_sha256':'b'*64,'starting_profiles':{'dep8':'c'*64}}}))
    return {'cell_id':'cell','run_dir':str(run_dir),'manifest':str(manifest),'image':'fixture','source_revision':'revision','layout':'dep8'}


class PilotRegistryTests(unittest.TestCase):
    def test_incomplete_attempt_blocks_next_and_terminal_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'attempts';config=fixture_config(d,'/data/dsv41-fixture')
            attempt=pilot_registry.begin(config,'a'*64,root,lambda pid:{'pid':pid,'start_ticks':1,'boot_id':'fixture'})
            pilot_registry.fail(attempt,ValueError('fixture'))
            with self.assertRaises(runtime.RuntimeErrorClosed):
                pilot_registry.begin(config,'a'*64,root,lambda pid:{'pid':pid})
            # A successful invalid pilot still has a terminal, non-repeatable record.
            result=Path(d)/'result.json';result.write_text(json.dumps({'status':'invalid','cell_id':'cell'}))
            pilot_registry.finish(attempt,result,{'status':'invalid','cell_id':'cell'})
            with self.assertRaises(runtime.RuntimeErrorClosed):
                pilot_registry.begin(config,'a'*64,root,lambda pid:{'pid':pid})
