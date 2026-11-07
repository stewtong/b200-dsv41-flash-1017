"""Independent native receipt guards; full owned-node capture remains live work."""
import base64,copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import runtime,verify_node_client_fixture as audit


class NodeClientFixtureTests(unittest.TestCase):
    def fixture(self):
        bodies={'p0':{'model':'/model','messages':[{'role':'user','content':'a'}],'stream':True,
            'stream_options':{'include_usage':True},'max_tokens':128}}
        client=json.dumps(bodies['p0'],sort_keys=True,separators=(',',':')).encode()
        backend=json.dumps({**audit.NATIVE_DEFAULTS,**bodies['p0']}).encode()
        requests={'p0':{'session_id':'fixture-session-0','request_body_sha256':hashlib.sha256(client).hexdigest(),'request_payload_bytes':len(client),
                        'start_ns':100,'terminal_ns':200}}
        arrival={'request_id':'p0','session_id':'fixture-session-0','backend_port':31000,'rank_headers':['1'],
                 'body_sha256':hashlib.sha256(backend).hexdigest(),'body_bytes':len(backend),
                 'body_base64':base64.b64encode(backend).decode(),'observed_ns':150,'server_cpus':[8,9]}
        return requests,arrival,bodies

    def test_native_population_affinity_cpu_and_wire_bindings_cannot_be_forged(self):
        requests,arrival,bodies=self.fixture();routes={}
        audit.audit_arrivals(requests,[arrival],[31000,31001],2,[8,9],routes,bodies)
        self.assertEqual(routes,{'fixture-session-0':(31000,1)})
        for change in ({'request_id':'unknown'},{'session_id':'foreign'},{'backend_port':31999},
                       {'body_sha256':'b'*64},{'body_bytes':0},{'observed_ns':99},{'server_cpus':[0,1]},
                       {'rank_headers':[]},{'rank_headers':['2']},{'rank_headers':['0','1']}):
            bad=dict(arrival,**change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                audit.audit_arrivals(requests,[bad],[31000,31001],2,[8,9],{},bodies)
        with self.assertRaisesRegex(ValueError,'lost or retried'):audit.audit_arrivals(requests,[],[31000],2,[8,9],{},bodies)
        with self.assertRaisesRegex(ValueError,'lost or retried'):audit.audit_arrivals(requests,[arrival,arrival],[31000],2,[8,9],{},bodies)
        with self.assertRaisesRegex(ValueError,'affinity changed'):
            audit.audit_arrivals(requests,[dict(arrival,rank_headers=['0'])],[31000],2,[8,9],routes,bodies)

    def test_backend_bytes_allow_only_pinned_native_defaults_and_preserve_explicit_fields(self):
        requests,arrival,bodies=self.fixture()
        original={**audit.NATIVE_DEFAULTS,**bodies['p0']}
        for changes in ({'max_tokens':127},{'model':'foreign'},{'ignore_eos':0},{'unknown_extension':True},
                        {'messages':[{'role':'user','content':'changed'}]}):
            payload=json.dumps({**original,**changes}).encode()
            bad=dict(arrival,body_sha256=hashlib.sha256(payload).hexdigest(),body_bytes=len(payload),
                body_base64=base64.b64encode(payload).decode())
            with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'native body transformation'):
                audit.audit_arrivals(requests,[bad],[31000],2,[8,9],{},bodies)
        for encoded in ('%%%','',None):
            with self.subTest(encoded=encoded),self.assertRaises(ValueError):
                audit.audit_arrivals(requests,[dict(arrival,body_base64=encoded)],[31000],2,[8,9],{},bodies)
        changed=copy.deepcopy(requests);changed['p0']['request_body_sha256']='a'*64
        with self.assertRaisesRegex(ValueError,'native body transformation'):
            audit.audit_arrivals(changed,[arrival],[31000],2,[8,9],{},bodies)

    def test_resealed_foreign_source_cannot_reach_a_node_qualified_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,value in [('run-config.json',{'toolkit_manifest_sha256':'a'*64}),('execution-state.json',{}),
                               ('cpu-partition.json',{}),('router-qualification.json',{'installed_packages':{},'router_executable_sha256':'b'*64}),
                               ('router-runtime-check.json',{})]:runtime.atomic_json(root/name,value)
            runtime.atomic_json(root/'router-body-contract.json',audit.BODY_CONTRACT)
            (root/'toolkit-MANIFEST.json').write_text('{}');runtime.seal_evidence(root)
            with self.assertRaisesRegex(ValueError,'source, CPU or router binding'):audit.verify(root)


if __name__=='__main__':unittest.main()
