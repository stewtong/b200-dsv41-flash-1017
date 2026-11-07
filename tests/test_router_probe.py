from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import router_probe


class RouterProbeTests(unittest.TestCase):
    def test_loopback_synthetic_backend_records_exact_request(self):
        server=router_probe.Backend(0)
        try:
            router_probe.request(server.port,'session','session-r0')
            with server.lock: rows=list(server.records)
            result=router_probe.verify(rows,['session'],1,1,[server.port])
            self.assertEqual(result['requests'],1)
            def records():
                with server.lock:return list(server.records)
            self.assertEqual(router_probe.verify_no_retry(server.port,records)['observed_backend_attempts'],1)
        finally:server.close()

    def test_exact_coverage_and_session_affinity(self):
        rows=[{'session_id':'s','request_id':'s-r0','rank_headers':['1'],'backend_port':31050},
              {'session_id':'s','request_id':'s-r1','rank_headers':['1'],'backend_port':31050}]
        self.assertEqual(router_probe.verify(rows,['s'],2,2,[31050])['sessions']['s']['rank'],1)
        rows[1]['rank_headers']=['0']
        with self.assertRaisesRegex(ValueError,'affinity'):router_probe.verify(rows,['s'],2,2,[31050])
        rows[1]['rank_headers']=[]
        with self.assertRaisesRegex(ValueError,'rank'):router_probe.verify(rows,['s'],2,2,[31050])
