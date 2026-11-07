"""Synthetic boundary settling; HTTP and live GPU acceptance stay separate."""
import base64,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import fixed_attribution,runtime,speculation_evidence as spec


def metrics(counts):
    lines=[]
    for name in spec.SCALARS:
        lines.append('# TYPE '+spec.PREFIX+name+' counter')
    lines.append('# TYPE '+spec.POSITION.removesuffix('_total')+' counter')
    for rank,count in enumerate(counts):
        labels='model_name="/model",engine="'+str(rank)+'"'
        for name,value in zip(spec.SCALARS,(4*count,12*count,6*count)):
            lines.append(spec.PREFIX+name+'_total{'+labels+'} '+str(value))
        for i,v in enumerate((3,2,1)):
            lines.append(spec.POSITION+'{'+labels+',position="'+str(i)+'"} '+str(v*count))
    return '\n'.join(lines)


def wire(task):
    event={'choices':[],'usage':{'prompt_tokens':8000,'completion_tokens':10},'metrics':{
        'speculative_decoding':{'num_spec_tokens':3,'num_spec_steps':4,
        'num_accepted_draft_tokens':6,'num_draft_tokens':12,'mean_acceptance_length':2.5,
        'draft_acceptance_rate':.5,'acceptance_histogram':[1,1,1,1],
        'per_step_accepted':[0,1,2,3],'per_step_drafted':[3,3,3,3]}}}
    data=('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode()
    return {'task_id':task,'raw':{'complete':True,'http_status':200,'error':None,
        'start_ns':0,'terminal_ns':2,'wire_chunks':[
        {'offset_ns':1,'base64':base64.b64encode(data).decode(),'eof':False},
        {'offset_ns':2,'base64':'','eof':True}], 'wire_byte_count':len(data),
        'wire_bytes_observed':len(data),'wire_limit_bytes':65536,'wire_truncated':False,
        'wire_response_kind':'sse','wire_sha256':hashlib.sha256(data).hexdigest(),
        'wire_expected_body_bytes':len(data),'events':[{'kind':'event','data':event,'offset_ns':1},
        {'kind':'done','data':None,'offset_ns':1}]}}


class SpeculativeBoundaryTests(unittest.TestCase):
    def test_request_steps_bind_every_counter_and_native_rank(self):
        ranks={'ep':[0,1]};rows={'s0':wire('s0')};placement={'s0':['ep','0']}
        before={'ep':metrics([0,0])};after={'ep':metrics([1,0])}
        result=spec.reconcile_requests(before,after,ranks,3,rows,placement)
        self.assertEqual(result['request_population_status'],'ordered_wire_steps_match_cumulative_counters_per_rank')
        self.assertIsNone(result['verified_token_count'])
        # Same pooled totals on the wrong rank cannot pass.
        for counts in ([0,1],[0,0],[2,0]):
            with self.assertRaises(spec.UnsettledSpeculationError):
                spec.reconcile_requests(before,{'ep':metrics(counts)},ranks,3,rows,placement)
        with self.assertRaisesRegex(ValueError,'reset'):
            spec.reconcile_requests(after,before,ranks,3,rows,placement)
        with self.assertRaisesRegex(ValueError,'population'):
            spec.reconcile_requests(before,after,ranks,3,rows,{})
        changed={'s0':wire('s0')};changed['s0']['raw']['events'][0]['data']['metrics']['speculative_decoding']['num_draft_tokens']=99
        with self.assertRaisesRegex(ValueError,'retained wire'):
            spec.reconcile_requests(before,after,ranks,3,changed,placement)

    def test_lagged_counters_wait_inside_same_deadline_and_preserve_each_read(self):
        self.boundary_control(permanently_lagged=False)

    def test_completed_requests_do_not_hide_unsettled_speculation_at_deadline(self):
        self.boundary_control(permanently_lagged=True)

    def boundary_control(self,permanently_lagged):
        clock=[1_000_000_000];reads=[];finished=[False]
        class Session:
            deadline_ns=200_000_000_000
            def ensure_healthy(self):pass
        def scrape(endpoint,names,timeout):
            reads.append(timeout)
            return metrics([1 if finished[0] and not permanently_lagged and len(reads)>2 else 0])
        def capsule(session,url,tasks,root,timeout,clock_id,namespace,ids,observe):
            observe('before_measured');(root/'measured').mkdir(parents=True)
            runtime.atomic_json(root/'measured/s0.json',wire('s0'));finished[0]=True
            observe('after_measured');return {'valid_requests':1}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'router.log';log.write_text('')
            with patch.object(fixed_attribution,'time',SimpleNamespace(monotonic_ns=lambda:clock[0],
                    sleep=lambda seconds:clock.__setitem__(0,clock[0]+int(seconds*1e9)))), \
                 patch.object(fixed_attribution.telemetry,'scrape',side_effect=scrape), \
                 patch.object(fixed_attribution.engine_counters,'snapshot'), \
                 patch.object(fixed_attribution.engine_counters,'differences',return_value={'ranks':[
                    {'finished_by_reason':{'stop':0,'length':1,'abort':0,'error':0,'repetition':0}}]}), \
                 patch.object(fixed_attribution.cache_evidence,'counters'), \
                 patch.object(fixed_attribution.fixed_session,'run',side_effect=capsule), \
                 patch('verify_fixed_attribution.verify',return_value={'fixture':'independent HTTP verification excluded'}):
                args=(Session(),'unused',[{'id':'s0','cache_designation':'cold'}],root/'capture',5,
                    'fixture','fx'+'a'*24,{'s0':'unit0'},{'ep':[0]},log,128)
                kwargs={'speculative_contract':{'max_spec_tokens':3,'placement':{'s0':['ep','0']}}}
                if permanently_lagged:
                    with self.assertRaisesRegex(runtime.RuntimeErrorClosed,'boundary deadline'):
                        fixed_attribution.capture(*args,**kwargs)
                    self.assertTrue((root/'capture/attribution-failure.json').exists())
                    self.assertFalse((root/'capture/attribution-result.json').exists())
                else:
                    fixed_attribution.capture(*args,**kwargs)
                    boundaries=json.loads((root/'capture/boundaries.json').read_text())
                    self.assertEqual(boundaries[-1]['snapshot_file'],'after_measured-001.json')
                    self.assertLess(boundaries[-1]['completed_ns'],boundaries[-1]['deadline_ns'])
                    self.assertTrue((root/'capture/after_measured-000.json').exists())
                    self.assertEqual(len(reads),3)


if __name__=='__main__':unittest.main()
