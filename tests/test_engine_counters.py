from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import engine_counters as counters


def export(prompt=0,cached=0,finished=None):
    finished=finished or {r:0 for r in counters.REASONS}
    values={};types={};requests=sum(finished.values())
    for (name,rank,dimension),(kind,integer) in counters.contract([0]).items():
        base=name.removesuffix('_total') if kind=='counter' else name.rsplit('_',1)[0];types[base]=kind
        dimension_name='source' if name=='vllm:prompt_tokens_by_source_total' else 'finished_reason'
        label='model_name="/model",engine="0"'+(','+dimension_name+'="'+dimension+'"' if dimension else '')
        value=0
        if name=='vllm:prompt_tokens_total':value=prompt
        if name=='vllm:prompt_tokens_cached_total':value=cached
        if name=='vllm:prompt_tokens_by_source_total':value={'local_compute':prompt-cached,'local_cache_hit':cached,'external_kv_transfer':0}[dimension]
        if name=='vllm:request_success_total':value=finished[dimension]
        if name.endswith('_count'):value=requests
        if name=='vllm:request_prefill_time_seconds_sum':value=requests*.25
        values[name+'{'+label+'}']=value
    return '\n'.join('# TYPE '+name+' '+kind for name,kind in types.items())+'\n'+'\n'.join(name+' '+str(value) for name,value in values.items())+'\n'


class EngineCounterTests(unittest.TestCase):
    def test_failed_requests_are_separate_and_cached_prompt_is_not_compute(self):
        reasons={r:0 for r in counters.REASONS};reasons.update(length=2,abort=1,error=1)
        result=counters.differences({'rank':export()},{'rank':export(1000,900,reasons)},{'rank':[0]})
        row=result['ranks'][0]
        self.assertEqual(row['successfully_finished_requests'],2)
        self.assertEqual(row['prompt_tokens_by_source']['local_compute'],100)
        self.assertEqual(row['finished_by_reason']['abort'],1)
        self.assertEqual(row['request_prefill_seconds_sum'],1.0)
        self.assertNotIn('prefill_tokens_per_second',row)

    def test_missing_duplicate_bad_type_reset_and_unsettled_counts_fail(self):
        original=export(1000,900);line=next(l for l in original.splitlines() if l.startswith('vllm:prompt_tokens_total'))
        bad=[original.replace(line+'\n',''),original+line+'\n',original.replace('# TYPE vllm:prompt_tokens counter','# TYPE vllm:prompt_tokens gauge'),
             original.replace(line,line.rsplit(' ',1)[0]+' nan'),original.replace(line,line.rsplit(' ',1)[0]+' 1000.5'),
             original.replace('source="local_compute"','source="unregistered"')]
        for text in bad:
            with self.subTest(text=text[-100:]),self.assertRaises(ValueError):counters.snapshot(text,[0])
        with self.assertRaisesRegex(ValueError,'reset'):counters.differences({'rank':original},{'rank':export()},{'rank':[0]})
        broken=export(1000,900).replace('source="local_compute"} 100','source="local_compute"} 101')
        with self.assertRaisesRegex(ValueError,'not settled'):counters.differences({'rank':export()},{'rank':broken},{'rank':[0]})


if __name__=='__main__':unittest.main()
