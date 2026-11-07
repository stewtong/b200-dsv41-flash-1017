"""Pinned completion-counter labels retain distinct failures and cancellations."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import telemetry
import verify_telemetry


class FinishedReasonMetricsTests(unittest.TestCase):
    endpoint='http://127.0.0.1:31000/metrics'
    reasons=('stop','length','abort','error','repetition')

    def snapshot(self):
        return '# TYPE vllm:request_success counter\n'+''.join(
            'vllm:request_success_total{model_name="/model",engine="%d",finished_reason="%s"} %d\n'%(rank,reason,index)
            for rank in (0,1) for index,reason in enumerate(self.reasons))

    def check(self,text):
        ranks={self.endpoint:[0,1]};snapshot={self.endpoint:text}
        telemetry.check_rank_coverage(snapshot,['vllm:request_success_total'],ranks)
        verify_telemetry.verify_metrics(snapshot,ranks,{'vllm:request_success_total':'counter'})

    def test_every_declared_finish_reason_is_required_on_every_rank(self):
        text=self.snapshot();self.check(text)
        missing='\n'.join(l for l in text.splitlines() if not ('engine="1"' in l and 'finished_reason="error"' in l))+'\n'
        duplicate=text+next(l for l in text.splitlines() if 'engine="0"' in l and 'finished_reason="abort"' in l)+'\n'
        unknown=text.replace('finished_reason="error"','finished_reason="unknown"')
        repeated_label=text.replace('finished_reason="error"','finished_reason="error",finished_reason="error"')
        missing_label=text.replace(',finished_reason="error"','')
        for altered in (missing,duplicate,unknown,repeated_label,missing_label):
            snapshot={self.endpoint:altered};ranks={self.endpoint:[0,1]}
            with self.subTest(altered=altered):
                with self.assertRaises(ValueError):telemetry.check_rank_coverage(snapshot,['vllm:request_success_total'],ranks)
                with self.assertRaises(ValueError):verify_telemetry.verify_metrics(snapshot,ranks,{'vllm:request_success_total':'counter'})

    def test_independent_types_and_counter_values_still_fail_closed(self):
        for text in (self.snapshot().replace(' counter\n',' gauge\n'),
                     self.snapshot().replace('} 3\n','} -1\n'),
                     self.snapshot().replace('} 3\n','} NaN\n')):
            with self.assertRaises(ValueError):verify_telemetry.verify_metrics({self.endpoint:text},{self.endpoint:[0,1]},{'vllm:request_success_total':'counter'})


if __name__=='__main__':unittest.main()
