from collections import Counter
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'reproduce'))
import campaign
import suite_controls


class SuiteControlTests(unittest.TestCase):
    def test_added_controls_preserve_fixed_denominators_and_pilot_bound(self):
        cells=campaign.expand(__import__('json').loads(Path(campaign.HERE/'matrix.json').read_text()))
        self.assertEqual(len({c['id'] for c in cells}),len(cells))
        self.assertEqual(sum(c['suite'] in ('core','output_sensitivity','historical_bridge_extra') for c in cells),1036)
        self.assertEqual(sum(c['suite'].startswith('pilot_') for c in cells),38)
        counts=Counter(c['suite'] for c in cells)
        self.assertEqual(counts['gsm8k_calibration'],8)
        self.assertEqual(counts['quality_calibration'],132)
        self.assertEqual(counts['mixed_load_short_control'],16)
        self.assertEqual(counts['natural_stop_acceptance'],24)
        self.assertEqual(counts['context_boundary'],8)
        self.assertEqual(counts['arrival_bracket'],8)
        self.assertEqual(counts['tuning_extra'],792)
        for c in suite_controls.controls():
            if c['suite'] in ('router_session_map','replica_calibration'):
                self.assertIn(c['layout'],('2xtp4','4xdep2','2xdep4','dep8','tp8','4xtp2'))
            else:self.assertTrue(c['layout'] in suite_controls.TOPS or c['layout']=='selected')
        self.assertEqual(counts['replica_calibration'],31)

    def test_traffic_and_thinking_contracts_are_explicit(self):
        cells=campaign.expand(__import__('json').loads(Path(campaign.HERE/'matrix.json').read_text()))
        for c in cells:
            contract=c['request_contract'];self.assertFalse(contract['automatic_retry'])
            if c['suite']=='arrival_ladder':
                self.assertEqual((contract['ramp_seconds'],contract['measured_seconds'],contract['supplemental_batch_ceiling']),(60,300,3))
            if c['suite']=='gsm8k':
                self.assertEqual(contract['thinking'],c['parameters']['thinking']=='on')
                self.assertEqual(contract['concurrency'],8)
                self.assertIn('output_cap',contract['unresolved'])
            if c['suite']=='api_protocol':
                self.assertEqual(contract['stream'],c['parameters']['mode']=='stream')
                self.assertEqual(contract['thinking'],c['parameters']['case']=='thinking')
            if c['suite']=='reasoning_effort':
                self.assertEqual(contract['reasoning_effort'],c['parameters']['effort'])
                self.assertEqual(contract['thinking'],c['parameters']['thinking']=='on')
            if c['suite']=='gsm8k_calibration':self.assertEqual(contract['thinking'],c['parameters']['thinking']=='on')
            if c['suite']=='continuous':self.assertEqual(contract['output_tokens'],c['parameters']['output'])

    def test_conditional_tuning_and_brackets_keep_unresolved_sources(self):
        rows=suite_controls.controls()
        for r in rows:
            if r['suite'] in ('tuning_extra','tuning_tail_validation','arrival_bracket','arrival_doubling_extension'):
                self.assertTrue(r['condition']);self.assertTrue(r['dependent_inputs'])
                self.assertEqual(r['status'],'planned')
