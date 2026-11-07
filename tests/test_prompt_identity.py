import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduce'))

import comparison_scope
import launcher
import prompt_identity


class PromptIdentityTests(unittest.TestCase):
    def test_all_scientific_input_populations_fit_actual_unchanged_launcher(self):
        seen = set()
        for cache, lengths, cs in [('cold', comparison_scope.INPUTS, comparison_scope.CONCURRENCIES),
                                   ('warm_prefix', comparison_scope.WARM_INPUTS, comparison_scope.WARM_CONCURRENCIES)]:
            for n in lengths:
                launcher.validate_run_id(prompt_identity.panel(n, cache, '20261104t0545z'))
                for c in cs:
                    for rep in range(3):
                        alias = prompt_identity.child(n, c, rep, cache, '20261104t0545z')
                        launcher.validate_run_id(alias)
                        self.assertNotIn(alias, seen)
                        seen.add(alias)
        self.assertEqual(len(seen), 252)
        with self.assertRaises(ValueError):
            launcher.validate_run_id('prompt-comparison-tp4-i128-c128-r0-20261103t0728z')

    def test_invalid_scientific_shape_or_timestamp_never_creates_alias(self):
        for args in [(128000, 128, 0, 'warm_prefix', '20261104t0545z'),
                     (64000, 1, 0, 'warm_prefix', '20261104t0545z'),
                     (128000, True, 0, 'cold', '20261104t0545z'),
                     (128000, 1, 3, 'cold', '20261104t0545z'),
                     (128000, 1, 0, 'cold', '20260230t0545z')]:
            with self.assertRaises(ValueError):
                prompt_identity.child(*args)
