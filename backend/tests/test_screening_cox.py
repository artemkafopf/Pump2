"""Unit tests for the pure helpers of the Phase C screening cascade.

The lifelines-fitting stages are exercised by the integration runs (C1/C2/C4);
here we pin the deterministic pruning / formula / handoff logic on synthetic data.
"""
import unittest

import numpy as np
import pandas as pd

from analysis.models.survival.screening_cox import (
    correlation_prune,
    vif_prune,
    _build_formula,
    population_ref_coeffs,
)


class CorrelationPruneTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        self.df = pd.DataFrame({
            "a": x,
            "b": x + rng.normal(scale=0.05, size=400),   # ~collinear with a
            "c": rng.normal(size=400),                    # independent
        })

    def test_drops_lower_cindex_of_correlated_pair(self):
        kept, pairs = correlation_prune(self.df, ["a", "b", "c"],
                                        cindex={"a": 0.7, "b": 0.6, "c": 0.55})
        self.assertIn("a", kept)      # higher C-index of the a/b pair
        self.assertNotIn("b", kept)   # dropped
        self.assertIn("c", kept)      # uncorrelated, survives
        self.assertEqual(pairs[0]["kept"], "a")
        self.assertEqual(pairs[0]["dropped"], "b")

    def test_no_drop_when_uncorrelated(self):
        kept, pairs = correlation_prune(self.df, ["a", "c"],
                                        cindex={"a": 0.7, "c": 0.6})
        self.assertEqual(set(kept), {"a", "c"})
        self.assertEqual(pairs, [])


class VifPruneTests(unittest.TestCase):
    def test_drops_redundant_column(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        df = pd.DataFrame({"x": x, "y": y, "xdup": x + rng.normal(scale=1e-3, size=500)})
        keep, vif_df = vif_prune(df, ["x", "y", "xdup"], thr=5.0)
        # one of the near-duplicate {x, xdup} must be removed; y stays
        self.assertIn("y", keep)
        self.assertTrue(("x" in keep) ^ ("xdup" in keep))
        self.assertTrue((vif_df["VIF"] <= 5.0).all())


class FormulaAndHandoffTests(unittest.TestCase):
    def test_formula_wraps_categoricals(self):
        f = _build_formula(["freq_mean", "load_mean"], ["install_period"])
        self.assertEqual(f, "freq_mean + load_mean + C(install_period)")

    def test_population_ref_coeffs_stamps_and_refs(self):
        summary = pd.DataFrame({
            "term": ["freq_mean", "C(install_period)[T.2023+]"],
            "coef": [0.05, 0.30], "se": [0.01, 0.1],
            "hr": [1.05, 1.35], "hr_ci_lo": [1.0, 1.1], "hr_ci_hi": [1.1, 1.6],
            "p": [0.001, 0.02],
        })
        df = pd.DataFrame({"freq_mean": [50.0, 52.0, 54.0]})
        out = population_ref_coeffs(summary, df, ["freq_mean"], block_label="block2")
        num = out[out["covariate"] == "freq_mean"].iloc[0]
        self.assertAlmostEqual(num["ref_value"], 52.0, places=6)
        self.assertEqual(num["block"], "block2")
        self.assertEqual(num["clock"], "ttf_mix")
        # categorical term carries no numeric ref
        cat = out[out["covariate"].str.startswith("C(")].iloc[0]
        self.assertTrue(np.isnan(cat["ref_value"]))


if __name__ == "__main__":
    unittest.main()
