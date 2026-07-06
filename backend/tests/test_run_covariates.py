"""Integration smoke test for the run-level covariate registry builder (T7).

Skipped automatically when the warehouse DB is not present (e.g. clean CI).
"""
import unittest

from analysis.data.run_covariates import (
    build_run_covariates,
    coverage_report,
    coverage_by_stratum,
    WAREHOUSE_DB,
)


@unittest.skipUnless(WAREHOUSE_DB.exists(), "warehouse pump2.db not available")
class RunCovariatesBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = build_run_covariates(tte_col="ttf_mix")

    def test_one_row_per_run(self):
        self.assertEqual(len(self.df), 2634)
        # row_id is the run key and must be unique
        self.assertEqual(self.df["row_id"].nunique(), 2634)

    def test_stratum_key_populated(self):
        self.assertTrue(self.df["stratum_key"].notna().all())
        self.assertTrue((self.df["stratum_key"].str.count("_") >= 2).all())

    def test_missing_indicators_present(self):
        for col in ("chloride_mg_l", "freq_mean", "stages"):
            self.assertIn(f"{col}_missing", self.df.columns)
            self.assertIn(f"{col}_imputed_src", self.df.columns)

    def test_log_transforms_finite(self):
        for col in ("log_chloride_mg_l", "log_motor_power_kw"):
            self.assertIn(col, self.df.columns)
            self.assertTrue(self.df[col].replace([float("inf"), float("-inf")], float("nan")).notna().any())

    def test_idle_frac_bounded(self):
        idle = self.df["idle_frac"].dropna()
        self.assertTrue((idle >= 0).all() and (idle <= 1).all())

    def test_run_seq_and_pump_parse(self):
        self.assertTrue((self.df["run_seq"] >= 1).all())
        # parser should cover the large majority of runs
        self.assertGreater((self.df["pump_series"] != "other").mean(), 0.9)

    def test_coverage_report_measured_le_total(self):
        cov = coverage_report(self.df)
        self.assertTrue((cov["n_measured"] <= cov["n_total"]).all())
        # completion covariates are near-complete
        stages_row = cov.loc[cov["covariate"] == "stages"].iloc[0]
        self.assertGreater(stages_row["pct_measured"], 90)

    def test_coverage_by_stratum_nonempty(self):
        cs = coverage_by_stratum(self.df)
        self.assertGreater(len(cs), 0)
        self.assertIn("low_coverage_flag", cs.columns)


if __name__ == "__main__":
    unittest.main()
