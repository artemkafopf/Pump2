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

    def test_phase_c_features_and_adjusters(self):
        # C0 early-window instability/exposure features
        for col in ("freq_std_early", "load_std_early", "freq_above_55hz_pct_early"):
            self.assertIn(col, self.df.columns)
        # mandatory adjusters
        self.assertIn("install_period", self.df.columns)
        self.assertEqual(set(self.df["install_period"].dropna().unique())
                         <= {"<=2019", "2020-2022", "2023+"}, True)
        self.assertEqual(set(self.df["has_telemetry"].unique()) <= {0, 1}, True)
        # has_telemetry mirrors ttf_true_source == 'missing'
        self.assertEqual(int((self.df["has_telemetry"] == 0).sum()),
                         int((self.df["ttf_true_source"] == "missing").sum()))

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
