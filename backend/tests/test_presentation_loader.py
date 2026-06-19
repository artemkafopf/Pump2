import unittest

import pandas as pd

from analysis.presentation_correlations import build_ttf_correlation_frame
from analysis.presentation_loader import detect_sheet_roles, prepare_presentation_dataset


class PresentationLoaderTests(unittest.TestCase):
    def test_detect_sheet_roles_uses_sheet_name_and_column_profile(self):
        workbook = {
            "runs": pd.DataFrame({"run_id": [1], "event": [1], "TTF_days": [50]}),
            "Sheet2": pd.DataFrame(
                {
                    "run_id": [1, 1],
                    "date": ["2024-01-01", "2024-01-15"],
                    "Qliq_m3d": [100, 95],
                    "frequency_hz": [50, 52],
                    "status": ["operating", "operating"],
                }
            ),
            "Telemetry Data": pd.DataFrame(
                {
                    "run_id": [1, 1],
                    "timestamp": ["2024-01-01 01:00:00", "2024-01-01 02:00:00"],
                    "frequency_hz": [50, 51],
                    "current_a": [20, 22],
                    "status": ["operating", "restart"],
                }
            ),
        }

        detected = detect_sheet_roles(workbook)

        self.assertEqual(detected["runs"][0], "runs")
        self.assertEqual(detected["daily_or_monthly_regime"][0], "Sheet2")
        self.assertEqual(detected["telemetry"][0], "Telemetry Data")

    def test_prepare_presentation_dataset_derives_ttf_and_merges_sources(self):
        workbook = {
            "runs": pd.DataFrame(
                {
                    "Run ID": ["R1", "R2"],
                    "Well": ["W-1", "W-2"],
                    "Field": ["Ya", "Ya"],
                    "Contractor": ["Borets", "Novomet"],
                    "Event": [1, 0],
                    "Start Date": ["2024-01-01", "2024-02-01"],
                    "Stop Date": ["2024-02-20", "2024-03-10"],
                    "Gas Handling Type": ["gas separator", "none"],
                }
            ),
            "design": pd.DataFrame(
                {
                    "run_id": ["R1", "R2"],
                    "design_date": ["2023-12-15", "2024-01-20"],
                    "design_Qliq_m3d": [110, 90],
                    "design_Kpod": [0.8, 0.75],
                }
            ),
            "daily_or_monthly_regime": pd.DataFrame(
                {
                    "run_id": ["R1", "R1", "R2"],
                    "date": ["2024-01-10", "2024-02-10", "2024-02-20"],
                    "Qliq_m3d": [100, 95, 80],
                    "frequency_hz": [50, 53, 49],
                    "status": ["operating", "shutdown", "operating"],
                }
            ),
            "gas_limits": pd.DataFrame(
                {
                    "gas_handling_type": ["gas_separator"],
                    "free_gas_limit_fraction": [0.22],
                    "description": ["Override"],
                }
            ),
        }

        result = prepare_presentation_dataset(workbook, infant_threshold_days=30)

        self.assertIn("TTF_days", result.merged_df.columns)
        self.assertAlmostEqual(float(result.merged_df.loc[0, "TTF_days"]), 50.0, delta=0.01)
        self.assertAlmostEqual(float(result.merged_df.loc[1, "TTF_days"]), 38.0, delta=0.01)
        self.assertIn("design_Qliq_m3d", result.merged_df.columns)
        self.assertIn("regime_Qliq_m3d_mean", result.merged_df.columns)
        self.assertIn("regime_frequency_hz_last", result.merged_df.columns)
        self.assertIn("free_gas_limit_fraction", result.merged_df.columns)
        self.assertEqual(float(result.merged_df.loc[0, "free_gas_limit_fraction"]), 0.22)
        self.assertEqual(float(result.merged_df.loc[1, "free_gas_limit_fraction"]), 0.05)
        self.assertFalse(bool(result.merged_df.loc[0, "infant_mortality_flag"]))

    def test_prepare_presentation_dataset_can_add_last_30_day_regime_ratios(self):
        workbook = {
            "runs": pd.DataFrame(
                {
                    "Run ID": ["R1"],
                    "Well": ["W-1"],
                    "Field": ["Ya"],
                    "Contractor": ["Borets"],
                    "Event": [1],
                    "Start Date": ["2024-01-01"],
                    "Stop Date": ["2024-03-31"],
                }
            ),
            "daily_or_monthly_regime": pd.DataFrame(
                {
                    "run_id": ["R1", "R1", "R1"],
                    "date": ["2024-01-10", "2024-03-10", "2024-03-25"],
                    "Qliq_m3d": [100.0, 50.0, 50.0],
                    "frequency_hz": [50.0, 40.0, 40.0],
                    "status": ["operating", "operating", "operating"],
                }
            ),
        }

        result = prepare_presentation_dataset(
            workbook,
            infant_threshold_days=30,
            include_regime_last_30d_ratios=True,
        )

        self.assertIn("regime_Qliq_m3d_last30d_to_mean_ratio", result.merged_df.columns)
        self.assertIn("regime_frequency_hz_last30d_to_mean_ratio", result.merged_df.columns)
        self.assertAlmostEqual(float(result.merged_df.loc[0, "regime_Qliq_m3d_mean"]), 200.0 / 3.0, delta=0.01)
        self.assertAlmostEqual(float(result.merged_df.loc[0, "regime_Qliq_m3d_last30d_to_mean_ratio"]), 0.75, delta=0.01)
        qliq_mean_index = result.merged_df.columns.get_loc("regime_Qliq_m3d_mean")
        qliq_ratio_index = result.merged_df.columns.get_loc("regime_Qliq_m3d_last30d_to_mean_ratio")
        self.assertEqual(qliq_ratio_index, qliq_mean_index + 1)

    def test_prepare_presentation_dataset_limits_regime_ratios_to_target_runs(self):
        workbook = {
            "runs": pd.DataFrame(
                {
                    "Run ID": ["R1"],
                    "Well": ["W-1"],
                    "Field": ["Ya"],
                    "Contractor": ["Borets"],
                    "Event": [1],
                    "Start Date": ["2024-01-01"],
                    "Stop Date": ["2024-03-31"],
                }
            ),
            "daily_or_monthly_regime": pd.DataFrame(
                {
                    "run_id": ["R1", "R1", "R2", "R2"],
                    "date": ["2024-01-10", "2024-03-25", "2024-03-10", "2024-03-20"],
                    "Qliq_m3d": [100.0, 50.0, 1.0, 1.0],
                    "frequency_hz": [50.0, 40.0, 1.0, 1.0],
                    "status": ["operating", "operating", "operating", "operating"],
                }
            ),
        }

        result = prepare_presentation_dataset(
            workbook,
            infant_threshold_days=30,
            include_regime_last_30d_ratios=True,
        )

        self.assertEqual(len(result.merged_df), 1)
        self.assertAlmostEqual(float(result.merged_df.loc[0, "regime_Qliq_m3d_mean"]), 75.0, delta=0.01)
        self.assertAlmostEqual(float(result.merged_df.loc[0, "regime_Qliq_m3d_last30d_to_mean_ratio"]), 2.0 / 3.0, delta=0.01)

    def test_prepare_presentation_dataset_maps_russian_runtime_group_columns(self):
        workbook = {
            "runs": pd.DataFrame(
                {
                    "run_id": ["R1"],
                    "Event": [1],
                    "Наработка (сут)": [42],
                    "Месторождение": ["Ya"],
                    "Принадлежность": ["Борец"],
                }
            )
        }

        result = prepare_presentation_dataset(workbook, infant_threshold_days=30)

        self.assertEqual(result.suggested_event_column, "event")
        self.assertEqual(result.suggested_duration_column, "TTF_days")
        self.assertEqual(result.suggested_group_columns, ["field", "contractor"])
        self.assertEqual(float(result.merged_df.loc[0, "TTF_days"]), 42.0)
        self.assertEqual(str(result.merged_df.loc[0, "field"]), "Ya")
        self.assertEqual(str(result.merged_df.loc[0, "contractor"]), "Борец")

    def test_build_ttf_correlation_frame_ranks_numeric_features(self):
        df = pd.DataFrame(
            {
                "TTF_days": [10, 20, 30, 40, 50, 60],
                "regime_Qliq_m3d_mean": [1, 2, 3, 4, 5, 6],
                "design_Kpod": [6, 5, 4, 3, 2, 1],
                "event": [1, 1, 1, 0, 0, 0],
                "run_id": ["R1", "R2", "R3", "R4", "R5", "R6"],
            }
        )

        frame = build_ttf_correlation_frame(df, target_column="TTF_days", min_pairs=4)

        self.assertEqual(frame.iloc[0]["feature"], "design_Kpod")
        self.assertEqual(frame.iloc[0]["feature_group"], "design")
        self.assertEqual(int(frame.iloc[0]["valid_pairs"]), 6)
        self.assertAlmostEqual(float(frame.loc[frame["feature"] == "regime_Qliq_m3d_mean", "pearson_corr"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(frame.loc[frame["feature"] == "design_Kpod", "pearson_corr"].iloc[0]), -1.0)
        self.assertNotIn("event", frame["feature"].tolist())


if __name__ == "__main__":
    unittest.main()
