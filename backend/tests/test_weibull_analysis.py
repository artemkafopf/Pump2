import unittest

import numpy as np
import pandas as pd

from analysis.plotting import (
    _empirical_bin_frame,
    _factual_histogram_frame,
    build_empirical_sensitivity_figure,
    build_model_sensitivity_figure,
    build_model_sensitivity_frame,
    build_sensitivity_figure,
    build_sensitivity_range_split_frame,
)
from analysis.data_utils import apply_filters
from analysis.stress_transforms import relative_squared_deviation
from analysis.transform_selection import rank_transform_candidates
from analysis.weibull_model import fit_weibull_stress_model


class WeibullAnalysisTests(unittest.TestCase):
    def test_apply_filters_supports_numeric_and_string_rules(self):
        df = pd.DataFrame(
            {
                "runtime": [45, 90, 120, 180],
                "failure_type": ["Early", "Lifecycle", "Lifecycle", None],
            }
        )

        filtered = apply_filters(
            df,
            [
                {"column": "runtime", "operator": ">=", "value": 90},
                {"column": "failure_type", "operator": "not_contains", "value": "early"},
                {"column": "failure_type", "operator": "not_null", "value": None},
            ],
        )

        self.assertEqual(filtered["runtime"].tolist(), [90, 120])

    def test_prepare_modeling_ignores_non_binary_event_flags(self):
        df = pd.DataFrame(
            {
                "duration": [45, 60, 75, 90],
                "event": [1, 0, -1, 2],
            }
        )

        result = fit_weibull_stress_model(df, duration_column="duration", event_column="event")

        self.assertTrue(result.success)
        self.assertEqual(len(result.prepared_df), 2)
        self.assertEqual(sorted(result.prepared_df["event"].astype(int).tolist()), [0, 1])

    def test_relative_squared_deviation_handles_zero_reference_safely(self):
        values = relative_squared_deviation(np.array([10.0, 10.0]), np.array([5.0, 0.0]))
        self.assertAlmostEqual(values[0], 1.0)
        self.assertTrue(np.isnan(values[1]))

    def test_fit_weibull_model_recovers_simple_global_parameters(self):
        rng = np.random.default_rng(7)
        true_beta = 1.7
        true_eta = 120.0

        raw_times = true_eta * np.power(-np.log(rng.uniform(size=400)), 1.0 / true_beta)
        censor_time = 150.0
        observed = np.minimum(raw_times, censor_time)
        event = (raw_times <= censor_time).astype(int)

        df = pd.DataFrame({"duration": observed, "event": event})
        result = fit_weibull_stress_model(df, duration_column="duration", event_column="event")

        self.assertTrue(result.success)
        self.assertIn("GLOBAL", result.beta_by_group)
        self.assertIn("GLOBAL", result.eta_by_group)
        self.assertAlmostEqual(result.beta_by_group["GLOBAL"], true_beta, delta=0.35)
        self.assertAlmostEqual(result.eta_by_group["GLOBAL"], true_eta, delta=20.0)
        local_fit = result.fit_group_only_weibull("GLOBAL")
        self.assertAlmostEqual(float(local_fit["beta"]), true_beta, delta=0.35)
        self.assertAlmostEqual(float(local_fit["eta"]), true_eta, delta=20.0)

    def test_fit_weibull_model_uses_group_specific_beta(self):
        rng = np.random.default_rng(11)
        beta_a = 0.9
        eta_a = 140.0
        beta_b = 2.2
        eta_b = 210.0

        raw_a = eta_a * np.power(-np.log(rng.uniform(size=260)), 1.0 / beta_a)
        raw_b = eta_b * np.power(-np.log(rng.uniform(size=260)), 1.0 / beta_b)
        censor_time = 240.0

        df = pd.DataFrame(
            {
                "duration": np.concatenate([np.minimum(raw_a, censor_time), np.minimum(raw_b, censor_time)]),
                "event": np.concatenate([(raw_a <= censor_time).astype(int), (raw_b <= censor_time).astype(int)]),
                "group": ["A"] * len(raw_a) + ["B"] * len(raw_b),
            }
        )

        result = fit_weibull_stress_model(
            df,
            duration_column="duration",
            event_column="event",
            group_columns=["group"],
            min_group_size=1,
        )

        self.assertIn("A", result.beta_by_group)
        self.assertIn("B", result.beta_by_group)
        self.assertNotAlmostEqual(result.beta_by_group["A"], result.beta_by_group["B"], places=1)

    def test_group_stats_expose_bucket_and_source_rows(self):
        df = pd.DataFrame(
            {
                "duration": [40, 50, 60, 70, 80, 90, 100, 110],
                "event": [1, 0, 1, 0, 1, 0, 1, 0],
                "field": ["Ya", "Ya", "Ya", "Ya", "Ya", "Ya", "Ya", "Ya"],
                "contractor": ["A", "A", "A", "B", "B", "B", "C", "C"],
            }
        )

        result = fit_weibull_stress_model(
            df,
            duration_column="duration",
            event_column="event",
            group_columns=["field", "contractor"],
            min_group_size=3,
        )

        self.assertIn("Ya | C", result.group_stats)
        self.assertEqual(result.group_stats["Ya | C"]["bucket_rows"], 2)
        self.assertEqual(result.group_stats["Ya | C"]["source_rows"], 2)
        self.assertEqual(len(result.group_source_rows("Ya | C")), 2)
        self.assertEqual(result.group_stats["Ya | C"]["eta_group"], "Ya")

    def test_empirical_bin_frame_outputs_probabilities(self):
        df = pd.DataFrame(
            {
                "duration": [20, 25, 30, 35, 40, 45, 50, 55],
                "event": [1, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        frame = _empirical_bin_frame(df, "duration", "event")

        self.assertFalse(frame.empty)
        self.assertTrue(frame["survival"].between(0.0, 1.0).all())
        self.assertTrue(frame["failure"].between(0.0, 1.0).all())
        self.assertTrue((frame["width"] == 30.0).all())

    def test_factual_histogram_frame_keeps_event_buckets_separate(self):
        df = pd.DataFrame(
            {
                "duration": [10, 20, 40, 50, 70, 80],
                "event": [1, 0, -1, 1, 0, -1],
            }
        )

        frame = _factual_histogram_frame(df, "duration", "event")

        self.assertFalse(frame.empty)
        self.assertEqual(int(frame["failures"].sum()), 2)
        self.assertEqual(int(frame["censored_zero"].sum()), 2)
        self.assertEqual(int(frame["other_flags"].sum()), 2)
        self.assertTrue((frame["total"] == (frame["failures"] + frame["censored_zero"] + frame["other_flags"])).all())

    def test_sensitivity_range_split_uses_threshold_ranges_for_single_value(self):
        df = pd.DataFrame({"stress": [10, 20, 30, 40, 50, 60]})

        split = build_sensitivity_range_split_frame(df, "stress", [35])

        self.assertEqual(
            split["sensitivity_range_label"].tolist(),
            ["x <= 35", "x <= 35", "x <= 35", "x > 35", "x > 35", "x > 35"],
        )

    def test_sensitivity_range_split_uses_ordered_threshold_ranges(self):
        df = pd.DataFrame({"stress": [10, 20, 30, 40, 50, 60]})

        split = build_sensitivity_range_split_frame(df, "stress", [25, 45])

        self.assertEqual(
            split["sensitivity_range_label"].tolist(),
            ["x <= 25", "x <= 25", "25 < x <= 45", "25 < x <= 45", "x > 45", "x > 45"],
        )

    def test_sensitivity_model_and_empirical_figures_are_both_available(self):
        rng = np.random.default_rng(21)
        durations = np.linspace(30, 240, 80)
        events = (durations < 180).astype(int)
        stress_values = np.clip(rng.normal(0.8, 0.12, size=80), 0.3, 1.2)
        df = pd.DataFrame(
            {
                "duration": durations,
                "event": events,
                "group": ["A"] * 80,
                "Kpod": stress_values,
            }
        )

        result = fit_weibull_stress_model(
            df,
            duration_column="duration",
            event_column="event",
            group_columns=["group"],
            stress_terms=[
                {
                    "name": "stress_low_Kpod",
                    "column": "Kpod",
                    "transform": "negative_excess",
                    "reference_mode": "fixed",
                    "reference_value": 0.7,
                    "coefficient_mode": "fit",
                    "coefficient_value": 0.05,
                    "coefficient_non_negative": True,
                    "coefficient_bounds": [0.0, None],
                    "scale": 0.1,
                }
            ],
            min_group_size=1,
        )

        model_frame = build_model_sensitivity_frame(result, "A", "stress_low_Kpod", [0.6, 0.8, 1.0])
        combined_fig = build_sensitivity_figure(result, "A", "stress_low_Kpod", [0.6, 0.8, 1.0])
        model_fig = build_model_sensitivity_figure(result, "A", "stress_low_Kpod", [0.6, 0.8, 1.0])
        empirical_fig = build_empirical_sensitivity_figure(result, "A", "stress_low_Kpod", [0.6, 0.8, 1.0])

        self.assertEqual(model_frame["scenario_value"].round(3).tolist(), [0.6, 0.8, 1.0])
        self.assertGreaterEqual(len(combined_fig.data), 7)
        self.assertEqual(len(model_fig.data), 6)
        self.assertGreaterEqual(len(empirical_fig.data), 1)

    def test_rank_transform_candidates_returns_ranked_results(self):
        df = pd.DataFrame(
            {
                "duration": [40, 55, 60, 75, 90, 110, 130, 150],
                "event": [1, 1, 0, 1, 0, 1, 0, 1],
                "group": ["A"] * 8,
                "Kpod": [0.45, 0.5, 0.58, 0.66, 0.72, 0.8, 0.88, 0.96],
            }
        )

        ranked = rank_transform_candidates(
            df,
            duration_column="duration",
            event_column="event",
            column="Kpod",
            group_columns=["group"],
            candidate_transforms=["negative_excess", "relative_negative_excess"],
            min_group_size=1,
        )

        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(item.transform in {"negative_excess", "relative_negative_excess"} for item in ranked))
        self.assertTrue(all(isinstance(item.success, bool) for item in ranked))


if __name__ == "__main__":
    unittest.main()
