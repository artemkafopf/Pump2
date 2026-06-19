import unittest

import pandas as pd

from analysis.stress_correlations import build_grouped_stress_correlation_frame, build_stress_term_observation_frame
from analysis.weibull_model import fit_weibull_stress_model


class StressCorrelationTests(unittest.TestCase):
    def test_build_stress_term_observation_frame_and_grouped_summary(self):
        df = pd.DataFrame(
            {
                "duration": [30, 40, 50, 60, 70, 80],
                "event": [1, 1, 0, 1, 0, 0],
                "field": ["Ya", "Ya", "Ya", "Bb", "Bb", "Bb"],
                "Kpod": [0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
            }
        )

        result = fit_weibull_stress_model(
            df,
            duration_column="duration",
            event_column="event",
            group_columns=["field"],
            min_group_size=1,
            stress_terms=[
                {
                    "name": "stress_low_Kpod",
                    "column": "Kpod",
                    "transform": "negative_excess",
                    "reference_mode": "fixed",
                    "reference_value": 0.7,
                    "coefficient_mode": "fixed",
                    "coefficient_value": 1.0,
                    "scale": 0.1,
                }
            ],
        )

        observation = build_stress_term_observation_frame(result, "stress_low_Kpod", target_column="duration")
        grouped = build_grouped_stress_correlation_frame(observation, value_column="weighted_stress_value", group_column="field")

        self.assertIn("weighted_stress_value", observation.columns)
        self.assertIn("target_value", observation.columns)
        self.assertIn("ALL", grouped["group"].tolist())
        self.assertIn("Ya", grouped["group"].tolist())
        self.assertTrue((observation["weighted_stress_value"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
