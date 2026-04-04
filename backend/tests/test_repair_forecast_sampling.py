import unittest

import numpy as np

from app.services.repair_forecast import postprocess_runtime_prediction, sample_tail_runtime


class RepairForecastSamplingTests(unittest.TestCase):
    def test_prediction_kept_when_above_fact(self):
        values = np.array([10, 12, 15, 18, 22, 25], dtype=float)
        result = postprocess_runtime_prediction(30.0, 20.0, values, random_state=7)
        self.assertEqual(result, 30.0)

    def test_tail_sample_always_above_fact(self):
        values = np.array([10, 12, 15, 18, 22, 25, 28, 31, 35, 40], dtype=float)
        result = sample_tail_runtime(values, 20.0, random_state=7, min_group_size=4, max_iter=200)
        self.assertGreater(result, 20.0)

    def test_sampling_is_stable_with_random_state(self):
        values = np.array([8, 10, 14, 17, 21, 24, 28, 33, 39, 45], dtype=float)
        first = sample_tail_runtime(values, 20.0, random_state=11, min_group_size=4, max_iter=200)
        second = sample_tail_runtime(values, 20.0, random_state=11, min_group_size=4, max_iter=200)
        self.assertAlmostEqual(first, second, places=8)

    def test_fallback_works_for_small_group(self):
        values = np.array([10, 11], dtype=float)
        result = sample_tail_runtime(values, 10.5, random_state=3, min_group_size=20, max_iter=10)
        self.assertGreater(result, 10.5)

    def test_no_outliers_beyond_quantile_cap(self):
        values = np.array([5, 8, 13, 21, 34, 55, 89, 144, 233, 377], dtype=float)
        result = sample_tail_runtime(values, 100.0, random_state=5, min_group_size=4, max_iter=200)
        self.assertLessEqual(result, float(np.quantile(values, 0.995)))


if __name__ == "__main__":
    unittest.main()
