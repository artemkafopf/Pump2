import numpy as np
import pandas as pd
import unittest

from backend.analysis.regularized_logistic import (
    RegularizedLogisticConfig,
    fit_regularized_logistic_model,
    predict_regularized_logistic_from_table,
)


class RegularizedLogisticTests(unittest.TestCase):
    def test_numeric_signal_recovers_positive_direction(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=400)
        logits = -0.4 + 1.3 * x
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, probabilities)

        df = pd.DataFrame({"row_id": np.arange(len(x)), "signal": x, "target": y})
        result = fit_regularized_logistic_model(
            df,
            target_column="target",
            numeric_features=["signal"],
            categorical_features=[],
            id_column="row_id",
            config=RegularizedLogisticConfig(l2_penalty=0.1, max_iter=200),
        )

        signal_row = result.coefficient_table.loc[result.coefficient_table["term"] == "signal"].iloc[0]
        self.assertGreater(float(signal_row["coefficient"]), 0.0)
        self.assertTrue(bool(result.metrics["converged"]))
        ordered = result.prediction_table.sort_values("predicted_probability")
        self.assertGreater(float(ordered["predicted_probability"].iloc[-1]), float(ordered["predicted_probability"].iloc[0]))

    def test_categorical_and_missing_terms_are_encoded(self):
        df = pd.DataFrame(
            {
                "row_id": np.arange(8),
                "signal": [1.0, 2.0, np.nan, 0.0, 3.0, np.nan, 4.0, 5.0],
                "category": ["A", "A", "B", "B", "B", None, "A", "B"],
                "target": [0, 0, 0, 1, 1, 0, 1, 1],
            }
        )
        result = fit_regularized_logistic_model(
            df,
            target_column="target",
            numeric_features=["signal"],
            categorical_features=["category"],
            id_column="row_id",
            config=RegularizedLogisticConfig(l2_penalty=0.5, max_iter=200, min_category_rows=1),
        )

        terms = set(result.coefficient_table["term"].tolist())
        self.assertIn("signal", terms)
        self.assertIn("signal__missing", terms)
        self.assertTrue(any(term.startswith("category[") for term in terms))
        self.assertTrue(np.isfinite(result.prediction_table["predicted_probability"]).all())

    def test_prediction_helper_reproduces_fit_probabilities(self):
        df = pd.DataFrame(
            {
                "row_id": np.arange(10),
                "signal": [0.2, 0.5, 1.0, np.nan, -0.2, 0.0, 0.8, 1.5, np.nan, -1.0],
                "category": ["A", "B", "A", "B", "A", None, "B", "A", "C", "C"],
                "target": [0, 0, 1, 0, 0, 1, 1, 1, 0, 0],
            }
        )
        result = fit_regularized_logistic_model(
            df,
            target_column="target",
            numeric_features=["signal"],
            categorical_features=["category"],
            id_column="row_id",
            config=RegularizedLogisticConfig(l2_penalty=0.5, max_iter=200, min_category_rows=1),
        )
        scored = predict_regularized_logistic_from_table(df, result.coefficient_table)
        self.assertTrue(
            np.allclose(
                scored["predicted_probability"].to_numpy(dtype=float),
                result.prediction_table["predicted_probability"].to_numpy(dtype=float),
                atol=1e-10,
            )
        )


if __name__ == "__main__":
    unittest.main()
