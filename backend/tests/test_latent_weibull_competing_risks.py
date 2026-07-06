import unittest

import numpy as np
import pandas as pd

from analysis.latent_weibull_competing_risks import (
    CompetingRiskModel,
    ThreeComponentLatentWeibullModel,
    build_survival_curve_weights,
    filter_km_frame_for_fit,
    fit_latent_weibull_to_survival_curve,
    fit_three_component_latent_weibull_to_survival_curve,
    fit_weibull_to_survival_curve,
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    competing_curve_frame,
    competing_next_window_probabilities,
    latent_next_window_failure_probability,
    latent_posterior_given_failure,
    latent_posterior_given_survival,
    latent_remaining_life_quantile,
    latent_survival,
    weibull_cdf,
    weibull_density,
    weibull_hazard,
    weibull_survival,
)


class LatentWeibullCompetingRiskTests(unittest.TestCase):
    def test_weibull_probability_identities_hold(self):
        params = WeibullParameters(beta=1.7, eta=180.0)
        times = np.linspace(1.0, 400.0, 50)

        survival = np.asarray(weibull_survival(times, params), dtype=float)
        cdf = np.asarray(weibull_cdf(times, params), dtype=float)
        density = np.asarray(weibull_density(times, params), dtype=float)
        hazard = np.asarray(weibull_hazard(times, params), dtype=float)

        self.assertTrue(np.allclose(survival + cdf, 1.0, atol=1e-10))
        self.assertTrue(np.allclose(hazard, density / survival, rtol=1e-8, atol=1e-10))

    def test_curve_weight_helpers_tail_emphasis_and_filtering_work(self):
        times = np.asarray([0.0, 50.0, 100.0, 150.0], dtype=float)
        n_risk = np.asarray([100.0, 60.0, 20.0, 4.0], dtype=float)
        weights = build_survival_curve_weights(
            times,
            n_risk=n_risk,
            weight_scheme="blended_tail",
            gamma=0.5,
            tail_lambda=1.0,
            tail_power=1.0,
        )
        self.assertEqual(weights.shape, times.shape)
        self.assertTrue(np.all(weights > 0.0))

        km_frame = np.asarray(
            [(0.0, 1.0, 100.0), (50.0, 0.8, 60.0), (100.0, 0.5, 20.0), (150.0, 0.3, 4.0)],
            dtype=[("time", float), ("survival", float), ("n_risk", float)],
        )
        filtered = filter_km_frame_for_fit(pd.DataFrame(km_frame), min_n_risk=5)
        self.assertEqual(len(filtered), 3)
        self.assertTrue((filtered["n_risk"] >= 5).all())

    def test_latent_posteriors_sum_to_one(self):
        model = TwoComponentLatentWeibullModel(
            weight_1=0.3,
            component_1=WeibullParameters(beta=0.9, eta=90.0, label="Early"),
            component_2=WeibullParameters(beta=1.8, eta=320.0, label="Normal"),
        )
        times = np.linspace(5.0, 300.0, 40)

        ps1, ps2 = latent_posterior_given_survival(times, model)
        pf1, pf2 = latent_posterior_given_failure(times, model)

        self.assertTrue(np.allclose(np.asarray(ps1) + np.asarray(ps2), 1.0, atol=1e-10))
        self.assertTrue(np.allclose(np.asarray(pf1) + np.asarray(pf2), 1.0, atol=1e-10))

    def test_latent_window_probability_and_remaining_life_are_well_formed(self):
        model = TwoComponentLatentWeibullModel(
            weight_1=0.45,
            component_1=WeibullParameters(beta=0.8, eta=110.0),
            component_2=WeibullParameters(beta=1.6, eta=360.0),
        )

        window_probability = latent_next_window_failure_probability(120.0, 90.0, model)
        remaining_life = latent_remaining_life_quantile(120.0, 0.5, model)

        self.assertGreaterEqual(window_probability, 0.0)
        self.assertLessEqual(window_probability, 1.0)
        self.assertTrue(np.isfinite(remaining_life))
        self.assertGreater(remaining_life, 0.0)
        self.assertLess(float(latent_survival(120.0 + remaining_life, model)), float(latent_survival(120.0, model)))

    def test_competing_risk_identity_holds_on_curve_grid(self):
        model = CompetingRiskModel(
            causes=(
                WeibullParameters(beta=1.3, eta=520.0, label="Electrical"),
                WeibullParameters(beta=1.8, eta=330.0, label="Mechanical"),
                WeibullParameters(beta=0.95, eta=760.0, label="Solids"),
            )
        )

        frame = competing_curve_frame(model, max_time=900.0, num_points=600)
        cif_columns = [column for column in frame.columns if column.startswith("cif_")]
        identity = frame["survival"].to_numpy() + frame[cif_columns].sum(axis=1).to_numpy()

        self.assertTrue(np.allclose(identity, 1.0, atol=5e-3))
        self.assertLess(frame["identity_error"].abs().max(), 5e-3)

    def test_competing_curve_frame_bounded_for_beta_below_one(self):
        """Regression (2026-07-06): with β<1 the density S·h is singular at t=0 and a
        trapezoid CIF diverged to ~27; the ΔH scheme must keep every CIF in [0, 1]."""
        model = CompetingRiskModel(
            causes=(
                WeibullParameters(beta=0.5, eta=90.0, label="Infant"),
                WeibullParameters(beta=1.7, eta=400.0, label="WearOut"),
            )
        )

        frame = competing_curve_frame(model, max_time=1200.0, num_points=500)
        for column in ("cif_1", "cif_2"):
            values = frame[column].to_numpy()
            self.assertGreaterEqual(values.min(), 0.0)
            self.assertLessEqual(values.max(), 1.0)
            self.assertTrue(np.all(np.diff(values) >= -1e-12), f"{column} must be monotone")
        self.assertLess(frame["identity_error"].abs().max(), 5e-3)

        # Cross-check against the independent B5 integrator (analysis.models.survival.cif).
        # Dense reference grid: its left-endpoint scheme overestimates on coarse grids
        # (first ΔH step is large for β<1), while the frame uses midpoint-averaged S.
        from analysis.models.survival.cif import parametric_competing_cif

        u, _, cifs = parametric_competing_cif([0.5, 1.7], [90.0, 400.0], max_time=1200.0, n_grid=40000)
        for column, reference in zip(("cif_1", "cif_2"), cifs):
            interpolated = np.interp(frame["time"].to_numpy(), u, reference)
            self.assertTrue(np.allclose(frame[column].to_numpy(), interpolated, atol=0.01))

    def test_competing_risk_window_probabilities_are_consistent(self):
        model = CompetingRiskModel(
            causes=(
                WeibullParameters(beta=1.4, eta=450.0, label="Electrical"),
                WeibullParameters(beta=1.1, eta=700.0, label="Mechanical"),
            )
        )

        window = competing_next_window_probabilities(150.0, 60.0, model)
        all_cause = float(window.attrs["all_cause_window_probability"])
        cause_sum = float(window["window_probability"].sum())

        self.assertGreaterEqual(all_cause, 0.0)
        self.assertLessEqual(all_cause, 1.0)
        self.assertAlmostEqual(cause_sum, all_cause, delta=5e-3)
        self.assertAlmostEqual(float(window["share_of_predicted_failures"].sum()), 1.0, delta=5e-6)

    def test_latent_curve_fit_matches_synthetic_target(self):
        target_model = TwoComponentLatentWeibullModel(
            weight_1=0.38,
            component_1=WeibullParameters(beta=0.78, eta=95.0, label="Short"),
            component_2=WeibullParameters(beta=1.85, eta=410.0, label="Long"),
        )
        times = np.linspace(30.0, 700.0, 40)
        survival = np.asarray(latent_survival(times, target_model), dtype=float)
        weights = np.linspace(100.0, 10.0, len(times))

        initial_model = TwoComponentLatentWeibullModel(
            weight_1=0.55,
            component_1=WeibullParameters(beta=1.10, eta=140.0, label="Short"),
            component_2=WeibullParameters(beta=1.40, eta=300.0, label="Long"),
        )
        fitted = fit_latent_weibull_to_survival_curve(
            times,
            survival,
            initial_model=initial_model,
            curve_weights=weights,
            max_iter=250,
        )

        self.assertTrue(fitted.success)
        self.assertLess(fitted.rmse, 0.03)
        self.assertLess(fitted.weighted_rmse, 0.02)
        fitted_survival = np.asarray(latent_survival(times, fitted.model), dtype=float)
        self.assertTrue(np.allclose(fitted_survival, survival, atol=0.05))

    def test_three_component_latent_curve_fit_matches_synthetic_target_and_orders_eta(self):
        target_model = ThreeComponentLatentWeibullModel(
            weight_1=0.18,
            weight_2=0.36,
            component_1=WeibullParameters(beta=0.72, eta=85.0, label="Class 1"),
            component_2=WeibullParameters(beta=1.25, eta=240.0, label="Class 2"),
            component_3=WeibullParameters(beta=2.10, eta=520.0, label="Class 3"),
        )
        times = np.linspace(20.0, 820.0, 55)
        survival = np.asarray(latent_survival(times, target_model), dtype=float)
        weights = np.linspace(160.0, 12.0, len(times))

        initial_model = ThreeComponentLatentWeibullModel(
            weight_1=0.40,
            weight_2=0.25,
            component_1=WeibullParameters(beta=1.80, eta=460.0, label="Class 1"),
            component_2=WeibullParameters(beta=0.95, eta=120.0, label="Class 2"),
            component_3=WeibullParameters(beta=1.35, eta=300.0, label="Class 3"),
        )
        fitted = fit_three_component_latent_weibull_to_survival_curve(
            times,
            survival,
            initial_model=initial_model,
            curve_weights=weights,
            num_starts=6,
            max_iter=350,
        )

        self.assertTrue(fitted.success)
        self.assertLess(fitted.rmse, 0.03)
        self.assertLess(fitted.weighted_rmse, 0.02)
        self.assertLessEqual(fitted.model.component_1.eta, fitted.model.component_2.eta)
        self.assertLessEqual(fitted.model.component_2.eta, fitted.model.component_3.eta)
        fitted_survival = np.asarray(latent_survival(times, fitted.model), dtype=float)
        self.assertTrue(np.allclose(fitted_survival, survival, atol=0.05))

    def test_classical_curve_fit_matches_synthetic_target(self):
        target_params = WeibullParameters(beta=1.65, eta=285.0, label="Classical")
        times = np.linspace(25.0, 650.0, 45)
        survival = np.asarray(weibull_survival(times, target_params), dtype=float)
        weights = np.linspace(120.0, 12.0, len(times))

        initial_params = WeibullParameters(beta=0.95, eta=180.0, label="Classical")
        fitted = fit_weibull_to_survival_curve(
            times,
            survival,
            initial_params=initial_params,
            curve_weights=weights,
            max_iter=250,
        )

        self.assertTrue(fitted.success)
        self.assertLess(fitted.rmse, 0.02)
        self.assertLess(fitted.weighted_rmse, 0.015)
        fitted_survival = np.asarray(weibull_survival(times, fitted.params), dtype=float)
        self.assertTrue(np.allclose(fitted_survival, survival, atol=0.03))

    def test_classical_curve_fit_respects_fixed_beta(self):
        target_params = WeibullParameters(beta=1.40, eta=320.0, label="Classical")
        times = np.linspace(25.0, 650.0, 45)
        survival = np.asarray(weibull_survival(times, target_params), dtype=float)

        initial_params = WeibullParameters(beta=1.10, eta=180.0, label="Classical")
        fitted = fit_weibull_to_survival_curve(
            times,
            survival,
            initial_params=initial_params,
            optimize_beta=False,
            optimize_eta=True,
            max_iter=250,
        )

        self.assertTrue(fitted.success)
        self.assertAlmostEqual(fitted.params.beta, initial_params.beta, places=9)
        self.assertNotAlmostEqual(fitted.params.eta, initial_params.eta, places=3)

    def test_latent_curve_fit_respects_fixed_component_weight(self):
        target_model = TwoComponentLatentWeibullModel(
            weight_1=0.42,
            component_1=WeibullParameters(beta=0.82, eta=105.0, label="Short"),
            component_2=WeibullParameters(beta=1.75, eta=395.0, label="Long"),
        )
        times = np.linspace(30.0, 700.0, 40)
        survival = np.asarray(latent_survival(times, target_model), dtype=float)

        initial_model = TwoComponentLatentWeibullModel(
            weight_1=0.55,
            component_1=WeibullParameters(beta=1.05, eta=150.0, label="Short"),
            component_2=WeibullParameters(beta=1.30, eta=260.0, label="Long"),
        )
        fitted = fit_latent_weibull_to_survival_curve(
            times,
            survival,
            initial_model=initial_model,
            optimize_weight_1=False,
            optimize_weight_2=False,
            max_iter=250,
        )

        self.assertTrue(fitted.success)
        self.assertAlmostEqual(fitted.model.weight_1, initial_model.weight_1, places=9)

    def test_latent_curve_fit_enforces_beta_split_bounds(self):
        target_model = TwoComponentLatentWeibullModel(
            weight_1=0.36,
            component_1=WeibullParameters(beta=0.88, eta=115.0, label="Short"),
            component_2=WeibullParameters(beta=1.62, eta=360.0, label="Long"),
        )
        times = np.linspace(30.0, 650.0, 45)
        survival = np.asarray(latent_survival(times, target_model), dtype=float)

        initial_model = TwoComponentLatentWeibullModel(
            weight_1=0.50,
            component_1=WeibullParameters(beta=1.40, eta=160.0, label="Short"),
            component_2=WeibullParameters(beta=0.70, eta=260.0, label="Long"),
        )
        fitted = fit_latent_weibull_to_survival_curve(
            times,
            survival,
            initial_model=initial_model,
            max_iter=250,
        )

        self.assertTrue(fitted.success)
        self.assertLessEqual(fitted.model.component_1.beta, 1.0 + 1e-9)
        self.assertGreaterEqual(fitted.model.component_2.beta, 1.0 - 1e-9)

    def test_latent_curve_fit_enforces_min_weight_floor(self):
        target_model = TwoComponentLatentWeibullModel(
            weight_1=0.02,
            component_1=WeibullParameters(beta=0.70, eta=90.0, label="Short"),
            component_2=WeibullParameters(beta=1.45, eta=300.0, label="Long"),
        )
        times = np.linspace(30.0, 650.0, 45)
        survival = np.asarray(latent_survival(times, target_model), dtype=float)

        initial_model = TwoComponentLatentWeibullModel(
            weight_1=0.30,
            component_1=WeibullParameters(beta=0.85, eta=120.0, label="Short"),
            component_2=WeibullParameters(beta=1.80, eta=420.0, label="Long"),
        )
        fitted = fit_latent_weibull_to_survival_curve(
            times,
            survival,
            initial_model=initial_model,
            min_weight_1=0.05,
            max_iter=250,
        )

        self.assertTrue(fitted.success)
        self.assertGreaterEqual(fitted.model.weight_1, 0.05 - 1e-9)

    def test_latent_curve_fit_preserves_component_identity_without_eta_sorting(self):
        model = TwoComponentLatentWeibullModel(
            weight_1=0.25,
            component_1=WeibullParameters(beta=0.80, eta=420.0, label="Short"),
            component_2=WeibullParameters(beta=1.60, eta=110.0, label="Long"),
        )
        times = np.linspace(30.0, 650.0, 45)
        survival = np.asarray(latent_survival(times, model), dtype=float)

        fitted = fit_latent_weibull_to_survival_curve(
            times,
            survival,
            initial_model=model,
            optimize_beta_1=False,
            optimize_eta_1=False,
            optimize_beta_2=False,
            optimize_eta_2=False,
            optimize_weight_1=False,
            optimize_weight_2=False,
        )

        self.assertEqual(fitted.model.component_1.label, "Short")
        self.assertEqual(fitted.model.component_2.label, "Long")
        self.assertAlmostEqual(fitted.model.component_1.eta, 420.0, places=9)
        self.assertAlmostEqual(fitted.model.component_2.eta, 110.0, places=9)

    def test_latent_curve_fit_reports_multistart_metadata(self):
        target_model = TwoComponentLatentWeibullModel(
            weight_1=0.34,
            component_1=WeibullParameters(beta=0.82, eta=95.0, label="Short"),
            component_2=WeibullParameters(beta=1.75, eta=390.0, label="Long"),
        )
        times = np.linspace(30.0, 650.0, 45)
        survival = np.asarray(latent_survival(times, target_model), dtype=float)
        initial_model = TwoComponentLatentWeibullModel(
            weight_1=0.60,
            component_1=WeibullParameters(beta=0.95, eta=160.0, label="Short"),
            component_2=WeibullParameters(beta=1.20, eta=600.0, label="Long"),
        )

        fitted = fit_latent_weibull_to_survival_curve(
            times,
            survival,
            initial_model=initial_model,
            num_starts=5,
            max_iter=200,
        )

        self.assertTrue(fitted.success)
        self.assertEqual(fitted.n_starts, 5)
        self.assertGreaterEqual(fitted.best_start_index, 0)
        self.assertLess(fitted.best_start_index, fitted.n_starts)


if __name__ == "__main__":
    unittest.main()
