"""
Tests for the Bayesian Latent Weibull Mixture implementation.

Covers:
- Likelihood correctness: failure → density term; censored → survival term.
- Posterior probability normalisation.
- KM / validation utilities.
- Simulation recovery for K=1 and K=2 models.
- Birth-death MCMC run (smoke + K distribution).
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from analysis.bayesian_latent_weibull import (
    BayesianLatentWeibullConfig,
    _log_weibull_density,
    _log_weibull_survival,
    fit_bayesian_latent_weibull,
    kaplan_meier_frame,
    mixture_log_likelihood,
    validate_survival_dataframe,
    weibull_density,
    weibull_survival,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(durations, events, ids=None):
    n = len(durations)
    return pd.DataFrame(
        {
            "id": ids if ids is not None else [f"unit_{i}" for i in range(n)],
            "duration": durations,
            "event": events,
        }
    )


def _simulate_weibull(n, beta, eta, censoring_frac, rng):
    """Simulate Weibull lifetimes with independent uniform censoring."""
    lifetimes = eta * rng.weibull(beta, size=n)
    max_life = float(np.quantile(lifetimes, 1.0 - censoring_frac / 2.0))
    censor_times = rng.uniform(0.0, max_life, size=n)
    durations = np.minimum(lifetimes, censor_times)
    events = (lifetimes <= censor_times).astype(int)
    durations = np.clip(durations, 1e-6, None)
    return durations, events


def _simulate_two_component(n, beta1, eta1, w1, beta2, eta2, rng, censoring_frac=0.20):
    """Simulate a 2-component Weibull mixture."""
    n1 = int(round(n * w1))
    n2 = n - n1
    t1 = eta1 * rng.weibull(beta1, size=n1)
    t2 = eta2 * rng.weibull(beta2, size=n2)
    lifetimes = np.concatenate([t1, t2])
    max_life = float(np.quantile(lifetimes, 1.0 - censoring_frac / 2.0))
    censor_times = rng.uniform(0.0, max_life, size=n)
    durations = np.clip(np.minimum(lifetimes, censor_times), 1e-6, None)
    events = (lifetimes <= censor_times).astype(int)
    return durations, events


# ---------------------------------------------------------------------------
# 1. Likelihood correctness
# ---------------------------------------------------------------------------

class LikelihoodCorrectnessTests(unittest.TestCase):

    def test_failure_observation_contributes_density(self):
        """event=1 -> log f(t) term, not log S(t)."""
        beta, eta = 1.5, 200.0
        t = 100.0
        expected = float(np.log(weibull_density(np.asarray([t]), beta=beta, eta=eta)[0]))
        computed = mixture_log_likelihood([t], [1], [1.0], [beta], [eta])
        self.assertAlmostEqual(computed, expected, places=10)

    def test_censored_observation_contributes_survival(self):
        """event=0 -> log S(t) term, not log f(t)."""
        beta, eta = 1.5, 200.0
        t = 150.0
        expected = float(np.log(weibull_survival(np.asarray([t]), beta=beta, eta=eta)[0]))
        computed = mixture_log_likelihood([t], [0], [1.0], [beta], [eta])
        self.assertAlmostEqual(computed, expected, places=10)

    def test_mixed_dataset_sum_is_correct(self):
        """Two observations: one failure, one censored."""
        beta, eta = 1.5, 200.0
        t_fail, t_cens = 100.0, 150.0

        log_density = float(np.log(weibull_density(np.asarray([t_fail]), beta, eta)[0]))
        log_survival = float(np.log(weibull_survival(np.asarray([t_cens]), beta, eta)[0]))
        expected = log_density + log_survival

        computed = mixture_log_likelihood([t_fail, t_cens], [1, 0], [1.0], [beta], [eta])
        self.assertAlmostEqual(computed, expected, places=10)

    def test_swapping_event_codes_changes_likelihood(self):
        """Treating a failure as censored gives a different likelihood."""
        beta, eta = 1.5, 200.0
        t = 100.0
        ll_failure = mixture_log_likelihood([t], [1], [1.0], [beta], [eta])
        ll_censored = mixture_log_likelihood([t], [0], [1.0], [beta], [eta])
        self.assertNotAlmostEqual(ll_failure, ll_censored, places=5)

    def test_all_censored_dataset_is_finite(self):
        """If all observations are censored the log-likelihood should be finite."""
        rng = np.random.default_rng(7)
        beta, eta = 1.2, 300.0
        n = 30
        t = rng.uniform(50.0, 400.0, size=n)
        ll = mixture_log_likelihood(t.tolist(), [0] * n, [1.0], [beta], [eta])
        self.assertTrue(np.isfinite(ll))

    def test_mixture_likelihood_weights_must_sum_to_one(self):
        """Weights not summing to 1 should raise ValueError."""
        with self.assertRaises(ValueError):
            mixture_log_likelihood([100.0], [1], [0.4, 0.4], [1.5, 2.0], [200.0, 300.0])

    def test_two_component_mixture_log_lik_is_finite(self):
        rng = np.random.default_rng(0)
        n = 50
        durations, events = _simulate_two_component(n, 0.8, 100.0, 0.4, 1.5, 300.0, rng)
        ll = mixture_log_likelihood(durations, events, [0.4, 0.6], [0.8, 1.5], [100.0, 300.0])
        self.assertTrue(np.isfinite(ll))

    def test_log_weibull_density_and_survival_are_consistent(self):
        """log f(t) = log h(t) + log S(t) for Weibull."""
        beta, eta = 1.8, 250.0
        t = np.linspace(10.0, 500.0, 20)
        log_f = _log_weibull_density(t, beta, eta)
        log_S = _log_weibull_survival(t, beta, eta)
        log_h = np.log(beta / eta) + (beta - 1.0) * (np.log(t) - np.log(eta))
        self.assertTrue(np.allclose(log_f, log_h + log_S, atol=1e-10))


# ---------------------------------------------------------------------------
# 2. Validation utilities
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):

    def test_non_positive_duration_removed(self):
        df = _make_df([100, -1, 0, 200], [1, 1, 0, 0])
        cleaned, report = validate_survival_dataframe(
            df, duration_column="duration", event_column="event", id_column="id"
        )
        self.assertEqual(report.removed_rows, 2)
        self.assertTrue((cleaned["duration"] > 0).all())

    def test_invalid_event_removed(self):
        df = _make_df([100, 200, 300], [1, 2, 0])
        cleaned, report = validate_survival_dataframe(
            df, duration_column="duration", event_column="event"
        )
        self.assertEqual(len(cleaned), 2)

    def test_failure_and_censored_counts_correct(self):
        df = _make_df([100, 200, 300, 400], [1, 1, 0, 0])
        _, report = validate_survival_dataframe(
            df, duration_column="duration", event_column="event"
        )
        self.assertEqual(report.failure_count, 2)
        self.assertEqual(report.censored_count, 2)

    def test_duplicate_ids_removed(self):
        df = _make_df([100, 200, 300], [1, 0, 1], ids=["a", "a", "b"])
        cleaned, report = validate_survival_dataframe(
            df, duration_column="duration", event_column="event", id_column="id"
        )
        self.assertNotIn("a", cleaned["id"].tolist())

    def test_date_derived_duration(self):
        df = pd.DataFrame(
            {
                "id": ["x", "y"],
                "start": ["2023-01-01", "2023-03-01"],
                "end": ["2023-04-01", "2023-06-01"],
                "event": [1, 0],
            }
        )
        cleaned, _ = validate_survival_dataframe(
            df,
            duration_column=None,
            event_column="event",
            start_date_column="start",
            end_date_column="end",
        )
        self.assertAlmostEqual(cleaned["duration"].iloc[0], 90.0, delta=1.0)


# ---------------------------------------------------------------------------
# 3. Kaplan-Meier
# ---------------------------------------------------------------------------

class KaplanMeierTests(unittest.TestCase):

    def test_km_survival_monotone_decreasing(self):
        rng = np.random.default_rng(1)
        durations, events = _simulate_weibull(100, 1.5, 200.0, 0.20, rng)
        km = kaplan_meier_frame(durations, events)
        s = km["survival"].to_numpy()
        self.assertTrue(np.all(np.diff(s) <= 1e-12))

    def test_km_survival_starts_at_one(self):
        durations = [50.0, 100.0, 200.0]
        events = [0, 1, 1]
        km = kaplan_meier_frame(durations, events)
        self.assertEqual(float(km["survival"].iloc[0]), 1.0)

    def test_km_all_censored_stays_one(self):
        durations = [50.0, 100.0, 200.0]
        events = [0, 0, 0]
        km = kaplan_meier_frame(durations, events)
        self.assertTrue((km["survival"] == 1.0).all())


# ---------------------------------------------------------------------------
# 4. Fixed-K Bayesian fitter -- basic integration
# ---------------------------------------------------------------------------

class FixedKFitterIntegrationTests(unittest.TestCase):

    def _fast_config(self, k):
        return BayesianLatentWeibullConfig(
            n_components=k,
            n_iter=600,
            burn_in=200,
            thin=2,
            n_chains=1,
            random_seed=99,
        )

    def test_fit_runs_without_error(self):
        rng = np.random.default_rng(10)
        durations, events = _simulate_weibull(60, 1.5, 200.0, 0.15, rng)
        df = _make_df(durations, events)
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=self._fast_config(2)
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result.chains), 1)

    def test_posterior_probabilities_sum_to_one(self):
        rng = np.random.default_rng(11)
        durations, events = _simulate_weibull(50, 1.5, 200.0, 0.20, rng)
        df = _make_df(durations, events)
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=self._fast_config(2)
        )
        probs = result.observation_posterior_probabilities()
        prob_cols = [c for c in probs.columns if c.endswith("_probability") and "component" in c]
        row_sums = probs[prob_cols].sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))

    def test_active_unit_predictions_in_0_1(self):
        rng = np.random.default_rng(12)
        durations, events = _simulate_weibull(60, 1.5, 200.0, 0.30, rng)
        df = _make_df(durations, events)
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=self._fast_config(1)
        )
        preds = result.active_unit_predictions(horizons=(30,))
        if not preds.empty:
            means = preds["next_30_day_failure_mean"].to_numpy()
            self.assertTrue((means >= 0.0).all())
            self.assertTrue((means <= 1.0).all())

    def test_survival_frame_starts_near_one(self):
        rng = np.random.default_rng(13)
        durations, events = _simulate_weibull(50, 1.5, 200.0, 0.20, rng)
        df = _make_df(durations, events)
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=self._fast_config(2)
        )
        sf = result.posterior_survival_frame(num_points=50)
        self.assertAlmostEqual(float(sf["survival_mean"].iloc[0]), 1.0, delta=0.01)

    def test_b50_life_quantile_is_positive(self):
        rng = np.random.default_rng(14)
        durations, events = _simulate_weibull(60, 1.5, 200.0, 0.15, rng)
        df = _make_df(durations, events)
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=self._fast_config(1)
        )
        q_frame = result.posterior_life_quantiles_frame(quantiles=(0.50,))
        b50_mean = float(q_frame["mean"].iloc[0])
        self.assertGreater(b50_mean, 0.0)
        self.assertTrue(np.isfinite(b50_mean))

    def test_mean_life_draws_are_positive(self):
        rng = np.random.default_rng(15)
        durations, events = _simulate_weibull(50, 1.2, 180.0, 0.20, rng)
        df = _make_df(durations, events)
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=self._fast_config(1)
        )
        mean_lives = result.posterior_mixture_mean_life_draws()
        self.assertTrue((mean_lives > 0.0).all())
        self.assertTrue(np.isfinite(mean_lives).all())

    def test_diagnostics_frame_has_expected_columns(self):
        rng = np.random.default_rng(16)
        durations, events = _simulate_weibull(50, 1.5, 200.0, 0.20, rng)
        df = _make_df(durations, events)
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=self._fast_config(2)
        )
        diag = result.diagnostics_frame()
        for col in ("chain", "saved_draws", "acceptance_eta_mean", "acceptance_beta_mean"):
            self.assertIn(col, diag.columns)


# ---------------------------------------------------------------------------
# 5. Simulation recovery -- K=1 single Weibull
# ---------------------------------------------------------------------------

class SingleWeibullRecoveryTests(unittest.TestCase):
    """Fit K=1 on pure Weibull data; posterior means should be near true params."""

    BETA_TRUE = 1.5
    ETA_TRUE = 200.0
    N = 200

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        durations, events = _simulate_weibull(cls.N, cls.BETA_TRUE, cls.ETA_TRUE, 0.15, rng)
        df = _make_df(durations, events)
        config = BayesianLatentWeibullConfig(
            n_components=1,
            n_iter=3_000,
            burn_in=1_000,
            thin=3,
            n_chains=1,
            mu_log_eta=5.0,
            sigma_log_eta=2.0,
            mu_log_beta=0.0,
            sigma_log_beta=1.0,
            random_seed=42,
        )
        cls.result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=config
        )

    def test_eta_posterior_mean_near_truth(self):
        eta_draws = self.result.eta_draws[:, 0]
        eta_mean = float(np.mean(eta_draws))
        self.assertAlmostEqual(eta_mean, self.ETA_TRUE, delta=self.ETA_TRUE * 0.30)

    def test_beta_posterior_mean_near_truth(self):
        beta_draws = self.result.beta_draws[:, 0]
        beta_mean = float(np.mean(beta_draws))
        self.assertAlmostEqual(beta_mean, self.BETA_TRUE, delta=self.BETA_TRUE * 0.35)

    def test_weight_is_essentially_one(self):
        w_draws = self.result.weights_draws[:, 0]
        self.assertGreater(float(np.mean(w_draws)), 0.95)

    def test_posterior_survival_covers_km(self):
        """KM estimate should broadly fall within the 95% credible band."""
        km = kaplan_meier_frame(
            self.result.cleaned_df[self.result.duration_column].to_numpy(),
            self.result.cleaned_df[self.result.event_column].to_numpy(),
        )
        sf = self.result.posterior_survival_frame(num_points=50)
        outside = 0
        for _, row in km.iterrows():
            t = float(row["time"])
            km_s = float(row["survival"])
            idx = int(np.argmin(np.abs(sf["time"].to_numpy() - t)))
            lo = float(sf["survival_q2_5"].iloc[idx])
            hi = float(sf["survival_q97_5"].iloc[idx])
            if not (lo - 0.10 <= km_s <= hi + 0.10):
                outside += 1
        # Allow at most 15% of time-points to be outside the widened band
        self.assertLess(outside / max(len(km), 1), 0.15)


# ---------------------------------------------------------------------------
# 6. Simulation recovery -- K=2 two-component mixture
# ---------------------------------------------------------------------------

class TwoComponentRecoveryTests(unittest.TestCase):
    """Fit K=2 on a 2-component mixture; parameters should be recoverable."""

    BETA1, ETA1, W1 = 0.8, 90.0, 0.35
    BETA2, ETA2 = 1.6, 300.0
    N = 250

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(55)
        durations, events = _simulate_two_component(
            cls.N, cls.BETA1, cls.ETA1, cls.W1, cls.BETA2, cls.ETA2, rng, censoring_frac=0.20
        )
        df = _make_df(durations, events)
        config = BayesianLatentWeibullConfig(
            n_components=2,
            n_iter=4_000,
            burn_in=1_500,
            thin=4,
            n_chains=1,
            mu_log_eta=4.5,
            sigma_log_eta=2.0,
            mu_log_beta=0.0,
            sigma_log_beta=1.0,
            random_seed=55,
        )
        cls.result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=config
        )

    def test_component_eta_ordering(self):
        """After relabeling, component 1 should have smaller median life."""
        summary = self.result.posterior_summary_frame()
        eta1_row = summary.loc[(summary["component"] == "component_1") & (summary["parameter"] == "eta")]
        eta2_row = summary.loc[(summary["component"] == "component_2") & (summary["parameter"] == "eta")]
        self.assertLess(float(eta1_row["mean"].iloc[0]), float(eta2_row["mean"].iloc[0]))

    def test_weight_sum_near_one(self):
        w_sum = self.result.weights_draws.sum(axis=1)
        self.assertTrue(np.allclose(w_sum, 1.0, atol=1e-8))

    def test_posterior_summary_has_two_components(self):
        summary = self.result.posterior_summary_frame()
        components = summary["component"].unique()
        self.assertIn("component_1", components)
        self.assertIn("component_2", components)

    def test_censoring_fraction_matches_data(self):
        report = self.result.validation_report
        total = report.failure_count + report.censored_count
        cens_frac = report.censored_count / total
        self.assertGreater(cens_frac, 0.05)
        self.assertLess(cens_frac, 0.50)


# ---------------------------------------------------------------------------
# 7. High-censoring scenario (70%)
# ---------------------------------------------------------------------------

class HighCensoringTests(unittest.TestCase):

    def test_high_censoring_fit_completes(self):
        rng = np.random.default_rng(77)
        durations, events = _simulate_weibull(100, 1.5, 200.0, 0.70, rng)
        df = _make_df(durations, events)
        config = BayesianLatentWeibullConfig(
            n_components=1, n_iter=600, burn_in=200, thin=2, n_chains=1, random_seed=77
        )
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=config
        )
        self.assertGreater(result.validation_report.censored_count, 50)
        sf = result.posterior_survival_frame(num_points=30)
        self.assertTrue(np.isfinite(sf["survival_mean"].to_numpy()).all())


# ---------------------------------------------------------------------------
# 8. Birth-death MCMC -- smoke tests
# ---------------------------------------------------------------------------

class BirthDeathMCMCTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(88)
        durations, events = _simulate_two_component(
            150, 0.8, 90.0, 0.4, 1.5, 280.0, rng, censoring_frac=0.20
        )
        cls.df = _make_df(durations, events)
        cls.config = BayesianLatentWeibullConfig(
            n_components=2,
            use_unknown_k=True,
            k_max=6,
            lambda_k=3.0,
            birth_death_probability=0.35,
            n_iter=1_500,
            burn_in=500,
            thin=5,
            n_chains=1,
            random_seed=88,
        )
        cls.result = fit_bayesian_latent_weibull(
            cls.df,
            duration_column="duration",
            event_column="event",
            config=cls.config,
        )

    def test_chains_produced(self):
        self.assertEqual(len(self.result.chains), 1)

    def test_draws_are_non_empty(self):
        self.assertGreater(len(self.result.k_draws), 0)

    def test_posterior_k_distribution_sums_to_one(self):
        k_frame = self.result.posterior_k_frame()
        self.assertAlmostEqual(float(k_frame["posterior_probability"].sum()), 1.0, places=10)

    def test_posterior_k_values_within_range(self):
        k_frame = self.result.posterior_k_frame()
        self.assertTrue((k_frame["K"] >= 1).all())
        self.assertTrue((k_frame["K"] <= self.config.k_max).all())

    def test_k_trace_length(self):
        expected_draws = (self.config.n_iter - self.config.burn_in) // self.config.thin
        self.assertEqual(len(self.result.chains[0].k_draws), expected_draws)

    def test_survival_frame_is_well_formed(self):
        sf = self.result.posterior_survival_frame(num_points=40)
        self.assertAlmostEqual(float(sf["survival_mean"].iloc[0]), 1.0, delta=0.02)
        self.assertTrue((sf["survival_mean"].to_numpy() >= 0.0).all())
        self.assertTrue((sf["survival_mean"].to_numpy() <= 1.0 + 1e-8).all())

    def test_life_quantiles_are_positive(self):
        q_frame = self.result.posterior_life_quantiles_frame(quantiles=(0.50,))
        self.assertGreater(float(q_frame["mean"].iloc[0]), 0.0)

    def test_draws_weights_sum_to_one(self):
        self.assertTrue(np.allclose(self.result.weights_draws.sum(axis=1), 1.0, atol=1e-8))

    def test_diagnostics_has_birth_death_counts(self):
        diag = self.result.diagnostics_frame()
        self.assertIn("birth_attempts", diag.columns)
        self.assertIn("death_attempts", diag.columns)

    def test_bd_births_and_deaths_occurred(self):
        chain = self.result.chains[0]
        total_attempts = chain.birth_attempts + chain.death_attempts
        self.assertGreater(total_attempts, 0, "No birth/death attempts occurred.")


# ---------------------------------------------------------------------------
# 9. K=3 three-component smoke test
# ---------------------------------------------------------------------------

class ThreeComponentSmokeTest(unittest.TestCase):

    def test_three_component_fit_runs(self):
        rng = np.random.default_rng(33)
        n = 180
        n1, n2, n3 = 60, 80, 40
        t1 = 80.0 * rng.weibull(0.7, size=n1)
        t2 = 220.0 * rng.weibull(1.5, size=n2)
        t3 = 500.0 * rng.weibull(2.5, size=n3)
        lifetimes = np.concatenate([t1, t2, t3])
        censor = rng.uniform(30.0, 600.0, size=n)
        durations = np.clip(np.minimum(lifetimes, censor), 1e-6, None)
        events = (lifetimes <= censor).astype(int)
        df = _make_df(durations, events)
        config = BayesianLatentWeibullConfig(
            n_components=3, n_iter=800, burn_in=300, thin=2, n_chains=1, random_seed=33
        )
        result = fit_bayesian_latent_weibull(
            df, duration_column="duration", event_column="event", config=config
        )
        self.assertEqual(result.config.n_components, 3)
        summary = result.posterior_summary_frame()
        self.assertIn("component_3", summary["component"].unique())


if __name__ == "__main__":
    unittest.main()
