"""Backward-compat shim — use analysis.models.survival.bayesian_latent_weibull directly."""
from analysis.models.survival.bayesian_latent_weibull import *  # noqa: F401, F403
from analysis.models.survival.bayesian_latent_weibull import (  # noqa: F401
    _log_weibull_density,
    _log_weibull_survival,
)
