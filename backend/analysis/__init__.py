"""
analysis — reusable computation library for Pump2.

Sub-packages (canonical import paths):
  analysis.common          — plotting, modeling_config
  analysis.data            — data_utils, presentation_loader, presentation_schema
  analysis.features        — derived_features, stress_transforms, transform_selection, …
  analysis.models.survival — weibull_model, bayesian_latent_weibull, …
  analysis.models          — regularized_logistic
  analysis.workflows       — vt_failure (and future workflows)

Backward-compatible flat imports still work via shim files:
  from analysis.weibull_model import fit_weibull_stress_model   # old
  from analysis.models.survival.weibull_model import ...         # new canonical

NOTE: exports are lazy (PEP 562).  ``from analysis import fit_weibull_stress_model``
still works, but the heavy scientific stack (scipy & co.) is imported only when such a
name is actually requested.  This keeps light consumers — e.g. the frozen
production-risk executable, which only needs ``analysis.paths`` and
``analysis.workflows.production_risk`` — fast to import and small to bundle.
"""
from __future__ import annotations

import importlib
from typing import Any

# name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    # models.survival.bayesian_latent_weibull
    "BayesianLatentWeibullConfig": "analysis.models.survival.bayesian_latent_weibull",
    "BayesianLatentWeibullFitResult": "analysis.models.survival.bayesian_latent_weibull",
    "SurvivalValidationReport": "analysis.models.survival.bayesian_latent_weibull",
    "fit_bayesian_latent_weibull": "analysis.models.survival.bayesian_latent_weibull",
    "kaplan_meier_frame": "analysis.models.survival.bayesian_latent_weibull",
    "load_survival_file": "analysis.models.survival.bayesian_latent_weibull",
    "mixture_log_likelihood": "analysis.models.survival.bayesian_latent_weibull",
    "validate_survival_dataframe": "analysis.models.survival.bayesian_latent_weibull",
    # models.survival.weibull_birth_death
    "BirthDeathDraw": "analysis.models.survival.weibull_birth_death",
    "BirthDeathFitResult": "analysis.models.survival.weibull_birth_death",
    "BirthDeathWeibullConfig": "analysis.models.survival.weibull_birth_death",
    "fit_birth_death_weibull": "analysis.models.survival.weibull_birth_death",
    # data.data_utils
    "apply_filters": "analysis.data.data_utils",
    "assign_group_fallback": "analysis.data.data_utils",
    "build_representative_row": "analysis.data.data_utils",
    "load_excel_sheet": "analysis.data.data_utils",
    "normalize_event_series": "analysis.data.data_utils",
    "prepare_modeling_dataframe": "analysis.data.data_utils",
    "read_excel_sheets": "analysis.data.data_utils",
    # features.derived_feature_presets
    "DerivedPreset": "analysis.features.derived_feature_presets",
    "StressTermPreset": "analysis.features.derived_feature_presets",
    "inspect_derived_presets": "analysis.features.derived_feature_presets",
    "suggest_derived_presets": "analysis.features.derived_feature_presets",
    # features.derived_features
    "DerivedColumnSpec": "analysis.features.derived_features",
    "apply_derived_columns": "analysis.features.derived_features",
    "evaluate_derived_formula": "analysis.features.derived_features",
    # models.survival.latent_weibull_competing_risks
    "ClassicalWeibullCurveFitResult": "analysis.models.survival.latent_weibull_competing_risks",
    "CompetingRiskModel": "analysis.models.survival.latent_weibull_competing_risks",
    "LatentWeibullCurveFitResult": "analysis.models.survival.latent_weibull_competing_risks",
    "ThreeComponentLatentWeibullModel": "analysis.models.survival.latent_weibull_competing_risks",
    "TwoComponentLatentWeibullModel": "analysis.models.survival.latent_weibull_competing_risks",
    "WeibullParameters": "analysis.models.survival.latent_weibull_competing_risks",
    "build_survival_curve_weights": "analysis.models.survival.latent_weibull_competing_risks",
    "competing_curve_frame": "analysis.models.survival.latent_weibull_competing_risks",
    "competing_next_window_probabilities": "analysis.models.survival.latent_weibull_competing_risks",
    "competing_overall_cdf": "analysis.models.survival.latent_weibull_competing_risks",
    "competing_overall_hazard": "analysis.models.survival.latent_weibull_competing_risks",
    "competing_overall_survival": "analysis.models.survival.latent_weibull_competing_risks",
    "filter_km_frame_for_fit": "analysis.models.survival.latent_weibull_competing_risks",
    "fit_latent_weibull_to_km_frame": "analysis.models.survival.latent_weibull_competing_risks",
    "fit_latent_weibull_to_survival_curve": "analysis.models.survival.latent_weibull_competing_risks",
    "fit_three_component_latent_weibull_to_km_frame": "analysis.models.survival.latent_weibull_competing_risks",
    "fit_three_component_latent_weibull_to_survival_curve": "analysis.models.survival.latent_weibull_competing_risks",
    "fit_weibull_to_km_frame": "analysis.models.survival.latent_weibull_competing_risks",
    "fit_weibull_to_survival_curve": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_curve_frame": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_density": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_hazard": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_life_quantile": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_next_window_failure_probability": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_posterior_given_failure": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_posterior_given_survival": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_remaining_life_quantile": "analysis.models.survival.latent_weibull_competing_risks",
    "latent_survival": "analysis.models.survival.latent_weibull_competing_risks",
    "weibull_cdf": "analysis.models.survival.latent_weibull_competing_risks",
    "weibull_cumulative_hazard": "analysis.models.survival.latent_weibull_competing_risks",
    "weibull_density": "analysis.models.survival.latent_weibull_competing_risks",
    "weibull_hazard": "analysis.models.survival.latent_weibull_competing_risks",
    "weibull_survival": "analysis.models.survival.latent_weibull_competing_risks",
    # data.presentation_loader
    "PresentationPreparationResult": "analysis.data.presentation_loader",
    "PresentationSheetReport": "analysis.data.presentation_loader",
    "detect_sheet_roles": "analysis.data.presentation_loader",
    "inspect_presentation_workbook": "analysis.data.presentation_loader",
    "map_role_columns": "analysis.data.presentation_loader",
    "prepare_presentation_dataset": "analysis.data.presentation_loader",
    "standardize_role_dataframe": "analysis.data.presentation_loader",
    # data.presentation_correlations
    "build_ttf_correlation_frame": "analysis.data.presentation_correlations",
    # data.presentation_schema
    "DEFAULT_GAS_LIMITS": "analysis.data.presentation_schema",
    "build_default_gas_limits_frame": "analysis.data.presentation_schema",
    "normalize_gas_handling_type": "analysis.data.presentation_schema",
    # models.regularized_logistic
    "RegularizedLogisticConfig": "analysis.models.regularized_logistic",
    "RegularizedLogisticResult": "analysis.models.regularized_logistic",
    "design_matrix_from_coefficient_table": "analysis.models.regularized_logistic",
    "fit_regularized_logistic_model": "analysis.models.regularized_logistic",
    "predict_regularized_logistic_from_table": "analysis.models.regularized_logistic",
    # features.stress_correlations
    "build_grouped_stress_correlation_frame": "analysis.features.stress_correlations",
    "build_stress_term_observation_frame": "analysis.features.stress_correlations",
    # features.stress_term_presets
    "StressTermPresetSuggestion": "analysis.features.stress_term_presets",
    "suggest_stress_term_presets": "analysis.features.stress_term_presets",
    # features.stress_transforms
    "TRANSFORM_LIBRARY": "analysis.features.stress_transforms",
    "apply_transform": "analysis.features.stress_transforms",
    # features.transform_selection
    "TransformSelectionResult": "analysis.features.transform_selection",
    "rank_transform_candidates": "analysis.features.transform_selection",
    # models.survival.weibull_model
    "FitStageSummary": "analysis.models.survival.weibull_model",
    "StressTermConfig": "analysis.models.survival.weibull_model",
    "WeibullStressFitResult": "analysis.models.survival.weibull_model",
    "fit_weibull_stress_model": "analysis.models.survival.weibull_model",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # PEP 562 lazy re-export
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
