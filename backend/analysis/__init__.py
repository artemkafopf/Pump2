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
"""
from .models.survival.bayesian_latent_weibull import (
    BayesianLatentWeibullConfig,
    BayesianLatentWeibullFitResult,
    SurvivalValidationReport,
    fit_bayesian_latent_weibull,
    kaplan_meier_frame,
    load_survival_file,
    mixture_log_likelihood,
    validate_survival_dataframe,
)
from .models.survival.weibull_birth_death import (
    BirthDeathDraw,
    BirthDeathFitResult,
    BirthDeathWeibullConfig,
    fit_birth_death_weibull,
)
from .data.data_utils import (
    apply_filters,
    assign_group_fallback,
    build_representative_row,
    load_excel_sheet,
    normalize_event_series,
    prepare_modeling_dataframe,
    read_excel_sheets,
)
from .features.derived_feature_presets import DerivedPreset, StressTermPreset, inspect_derived_presets, suggest_derived_presets
from .features.derived_features import DerivedColumnSpec, apply_derived_columns, evaluate_derived_formula
from .models.survival.latent_weibull_competing_risks import (
    ClassicalWeibullCurveFitResult,
    CompetingRiskModel,
    LatentWeibullCurveFitResult,
    ThreeComponentLatentWeibullModel,
    TwoComponentLatentWeibullModel,
    WeibullParameters,
    build_survival_curve_weights,
    competing_curve_frame,
    competing_next_window_probabilities,
    competing_overall_cdf,
    competing_overall_hazard,
    competing_overall_survival,
    filter_km_frame_for_fit,
    fit_latent_weibull_to_km_frame,
    fit_latent_weibull_to_survival_curve,
    fit_three_component_latent_weibull_to_km_frame,
    fit_three_component_latent_weibull_to_survival_curve,
    fit_weibull_to_km_frame,
    fit_weibull_to_survival_curve,
    latent_curve_frame,
    latent_density,
    latent_hazard,
    latent_life_quantile,
    latent_next_window_failure_probability,
    latent_posterior_given_failure,
    latent_posterior_given_survival,
    latent_remaining_life_quantile,
    latent_survival,
    weibull_cdf,
    weibull_cumulative_hazard,
    weibull_density,
    weibull_hazard,
    weibull_survival,
)
from .data.presentation_loader import (
    PresentationPreparationResult,
    PresentationSheetReport,
    detect_sheet_roles,
    inspect_presentation_workbook,
    map_role_columns,
    prepare_presentation_dataset,
    standardize_role_dataframe,
)
from .data.presentation_correlations import build_ttf_correlation_frame
from .data.presentation_schema import DEFAULT_GAS_LIMITS, build_default_gas_limits_frame, normalize_gas_handling_type
from .models.regularized_logistic import (
    RegularizedLogisticConfig,
    RegularizedLogisticResult,
    design_matrix_from_coefficient_table,
    fit_regularized_logistic_model,
    predict_regularized_logistic_from_table,
)
from .features.stress_correlations import build_grouped_stress_correlation_frame, build_stress_term_observation_frame
from .features.stress_term_presets import StressTermPresetSuggestion, suggest_stress_term_presets
from .features.stress_transforms import TRANSFORM_LIBRARY, apply_transform
from .features.transform_selection import TransformSelectionResult, rank_transform_candidates
from .models.survival.weibull_model import FitStageSummary, StressTermConfig, WeibullStressFitResult, fit_weibull_stress_model

__all__ = [
    "BirthDeathDraw",
    "BirthDeathFitResult",
    "BirthDeathWeibullConfig",
    "ClassicalWeibullCurveFitResult",
    "CompetingRiskModel",
    "DEFAULT_GAS_LIMITS",
    "BayesianLatentWeibullConfig",
    "BayesianLatentWeibullFitResult",
    "DerivedColumnSpec",
    "DerivedPreset",
    "FitStageSummary",
    "LatentWeibullCurveFitResult",
    "PresentationPreparationResult",
    "PresentationSheetReport",
    "RegularizedLogisticConfig",
    "RegularizedLogisticResult",
    "StressTermPreset",
    "StressTermConfig",
    "StressTermPresetSuggestion",
    "SurvivalValidationReport",
    "TransformSelectionResult",
    "TRANSFORM_LIBRARY",
    "ThreeComponentLatentWeibullModel",
    "TwoComponentLatentWeibullModel",
    "WeibullParameters",
    "WeibullStressFitResult",
    "apply_derived_columns",
    "apply_filters",
    "apply_transform",
    "assign_group_fallback",
    "build_survival_curve_weights",
    "build_default_gas_limits_frame",
    "build_representative_row",
    "build_grouped_stress_correlation_frame",
    "build_stress_term_observation_frame",
    "build_ttf_correlation_frame",
    "competing_curve_frame",
    "competing_next_window_probabilities",
    "competing_overall_cdf",
    "competing_overall_hazard",
    "competing_overall_survival",
    "design_matrix_from_coefficient_table",
    "detect_sheet_roles",
    "evaluate_derived_formula",
    "filter_km_frame_for_fit",
    "fit_bayesian_latent_weibull",
    "fit_birth_death_weibull",
    "fit_latent_weibull_to_km_frame",
    "fit_latent_weibull_to_survival_curve",
    "fit_regularized_logistic_model",
    "fit_three_component_latent_weibull_to_km_frame",
    "fit_three_component_latent_weibull_to_survival_curve",
    "fit_weibull_to_km_frame",
    "fit_weibull_to_survival_curve",
    "fit_weibull_stress_model",
    "inspect_derived_presets",
    "inspect_presentation_workbook",
    "kaplan_meier_frame",
    "latent_curve_frame",
    "latent_density",
    "latent_hazard",
    "latent_life_quantile",
    "latent_next_window_failure_probability",
    "latent_posterior_given_failure",
    "latent_posterior_given_survival",
    "latent_remaining_life_quantile",
    "latent_survival",
    "load_excel_sheet",
    "load_survival_file",
    "map_role_columns",
    "mixture_log_likelihood",
    "normalize_gas_handling_type",
    "normalize_event_series",
    "prepare_presentation_dataset",
    "prepare_modeling_dataframe",
    "predict_regularized_logistic_from_table",
    "rank_transform_candidates",
    "read_excel_sheets",
    "standardize_role_dataframe",
    "suggest_stress_term_presets",
    "suggest_derived_presets",
    "validate_survival_dataframe",
    "weibull_cdf",
    "weibull_cumulative_hazard",
    "weibull_density",
    "weibull_hazard",
    "weibull_survival",
]
