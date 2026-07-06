from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .derived_feature_presets import EXTRA_COLUMN_ALIASES, StressTermPreset, _aliases, _find_first_column


@dataclass(frozen=True, slots=True)
class StressTermPresetSuggestion:
    key: str
    category: str
    source_type: str
    name: str
    description: str
    term: StressTermPreset


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _median(df: pd.DataFrame, column: str, fallback: float) -> float:
    value = float(_numeric_series(df, column).median())
    return value if pd.notna(value) else fallback


def _quantile(df: pd.DataFrame, column: str, q: float, fallback: float) -> float:
    value = float(_numeric_series(df, column).quantile(q))
    return value if pd.notna(value) else fallback


def _append_unique(suggestions: list[StressTermPresetSuggestion], suggestion: StressTermPresetSuggestion) -> None:
    identity = (suggestion.term.name, suggestion.term.column, suggestion.term.transform, suggestion.term.reference_mode)
    for existing in suggestions:
        if (existing.term.name, existing.term.column, existing.term.transform, existing.term.reference_mode) == identity:
            return
    suggestions.append(suggestion)


def suggest_stress_term_presets(df: pd.DataFrame) -> list[StressTermPresetSuggestion]:
    columns = [str(column) for column in df.columns.tolist()]
    suggestions: list[StressTermPresetSuggestion] = []

    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    nominal = _find_first_column(columns, _aliases("pump_nominal_rate_m3d", extra=["pump_nominal_rate", "nominal_rate_m3d"]))
    design_qliq = _find_first_column(columns, _aliases("design_Qliq_m3d", extra=["design qliq", "design_Qliq"]))
    glf = _find_first_column(columns, _aliases("GLF_m3m3", extra=["GLF", "actual_GLF"]))
    design_glf = _find_first_column(columns, _aliases("design_GLF_m3m3", extra=["design_GLF", "design glf"]))
    free_gas = _find_first_column(columns, _aliases("free_gas_intake_percent", extra=["actual_free_gas", "free_gas_percent"]))
    design_free_gas = _find_first_column(columns, _aliases("design_free_gas_intake_percent", extra=["design_free_gas"]))
    frequency = _find_first_column(columns, _aliases("frequency_hz", extra=["frequency", "freq_hz"]))
    reference_frequency = _find_first_column(
        columns,
        _aliases("reference_frequency_hz", "design_frequency_hz", extra=["reference frequency", "base_frequency_hz"]),
    )
    motor_load = _find_first_column(columns, _aliases("motor_load_percent", extra=["motor_load", "PED_load", "ped_load"]))
    design_motor_load = _find_first_column(columns, _aliases("design_motor_load_percent", extra=["design motor load"]))
    watercut = _find_first_column(columns, _aliases("watercut_percent", extra=["watercut", "actual_watercut"]))
    design_watercut = _find_first_column(columns, _aliases("design_watercut_percent", extra=["design_watercut"]))
    motor_power = _find_first_column(columns, _aliases("motor_power_kw", extra=["motor power kw", "ped_power_kw"]))
    kpod = _find_first_column(columns, _aliases("Kpod", extra=["Кпод", "Кпрод.", "Кпрод"]))
    work_in_curvature = _find_first_column(columns, _aliases("work_in_curvature"))
    pressure_ratio = _find_first_column(columns, _aliases("pressure_ratio"))
    gas_limit_fraction = _find_first_column(columns, _aliases("free_gas_limit_fraction", extra=["gas_limit_fraction"]))
    solids = _find_first_column(columns, _aliases("solids_mg_l", extra=["solids"]))

    if kpod:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_kpod_low",
                category="Hydraulic",
                source_type="Raw / direct",
                name="stress_low_Kpod",
                description="Low Kpod stress directly from the input column.",
                term=StressTermPreset(
                    name="stress_low_Kpod",
                    column=kpod,
                    description="Low Kpod stress directly from the input column.",
                    transform="negative_excess",
                    reference_mode="fixed",
                    reference_value=0.7,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=0.1,
                ),
            ),
        )

    if qliq and nominal and not kpod:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_qliq_vs_nominal",
                category="Hydraulic",
                source_type="Raw / direct",
                name="stress_qliq_vs_nominal",
                description="Underproduction relative to nominal pump rate using raw columns.",
                term=StressTermPreset(
                    name="stress_qliq_vs_nominal",
                    column=qliq,
                    description="Underproduction relative to nominal pump rate using raw columns.",
                    transform="relative_negative_excess",
                    reference_mode="column",
                    reference_column=nominal,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=1.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=1.0,
                ),
            ),
        )

    if work_in_curvature:
        median = _median(df, work_in_curvature, 0.0)
        upper = _quantile(df, work_in_curvature, 0.9, median + 1.0)
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="curve_work_fit",
                category="Trajectory",
                source_type="Raw / direct",
                name="stress_high_curve_work",
                description="Work-in-curvature threshold fitted from the data.",
                term=StressTermPreset(
                    name="stress_high_curve_work",
                    column=work_in_curvature,
                    description="Work-in-curvature threshold fitted from the data.",
                    transform="positive_excess",
                    reference_mode="fit",
                    reference_init=median,
                    reference_bounds=(_quantile(df, work_in_curvature, 0.25, median), upper),
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=max(abs(median) * 0.1, 1.0),
                ),
            ),
        )

    if qliq and design_qliq:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_qliq_vs_design",
                category="Design Mismatch",
                source_type="Raw / direct",
                name="stress_qliq_vs_design",
                description="Underproduction relative to design liquid rate using raw columns.",
                term=StressTermPreset(
                    name="stress_qliq_vs_design",
                    column=qliq,
                    description="Underproduction relative to design liquid rate using raw columns.",
                    transform="relative_negative_excess",
                    reference_mode="column",
                    reference_column=design_qliq,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=1.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=1.0,
                ),
            ),
        )

    if glf and design_glf:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_glf_vs_design",
                category="Design Mismatch",
                source_type="Raw / direct",
                name="stress_glf_over_design",
                description="GLF above design GLF using raw columns.",
                term=StressTermPreset(
                    name="stress_glf_over_design",
                    column=glf,
                    description="GLF above design GLF using raw columns.",
                    transform="relative_positive_excess",
                    reference_mode="column",
                    reference_column=design_glf,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=1.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=1.0,
                ),
            ),
        )

    if free_gas and design_free_gas:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_free_gas_vs_design",
                category="Design Mismatch",
                source_type="Raw / direct",
                name="stress_free_gas_over_design",
                description="Free gas above design free-gas value using raw columns.",
                term=StressTermPreset(
                    name="stress_free_gas_over_design",
                    column=free_gas,
                    description="Free gas above design free-gas value using raw columns.",
                    transform="positive_excess",
                    reference_mode="column",
                    reference_column=design_free_gas,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=1.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=5.0,
                ),
            ),
        )

    if frequency and reference_frequency:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_frequency_over_reference",
                category="Control",
                source_type="Raw / direct",
                name="stress_frequency_over_reference",
                description="Frequency above nominal/reference frequency using raw columns.",
                term=StressTermPreset(
                    name="stress_frequency_over_reference",
                    column=frequency,
                    description="Frequency above nominal/reference frequency using raw columns.",
                    transform="positive_excess",
                    reference_mode="column",
                    reference_column=reference_frequency,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=1.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=2.0,
                ),
            ),
        )

    if motor_load and design_motor_load:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_motor_load_over_design",
                category="Load",
                source_type="Raw / direct",
                name="stress_motor_load_over_design",
                description="Motor load above design motor load using raw columns.",
                term=StressTermPreset(
                    name="stress_motor_load_over_design",
                    column=motor_load,
                    description="Motor load above design motor load using raw columns.",
                    transform="positive_excess",
                    reference_mode="column",
                    reference_column=design_motor_load,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=1.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=5.0,
                ),
            ),
        )

    if watercut and design_watercut:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_watercut_vs_design",
                category="Fluids",
                source_type="Raw / direct",
                name="stress_watercut_vs_design",
                description="Watercut mismatch relative to design watercut using raw columns.",
                term=StressTermPreset(
                    name="stress_watercut_vs_design",
                    column=watercut,
                    description="Watercut mismatch relative to design watercut using raw columns.",
                    transform="relative_abs_deviation",
                    reference_mode="column",
                    reference_column=design_watercut,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=1.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=1.0,
                ),
            ),
        )

    if free_gas and gas_limit_fraction:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="raw_free_gas_vs_limit",
                category="Pressure / Gas",
                source_type="Raw / direct",
                name="stress_free_gas_over_limit",
                description="Free gas above allowed gas limit using raw columns.",
                term=StressTermPreset(
                    name="stress_free_gas_over_limit",
                    column=free_gas,
                    description="Free gas above allowed gas limit using raw columns.",
                    transform="positive_excess",
                    reference_mode="column",
                    reference_column=gas_limit_fraction,
                    reference_multiplier_mode="fixed",
                    reference_multiplier_value=100.0,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=5.0,
                ),
            ),
        )

    if pressure_ratio:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="pressure_ratio_existing",
                category="Pressure / Gas",
                source_type="Derived / existing",
                name="stress_low_pressure_ratio",
                description="Low pressure-ratio stress from the existing pressure-ratio column.",
                term=StressTermPreset(
                    name="stress_low_pressure_ratio",
                    column=pressure_ratio,
                    description="Low pressure-ratio stress from the existing pressure-ratio column.",
                    transform="negative_excess",
                    reference_mode="fixed",
                    reference_value=0.7,
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=0.1,
                ),
            ),
        )

    if motor_load:
        median = _median(df, motor_load, 70.0)
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="motor_load_fit",
                category="Load",
                source_type="Raw / direct",
                name="stress_high_motor_load",
                description="Motor-load threshold fitted from the data.",
                term=StressTermPreset(
                    name="stress_high_motor_load",
                    column=motor_load,
                    description="Motor-load threshold fitted from the data.",
                    transform="positive_excess",
                    reference_mode="fit",
                    reference_init=median,
                    reference_bounds=(_quantile(df, motor_load, 0.25, median), _quantile(df, motor_load, 0.9, median + 10.0)),
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=5.0,
                ),
            ),
        )

    if frequency:
        median = _median(df, frequency, 50.0)
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="frequency_fit",
                category="Control",
                source_type="Raw / direct",
                name="stress_high_frequency",
                description="Frequency threshold fitted from the data.",
                term=StressTermPreset(
                    name="stress_high_frequency",
                    column=frequency,
                    description="Frequency threshold fitted from the data.",
                    transform="positive_excess",
                    reference_mode="fit",
                    reference_init=median,
                    reference_bounds=(_quantile(df, frequency, 0.25, median), _quantile(df, frequency, 0.9, median + 5.0)),
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=2.0,
                ),
            ),
        )

    if glf:
        median = _median(df, glf, 100.0)
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="glf_fit",
                category="Pressure / Gas",
                source_type="Raw / direct",
                name="stress_high_GLF",
                description="GLF threshold fitted from the data.",
                term=StressTermPreset(
                    name="stress_high_GLF",
                    column=glf,
                    description="GLF threshold fitted from the data.",
                    transform="positive_excess",
                    reference_mode="fit",
                    reference_init=median,
                    reference_bounds=(_quantile(df, glf, 0.25, median), _quantile(df, glf, 0.9, median * 1.25)),
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=max(median * 0.1, 10.0),
                ),
            ),
        )

    if solids:
        median = _median(df, solids, 10.0)
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="solids_fit",
                category="Chemistry",
                source_type="Raw / direct",
                name="stress_high_solids",
                description="High solids threshold fitted from the data.",
                term=StressTermPreset(
                    name="stress_high_solids",
                    column=solids,
                    description="High solids threshold fitted from the data.",
                    transform="positive_excess",
                    reference_mode="fit",
                    reference_init=median,
                    reference_bounds=(_quantile(df, solids, 0.25, median), _quantile(df, solids, 0.9, median * 1.5)),
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=max(median * 0.1, 1.0),
                ),
            ),
        )

    if qliq and motor_power:
        _append_unique(
            suggestions,
            StressTermPresetSuggestion(
                key="qliq_per_kw_fit",
                category="Hydraulic",
                source_type="Raw / direct",
                name="stress_low_qliq_per_kw",
                description="Low liquid-rate-per-kW efficiency fitted from the data.",
                term=StressTermPreset(
                    name="stress_low_qliq_per_kw",
                    column=qliq,
                    description="Low liquid-rate-per-kW efficiency fitted from the data.",
                    transform="relative_negative_excess",
                    reference_mode="column",
                    reference_column=motor_power,
                    reference_multiplier_mode="fit",
                    reference_multiplier_value=1.0,
                    reference_multiplier_init=1.0,
                    reference_multiplier_bounds=(0.1, 10.0),
                    coefficient_mode="fit",
                    coefficient_value=0.05,
                    coefficient_non_negative=True,
                    coefficient_bounds=(0.0, None),
                    scale=1.0,
                ),
            ),
        )

    return suggestions
