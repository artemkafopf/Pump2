from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


DEFAULT_GAS_LIMITS = {
    "none": 0.05,
    "standard_gas_separator": 0.25,
    "gas_separator": 0.25,
    "gas_separator_disperser": 0.45,
    "disperser": 0.45,
    "multiphase_section": 0.75,
    "custom": None,
}


def normalize_name(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[\W_]+", "", str(value).strip().casefold())


def _alias_map(values: list[str]) -> dict[str, list[str]]:
    return {item: aliases[:] for item, aliases in values}


@dataclass(frozen=True, slots=True)
class PresentationSheetSpec:
    role: str
    required: bool
    aliases: tuple[str, ...]
    key_columns: tuple[str, ...]
    recommended_columns: tuple[str, ...]


ROLE_COLUMN_ALIASES: dict[str, dict[str, list[str]]] = {
    "runs": _alias_map(
        [
            ("run_id", ["run_id", "run id", "runid", "esp_run_id", "id_run"]),
            ("well", ["well", "well_name", "well name", "скважина", "well_no"]),
            ("field", ["field", "месторождение"]),
            ("region", ["region", "регион"]),
            ("pad", ["pad", "cluster", "куст"]),
            ("contractor", ["contractor", "esp contractor", "принадлежность", "подрядчик"]),
            ("pump_family", ["pump_family", "pump family", "тип насоса"]),
            ("pump_model", ["pump_model", "pump model", "модель насоса"]),
            ("pump_nominal_rate_m3d", ["pump_nominal_rate_m3d", "pump nominal rate", "nominal_rate_m3d", "ном. произв. м₃/сут", "ном. произв. м3/сут"]),
            ("reference_frequency_hz", ["reference_frequency_hz", "reference frequency", "base_frequency_hz", "номинальная частота, гц"]),
            ("motor_power_kw", ["motor_power_kw", "ped power", "motor power", "мощность пэд", "мощность, квт"]),
            ("gas_handling_type", ["gas_handling_type", "gas handling type", "gas separator type", "тип газозащиты"]),
            ("install_date", ["install_date", "install date", "esp install date", "дата монтажа"]),
            ("start_date", ["start_date", "start date", "launch_date", "дата запуска"]),
            ("stop_date", ["stop_date", "stop date", "end_date", "дата останова"]),
            ("failure_date", ["failure_date", "failure date", "дата отказа"]),
            ("event", ["event", "failure_event", "Event", "Failure Flag", "failure flag", "отказ", "авария"]),
            ("failure_cause_original", ["failure_cause_original", "failure cause", "причина отказа"]),
            ("failure_node_original", ["failure_node_original", "failed node", "узел отказа"]),
            ("stop_reason_original", ["stop_reason_original", "stop reason", "причина останова"]),
            ("failure_category_standard", ["failure_category_standard", "standard failure category"]),
            ("failed_node_standard", ["failed_node_standard", "standard failed node"]),
            ("stop_reason_standard", ["stop_reason_standard", "standard stop reason"]),
            ("TTF_days", ["TTF_days", "ttf", "ttf_days", "runtime_days", "run life days", "наработка (сут)", "наработка", "наработкасут"]),
            ("infant_mortality_flag", ["infant_mortality_flag", "early_failure_flag", "младенческий отказ"]),
            ("planned_stop_flag", ["planned_stop_flag", "planned_stop", "плановый останов"]),
            ("workover_type", ["workover_type", "workover", "гтм"]),
            ("first_run_after_workover", ["first_run_after_workover", "first run after workover"]),
            ("previous_failure_cause", ["previous_failure_cause", "previous failure cause"]),
            ("previous_TTF_days", ["previous_TTF_days", "previous ttf", "previous runtime"]),
            ("h2s_ppm", ["h2s_ppm", "h2s"]),
            ("co2_percent", ["co2_percent", "co2"]),
            ("solids_mg_l", ["solids_mg_l", "solids", "мехпримеси"]),
            ("water_density_gcc", ["water_density_gcc", "water density"]),
            ("oil_density_gcc", ["oil_density_gcc", "oil density"]),
            ("oil_viscosity_cp", ["oil_viscosity_cp", "oil viscosity"]),
            ("watercut_percent", ["watercut_percent", "watercut", "обводненность"]),
            ("salinity_mg_l", ["salinity_mg_l", "salinity"]),
            ("scale_risk_flag", ["scale_risk_flag", "scale risk"]),
            ("corrosion_risk_flag", ["corrosion_risk_flag", "corrosion risk"]),
        ]
    ),
    "daily_or_monthly_regime": _alias_map(
        [
            ("run_id", ["run_id", "run id", "esp_run_id"]),
            ("well", ["well", "well_name", "скважина"]),
            ("date", ["date", "record_date", "дата"]),
            ("Qliq_m3d", ["Qliq_m3d", "qliq", "liquid_rate_m3d", "дебит жидкости", "дебит жидк."]),
            ("Qoil_td", ["Qoil_td", "qoil", "oil_rate_td"]),
            ("Qgas_m3d", ["Qgas_m3d", "qgas", "gas_rate_m3d"]),
            ("watercut_percent", ["watercut_percent", "watercut", "обводненность"]),
            ("GLF_m3m3", ["GLF_m3m3", "glf", "gzf", "гжф"]),
            ("GOR_m3t", ["GOR_m3t", "gor"]),
            ("P_res_atm", ["P_res_atm", "pres_atm", "reservoir_pressure_atm"]),
            ("P_bhp_atm", ["P_bhp_atm", "pbhp_atm", "bottomhole_pressure_atm", "рзаб"]),
            ("P_intake_atm", ["P_intake_atm", "pintake_atm", "pump_intake_pressure_atm"]),
            ("P_line_atm", ["P_line_atm", "line_pressure_atm"]),
            ("P_annulus_atm", ["P_annulus_atm", "annulus_pressure_atm"]),
            ("Pbubble_atm", ["Pbubble_atm", "pbubble_atm", "bubblepoint_pressure_atm", "рнас", "дав. нас"]),
            ("dynamic_level_m", ["dynamic_level_m", "dynamic_level"]),
            ("frequency_hz", ["frequency_hz", "frequency", "freq_hz", "частота"]),
            ("current_a", ["current_a", "current", "ток"]),
            ("voltage_v", ["voltage_v", "voltage", "напряжение"]),
            ("motor_load_percent", ["motor_load_percent", "motor_load", "ped_load", "загрузка пэд", "загр, двиг,"]),
            ("winding_temp_c", ["winding_temp_c", "winding_temp"]),
            ("shaft_load_percent", ["shaft_load_percent", "shaft_load"]),
            ("protector_thrust_load_percent", ["protector_thrust_load_percent", "protector_thrust_load"]),
            ("status", ["status", "state", "режим"]),
        ]
    ),
    "design": _alias_map(
        [
            ("run_id", ["run_id", "run id", "esp_run_id"]),
            ("design_date", ["design_date", "design date", "selection_date"]),
            ("design_Qliq_m3d", ["design_Qliq_m3d", "design qliq", "planned_qliq_m3d"]),
            ("design_Qoil_td", ["design_Qoil_td", "design qoil"]),
            ("design_watercut_percent", ["design_watercut_percent", "design watercut"]),
            ("design_GLF_m3m3", ["design_GLF_m3m3", "design glf", "design gzf"]),
            ("design_Pintake_atm", ["design_Pintake_atm", "design pintake"]),
            ("design_Pbhp_atm", ["design_Pbhp_atm", "design pbhp"]),
            ("design_Pbubble_atm", ["design_Pbubble_atm", "design pbubble", "дав. нас по подбору"]),
            ("design_Kpod", ["design_Kpod", "design kpod"]),
            ("design_free_gas_intake_percent", ["design_free_gas_intake_percent", "design free gas intake"]),
            ("design_free_gas_pump_percent", ["design_free_gas_pump_percent", "design free gas pump"]),
            ("design_motor_load_percent", ["design_motor_load_percent", "design motor load"]),
            ("design_frequency_hz", ["design_frequency_hz", "design frequency"]),
            ("design_pump_head_m", ["design_pump_head_m", "design pump head"]),
            ("design_stage_count", ["design_stage_count", "design stage count"]),
            ("design_gas_separator_efficiency", ["design_gas_separator_efficiency", "design gas separator efficiency"]),
        ]
    ),
    "telemetry": _alias_map(
        [
            ("run_id", ["run_id", "run id", "esp_run_id"]),
            ("well", ["well", "well_name", "скважина"]),
            ("timestamp", ["timestamp", "datetime", "time", "время"]),
            ("frequency_hz", ["frequency_hz", "frequency", "freq_hz", "частота"]),
            ("current_a", ["current_a", "current", "ток"]),
            ("voltage_v", ["voltage_v", "voltage", "напряжение"]),
            ("motor_load_percent", ["motor_load_percent", "motor_load", "ped_load"]),
            ("P_intake_atm", ["P_intake_atm", "pintake_atm", "pump_intake_pressure_atm"]),
            ("P_discharge_atm", ["P_discharge_atm", "pdischarge_atm", "pump_discharge_pressure_atm"]),
            ("winding_temp_c", ["winding_temp_c", "winding_temp"]),
            ("status", ["status", "state", "режим"]),
            ("Qliq_m3d", ["Qliq_m3d", "qliq", "liquid_rate_m3d"]),
        ]
    ),
    "gas_limits": _alias_map(
        [
            ("gas_handling_type", ["gas_handling_type", "gas handling type", "equipment type", "тип газозащиты"]),
            ("free_gas_limit_fraction", ["free_gas_limit_fraction", "free gas limit fraction", "gas_limit_fraction"]),
            ("description", ["description", "comment", "описание"]),
        ]
    ),
}


ROLE_DATE_COLUMNS = {
    "runs": ["install_date", "start_date", "stop_date", "failure_date"],
    "daily_or_monthly_regime": ["date"],
    "design": ["design_date"],
    "telemetry": ["timestamp"],
    "gas_limits": [],
}


ROLE_NUMERIC_COLUMNS = {
    "runs": [
        "pump_nominal_rate_m3d",
        "reference_frequency_hz",
        "motor_power_kw",
        "TTF_days",
        "previous_TTF_days",
        "h2s_ppm",
        "co2_percent",
        "solids_mg_l",
        "water_density_gcc",
        "oil_density_gcc",
        "oil_viscosity_cp",
        "watercut_percent",
        "salinity_mg_l",
    ],
    "daily_or_monthly_regime": [
        "Qliq_m3d",
        "Qoil_td",
        "Qgas_m3d",
        "watercut_percent",
        "GLF_m3m3",
        "GOR_m3t",
        "P_res_atm",
        "P_bhp_atm",
        "P_intake_atm",
        "P_line_atm",
        "P_annulus_atm",
        "Pbubble_atm",
        "dynamic_level_m",
        "frequency_hz",
        "current_a",
        "voltage_v",
        "motor_load_percent",
        "winding_temp_c",
        "shaft_load_percent",
        "protector_thrust_load_percent",
    ],
    "design": [
        "design_Qliq_m3d",
        "design_Qoil_td",
        "design_watercut_percent",
        "design_GLF_m3m3",
        "design_Pintake_atm",
        "design_Pbhp_atm",
        "design_Pbubble_atm",
        "design_Kpod",
        "design_free_gas_intake_percent",
        "design_free_gas_pump_percent",
        "design_motor_load_percent",
        "design_frequency_hz",
        "design_pump_head_m",
        "design_stage_count",
        "design_gas_separator_efficiency",
    ],
    "telemetry": [
        "frequency_hz",
        "current_a",
        "voltage_v",
        "motor_load_percent",
        "P_intake_atm",
        "P_discharge_atm",
        "winding_temp_c",
        "Qliq_m3d",
    ],
    "gas_limits": ["free_gas_limit_fraction"],
}


GAS_HANDLING_ALIASES = {
    "none": ["none", "безгазозащиты", "no_separator"],
    "standard_gas_separator": ["standard_gas_separator", "standard gas separator"],
    "gas_separator": ["gas_separator", "gas separator", "separator"],
    "gas_separator_disperser": ["gas_separator_disperser", "gas separator disperser", "separator disperser"],
    "disperser": ["disperser"],
    "multiphase_section": ["multiphase_section", "multiphase section"],
    "custom": ["custom"],
}


SHEET_SPECS = {
    "runs": PresentationSheetSpec(
        role="runs",
        required=True,
        aliases=("runs", "run", "main_runs", "esp_runs", "основной", "запуски"),
        key_columns=("run_id", "well", "field", "contractor", "event", "TTF_days", "start_date", "stop_date"),
        recommended_columns=("failure_date", "gas_handling_type", "failure_cause_original", "failure_node_original"),
    ),
    "daily_or_monthly_regime": PresentationSheetSpec(
        role="daily_or_monthly_regime",
        required=False,
        aliases=("daily_or_monthly_regime", "regime", "daily_regime", "monthly_regime", "режим", "режимы"),
        key_columns=("run_id", "date", "Qliq_m3d", "P_intake_atm", "Pbubble_atm", "frequency_hz", "status"),
        recommended_columns=("GLF_m3m3", "motor_load_percent", "current_a"),
    ),
    "design": PresentationSheetSpec(
        role="design",
        required=False,
        aliases=("design", "selection", "esp_design", "подбор", "дизайн"),
        key_columns=("run_id", "design_date", "design_Qliq_m3d", "design_GLF_m3m3", "design_Kpod"),
        recommended_columns=("design_Pintake_atm", "design_Pbubble_atm", "design_frequency_hz"),
    ),
    "telemetry": PresentationSheetSpec(
        role="telemetry",
        required=False,
        aliases=("telemetry", "tm", "high_freq_telemetry", "телеметрия"),
        key_columns=("run_id", "timestamp", "frequency_hz", "current_a", "motor_load_percent", "status"),
        recommended_columns=("P_intake_atm", "Qliq_m3d", "winding_temp_c"),
    ),
    "gas_limits": PresentationSheetSpec(
        role="gas_limits",
        required=False,
        aliases=("gas_limits", "gas limit", "gas_limits_override", "лимиты газа"),
        key_columns=("gas_handling_type", "free_gas_limit_fraction"),
        recommended_columns=("description",),
    ),
}


def build_default_gas_limits_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gas_handling_type": key,
                "free_gas_limit_fraction": value,
                "description": "Default prompt limit",
            }
            for key, value in DEFAULT_GAS_LIMITS.items()
        ]
    )


def normalize_gas_handling_type(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    normalized = normalize_name(value)
    for canonical, aliases in GAS_HANDLING_ALIASES.items():
        if normalized == normalize_name(canonical):
            return canonical
        if normalized in {normalize_name(alias) for alias in aliases}:
            return canonical
    return str(value).strip().casefold().replace(" ", "_")
