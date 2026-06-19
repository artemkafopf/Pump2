from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .presentation_schema import ROLE_COLUMN_ALIASES, normalize_name


def _find_first_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_columns = [(str(column), normalize_name(column)) for column in columns]
    exact_matches = {normalized: original for original, normalized in normalized_columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in exact_matches:
            return exact_matches[key]

    partial_matches: list[tuple[int, int, str]] = []
    for candidate in candidates:
        key = normalize_name(candidate)
        if not key:
            continue
        for original, normalized in normalized_columns:
            if not normalized:
                continue
            if not key.startswith("design") and normalized.startswith("design"):
                continue
            if key in normalized:
                partial_matches.append((abs(len(normalized) - len(key)), len(normalized), original))
    if partial_matches:
        partial_matches.sort()
        return partial_matches[0][2]
    return None


def _aliases(*canonical_names: str, extra: list[str] | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for canonical_name in canonical_names:
        alias_values: list[str] = [canonical_name]
        for role_aliases in ROLE_COLUMN_ALIASES.values():
            alias_values.extend(role_aliases.get(canonical_name, []))
        alias_values.extend(EXTRA_COLUMN_ALIASES.get(canonical_name, []))
        for value in alias_values:
            normalized = normalize_name(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                values.append(value)
    for value in extra or []:
        normalized = normalize_name(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(value)
    return values


def _has_column(columns: list[str], target: str) -> bool:
    return _find_first_column(columns, [target]) is not None


EXTRA_COLUMN_ALIASES: dict[str, list[str]] = {
    "Qliq_m3d": [
        "дебит жидкости, м3/сут",
        "дебит жидкости м3/сут",
        "дебит жид",
        "дебит жидк.",
        "дебж",
        "qж",
        "qжид",
    ],
    "pump_nominal_rate_m3d": [
        "ном. произв. м₃/сут",
        "ном. произв. м3/сут",
        "подача насоса",
        "номинальная подача насоса",
        "номинальный дебит насоса",
        "производительность насоса",
    ],
    "reference_frequency_hz": [
        "номинальная частота, гц",
        "опорная частота",
        "базовая частота",
        "номинальная частота",
        "reference frequency hz",
    ],
    "frequency_hz": [
        "частота, гц",
        "частота гц",
        "частота тока",
        "f",
    ],
    "P_intake_atm": [
        "давление на приеме",
        "давление на приёме",
        "давление приема",
        "давление приёма",
        "рприем",
        "рприём",
    ],
    "P_bhp_atm": [
        "забойное давление",
        "р заб",
        "давление забойное",
    ],
    "Pbubble_atm": [
        "дав. нас",
        "давление насыщения",
        "давление насыщения нефти",
    ],
    "GLF_m3m3": [
        "гжф",
        "гжф, м3/м3",
        "гжф м3/м3",
        "газожидкостный фактор",
    ],
    "free_gas_intake_percent": [
        "свободный газ на приеме, %",
        "свободный газ на приеме %",
        "свободный газ на приёме, %",
        "свободный газ на приёме %",
        "свободный газ на приеме",
        "свободный газ на приёме",
    ],
    "design_free_gas_intake_percent": [
        "свободный газ на приеме по подбору, %",
        "свободный газ на приёме по подбору, %",
        "свободный газ на приеме подбор, %",
        "design free gas intake percent",
    ],
    "free_gas_limit_fraction": [
        "предельный свободный газ, доля",
        "лимит свободного газа",
        "допустимый свободный газ",
    ],
    "motor_load_percent": [
        "загр, двиг,",
        "загрузка пэд, %",
        "загрузка пэд %",
        "загрузка пед, %",
        "загрузка пед %",
        "нагрузка пэд",
    ],
    "design_motor_load_percent": [
        "загрузка пэд по подбору, %",
        "нагрузка пэд по подбору, %",
        "design motor load percent",
    ],
    "watercut_percent": [
        "обводненность, %",
        "обводненность %",
    ],
    "design_watercut_percent": [
        "обводненность по подбору, %",
        "design watercut percent",
    ],
    "motor_power_kw": [
        "мощность, квт",
        "мощность пэд, квт",
        "мощность пед, квт",
        "мощность двигателя, квт",
    ],
    "Kpod": [
        "кпрод.",
        "кпрод",
    ],
    "design_Kpod": [
        "кпод по подбору",
        "design kpod ratio",
    ],
    "design_Qliq_m3d": [
        "дебит жидкости по подбору, м3/сут",
        "дебит жидкости подбор, м3/сут",
        "подача по подбору, м3/сут",
    ],
    "design_GLF_m3m3": [
        "гжф по подбору, м3/м3",
        "гжф подбор, м3/м3",
    ],
    "design_Pintake_atm": [
        "давление на приеме по подбору",
        "давление на приёме по подбору",
        "рприем по подбору",
        "рприём по подбору",
    ],
    "design_Pbhp_atm": [
        "рзаб по подбору",
        "забойное давление по подбору",
    ],
    "design_Pbubble_atm": [
        "дав. нас по подбору",
        "рнас по подбору",
        "давление насыщения по подбору",
    ],
    "design_frequency_hz": [
        "частота по подбору, гц",
        "частота подбор, гц",
    ],
    "work_in_curvature": [
        "работа в кривизне",
        "работа в кривизне, %",
        "работа в кривизне %",
    ],
    "actual_current_a": [
        "ток x.x",
        "ток х.х",
        "ток xx",
        "рабочий ток",
        "фактический ток",
    ],
    "nominal_current_a": [
        "ном. ток/ a",
        "ном ток/ a",
        "номинальный ток, a",
        "номинальный ток a",
    ],
    "install_depth_m": [
        "глубина спуска уэцн, по нкт",
        "глубина спуска уэцн",
        "глубина спуска",
    ],
    "dynamic_level_m": [
        "вг, м",
        "вг",
        "динамический уровень, м",
    ],
    "reservoir_pressure_atm": [
        "рпл.",
        "рпл",
        "пластовое давление",
    ],
    "nominal_head_m": [
        "ном.напор (50гц)",
        "ном напор (50гц)",
        "номинальный напор (50гц)",
        "номинальный напор, м",
    ],
    "duration_days": [
        "наработка (сут)",
        "наработка, сут",
        "ttf_days",
        "duration_days",
        "run_days",
        "мрп",
    ],
    "chloride_mg_l": [
        "cl⁻, мг/л",
        "cl-, мг/л",
        "cl мг/л",
        "хлориды, мг/л",
    ],
    "sulfate_mg_l": [
        "so₄²⁻, мг/л",
        "so4²⁻, мг/л",
        "so4, мг/л",
        "сульфаты, мг/л",
    ],
    "calcium_mg_l": [
        "ca₂⁺, мг/л",
        "ca2+, мг/л",
        "ca, мг/л",
        "кальций, мг/л",
    ],
    "stage_count": [
        "кол.ступеней",
        "кол ступеней",
        "число ступеней",
    ],
    "pump_length_m": [
        "длина уэцн/метр",
        "длина уэцн, м",
        "длина уэцн",
    ],
}


@dataclass(frozen=True, slots=True)
class StressTermPreset:
    name: str
    column: str
    description: str
    transform: str
    reference_mode: str = "fixed"
    reference_value: float | None = None
    reference_column: str | None = None
    reference_init: float | None = None
    reference_bounds: tuple[float | None, float | None] = (None, None)
    reference_multiplier_mode: str = "fixed"
    reference_multiplier_value: float = 1.0
    reference_multiplier_init: float = 1.0
    reference_multiplier_bounds: tuple[float | None, float | None] = (0.1, 10.0)
    coefficient_mode: str = "fit"
    coefficient_value: float = 0.05
    coefficient_non_negative: bool = True
    coefficient_bounds: tuple[float | None, float | None] = (0.0, None)
    scale: float = 1.0

    def to_term_dict(self) -> dict:
        payload = {
            "name": self.name,
            "column": self.column,
            "transform": self.transform,
            "reference_mode": self.reference_mode,
            "coefficient_mode": self.coefficient_mode,
            "coefficient_value": self.coefficient_value,
            "coefficient_non_negative": self.coefficient_non_negative,
            "coefficient_bounds": list(self.coefficient_bounds),
            "scale": self.scale,
        }
        if self.reference_value is not None:
            payload["reference_value"] = self.reference_value
        if self.reference_column is not None:
            payload["reference_column"] = self.reference_column
        if self.reference_mode == "fit":
            if self.reference_init is not None:
                payload["reference_init"] = self.reference_init
            payload["reference_bounds"] = list(self.reference_bounds)
        if self.reference_mode == "column":
            payload["reference_multiplier_mode"] = self.reference_multiplier_mode
            payload["reference_multiplier_value"] = self.reference_multiplier_value
            if self.reference_multiplier_mode == "fit":
                payload["reference_multiplier_init"] = self.reference_multiplier_init
                payload["reference_multiplier_bounds"] = list(self.reference_multiplier_bounds)
        return payload

    @property
    def summary(self) -> str:
        if self.reference_mode == "fixed":
            reference_text = f"ref={self.reference_value:g}" if self.reference_value is not None else "ref=fixed"
        elif self.reference_mode == "column":
            reference_text = f"ref={self.reference_column} * {self.reference_multiplier_value:g}"
        else:
            reference_text = "ref=fit"
        return f"{self.transform}, {reference_text}, scale={self.scale:g}"


@dataclass(frozen=True, slots=True)
class DerivedPreset:
    key: str
    category: str
    name: str
    description: str
    formula: str
    stress_preset: StressTermPreset | None = None
    auto_include: bool = True
    auto_stress: bool = True
    available: bool = True
    availability_note: str = "Available"
    required_inputs: tuple[str, ...] = ()


def _build_fixed_stress(
    column_name: str,
    description: str,
    transform: str,
    reference_value: float,
    scale: float,
    term_name: str | None = None,
) -> StressTermPreset:
    return StressTermPreset(
        name=term_name or f"{column_name}_stress",
        column=column_name,
        description=description,
        transform=transform,
        reference_mode="fixed",
        reference_value=reference_value,
        coefficient_mode="fit",
        coefficient_value=0.05,
        coefficient_non_negative=True,
        coefficient_bounds=(0.0, None),
        scale=scale,
    )


def _resolve_kpod_formula(columns: list[str]) -> str | None:
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    nominal = _find_first_column(columns, _aliases("pump_nominal_rate_m3d", extra=["pump_nominal_rate", "nominal_rate_m3d"]))
    if qliq and nominal:
        return f"[{qliq}] / [{nominal}]"
    return None


def _resolve_kpod_freq_adjusted_formula(columns: list[str]) -> str | None:
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    nominal = _find_first_column(columns, _aliases("pump_nominal_rate_m3d", extra=["pump_nominal_rate", "nominal_rate_m3d"]))
    reference_frequency = _find_first_column(
        columns,
        _aliases("reference_frequency_hz", "design_frequency_hz", extra=["reference frequency", "base_frequency_hz"]),
    )
    frequency = _find_first_column(columns, _aliases("frequency_hz", extra=["frequency", "freq_hz"]))
    if qliq and nominal and reference_frequency and frequency:
        return f"([{qliq}] / [{nominal}]) * ([{reference_frequency}] / [{frequency}])"
    return None


def _resolve_pressure_ratio_formula(columns: list[str]) -> str | None:
    pbubble = _find_first_column(columns, _aliases("Pbubble_atm", extra=["Pbubble", "pbubble"]))
    pintake = _find_first_column(columns, _aliases("P_intake_atm", extra=["p_intake_atm", "Pintake_atm", "Pintake"]))
    pbhp = _find_first_column(columns, _aliases("P_bhp_atm", extra=["p_bhp_atm", "Pbhp_atm", "Pbhp"]))
    if pbubble and pintake:
        return f"[{pintake}] / [{pbubble}]"
    if pbubble and pbhp:
        return f"[{pbhp}] / [{pbubble}]"
    return None


def _resolve_design_pressure_ratio_formula(columns: list[str]) -> str | None:
    pbubble = _find_first_column(columns, _aliases("design_Pbubble_atm", extra=["design_pbubble_atm", "design_Pbubble"]))
    pintake = _find_first_column(columns, _aliases("design_Pintake_atm", extra=["design_pintake_atm", "design_Pintake"]))
    pbhp = _find_first_column(columns, _aliases("design_Pbhp_atm", extra=["design_pbhp_atm", "design_Pbhp"]))
    if pbubble and pintake:
        return f"[{pintake}] / [{pbubble}]"
    if pbubble and pbhp:
        return f"[{pbhp}] / [{pbubble}]"
    return None


def _resolve_qliq_to_design_ratio_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("Qliq_m3d", extra=["actual_Qliq_mean", "actual_qliq_mean", "Qliq"]))
    design = _find_first_column(columns, _aliases("design_Qliq_m3d", extra=["design qliq", "design_Qliq"]))
    if actual and design:
        return f"[{actual}] / [{design}]"
    return None


def _resolve_glf_to_design_ratio_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("GLF_m3m3", extra=["GLF", "actual_GLF"]))
    design = _find_first_column(columns, _aliases("design_GLF_m3m3", extra=["design_GLF", "design glf"]))
    if actual and design:
        return f"[{actual}] / [{design}]"
    return None


def _resolve_free_gas_to_design_ratio_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(
        columns,
        _aliases("free_gas_intake_percent", extra=["actual_free_gas", "free_gas_percent", "free_gas_intake_fraction"]),
    )
    design = _find_first_column(columns, _aliases("design_free_gas_intake_percent", extra=["design_free_gas", "design free gas intake"]))
    if actual and design:
        return f"[{actual}] / [{design}]"
    return None


def _resolve_gas_excess_percent_formula(columns: list[str]) -> str | None:
    actual_percent = _find_first_column(columns, _aliases("free_gas_intake_percent", extra=["free_gas_percent", "actual_free_gas"]))
    limit_percent = _find_first_column(columns, ["free_gas_limit_percent", "gas_limit_percent", "лимит свободного газа, %"])
    if actual_percent and limit_percent:
        return f"max(0, [{actual_percent}] - [{limit_percent}])"

    actual_percent = _find_first_column(columns, _aliases("free_gas_intake_percent", extra=["free_gas_percent", "actual_free_gas"]))
    limit_fraction = _find_first_column(columns, _aliases("free_gas_limit_fraction", extra=["gas_limit_fraction"]))
    if actual_percent and limit_fraction:
        return f"max(0, [{actual_percent}] - (100 * [{limit_fraction}]))"

    return None


def _resolve_gas_excess_fraction_formula(columns: list[str]) -> str | None:
    actual_fraction = _find_first_column(columns, ["free_gas_intake_fraction", "free_gas_fraction", "свободный газ на приеме, доля"])
    limit_fraction = _find_first_column(columns, _aliases("free_gas_limit_fraction", extra=["gas_limit_fraction"]))
    if actual_fraction and limit_fraction:
        return f"max(0, [{actual_fraction}] - [{limit_fraction}])"
    return None


def _resolve_motor_load_to_design_ratio_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("motor_load_percent", extra=["motor_load", "PED_load", "ped_load"]))
    design = _find_first_column(columns, _aliases("design_motor_load_percent", extra=["design motor load"]))
    if actual and design:
        return f"[{actual}] / [{design}]"
    return None


def _resolve_frequency_to_reference_ratio_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("frequency_hz", extra=["frequency", "freq_hz"]))
    reference = _find_first_column(columns, _aliases("reference_frequency_hz", "design_frequency_hz", extra=["reference frequency"]))
    if actual and reference:
        return f"[{actual}] / [{reference}]"
    return None


def _resolve_free_gas_limit_percent_formula(columns: list[str]) -> str | None:
    limit_fraction = _find_first_column(columns, _aliases("free_gas_limit_fraction", extra=["gas_limit_fraction"]))
    if limit_fraction:
        return f"100 * [{limit_fraction}]"
    return None


def _resolve_kpod_to_design_ratio_formula(columns: list[str]) -> str | None:
    design_kpod = _find_first_column(columns, _aliases("design_Kpod", extra=["design_kpod"]))
    if _has_column(columns, "Kpod") and design_kpod:
        return f"[Kpod] / [{design_kpod}]"
    return None


def _resolve_pressure_ratio_to_design_ratio_formula(columns: list[str]) -> str | None:
    if _has_column(columns, "pressure_ratio") and _has_column(columns, "design_pressure_ratio"):
        return "[pressure_ratio] / [design_pressure_ratio]"
    return None


def _resolve_motor_load_over_design_pp_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("motor_load_percent", extra=["motor_load", "PED_load", "ped_load"]))
    design = _find_first_column(columns, _aliases("design_motor_load_percent", extra=["design motor load"]))
    if actual and design:
        return f"[{actual}] - [{design}]"
    return None


def _resolve_frequency_over_reference_hz_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("frequency_hz", extra=["frequency", "freq_hz"]))
    reference = _find_first_column(columns, _aliases("reference_frequency_hz", "design_frequency_hz", extra=["reference frequency"]))
    if actual and reference:
        return f"[{actual}] - [{reference}]"
    return None


def _resolve_watercut_to_design_ratio_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("watercut_percent", extra=["watercut", "actual_watercut"]))
    design = _find_first_column(columns, _aliases("design_watercut_percent", extra=["design_watercut"]))
    if actual and design:
        return f"[{actual}] / [{design}]"
    return None


def _resolve_qliq_per_hz_formula(columns: list[str]) -> str | None:
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    frequency = _find_first_column(columns, _aliases("frequency_hz", extra=["frequency", "freq_hz"]))
    if qliq and frequency:
        return f"[{qliq}] / [{frequency}]"
    return None


def _resolve_qliq_per_kw_formula(columns: list[str]) -> str | None:
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    power = _find_first_column(columns, _aliases("motor_power_kw", extra=["motor power kw", "ped_power_kw"]))
    if qliq and power:
        return f"[{qliq}] / [{power}]"
    return None


def _resolve_current_to_nominal_ratio_formula(columns: list[str]) -> str | None:
    actual = _find_first_column(columns, _aliases("actual_current_a", extra=["actual current", "current_a"]))
    nominal = _find_first_column(columns, _aliases("nominal_current_a", extra=["nominal current", "nominal_current_a"]))
    if actual and nominal:
        return f"[{actual}] / [{nominal}]"
    return None


def _resolve_qliq_per_current_formula(columns: list[str]) -> str | None:
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    actual = _find_first_column(columns, _aliases("actual_current_a", extra=["actual current", "current_a"]))
    if qliq and actual:
        return f"[{qliq}] / [{actual}]"
    return None


def _resolve_bhp_to_reservoir_ratio_formula(columns: list[str]) -> str | None:
    pbhp = _find_first_column(columns, _aliases("P_bhp_atm", extra=["p_bhp_atm", "Pbhp_atm", "Pbhp"]))
    reservoir = _find_first_column(columns, _aliases("reservoir_pressure_atm", extra=["pres_atm", "reservoir_pressure"]))
    if pbhp and reservoir:
        return f"[{pbhp}] / [{reservoir}]"
    return None


def _resolve_pressure_margin_to_bubble_formula(columns: list[str]) -> str | None:
    pbhp = _find_first_column(columns, _aliases("P_bhp_atm", extra=["p_bhp_atm", "Pbhp_atm", "Pbhp"]))
    pbubble = _find_first_column(columns, _aliases("Pbubble_atm", extra=["Pbubble", "pbubble"]))
    if pbhp and pbubble:
        return f"[{pbhp}] - [{pbubble}]"
    return None


def _resolve_submergence_margin_m_formula(columns: list[str]) -> str | None:
    depth = _find_first_column(columns, _aliases("install_depth_m", extra=["pump_depth_m", "install depth"]))
    dynamic = _find_first_column(columns, _aliases("dynamic_level_m", extra=["dynamic_level", "dynamic level"]))
    if depth and dynamic:
        return f"[{depth}] - [{dynamic}]"
    return None


def _resolve_submergence_margin_ratio_formula(columns: list[str]) -> str | None:
    depth = _find_first_column(columns, _aliases("install_depth_m", extra=["pump_depth_m", "install depth"]))
    dynamic = _find_first_column(columns, _aliases("dynamic_level_m", extra=["dynamic_level", "dynamic level"]))
    if depth and dynamic:
        return f"([{depth}] - [{dynamic}]) / [{depth}]"
    return None


def _resolve_nominal_head_per_stage_formula(columns: list[str]) -> str | None:
    head = _find_first_column(columns, _aliases("nominal_head_m", extra=["nominal_head", "head_50hz"]))
    stages = _find_first_column(columns, _aliases("stage_count", extra=["stages", "stage count"]))
    if head and stages:
        return f"[{head}] / [{stages}]"
    return None


def _resolve_motor_load_per_hz_formula(columns: list[str]) -> str | None:
    motor_load = _find_first_column(columns, _aliases("motor_load_percent", extra=["motor_load", "PED_load", "ped_load"]))
    frequency = _find_first_column(columns, _aliases("frequency_hz", extra=["frequency", "freq_hz"]))
    if motor_load and frequency:
        return f"[{motor_load}] / [{frequency}]"
    return None


def _resolve_curve_work_per_meter_formula(columns: list[str]) -> str | None:
    curve = _find_first_column(columns, _aliases("work_in_curvature"))
    length = _find_first_column(columns, _aliases("pump_length_m", extra=["pump_length", "pump_length_m"]))
    if curve and length:
        return f"[{curve}] / [{length}]"
    return None


def _resolve_cumulative_ion_load_formula(columns: list[str], ion_key: str) -> str | None:
    ion = _find_first_column(columns, _aliases(ion_key))
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    duration = _find_first_column(columns, _aliases("duration_days"))
    if ion and qliq and duration:
        # mg/L * m3/day * day / 1000 -> kg over the run (1 m3 = 1000 L).
        return f"([{ion}] * [{qliq}] * [{duration}]) / 1000"
    return None


def _resolve_total_salt_load_formula(columns: list[str]) -> str | None:
    calcium = _find_first_column(columns, _aliases("calcium_mg_l"))
    chloride = _find_first_column(columns, _aliases("chloride_mg_l"))
    sulfate = _find_first_column(columns, _aliases("sulfate_mg_l"))
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    duration = _find_first_column(columns, _aliases("duration_days"))
    if calcium and chloride and sulfate and qliq and duration:
        return f"([{calcium}] + [{chloride}] + [{sulfate}]) * [{qliq}] * [{duration}] / 1000"
    return None


def _resolve_gypsum_scale_proxy_formula(columns: list[str]) -> str | None:
    calcium = _find_first_column(columns, _aliases("calcium_mg_l"))
    sulfate = _find_first_column(columns, _aliases("sulfate_mg_l"))
    qliq = _find_first_column(columns, _aliases("Qliq_m3d", extra=["Qliq", "qliq"]))
    duration = _find_first_column(columns, _aliases("duration_days"))
    if calcium and sulfate and qliq and duration:
        # Interaction proxy kept in scaled arbitrary units to stay numerically manageable.
        return f"([{calcium}] * [{sulfate}] * [{qliq}] * [{duration}]) / 1000000"
    return None


def _resolve_glf_excess_over_design_formula(columns: list[str]) -> str | None:
    if _has_column(columns, "glf_to_design_ratio"):
        return "max(0, [glf_to_design_ratio] - 1)"
    return None


def _resolve_free_gas_excess_over_design_formula(columns: list[str]) -> str | None:
    if _has_column(columns, "free_gas_to_design_ratio"):
        return "max(0, [free_gas_to_design_ratio] - 1)"
    return None


PRESET_RESOLVERS = [
    {
        "key": "kpod",
        "category": "Hydraulic",
        "name": "Kpod",
        "description": "Pump delivery coefficient from liquid rate and nominal pump rate.",
        "required_inputs": ("Qliq_m3d", "pump_nominal_rate_m3d"),
        "resolver": _resolve_kpod_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Low Kpod stress with the standard 0.7 threshold.",
            "negative_excess",
            0.7,
            0.1,
            term_name="stress_low_Kpod",
        ),
    },
    {
        "key": "kpod_freq_adjusted",
        "category": "Hydraulic",
        "name": "Kpod_freq_adjusted",
        "description": "Frequency-adjusted Kpod using reference and actual frequency.",
        "required_inputs": ("Qliq_m3d", "pump_nominal_rate_m3d", "reference_frequency_hz or design_frequency_hz", "frequency_hz"),
        "resolver": _resolve_kpod_freq_adjusted_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Low frequency-adjusted Kpod stress with the standard 0.7 threshold.",
            "negative_excess",
            0.7,
            0.1,
            term_name="stress_low_Kpod_freq_adjusted",
        ),
    },
    {
        "key": "pressure_ratio",
        "category": "Pressure / Gas",
        "name": "pressure_ratio",
        "description": "Pressure ratio using Pintake/Pbubble if available, otherwise Pbhp/Pbubble.",
        "required_inputs": ("Pbubble_atm", "P_intake_atm or P_bhp_atm"),
        "resolver": _resolve_pressure_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Low pressure-ratio stress with the standard 0.7 threshold.",
            "negative_excess",
            0.7,
            0.1,
            term_name="stress_low_pressure_ratio",
        ),
    },
    {
        "key": "design_pressure_ratio",
        "category": "Pressure / Gas",
        "name": "design_pressure_ratio",
        "description": "Design pressure ratio using design Pintake/Pbubble if available, otherwise design Pbhp/Pbubble.",
        "required_inputs": ("design_Pbubble_atm", "design_Pintake_atm or design_Pbhp_atm"),
        "resolver": _resolve_design_pressure_ratio_formula,
        "stress_factory": None,
    },
    {
        "key": "qliq_to_design_ratio",
        "category": "Design Mismatch",
        "name": "qliq_to_design_ratio",
        "description": "Actual-to-design liquid-rate ratio.",
        "required_inputs": ("Qliq_m3d", "design_Qliq_m3d"),
        "resolver": _resolve_qliq_to_design_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Underproduction stress relative to the design liquid rate.",
            "negative_excess",
            1.0,
            0.1,
            term_name="stress_underproduction",
        ),
    },
    {
        "key": "glf_to_design_ratio",
        "category": "Design Mismatch",
        "name": "glf_to_design_ratio",
        "description": "Actual-to-design gas-liquid-factor ratio.",
        "required_inputs": ("GLF_m3m3", "design_GLF_m3m3"),
        "resolver": _resolve_glf_to_design_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Gas-design mismatch stress when actual GLF exceeds design GLF.",
            "positive_excess",
            1.0,
            0.25,
            term_name="stress_gas_design_miss",
        ),
    },
    {
        "key": "free_gas_to_design_ratio",
        "category": "Design Mismatch",
        "name": "free_gas_to_design_ratio",
        "description": "Actual-to-design free-gas ratio at intake.",
        "required_inputs": ("free_gas_intake_percent", "design_free_gas_intake_percent"),
        "resolver": _resolve_free_gas_to_design_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Free-gas design mismatch stress when actual free gas exceeds design free gas.",
            "positive_excess",
            1.0,
            0.1,
            term_name="stress_free_gas_design_miss",
        ),
    },
    {
        "key": "gas_excess_percent",
        "category": "Pressure / Gas",
        "name": "gas_excess_percent",
        "description": "Gas above the equipment limit expressed in percentage points.",
        "required_inputs": ("free_gas_intake_percent", "free_gas_limit_percent or free_gas_limit_fraction"),
        "resolver": _resolve_gas_excess_percent_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Gas-excess stress for percentage-point exceedance above the equipment limit.",
            "positive_excess",
            0.0,
            5.0,
            term_name="stress_gas_excess_percent",
        ),
    },
    {
        "key": "gas_excess_fraction",
        "category": "Pressure / Gas",
        "name": "gas_excess_fraction",
        "description": "Gas above the equipment limit expressed as a fraction.",
        "required_inputs": ("free_gas_intake_fraction", "free_gas_limit_fraction"),
        "resolver": _resolve_gas_excess_fraction_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Gas-excess stress for fraction exceedance above the equipment limit.",
            "positive_excess",
            0.0,
            0.05,
            term_name="stress_gas_excess_fraction",
        ),
    },
    {
        "key": "motor_load_to_design_ratio",
        "category": "Load",
        "name": "motor_load_to_design_ratio",
        "description": "Actual-to-design motor-load ratio.",
        "required_inputs": ("motor_load_percent", "design_motor_load_percent"),
        "resolver": _resolve_motor_load_to_design_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Motor-load overload stress when actual load exceeds the design load.",
            "positive_excess",
            1.0,
            0.05,
            term_name="stress_motor_load_over_design",
        ),
    },
    {
        "key": "frequency_to_reference_ratio",
        "category": "Control",
        "name": "frequency_to_reference_ratio",
        "description": "Actual-to-reference frequency ratio.",
        "required_inputs": ("frequency_hz", "reference_frequency_hz or design_frequency_hz"),
        "resolver": _resolve_frequency_to_reference_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Frequency-up stress when actual frequency exceeds the reference frequency.",
            "positive_excess",
            1.0,
            0.05,
            term_name="stress_frequency_over_reference",
        ),
    },
    {
        "key": "free_gas_limit_percent",
        "category": "Pressure / Gas",
        "name": "free_gas_limit_percent",
        "description": "Equipment free-gas limit expressed in percentage points.",
        "required_inputs": ("free_gas_limit_fraction",),
        "resolver": _resolve_free_gas_limit_percent_formula,
        "stress_factory": None,
    },
    {
        "key": "kpod_to_design_ratio",
        "category": "Design Mismatch",
        "name": "kpod_to_design_ratio",
        "description": "Actual Kpod relative to design Kpod.",
        "required_inputs": ("Kpod", "design_Kpod"),
        "resolver": _resolve_kpod_to_design_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Stress when actual Kpod falls below design Kpod.",
            "negative_excess",
            1.0,
            0.1,
            term_name="stress_kpod_design_miss",
        ),
    },
    {
        "key": "pressure_ratio_to_design_ratio",
        "category": "Design Mismatch",
        "name": "pressure_ratio_to_design_ratio",
        "description": "Actual pressure ratio relative to design pressure ratio.",
        "required_inputs": ("pressure_ratio", "design_pressure_ratio"),
        "resolver": _resolve_pressure_ratio_to_design_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Stress when actual pressure ratio falls below design pressure ratio.",
            "negative_excess",
            1.0,
            0.1,
            term_name="stress_pressure_ratio_design_miss",
        ),
    },
    {
        "key": "motor_load_over_design_pp",
        "category": "Load",
        "name": "motor_load_over_design_pp",
        "description": "Actual motor load minus design motor load in percentage points.",
        "required_inputs": ("motor_load_percent", "design_motor_load_percent"),
        "resolver": _resolve_motor_load_over_design_pp_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Overload stress when actual motor load exceeds design motor load.",
            "positive_excess",
            0.0,
            5.0,
            term_name="stress_motor_load_over_design_pp",
        ),
    },
    {
        "key": "frequency_over_reference_hz",
        "category": "Control",
        "name": "frequency_over_reference_hz",
        "description": "Actual frequency minus the reference frequency in Hz.",
        "required_inputs": ("frequency_hz", "reference_frequency_hz or design_frequency_hz"),
        "resolver": _resolve_frequency_over_reference_hz_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Frequency-up stress in absolute Hz above the reference frequency.",
            "positive_excess",
            0.0,
            2.0,
            term_name="stress_frequency_over_reference_hz",
        ),
    },
    {
        "key": "watercut_to_design_ratio",
        "category": "Fluids",
        "name": "watercut_to_design_ratio",
        "description": "Actual watercut relative to design watercut.",
        "required_inputs": ("watercut_percent", "design_watercut_percent"),
        "resolver": _resolve_watercut_to_design_ratio_formula,
        "stress_factory": None,
    },
    {
        "key": "qliq_per_hz",
        "category": "Hydraulic",
        "name": "qliq_per_hz",
        "description": "Liquid rate per Hz of operating frequency.",
        "required_inputs": ("Qliq_m3d", "frequency_hz"),
        "resolver": _resolve_qliq_per_hz_formula,
        "stress_factory": None,
    },
    {
        "key": "qliq_per_kw",
        "category": "Hydraulic",
        "name": "qliq_per_kw",
        "description": "Liquid rate per kW of installed motor power.",
        "required_inputs": ("Qliq_m3d", "motor_power_kw"),
        "resolver": _resolve_qliq_per_kw_formula,
        "stress_factory": None,
    },
    {
        "key": "current_to_nominal_ratio",
        "category": "Electrical",
        "name": "current_to_nominal_ratio",
        "description": "Actual current relative to the nominal motor current.",
        "required_inputs": ("actual_current_a", "nominal_current_a"),
        "resolver": _resolve_current_to_nominal_ratio_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Electrical overload stress when actual current exceeds nominal current.",
            "positive_excess",
            1.0,
            0.05,
            term_name="stress_current_over_nominal",
        ),
    },
    {
        "key": "qliq_per_current",
        "category": "Hydraulic",
        "name": "qliq_per_current",
        "description": "Liquid rate per ampere of actual current.",
        "required_inputs": ("Qliq_m3d", "actual_current_a"),
        "resolver": _resolve_qliq_per_current_formula,
        "stress_factory": None,
    },
    {
        "key": "bhp_to_reservoir_ratio",
        "category": "Pressure / Drawdown",
        "name": "bhp_to_reservoir_ratio",
        "description": "Bottom-hole pressure relative to reservoir pressure.",
        "required_inputs": ("P_bhp_atm", "reservoir_pressure_atm"),
        "resolver": _resolve_bhp_to_reservoir_ratio_formula,
        "stress_factory": None,
    },
    {
        "key": "pressure_margin_to_bubble",
        "category": "Pressure / Gas",
        "name": "pressure_margin_to_bubble",
        "description": "Absolute pressure margin above bubble-point pressure.",
        "required_inputs": ("P_bhp_atm", "Pbubble_atm"),
        "resolver": _resolve_pressure_margin_to_bubble_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Gas-breakout stress when bottom-hole pressure falls below bubble-point pressure.",
            "negative_excess",
            0.0,
            5.0,
            term_name="stress_low_pressure_margin_to_bubble",
        ),
    },
    {
        "key": "submergence_margin_m",
        "category": "Completion",
        "name": "submergence_margin_m",
        "description": "Pump submergence margin in meters using installation depth minus dynamic level.",
        "required_inputs": ("install_depth_m", "dynamic_level_m"),
        "resolver": _resolve_submergence_margin_m_formula,
        "stress_factory": None,
    },
    {
        "key": "submergence_margin_ratio",
        "category": "Completion",
        "name": "submergence_margin_ratio",
        "description": "Normalized pump submergence margin relative to installation depth.",
        "required_inputs": ("install_depth_m", "dynamic_level_m"),
        "resolver": _resolve_submergence_margin_ratio_formula,
        "stress_factory": None,
    },
    {
        "key": "nominal_head_per_stage",
        "category": "Equipment",
        "name": "nominal_head_per_stage",
        "description": "Nominal pump head per stage at 50 Hz.",
        "required_inputs": ("nominal_head_m", "stage_count"),
        "resolver": _resolve_nominal_head_per_stage_formula,
        "stress_factory": None,
    },
    {
        "key": "motor_load_per_hz",
        "category": "Electrical",
        "name": "motor_load_per_hz",
        "description": "Motor-load percentage per Hz of operating frequency.",
        "required_inputs": ("motor_load_percent", "frequency_hz"),
        "resolver": _resolve_motor_load_per_hz_formula,
        "stress_factory": None,
    },
    {
        "key": "curve_work_per_meter",
        "category": "Trajectory",
        "name": "curve_work_per_meter",
        "description": "Work-in-curvature normalized by pump length.",
        "required_inputs": ("work_in_curvature", "pump_length_m"),
        "resolver": _resolve_curve_work_per_meter_formula,
        "stress_factory": None,
    },
    {
        "key": "cum_calcium_load_kg",
        "category": "Salts / Scale",
        "name": "cum_calcium_load_kg",
        "description": "Run-level calcium throughput proxy in kg: Ca concentration × liquid rate × run duration.",
        "required_inputs": ("calcium_mg_l", "Qliq_m3d", "duration_days"),
        "resolver": lambda columns: _resolve_cumulative_ion_load_formula(columns, "calcium_mg_l"),
        "stress_factory": None,
    },
    {
        "key": "cum_chloride_load_kg",
        "category": "Salts / Scale",
        "name": "cum_chloride_load_kg",
        "description": "Run-level chloride throughput proxy in kg; useful as a halite/brine salinity exposure indicator.",
        "required_inputs": ("chloride_mg_l", "Qliq_m3d", "duration_days"),
        "resolver": lambda columns: _resolve_cumulative_ion_load_formula(columns, "chloride_mg_l"),
        "stress_factory": None,
    },
    {
        "key": "cum_sulfate_load_kg",
        "category": "Salts / Scale",
        "name": "cum_sulfate_load_kg",
        "description": "Run-level sulfate throughput proxy in kg; useful as a sulfate-scale exposure indicator.",
        "required_inputs": ("sulfate_mg_l", "Qliq_m3d", "duration_days"),
        "resolver": lambda columns: _resolve_cumulative_ion_load_formula(columns, "sulfate_mg_l"),
        "stress_factory": None,
    },
    {
        "key": "cum_salt_load_kg",
        "category": "Salts / Scale",
        "name": "cum_salt_load_kg",
        "description": "Combined salt-throughput proxy in kg using Ca + Cl + SO4 concentrations with liquid rate and run duration.",
        "required_inputs": ("calcium_mg_l", "chloride_mg_l", "sulfate_mg_l", "Qliq_m3d", "duration_days"),
        "resolver": _resolve_total_salt_load_formula,
        "stress_factory": None,
    },
    {
        "key": "gypsum_scale_proxy",
        "category": "Salts / Scale",
        "name": "gypsum_scale_proxy",
        "description": "Scaled Ca×SO4 interaction proxy for gypsum/anhydrite exposure using liquid rate and run duration.",
        "required_inputs": ("calcium_mg_l", "sulfate_mg_l", "Qliq_m3d", "duration_days"),
        "resolver": _resolve_gypsum_scale_proxy_formula,
        "stress_factory": None,
    },
    {
        "key": "glf_excess_over_design",
        "category": "Design Mismatch",
        "name": "glf_excess_over_design",
        "description": "Positive exceedance of actual GLF relative to design GLF ratio.",
        "required_inputs": ("glf_to_design_ratio",),
        "resolver": _resolve_glf_excess_over_design_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Stress for GLF exceedance above design conditions.",
            "positive_excess",
            0.0,
            0.1,
            term_name="stress_glf_excess_over_design",
        ),
    },
    {
        "key": "free_gas_excess_over_design",
        "category": "Design Mismatch",
        "name": "free_gas_excess_over_design",
        "description": "Positive exceedance of actual free gas relative to design free gas ratio.",
        "required_inputs": ("free_gas_to_design_ratio",),
        "resolver": _resolve_free_gas_excess_over_design_formula,
        "stress_factory": lambda name: _build_fixed_stress(
            name,
            "Stress for free-gas exceedance above design conditions.",
            "positive_excess",
            0.0,
            0.1,
            term_name="stress_free_gas_excess_over_design",
        ),
    },
]


def inspect_derived_presets(df: pd.DataFrame) -> list[DerivedPreset]:
    available_columns = [str(column) for column in df.columns.tolist()]
    presets: list[DerivedPreset] = []

    for item in PRESET_RESOLVERS:
        if _has_column(available_columns, item["name"]):
            presets.append(
                DerivedPreset(
                    key=item["key"],
                    category=item["category"],
                    name=item["name"],
                    description=item["description"],
                    formula="",
                    stress_preset=None,
                    auto_include=False,
                    auto_stress=False,
                    available=False,
                    availability_note="Already present in the uploaded sheet.",
                    required_inputs=tuple(item.get("required_inputs", ())),
                )
            )
            continue
        formula = item["resolver"](available_columns)
        if formula is None:
            presets.append(
                DerivedPreset(
                    key=item["key"],
                    category=item["category"],
                    name=item["name"],
                    description=item["description"],
                    formula="",
                    stress_preset=None,
                    auto_include=False,
                    auto_stress=False,
                    available=False,
                    availability_note="Not currently derivable from recognized source columns on this sheet.",
                    required_inputs=tuple(item.get("required_inputs", ())),
                )
            )
            continue
        stress_preset = item["stress_factory"](item["name"]) if item.get("stress_factory") else None
        presets.append(
            DerivedPreset(
                key=item["key"],
                category=item["category"],
                name=item["name"],
                description=item["description"],
                formula=formula,
                stress_preset=stress_preset,
                auto_include=True,
                auto_stress=stress_preset is not None,
                available=True,
                availability_note="Available",
                required_inputs=tuple(item.get("required_inputs", ())),
            )
        )
        available_columns.append(item["name"])

    return presets


def suggest_derived_presets(df: pd.DataFrame) -> list[DerivedPreset]:
    return [preset for preset in inspect_derived_presets(df) if preset.available]
