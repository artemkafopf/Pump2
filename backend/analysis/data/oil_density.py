"""Плотность нефти по (месторождение, ЛУ) — для перевода между газовыми осями.

Газовый фактор существует в двух базах, и они связаны точным тождеством:

    ГЖФ [м³/м³ жидкости] = ГФ [м³/т нефти] × ρ_нефти × (1 − обводнённость)

⚠⚠ **Плотность берётся ИЗ СПРАВОЧНИКА, а не из отношения источников.** Отношение
``ГЖФ / (ГФ × (1 − ХВ))`` на данных действительно выходит константой по группе до
третьего знака, и соблазн взять его как ρ велик — но это подгонка под сам предмет
измерения: если ось потом сравнивается с источником, из которого выведен коэффициент,
согласие гарантировано по построению и ничего не проверяет.

Поэтому здесь ровно наоборот: справочник даёт ЗНАЧЕНИЕ, а наблюдённое отношение служит
**контролем стыковки** (:func:`validate_against_observed`). Расхождение значит, что пара
подобрана неверно, — и его надо увидеть, а не сгладить.

⚠ Ключ — ПАРА (месторождение, ЛУ), не месторождение: у ``Bt``, ``Ya`` и группы «Без
месторождения» плотность различается по участкам (``Bt_Vt`` 0.811 против ``Bt_Bt`` 0.821,
``Ya_Ki`` 0.829 против ``Ya_Au`` 0.833).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from analysis.paths import resolve_oil_density_path

#: Насколько наблюдённое отношение может отличаться от справочного, прежде чем это
#: считается ошибкой стыковки, а не шумом. Плотности разнесены на 0.005–0.01, поэтому
#: 1 % — уже уровень «перепутан участок».
DENSITY_TOLERANCE = 0.01

#: Физически возможный диапазон плотности нефти, т/м³. Значение вне него — признак
#: того, что в колонку попало не то.
DENSITY_RANGE = (0.75, 0.95)


@dataclass(frozen=True)
class DensityLookup:
    """Справочник плотности с явным порядком разрешения ключа."""

    by_pair: dict[tuple[str, str], float]

    def get(self, field: object, lu: object, well_prefix: object = None) -> Optional[float]:
        """Плотность по паре, с документированным каскадом кандидатов.

        Справочник смешивает длинные имена («Большетирское НМ») и короткие коды
        («Bt»), причём в обеих колонках, поэтому одного варианта ключа не хватает.
        Каскад фиксирован и проверяем; ``None`` означает «в справочнике этого нет» —
        и это НЕ повод подставить соседнее значение.
        """
        candidates = [
            (field, lu),
            (well_prefix, lu),
            (field, well_prefix),
            (well_prefix, well_prefix),
        ]
        for raw_field, raw_lu in candidates:
            if raw_field is None or raw_lu is None:
                continue
            # ⚠⚠ Сравнение РЕГИСТРОНЕЗАВИСИМОЕ. Телеметрия отдаёт скважину как
            # ``Ya_403``, техрежим — как ``ya_403``, поэтому префикс приходит то
            # ``Ya``, то ``ya``, а справочник знает только ``Ya``. С точным
            # сравнением плотность молча не находилась у 70 530 строк одного лишь
            # Ярактинского — при том что пара в справочнике ЕСТЬ.
            key = (str(raw_field).strip().casefold(), str(raw_lu).strip().casefold())
            if key in self.by_pair:
                return self.by_pair[key]
        return None

    def __len__(self) -> int:  # pragma: no cover - удобство отладки
        return len(self.by_pair)


@lru_cache(maxsize=2)
def load_density_lookup(path: Optional[str] = None) -> DensityLookup:
    """Прочитать справочник плотностей."""
    source = pd.read_csv(path or resolve_oil_density_path())
    missing = {"field", "lu", "odens_t_m3"} - set(source.columns)
    if missing:
        raise KeyError(f"oil_density: в справочнике нет колонок {sorted(missing)}")

    by_pair: dict[tuple[str, str], float] = {}
    for _, row in source.iterrows():
        density = pd.to_numeric(row["odens_t_m3"], errors="coerce")
        if pd.isna(density):
            continue
        low, high = DENSITY_RANGE
        if not (low <= float(density) <= high):
            raise ValueError(
                f"oil_density: {row['field']}_{row['lu']} = {density} т/м³ вне диапазона "
                f"{DENSITY_RANGE} — в колонке не плотность"
            )
        # Ключи хранятся сложенными по регистру — см. оговорку в ``DensityLookup.get``.
        by_pair[(str(row["field"]).strip().casefold(), str(row["lu"]).strip().casefold())] = float(density)
    return DensityLookup(by_pair=by_pair)


def attach_density(
    frame: pd.DataFrame,
    *,
    field_col: str = "field",
    lu_col: str = "lu",
    well_col: Optional[str] = "well",
    out_col: str = "oil_density_t_m3",
    lookup: Optional[DensityLookup] = None,
) -> pd.DataFrame:
    """Приписать плотность каждой строке; ``NaN`` там, где справочник молчит.

    ⚠ Пропуск НЕ заполняется ни средним, ни соседним участком: неверная плотность
    молча масштабирует всю газовую ось на несколько процентов, и отличить это потом
    от физики нельзя.
    """
    lookup = lookup or load_density_lookup()
    result = frame.copy()
    prefix = (
        result[well_col].astype("string").str.extract(r"^([A-Za-zА-Яа-яЁё]+)_", expand=False)
        if well_col and well_col in result.columns
        else pd.Series([None] * len(result), index=result.index)
    )
    fields = result[field_col] if field_col in result.columns else pd.Series([None] * len(result), index=result.index)
    lus = result[lu_col] if lu_col in result.columns else pd.Series([None] * len(result), index=result.index)
    result[out_col] = [
        lookup.get(field, lu, well_prefix)
        for field, lu, well_prefix in zip(fields, lus, prefix)
    ]
    return result


def gas_liquid_ratio_from_gor(
    gas_factor_m3t, oil_density_t_m3, watercut_percent
) -> pd.Series:
    """ГЖФ [м³/м³] из ГФ [м³/т] по точному тождеству.

    ⚠ Обводнённость в ПРОЦЕНТАХ. При ХВ = 100 % нефти нет вовсе и величина не
    определена — возвращается ``NaN``, а не ноль: ноль означал бы «газа нет».
    """
    gor = pd.to_numeric(gas_factor_m3t, errors="coerce")
    density = pd.to_numeric(oil_density_t_m3, errors="coerce")
    watercut = pd.to_numeric(watercut_percent, errors="coerce")
    oil_share = 1.0 - watercut / 100.0
    oil_share = oil_share.where((oil_share > 0) & (oil_share <= 1.0))
    return gor * density * oil_share


def gor_from_gas_liquid_ratio(
    gas_liquid_ratio_m3m3, oil_density_t_m3, watercut_percent
) -> pd.Series:
    """Обратный перевод: ГФ [м³/т] из ГЖФ [м³/м³]."""
    glf = pd.to_numeric(gas_liquid_ratio_m3m3, errors="coerce")
    density = pd.to_numeric(oil_density_t_m3, errors="coerce")
    watercut = pd.to_numeric(watercut_percent, errors="coerce")
    oil_share = 1.0 - watercut / 100.0
    denominator = (density * oil_share).where((oil_share > 0) & (density > 0))
    return glf / denominator


def validate_against_observed(
    frame: pd.DataFrame,
    *,
    gor_col: str = "gas_factor_m3t",
    glf_col: str = "gas_liquid_ratio_m3m3",
    watercut_col: str = "watercut_percent",
    density_col: str = "oil_density_t_m3",
    group_cols: Iterable[str] = ("field", "lu"),
    tolerance: float = DENSITY_TOLERANCE,
) -> pd.DataFrame:
    """Сверить справочную плотность с наблюдённым отношением ГЖФ/(ГФ×(1−ХВ)).

    Контроль стыковки, а не источник значения: отношение считается только там, где
    ОБЕ величины измерены независимо. Строка отчёта на группу; ``расхождение`` за
    пределами допуска означает, что пара подобрана неверно.
    """
    group_cols = [c for c in group_cols if c in frame.columns]
    gor = pd.to_numeric(frame.get(gor_col), errors="coerce")
    glf = pd.to_numeric(frame.get(glf_col), errors="coerce")
    watercut = pd.to_numeric(frame.get(watercut_col), errors="coerce")
    oil_share = 1.0 - watercut / 100.0

    usable = gor.gt(0) & glf.gt(0) & oil_share.gt(0) & oil_share.le(1.0)
    observed = (glf / (gor * oil_share)).where(usable)

    working = frame.loc[:, group_cols].copy()
    working["_observed"] = observed
    working["_reference"] = pd.to_numeric(frame.get(density_col), errors="coerce")

    rows = []
    grouped = working.dropna(subset=["_observed"]).groupby(group_cols, dropna=False) if group_cols else []
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        reference = group["_reference"].dropna()
        reference_value = float(reference.median()) if not reference.empty else np.nan
        observed_value = float(group["_observed"].median())
        deviation = (
            abs(observed_value - reference_value) / reference_value
            if reference_value and not np.isnan(reference_value)
            else np.nan
        )
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "строк": int(len(group)),
                "наблюдённая": round(observed_value, 4),
                "справочная": round(reference_value, 4) if not np.isnan(reference_value) else None,
                "расхождение": round(deviation, 4) if not np.isnan(deviation) else None,
                "статус": (
                    "нет в справочнике"
                    if np.isnan(reference_value)
                    else "ОК" if deviation <= tolerance else "РАСХОЖДЕНИЕ"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("строк", ascending=False).reset_index(drop=True)


__all__ = [
    "DENSITY_RANGE",
    "DENSITY_TOLERANCE",
    "DensityLookup",
    "attach_density",
    "gas_liquid_ratio_from_gor",
    "gor_from_gas_liquid_ratio",
    "load_density_lookup",
    "validate_against_observed",
]
