"""Equipment-passport loader for WellsArtificialLiftBig.xlsx (one row per спуск).

The workbook has a two-level header: row A = equipment section («Насос (50Гц)»,
«Газосепаратор / Модуль входной», «Мультифазная секция», «Гидрозащита», «ПЭД»,
«ТМС», «ФА/НКТ», «Погружная кабельная линия», …), row B = the parameter within
that section («Тип ГНО», «Габарит УЭЦН», «Коррозионная стойкость», «Покрытие»,
«нов/рем», …). Columns are therefore addressed as (section, parameter) pairs —
never by positional index — so the loader survives column insertions upstream.

What is extracted (see results/wellsbig_review/2026-07-07/ for the full survey):

* run identity + the date quartet (монтаж / запуск / отказ / демонтаж) and
  recorded operating time (ННО) — the fail-vs-pull gap and clock arbiter;
* pump section: type, габарит, curvature, nominals, execution group,
  **corrosion class + coating**;
* per-component **corrosion classes (К0–К3) and coatings (монель/да/нет)** for
  gassep / protector / ПЭД / ТМС / НКТ / cable — the corrosion-execution profile
  the user flagged as first-class information;
* НКТ diameter/марка, cable сечение/length/max-t°, ПЭД max-t°, gassep type.

Execution letters embedded in Novomet-style type strings (ЭЦНДИ**К** = corrosion-
resistant, ЭЦНД**И**К = wear-resistant) are parsed into flags as well: the
dedicated corrosion-class column is filled on only ~1% of rows, while the type
string carries the execution marking for the bulk of the Russian fleet.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.paths import resolve_equipment_big_path
from analysis.data.pump_type_parser import GABARIT_OD_MM

# REDA series → OD mm (catalog-verified 2026-07-07; 540 is the naming exception).
REDA_SERIES_OD_MM: dict[str, float] = {
    "338": 85.9, "387": 98.3, "400": 101.6, "456": 115.8,
    "513": 130.3, "538": 136.7, "540": 130.3, "562": 142.7,
}

_RUS_GABARITS = {"2", "2А", "3", "4", "5", "5А", "6", "6А", "6Б", "7", "7А", "8", "9"}

# (output column) -> (section substring, parameter substring | None)
# Matched case-insensitively on normalized whitespace; None parameter = single-level column.
_SELECT: dict[str, tuple[str, str | None]] = {
    "well": ("Скважина", None),
    "field_raw": ("Месторождение", None),
    "pad": ("Куст", None),
    "run_no": ("№ спуска", None),
    "install_date": ("Дата монтажа", None),
    "launch_date": ("Дата запуска", None),
    "fail_date": ("Дата отказа", None),
    "pull_date": ("Дата демонтажа", None),
    "nno_days": ("ННО", None),
    "pull_reason": ("Причина подъема", None),
    # pump
    "gno_type": ("Насос", "Тип ГНО"),
    "pump_gabarit_raw": ("Насос", "Габарит УЭЦН"),
    "pump_length_m": ("Насос", "Длина УЭЦН"),
    "curvature": ("Насос", "Работа в кривизне"),
    "q_nom_m3d": ("Насос", "Ном. Произв"),
    "head_nom_m": ("Насос", "Ном.напор"),
    "freq_nom_hz": ("Насос", "Номинальная частота"),
    "stages": ("Насос", "Кол.ступеней"),
    "pump_exec_group_raw": ("Насос", "Группа исполнения"),
    "pump_napornost": ("Насос", "Напорность"),
    "pump_corr_raw": ("Насос", "Коррозионная стойкость"),
    "pump_coating_raw": ("Насос", "Покрытие"),
    # gas separator / intake
    "gassep_type": ("Газосепаратор", "Тип Газосепаратора"),
    "gassep_gabarit_raw": ("Газосепаратор", "Габарит ГС"),
    "intake_module": ("Газосепаратор", "Входной модуль"),
    "gassep_corr_raw": ("Газосепаратор", "Коррозионностойкость"),
    "gassep_coating_raw": ("Газосепаратор", "Покрытие"),
    # multiphase section
    "multiphase_type": ("Мультифазная", "Тип мультиф"),
    # protector
    "protector_type": ("Гидрозащита", "Тип Гидрозащиты"),
    "protector_corr_raw": ("Гидрозащита", "Коррозионностойкость"),
    "protector_coating_raw": ("Гидрозащита", "Покрытие"),
    # motor
    "ped_type": ("ПЭД", "Тип ПЭД"),
    "ped_oil": ("ПЭД", "Тип масла"),
    "ped_max_temp_c": ("ПЭД", "Макс. темп"),
    "ped_power_kw": ("ПЭД", "Мощность"),
    "ped_corr_raw": ("ПЭД", "Коррозионностойкость"),
    "ped_coating_raw": ("ПЭД", "Покрытие"),
    # downhole sensor
    "tms_type": ("ТМС", "Тип ТМС"),
    "tms_corr_raw": ("ТМС", "Коррозионностойкость"),
    # tubing
    "pump_depth_m": ("ФА/НКТ", "Глубина спуска УЭЦН"),
    "vg_m": ("ФА/НКТ", "ВГ, м"),
    "nkt_diam_mm": ("ФА/НКТ", "Диаметр НКТ"),
    "nkt_marka": ("ФА/НКТ", "Марка НКТ"),
    "nkt_corr_raw": ("ФА/НКТ", "Коррозионностойкость НКТ"),
    # cable
    "cable_model": ("кабельная линия", "Модель кабеля"),
    "cable_section_mm2": ("кабельная линия", "Сечение"),
    "cable_max_temp_c": ("кабельная линия", "Максимально t"),
    "cable_length_m": ("кабельная линия", "Длина"),
    "cable_corr_raw": ("кабельная линия", "Коррозионностойкость"),
}

_NON_ESP_PREFIXES = ("ВОРОНКА", "УГРП", "ОТСУТСТВ", "ПАКЕР")

_COMPONENTS = ("pump", "gassep", "protector", "ped", "tms", "nkt", "cable")


# ---------------------------------------------------------------------------
# Normalizers (pure — unit-tested)
# ---------------------------------------------------------------------------

def to_num(series: pd.Series) -> pd.Series:
    """Robust numeric parse: strips space/NBSP thousands separators and accepts
    comma decimals ('2 605,10' → 2605.10) before coercion."""
    s = series.astype(str).str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _norm_text(v: object) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s or None


def norm_corrosion_class(v: object) -> float | None:
    """К0/К1/К2/К3 (Cyrillic or Latin K) → numeric class 0–3; else None."""
    s = _norm_text(v)
    if s is None:
        return None
    m = re.fullmatch(r"[КK]\s*([0-3])", s.upper())
    return float(m.group(1)) if m else None


def norm_coating(v: object) -> str | None:
    """Coating → 'monel' / 'yes' / 'no' / None (unknown)."""
    s = _norm_text(v)
    if s is None:
        return None
    low = s.lower()
    if "монел" in low or "monel" in low:
        return "monel"
    if low in {"да", "есть", "yes"}:
        return "yes"
    if low in {"нет", "без покрытия", "-", "нет покрытия", "no"}:
        return "no"
    return None


def norm_exec_group(v: object) -> tuple[str | None, bool]:
    """«Группа исполнения УЭЦН» → (group, lch_flag).

    Vocabulary observed: Н2-ЛЧ, Н3-ЛЧ, Н2, Н3 and Latin twins N2/N3/2N/3N.
    Returns e.g. ('H2', True) for 'Н2-ЛЧ'; unknown values → (raw, False).
    """
    s = _norm_text(v)
    if s is None:
        return None, False
    u = s.upper().replace("Н", "H")  # Cyrillic Н → Latin H
    lch = "ЛЧ" in u or "LC" in u
    m = re.search(r"([HN])\s*-?\s*([23])|([23])\s*([HN])", u)
    if m:
        digit = m.group(2) or m.group(3)
        return f"H{digit}", lch
    return s, lch


def parse_type_execution_flags(gno_type: object) -> tuple[bool, bool]:
    """(wear_resistant И, corrosion_resistant К) from the ЭЦН suffix letters.

    Novomet-style names carry the execution in the letters after «ЭЦН»:
    «ЭЦНДИК» → И (износостойкое) + К (коррозионностойкое); «ЭЦНД» → neither.
    Only the letter cluster immediately following «ЭЦН» is examined, so field
    text elsewhere cannot false-trigger.
    """
    s = _norm_text(gno_type)
    if s is None:
        return False, False
    m = re.search(r"ЭЦН([А-ЯЁ]{0,6})", s.upper())
    if not m:
        return False, False
    suffix = m.group(1)
    return "И" in suffix, "К" in suffix


def norm_gabarit(v: object) -> tuple[str | None, float | None]:
    """Recorded габарит cell → (normalized tag, OD mm).

    Handles three vocabularies: Russian габарит codes (5А/6/…, Latin A accepted),
    REDA series numbers (387/400/538/540/…), and direct mm entries ('101ММ').
    """
    s = _norm_text(v)
    if s is None:
        return None, None
    u = s.upper().replace("A", "А")  # Latin A → Cyrillic inside gabarit codes
    mm = u.replace("ММ", "MM")
    if mm.endswith("MM"):
        try:
            val = float(mm[:-2].replace(",", "."))
            return f"MM:{val:g}", val
        except ValueError:
            return None, None
    ser = u.replace("А", "A")  # digits only anyway for series
    if ser in REDA_SERIES_OD_MM:
        return f"SER:{ser}", REDA_SERIES_OD_MM[ser]
    if u in _RUS_GABARITS:
        latin = u.replace("А", "A").replace("Б", "B")
        return f"GAB:{u}", GABARIT_OD_MM.get(latin)
    return None, None


def is_esp_row(gno_type: object) -> bool:
    s = _norm_text(gno_type)
    if s is None:
        return False
    return not s.upper().startswith(_NON_ESP_PREFIXES)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _header_map(raw: pd.DataFrame) -> dict[int, tuple[str, str | None]]:
    """Column index → (section, parameter) from the two header rows."""
    top = raw.iloc[0].ffill()
    sub = raw.iloc[1]
    out: dict[int, tuple[str, str | None]] = {}
    for i in range(raw.shape[1]):
        sec = _norm_text(top.iloc[i])
        par = _norm_text(sub.iloc[i])
        if sec is not None:
            out[i] = (sec, par)
    return out


def _find_column(hmap: dict[int, tuple[str, str | None]], section: str, param: str | None) -> int | None:
    """First column whose (section, parameter) contains the requested substrings."""
    sec_l, par_l = section.lower(), (param.lower() if param else None)
    for i, (sec, par) in sorted(hmap.items()):
        if sec_l not in sec.lower():
            continue
        if par_l is None:
            return i
        if par is not None and par_l in par.lower():
            return i
    return None


def load_equipment_big(path: Path | None = None) -> pd.DataFrame:
    """Load and normalize the workbook. One row per спуск, selected columns only."""
    from scripts.data_utils import normalize_well_key  # local import: scripts dep

    src = Path(path) if path is not None else resolve_equipment_big_path()
    raw = pd.read_excel(src, header=None)
    hmap = _header_map(raw.iloc[:2])
    body = raw.iloc[2:].reset_index(drop=True)

    missing: list[str] = []
    data: dict[str, pd.Series] = {}
    for out_col, (section, param) in _SELECT.items():
        idx = _find_column(hmap, section, param)
        if idx is None:
            missing.append(out_col)
            continue
        data[out_col] = body[idx]
    if missing:
        raise KeyError(
            f"equipment_big: header lookup failed for {missing} — workbook layout changed?"
        )

    df = pd.DataFrame(data)
    df = df[df["well"].notna()].reset_index(drop=True)

    # -- identity / dates -----------------------------------------------------
    df["well_key"] = df["well"].astype(str).map(normalize_well_key)
    for col in ("install_date", "launch_date", "fail_date", "pull_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    for col in ("nno_days", "run_no", "pump_length_m", "q_nom_m3d", "head_nom_m",
                "freq_nom_hz", "stages", "ped_max_temp_c", "ped_power_kw",
                "pump_depth_m", "vg_m", "nkt_diam_mm", "cable_section_mm2",
                "cable_max_temp_c", "cable_length_m", "curvature"):
        df[col] = to_num(df[col])

    df["is_esp"] = df["gno_type"].map(is_esp_row)
    df["pull_fail_gap_d"] = (df["pull_date"] - df["fail_date"]).dt.days
    df["launch_delay_d"] = (df["launch_date"] - df["install_date"]).dt.days

    # -- pump size --------------------------------------------------------------
    gab = df["pump_gabarit_raw"].map(norm_gabarit)
    df["pump_gabarit"] = gab.map(lambda t: t[0])
    df["pump_od_mm"] = gab.map(lambda t: t[1])

    # -- corrosion-execution profile ---------------------------------------------
    for comp in _COMPONENTS:
        raw_col = f"{comp}_corr_raw"
        if raw_col in df.columns:
            df[f"{comp}_corr_class"] = df[raw_col].map(norm_corrosion_class)
    for comp in ("pump", "gassep", "protector", "ped"):
        df[f"{comp}_coating"] = df[f"{comp}_coating_raw"].map(norm_coating)

    eg = df["pump_exec_group_raw"].map(norm_exec_group)
    df["pump_exec_group"] = eg.map(lambda t: t[0])
    df["pump_exec_lch"] = eg.map(lambda t: t[1]).astype("boolean")

    flags = df["gno_type"].map(parse_type_execution_flags)
    df["type_wear_resistant"] = flags.map(lambda t: t[0]).astype("boolean")
    df["type_corr_resistant"] = flags.map(lambda t: t[1]).astype("boolean")

    coat_cols = [f"{c}_coating" for c in ("pump", "gassep", "protector", "ped")]
    df["n_components_monel"] = (df[coat_cols] == "monel").sum(axis=1).astype("int64")
    corr_cols = [c for c in df.columns if c.endswith("_corr_class")]
    df["max_corr_class"] = df[corr_cols].max(axis=1)
    df["any_corr_protection"] = (
        (df["n_components_monel"] > 0)
        | (df["max_corr_class"] > 0)
        | df["type_corr_resistant"].fillna(False)
    ).astype("boolean")

    df["row_id"] = df.index.astype(int)
    return df


__all__ = [
    "load_equipment_big",
    "norm_corrosion_class",
    "norm_coating",
    "norm_exec_group",
    "norm_gabarit",
    "parse_type_execution_flags",
    "is_esp_row",
    "REDA_SERIES_OD_MM",
]
