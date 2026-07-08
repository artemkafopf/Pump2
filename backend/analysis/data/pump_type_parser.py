"""Parser for the free-text ``pump_type`` field in ``raw__v03_runs``.

The warehouse carries 1,254 distinct ``pump_type`` strings mixing three naming
conventions:

* **Russian ЭЦН** — ``gabarit-q_design-head`` e.g. ``5а-800-2000`` (gabarit ``5а``,
  design flow 800 m³/d, head 2000 m). May carry a Russian series prefix
  (``ЭЦНДИК``, ``ВНН``, ``УЭЦН`` …) and/or a leading OD dimension (``10.2ЭЦНДИК5а-80-1600``).
* **Western REDA** — ``<series><flow>`` e.g. ``GN6200`` (series ``GN``, design flow
  6200 **bbl/d** → ×0.159 ≈ 986 m³/d).
* **Modular MT** — ``MT5A-100DP`` (gabarit ``5A``, design flow 100 m³/d). The prefix
  appears in both Latin (``MT``) and Cyrillic (``МТ``) and mixed forms (``MT5А``).

This module extracts, per string:
  ``pump_series``    — naming family (russian_ecn / reda / mt / other)
  ``pump_gabarit``   — normalised OD size token (e.g. ``5A``) or None
  ``od_group_mm``    — outer-diameter group in mm via a *reviewable* mapping
  ``q_design_m3d``   — design flow in m³/d (REDA bbl/d converted)
  ``head_design_m``  — design head in m (Russian pattern only)
  ``parsed``         — bool, whether any structured field was recovered

**Reviewable constants** (``GABARIT_OD_MM``, ``REDA_SERIES_OD_INCH``) encode
diameter mappings that must be confirmed with the data owner — see the module-level
OPEN_QUESTIONS note and the Phase A registry report.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict

import pandas as pd

# ---------------------------------------------------------------------------
# Reviewable diameter mappings  (CONFIRM WITH USER — see OPEN_QUESTIONS)
# ---------------------------------------------------------------------------

# Russian ЭЦН gabarit → housing outer diameter (mm).
# Verified 2026-07-07 against the standard габарит table (ru.wikipedia.org/wiki/ЭЦН;
# 5/5А/6 casing constraints cross-checked on the neftegaz.ru tech library) and
# empirically against the per-run recorded 'Габарит УЭЦН' in
# WellsArtificialLiftBig.xlsx (2,379 runs joined to raw__v03_runs — see
# results/wellsbig_review/2026-07-07/).
GABARIT_OD_MM: dict[str, float] = {
    "2": 55.0,
    "2A": 69.0,
    "3": 81.0,
    "4": 86.0,
    "5": 92.0,    # casing ID >= 121.7 mm
    "5A": 103.0,  # casing ID >= 130 mm (Cyrillic 5а)
    "6": 114.0,   # casing ID >= 148.3 mm
    "6A": 123.0,
    "6B": 130.0,  # 6Б — same housing as габарит 7
    "7": 130.0,
    "7A": 136.0,
    "8": 172.0,
    "9": 185.0,
}

# Western REDA series letter → housing outer diameter (inches).
# Verified 2026-07-07 (REDA ESP technology catalog): 538 series = 5.38 in [136.65 mm];
# 540 series = 5.13 in [130.30 mm] — the naming exception, NOT 5.40; 400 series =
# 4.00 in [101.6 mm] with a slimline 3.87-in [98.3 mm] housing option. Empirically
# (recorded габарит), our DN fleet is predominantly the 387 slimline (104 vs 13 runs);
# GN → 540, SN → 538. Per-model truth:
# results/wellsbig_review/2026-07-07/tables/pump_size_reference.csv.
REDA_SERIES_OD_INCH: dict[str, float] = {
    "A": 3.38,
    "D": 3.87,  # 400-series slimline housing — fleet-dominant for DN models
    "G": 5.13,  # 540 series
    "S": 5.38,  # 538 series
}

# bbl/d → m³/d
BBL_TO_M3 = 0.159

# Open questions surfaced to the report (not silently resolved).
OPEN_QUESTIONS = (
    "GABARIT_OD_MM and REDA_SERIES_OD_INCH verified 2026-07-07 (sources in comments); "
    "remaining ambiguity: individual DN models may be 400-series standard (101.6 mm) "
    "rather than slimline (98.3 mm) — per-model empirical gabarit in "
    "results/wellsbig_review/2026-07-07/tables/pump_size_reference.csv takes precedence."
)

# Russian ESP series prefixes (order matters: longest first for greedy match).
_RUSSIAN_SERIES_PREFIXES = [
    "УЭЦВН", "ЭЦНДИКЭ", "ЭЦНДИКэ", "ЭЦНДИК", "ЭЦНМИКэ", "ЭЦНМИК", "ЭЦНДИэ",
    "ЭЦНДИ", "ЭЦНМИэ", "ЭЦНМИ", "УЭЦНДИК", "УЭЦН", "УВНН", "ВННП", "ВНН",
    "ЭЦНА", "ЭЦН", "ЭОВНБ",
]

# Homoglyph map: Cyrillic letters that visually equal Latin, used only for
# gabarit-suffix normalisation (а/А) and MT-prefix detection.
_CYR_TO_LAT = str.maketrans({
    "А": "A", "а": "A", "В": "B", "С": "C", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "Т": "T", "Х": "X",
})

# Russian gabarit-flow-head core, e.g. "5а-800-2000" or "3-320-2400".
_RU_CORE = re.compile(r"(\d+\s*[аАaA]?)\s*[- ]\s*(\d+)\s*[- ]\s*(\d+)")
# MT modular, e.g. "MT5A-100DP" / "МТ5А-200DP" / "МТ-5А-100DP" (Latin-normalised).
_MT_CORE = re.compile(r"^MT\s*-?\s*(\d+A?)\s*[-/ ]\s*(\d+)")
# Western REDA, e.g. "GN6200", "DN3500", "SN8000", "D460".
_REDA_CORE = re.compile(r"^([DGSA])(N)?\s*0*(\d+)")


@dataclass(frozen=True)
class PumpTypeParse:
    pump_type: str
    pump_series: str            # russian_ecn | reda | mt | other
    pump_series_detail: str | None   # e.g. ЭЦНДИК, GN, MT
    pump_gabarit: str | None    # normalised, e.g. 5A
    od_group_mm: float | None
    q_design_m3d: float | None
    head_design_m: float | None
    parsed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _norm_gabarit(raw: str) -> str:
    """Normalise a gabarit token: strip spaces, map Cyrillic а/А→A, uppercase."""
    g = raw.strip().translate(_CYR_TO_LAT).upper().replace(" ", "")
    return g


def _gabarit_od(gabarit: str | None) -> float | None:
    if gabarit is None:
        return None
    return GABARIT_OD_MM.get(gabarit)


def parse_pump_type(raw: object) -> PumpTypeParse:
    """Parse one ``pump_type`` string into structured fields.

    Never raises; unparseable input returns ``pump_series='other', parsed=False``.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return PumpTypeParse("", "other", None, None, None, None, None, False)

    s = str(raw).strip()
    if not s:
        return PumpTypeParse(s, "other", None, None, None, None, None, False)

    # NFKC folds full-width / compatibility chars; keep Cyrillic intact.
    s_norm = unicodedata.normalize("NFKC", s)
    s_lat = s_norm.translate(_CYR_TO_LAT)  # Cyrillic homoglyphs → Latin

    # ── MT modular (Latin or Cyrillic МТ) ────────────────────────────────────
    mt = _MT_CORE.match(s_lat.upper())
    if mt:
        gabarit = _norm_gabarit(mt.group(1))
        q = float(mt.group(2))
        return PumpTypeParse(
            s, "mt", "MT", gabarit, _gabarit_od(gabarit), q, None, True
        )

    # ── Western REDA (leading D/G/S/A optionally + N) ────────────────────────
    reda = _REDA_CORE.match(s_lat.upper())
    if reda and not s_lat[:1].isdigit():
        letter = reda.group(1)
        series = letter + (reda.group(2) or "")
        flow_bbl = float(reda.group(3))
        q_m3d = round(flow_bbl * BBL_TO_M3, 1)
        od_inch = REDA_SERIES_OD_INCH.get(letter)
        od_mm = round(od_inch * 25.4, 1) if od_inch is not None else None
        return PumpTypeParse(
            s, "reda", series, None, od_mm, q_m3d, None, True
        )

    # ── Russian ЭЦН family (with or without series prefix) ────────────────────
    series_detail: str | None = None
    for prefix in _RUSSIAN_SERIES_PREFIXES:
        if prefix in s_norm:
            series_detail = prefix
            break

    core = _RU_CORE.search(s_norm)
    if core:
        gabarit = _norm_gabarit(core.group(1))
        q = float(core.group(2))
        head = float(core.group(3))
        series = "russian_ecn"
        return PumpTypeParse(
            s, series, series_detail, gabarit, _gabarit_od(gabarit), q, head, True
        )

    # A bare series prefix with no parsable core still tells us the family.
    if series_detail is not None:
        return PumpTypeParse(s, "russian_ecn", series_detail, None, None, None, None, False)

    return PumpTypeParse(s, "other", None, None, None, None, None, False)


def parse_pump_type_series(raw_series: pd.Series) -> pd.DataFrame:
    """Vectorised parse of a Series of ``pump_type`` strings.

    Returns a DataFrame aligned to ``raw_series.index`` with the parse columns.
    """
    records = [parse_pump_type(v).as_dict() for v in raw_series]
    out = pd.DataFrame.from_records(records, index=raw_series.index)
    return out


def parse_report(raw_series: pd.Series) -> dict:
    """Summarise parse coverage for the registry report."""
    parsed = parse_pump_type_series(raw_series)
    n = len(parsed)
    by_series = parsed["pump_series"].value_counts().to_dict()
    n_parsed = int(parsed["parsed"].sum())
    return {
        "n_total": n,
        "n_parsed": n_parsed,
        "parse_rate": round(n_parsed / n, 4) if n else 0.0,
        "by_series": by_series,
        "n_gabarit": int(parsed["pump_gabarit"].notna().sum()),
        "n_q_design": int(parsed["q_design_m3d"].notna().sum()),
    }


__all__ = [
    "PumpTypeParse",
    "parse_pump_type",
    "parse_pump_type_series",
    "parse_report",
    "GABARIT_OD_MM",
    "REDA_SERIES_OD_INCH",
    "OPEN_QUESTIONS",
    "BBL_TO_M3",
]
