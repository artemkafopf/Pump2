"""Canonical pump nameplate (Qnom) from the vendor designation.

The recorded ``Ном. Произв`` is not always the vendor's own nameplate class.  Three naming
families appear in the fleet, and they behave differently:

======================  ==========================================  =========================
family                  designation examples                        canonical nominal
======================  ==========================================  =========================
``mt``   (Шлюмберже)    ``MT5-25DP``, ``MT-5A-320DP``, ``MT5A-500DP``  **from the name**
``reda`` (Шлюмберже)    ``D2400N``, ``S8000N``, ``GN10000``, ``H15500N``  recorded value
``russian``             ``ЭЦНКИД``, ``30.2 ЭЦНДИК Э``, ``ЭЦН-320``     recorded value
======================  ==========================================  =========================

**Why MT is special.**  The number in an MT designation is the vendor's size class in m³/d
(25/60/100/125/160/200/250/320/400/500/700 — the operator's own catalogue sheet), but the
recorded ``Ном. Произв`` is that build's actual BEP flow, which differs by up to ±12 %:
``MT-5A-500DP`` is recorded as 553, ``MT-5A-400DP`` as 437, ``MT-5A-100DP`` as 93.  The
difference is a fixed per-model catalogue offset, **not** a frequency effect —
``corr(ratio, freq) = −0.08`` over 3,726 tech-regime rows, and each model maps to the same
recorded number at every operating frequency.  Either is defensible; the vendor class is used
because it is the number the operator plans with.

**Why REDA is NOT rewritten.**  REDA model numbers are bbl/day at 60 Hz, and the recorded m³/d
is the operator's own 50 Hz conversion — ``D2400N`` → 2400 × 0.159 × (50/60) = 318 ≈ the
recorded 320; the tech-regime designation embeds it directly (``S8000N-1069-1943``).  So the
recorded value already *is* the vendor nameplate, on the same 50 Hz m³/d basis as the metric
pumps.  The operator's catalogue sheet leaves their round-nominal column blank, so there is
nothing to map them to.  ⚠ Do not "round" REDA values onto the metric grid: 567/563 are
``D4300N``, not an MT-500, and a value-based remap silently mis-assigns them.

**Борец 700 vs 800.**  ``ЭЦН-800(700)`` and the separate ``700`` row are the same pump
(operator confirmation 2026-07-27), so 700 folds into the 800 nameplate.
"""
from __future__ import annotations

import re
import unicodedata

#: MT size classes from the operator's catalogue sheet (m³/d).
MT_CLASSES = (25.0, 60.0, 100.0, 125.0, 160.0, 200.0, 250.0, 320.0, 400.0, 500.0, 700.0)

#: Designations that denote the same physical pump and must share one nameplate.
MERGED_NOMINALS = {700.0: 800.0}      # Борец ЭЦН-800(700)

#: ``MT`` / ``МТ``, optional gabarit token, then the size class, then ``DP``.
_MT_RE = re.compile(r"^\s*(?:MT|МТ)\s*-?\s*\d*[A-ZА-Я]?\s*-?\s*(\d{2,4})\s*DP", re.IGNORECASE)
#: REDA family: a letter series then a bbl/day figure (``D2400N``, ``GN10000``, ``S8000N``).
_REDA_RE = re.compile(r"\b(?:D|S|G|GN|H|ESP)\s?\d{3,5}\s?[A-Z]{0,2}\b", re.IGNORECASE)
#: The bbl/day figure inside a REDA designation.  ``ESP 538-7000`` → 7000 (538 is the series
#: OD, so a trailing ``-<N>`` wins over the leading number).
_REDA_BPD_RE = re.compile(r"(?:D|S|G|GN|H)\s?(\d{3,5})\s?[A-Z]{0,2}\b|ESP\s*\d{3}\s*-\s*(\d{3,5})",
                          re.IGNORECASE)

#: bbl/day (at 60 Hz, the REDA model-number basis) → m³/d at 50 Hz.
#: 1 bbl = 0.1589873 m³; the passport column is «Насос (50 Гц)», hence the 50/60 derate.
#: Observed recorded/model ratios span 0.129–0.143 (series differ in where their BEP sits
#: relative to the model number), and this factor sits in the middle of that band.
REDA_BPD_TO_M3D_50HZ = 0.1589873 * (50.0 / 60.0)

#: A recorded nameplate at least this fraction of the model's bbl/day number was never
#: converted (real conversions land near 0.13×, so there is a wide margin and no ambiguity).
_UNCONVERTED_RATIO = 0.80


def reda_model_bpd(designation) -> float | None:
    """The bbl/day figure embedded in a REDA designation, or None."""
    m = _REDA_BPD_RE.search(_norm(designation))
    if not m:
        return None
    return float(m.group(1) or m.group(2))


def _norm(s) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", str(s)).strip()


def pump_family(designation) -> str:
    """``mt`` | ``reda`` | ``russian`` | ``unknown`` from the free-text designation."""
    d = _norm(designation)
    if not d or d.lower() == "nan":
        return "unknown"
    if _MT_RE.match(d):
        return "mt"
    if _REDA_RE.search(d):
        return "reda"
    if re.search(r"[ЭЦНВ]", d):          # ЭЦН / ЭЦНДИК / ВНН / УЭЦН …
        return "russian"
    return "unknown"


def canonical_qnom(designation, recorded_qnom: float | None) -> tuple[float | None, str, str]:
    """Return ``(qnom, family, source)``.

    ``source`` is ``"vendor_class"`` when the value came from the designation, ``"recorded"``
    when the recorded nameplate was kept, and ``"merged"`` when a merge rule applied.
    """
    fam = pump_family(designation)
    q = None if recorded_qnom is None else float(recorded_qnom)

    if fam == "mt":
        m = _MT_RE.match(_norm(designation))
        cls = float(m.group(1))
        # snap to the catalogue class list; an unlisted number means a designation we have
        # not seen, so keep it rather than inventing a class
        if cls in MT_CLASSES:
            return MERGED_NOMINALS.get(cls, cls), fam, "vendor_class"
        return (MERGED_NOMINALS.get(q, q) if q is not None else None), fam, "recorded"

    if fam == "reda" and q is not None:
        # Unit repair: a handful of rows carry the raw bbl/day model number instead of the
        # converted m³/d.  `ESP 538-7000` is recorded as 7000 while its sibling
        # `ESP 538-9000` is correctly recorded as 1200 — same series, same workbook.
        bpd = reda_model_bpd(designation)
        if bpd and q >= _UNCONVERTED_RATIO * bpd:
            return round(bpd * REDA_BPD_TO_M3D_50HZ, 1), fam, "unit_repair"

    if q is not None and q in MERGED_NOMINALS:
        return MERGED_NOMINALS[q], fam, "merged"
    return q, fam, "recorded"
