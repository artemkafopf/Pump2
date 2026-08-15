"""Well identity and non-oil-bore classification.

The failure register is a registry of *oil-production* ESP runs. Two failure
modes leak non-oil bores into it and must be guarded against during assembly:

1. **Bore identity collapse.** A brine (``рс`` -- рассольная) bore such as
   ``vt_2704рс`` is a physically distinct wellbore from the oil well
   ``vt_2704``. A downstream normalizer that strips the ``рс`` suffix collapses
   the two into one identity, producing a phantom "living oil pump". The build
   must keep the bore suffix in the identity key so the two never merge, while
   still folding away cosmetic differences (case, spaces, latin/cyrillic
   homoglyphs) so the *same* bore always maps to one key.

2. **Non-oil bores.** Injection, piezometric, water-intake, conservation and
   brine bores are not oil production even when an ESP is installed and the
   ``Цель спуска`` reads ``Мех. добыча``. They are excluded by bore purpose and
   by the ``рс`` suffix -- never by the bare well number.
"""

import re
from typing import Any, Optional

from ..normalize import normalize_text, normalize_well


# Cyrillic characters that are visual homoglyphs of latin ones. Folding these to
# a single script lets a brine suffix typed as cyrillic ``рс`` and latin ``pc``
# collapse to one identity key without corrupting numeric or structural parts.
_CYRILLIC_TO_LATIN_HOMOGLYPHS = {
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s",
}
_HOMOGLYPH_TABLE = str.maketrans(_CYRILLIC_TO_LATIN_HOMOGLYPHS)


def _fold_homoglyphs(text: str) -> str:
    return text.translate(_HOMOGLYPH_TABLE)


def normalize_well_key(x: Any) -> str:
    """Return a canonical, **suffix-preserving** identity key for a well.

    The key keeps the bore suffix (``рс``, side-tracks, ``рс2`` ...) so distinct
    bores never merge, while folding away case, spaces and latin/cyrillic
    homoglyphs so the same bore always maps to one key. ``vt_2704рс`` and the
    oil well ``vt_2704`` therefore produce **different** keys, but
    ``VT_2704 РС`` / ``VT_2704рс`` / ``VT_2704pc`` all produce the **same** key.
    """
    normalized = normalize_well(x)
    if not normalized:
        return ""
    return _fold_homoglyphs(normalized)


def well_number_key(x: Any) -> str:
    """Return the physical-well key (prefix + number) with the bore suffix removed.

    Distinct bores of one physical well (``vt_2704``, ``vt_2704рс``,
    ``vt_100сч``) share a number key. This is the grouping for *validity* checks
    that reason about "the same well" -- e.g. a live run may not predate the same
    well's last closed run -- while ``normalize_well_key`` keeps bores distinct so
    their runs never merge.
    """
    normalized = normalize_well(x)
    if not normalized:
        return ""
    number_only = re.sub(r"(\d)[a-zа-яё]+\d*$", r"\1", normalized)
    return _fold_homoglyphs(number_only)


def bore_suffix(x: Any) -> str:
    """Return the trailing bore suffix (letters after the well number), or ``""``.

    ``vt_2704рс`` -> ``рс``; ``vt_2704рс2`` -> ``рс2``; ``vt_2704`` -> ``""``.
    Trailing separators are ignored so ``Yaw_400рс_`` / ``Yaw_400 рс`` still
    resolve to ``рс``.
    """
    normalized = normalize_well(x)
    # Drop trailing separators (underscore, dash, dot, space) that would
    # otherwise hide the suffix (e.g. a stray ``_`` after ``рс``).
    normalized = re.sub(r"[\s_\-.]+$", "", normalized)
    match = re.search(r"\d+([a-zа-яё]+\d*)$", normalized)
    return match.group(1) if match else ""


# A brine (рассольная) bore suffix, in cyrillic (``рс``) or the latin homoglyph
# form (``pc``), optionally followed by a bore index (``рс2``).
_RASSOL_SUFFIX_RE = re.compile(r"(?:рс|pc|рc|pс)\d*$", re.IGNORECASE)


def is_rassol_bore(x: Any) -> bool:
    """True when the well identifier carries a brine (``рс``) bore suffix."""
    return bool(_RASSOL_SUFFIX_RE.search(bore_suffix(x)))


# Substrings (normalized) that mark a non-oil bore purpose / assignment. These
# mirror the назначение values that must never enter the oil-production register.
_NON_OIL_PURPOSE_MARKERS = (
    "нагнетательн",   # injection
    "пьезометр",      # piezometric
    "водозаборн",     # water intake (ВД/НД)
    "поглощающ",      # absorbing
    "консерв",        # conservation
    "ликвид",         # liquidated
    "техн. операц",   # technical operations
    "техн операц",
    "техоперац",
)


def is_non_oil_purpose(value: Any) -> bool:
    """True when a bore purpose / assignment marks a non-oil bore."""
    text = normalize_text(value)
    if not text:
        return False
    return any(marker in text for marker in _NON_OIL_PURPOSE_MARKERS)


def non_oil_reason(well: Any, purpose: Any) -> Optional[str]:
    """Return a short reason a run is non-oil, or ``None`` when it is oil.

    The decision is by bore suffix and by purpose -- never by the bare well
    number. The returned label is used for drop-count logging.
    """
    if is_rassol_bore(well):
        return "рассольный ствол (рс)"
    if is_non_oil_purpose(purpose):
        return normalize_text(purpose) or "не-нефтяное назначение"
    return None


__all__ = [
    "normalize_well_key",
    "well_number_key",
    "bore_suffix",
    "is_rassol_bore",
    "is_non_oil_purpose",
    "non_oil_reason",
]
