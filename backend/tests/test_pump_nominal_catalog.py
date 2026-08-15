"""Contract tests for the vendor pump-nameplate catalogue.

The rules encoded here came from the operator's own catalogue sheet (2026-07-27).  They are
easy to "simplify" into a value-based lookup, which is exactly the mistake these tests exist
to prevent — see :func:`test_reda_values_are_never_snapped_onto_the_metric_grid`.
"""
import pytest

from analysis.data.pump_nominal_catalog import (  # noqa: F401
    reda_model_bpd,
    MERGED_NOMINALS, MT_CLASSES, canonical_qnom, pump_family,
)


@pytest.mark.parametrize("designation,expected", [
    ("MT5-25DP", "mt"), ("MT5A-60DP", "mt"), ("MT-5A-100DP", "mt"),
    ("MT-5-200DP", "mt"), ("MT-5A-700DP", "mt"), ("МТ5А-320DP", "mt"),   # Cyrillic МТ
    ("D2400N", "reda"), ("S8000N", "reda"), ("GN10000", "reda"),
    ("G6200N", "reda"), ("H15500N", "reda"), ("ESP 538-9000", "reda"),
    ("ЭЦНКИД", "russian"), ("30.2 ЭЦНДИК Э", "russian"), ("ЭЦН-320", "russian"),
    ("", "unknown"), (None, "unknown"), ("nan", "unknown"),
])
def test_pump_family_classification(designation, expected):
    assert pump_family(designation) == expected


@pytest.mark.parametrize("designation,recorded,expected", [
    # the operator's class replaces the build's BEP flow
    ("MT-5A-500DP", 553.0, 500.0),
    ("MT-5A-400DP", 437.0, 400.0),
    ("MT-5A-250DP", 244.0, 250.0),
    ("MT-5A-200DP", 215.0, 200.0),
    ("MT-5A-160DP", 156.0, 160.0),
    ("MT-5A-100DP", 93.0, 100.0),
    ("MT5A-60DP", 67.0, 60.0),
    ("MT-5A-320DP", 321.0, 320.0),
])
def test_mt_takes_the_vendor_class(designation, recorded, expected):
    q, fam, src = canonical_qnom(designation, recorded)
    assert (q, fam, src) == (expected, "mt", "vendor_class")


@pytest.mark.parametrize("designation,recorded", [
    ("D4300N", 567.0),      # NOT an MT-500 despite sitting near 553
    ("D4300N", 563.0),
    ("S8000N", 1069.0),
    ("G6200N", 889.0),
    ("S4000N", 571.0),
    ("GN10000", 1291.0),
    ("D3500N", 464.0),
])
def test_reda_values_are_never_snapped_onto_the_metric_grid(designation, recorded):
    """REDA nameplates are the operator's own 50 Hz conversion — keep them verbatim.

    567/563 are D4300N.  A value-based remap would mis-file them as MT-500DP.
    """
    q, fam, src = canonical_qnom(designation, recorded)
    assert q == recorded
    assert fam == "reda"
    assert src == "recorded"


def test_borets_700_folds_into_the_800_nameplate():
    """ЭЦН-800(700) and the standalone 700 row are the same pump (operator, 2026-07-27)."""
    assert MERGED_NOMINALS[700.0] == 800.0
    q, fam, src = canonical_qnom("ЭЦН-800(700)", 700.0)
    assert (q, fam, src) == (800.0, "russian", "merged")
    # an MT-700 class merges the same way, so the two routes cannot disagree
    assert canonical_qnom("MT-5A-700DP", 700.0)[0] == 800.0


def test_russian_designations_keep_their_recorded_nominal():
    """Russian names carry no nominal, and the recorded value is already the round class."""
    for d, q in (("ЭЦНКИД", 320.0), ("30.2 ЭЦНДИК Э", 80.0), ("ЭЦНМИК Э АСП", 500.0)):
        assert canonical_qnom(d, q) == (q, "russian", "recorded")


def test_unmatched_or_missing_inputs_fall_back_to_the_recorded_value():
    assert canonical_qnom(None, 300.0) == (300.0, "unknown", "recorded")
    assert canonical_qnom("WR", 300.0) == (300.0, "unknown", "recorded")
    assert canonical_qnom("ЭЦНКИД", None) == (None, "russian", "recorded")


def test_unconverted_reda_bbl_per_day_is_repaired():
    """`ESP 538-7000` carries the raw bbl/day figure; its sibling `-9000` was converted.

    9000 bbl/d is recorded as 1200 m³/d, so the same rule must turn a recorded 7000 into
    ~930 rather than leaving an implausible 7000 m³/d ESP nameplate.
    """
    q, fam, src = canonical_qnom("ESP 538-7000", 7000.0)
    assert fam == "reda" and src == "unit_repair"
    assert q == pytest.approx(927.4, abs=1.0)
    # cross-check against the correctly-converted sibling: 7000/9000 * 1200 = 933
    assert q == pytest.approx(7000.0 / 9000.0 * 1200.0, rel=0.01)


@pytest.mark.parametrize("designation,recorded", [
    ("ESP 538-9000", 1200.0),    # already converted — must not be touched
    ("S8000N", 1069.0),
    ("D460N", 65.0),             # small recorded value, far below the model number
    ("GN10000", 1291.0),
])
def test_correctly_converted_reda_rows_are_left_alone(designation, recorded):
    q, fam, src = canonical_qnom(designation, recorded)
    assert (q, src) == (recorded, "recorded")


def test_reda_model_bpd_extraction():
    assert reda_model_bpd("ESP 538-7000") == 7000.0     # trailing figure, not the 538 series
    assert reda_model_bpd("S8000N") == 8000.0
    assert reda_model_bpd("GN10000") == 10000.0
    assert reda_model_bpd("ЭЦНКИД") is None


def test_unlisted_mt_class_is_not_invented():
    """A size class absent from the operator's sheet must not become a nameplate."""
    q, fam, src = canonical_qnom("MT-5A-999DP", 950.0)
    assert fam == "mt"
    assert q == 950.0 and src == "recorded"      # falls back, does not emit 999
    assert 999.0 not in MT_CLASSES
