"""Shared constants for the VT failure analysis."""
from __future__ import annotations

# Frequency group boundaries (Hz) — mean-based definition
FREQ_LOW_MAX = 50.0
FREQ_HIGH_MIN = 55.0
FREQ_VERY_HIGH_MIN = 58.0  # "true 60 Hz" — report if n_fail >= 50

# Proportion-based high-frequency definition:
# freq_group_pct = "High" when freq_above_55hz_pct > this threshold
FREQ_HIGH_PCT_THRESHOLD = 0.50  # majority (>50%) of run time above 55 Hz

# Infant mortality thresholds to sweep (days)
INFANT_THRESHOLDS = [45, 60, 90, 120]
INFANT_THRESHOLD_PRIMARY = 90  # used as default in most tables

# KM / survival: minimum failures to show a curve
MIN_FAILURES_KM = 20
MIN_FAILURES_WEIBULL = 50
MIN_FAILURES_COX = 50
MIN_FAILURES_CIF = 100

# Vt field code as it appears in mart__vt_freq55.field
VT_FIELD = "Vt"

# Major fields (enough failures for survival analysis)
MAJOR_FIELDS = ["Ya", "Vt", "Az", "Za", "Ic"]

# Ordered failure categories (classifier output)
FAILURE_CATEGORIES = [
    "КЛ (R-0)",
    "ПЭД (R-0)",
    "Слом вала",
    "Засорение РО",
    "НКТ",
    "Износ РО",
    "Износ/негермет.гидрозащиты",
]

# Domain-knowledge Weibull β priors for each category
CATEGORY_BETA_PRIOR = {
    "КЛ (R-0)": "< 1",
    "ПЭД (R-0)": "≈ 1",
    "Слом вала": "< 1 or ≈ 1",
    "Засорение РО": "> 1",
    "НКТ": "≈ 1",
    "Износ РО": "> 1",
    "Износ/негермет.гидрозащиты": "> 1",
}

# Colour palette — consistent across all figures
FREQ_GROUP_COLORS = {
    "Low (≤50 Hz)": "#4878CF",
    "Normal (50–55 Hz)": "#6ACC65",
    "High (>55 Hz)": "#D65F5F",
    "Very High (>58 Hz)": "#B47CC7",
}

CATEGORY_COLORS = {
    "КЛ (R-0)": "#1f77b4",
    "ПЭД (R-0)": "#ff7f0e",
    "Слом вала": "#2ca02c",
    "Засорение РО": "#d62728",
    "НКТ": "#9467bd",
    "Износ РО": "#8c564b",
    "Износ/негермет.гидрозащиты": "#e377c2",
    "<missing>": "#B6B6B6",
}

H2S_COLORS = {"Кислый": "#d62728", "Некислый": "#2ca02c", "<missing>": "#B6B6B6"}

# Output directory (relative to repo root — set in run.py)
OUTPUT_SUBDIR = "analysis_outputs/vt_failure_analysis"
