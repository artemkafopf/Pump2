"""Backward-compatible shim — import from analysis.paths directly."""
from analysis.paths import (
    DEFAULT_PRESENTATION_EXTERNAL,
    DEFAULT_V03_ALL_EXTERNAL,
    DEFAULT_V03_FAILURES_EXTERNAL,
    LOCAL_INPUT_DIR,
    resolve_presentation_path,
    resolve_v03_all_path,
    resolve_v03_failures_path,
)

__all__ = [
    "DEFAULT_PRESENTATION_EXTERNAL",
    "DEFAULT_V03_ALL_EXTERNAL",
    "DEFAULT_V03_FAILURES_EXTERNAL",
    "LOCAL_INPUT_DIR",
    "resolve_presentation_path",
    "resolve_v03_all_path",
    "resolve_v03_failures_path",
]
