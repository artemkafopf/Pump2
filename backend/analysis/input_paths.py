from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_INPUT_DIR = REPO_ROOT / "data" / "inputs"

DEFAULT_V03_ALL_EXTERNAL = Path(r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_all.xlsx")
DEFAULT_V03_FAILURES_EXTERNAL = Path(r"D:\Projects\Pumps\data\target\Отказы свод с анализом_БДА_V03_failures.xlsx")
DEFAULT_PRESENTATION_EXTERNAL = Path(r"C:\Users\alexe\Downloads\Отказность Аналитика(1).pptx")


def _resolve_path(*, env_var: str, local_name: str, external_default: Path) -> Path:
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value)

    local_path = LOCAL_INPUT_DIR / local_name
    if local_path.exists():
        return local_path

    return external_default


def resolve_v03_all_path() -> Path:
    return _resolve_path(
        env_var="PUMP2_V03_ALL_PATH",
        local_name="Отказы свод с анализом_БДА_V03_all.xlsx",
        external_default=DEFAULT_V03_ALL_EXTERNAL,
    )


def resolve_v03_failures_path() -> Path:
    return _resolve_path(
        env_var="PUMP2_V03_FAILURES_PATH",
        local_name="Отказы свод с анализом_БДА_V03_failures.xlsx",
        external_default=DEFAULT_V03_FAILURES_EXTERNAL,
    )


def resolve_presentation_path() -> Path:
    return _resolve_path(
        env_var="PUMP2_PRESENTATION_PATH",
        local_name="Отказность Аналитика(1).pptx",
        external_default=DEFAULT_PRESENTATION_EXTERNAL,
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
