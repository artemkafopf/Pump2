"""Persist and reload field-simulation experiments.

Each experiment is a directory under ``results/field_sim/<experiment_id>/``:
  - ``config.json``    parameters + label + created timestamp + meta
  - ``fit_table.csv``  per-(seed, snapshot) fitted params and counts
  - ``km_curves.json`` pooled KM curve arrays per snapshot year
  - ``bin_table.csv``  per-(seed, snapshot, layer, bin) life summaries — only
                       written when a hazard layer is enabled

The Streamlit app lists these directories so past runs can be re-opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.paths import RESULTS_ROOT

from .config import SimConfig
from .experiment import ExperimentResult

FIELD_SIM_ROOT: Path = RESULTS_ROOT / "field_sim"

# The explorer's per-user input defaults.  A single file next to the experiments
# rather than session state, so they survive a restart of the app; it lives under
# the gitignored results tree because it is one machine's preference, not data.
APP_DEFAULTS_PATH: Path = FIELD_SIM_ROOT / "app_defaults.json"


def experiment_dir(experiment_id: str) -> Path:
    return FIELD_SIM_ROOT / experiment_id


def load_app_defaults() -> dict:
    """Saved input defaults for the Streamlit explorer; ``{}`` when there are none.

    Never raises: a missing or corrupt file just means "no overrides", which is
    the right behaviour for a preferences file the app can always rebuild.
    """
    try:
        payload = json.loads(APP_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_app_defaults(values: dict) -> Path | None:
    """Persist the explorer's input defaults; an empty dict removes the file."""
    if not values:
        APP_DEFAULTS_PATH.unlink(missing_ok=True)
        return None
    APP_DEFAULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_DEFAULTS_PATH.write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return APP_DEFAULTS_PATH


def save_experiment(result: ExperimentResult) -> Path:
    d = experiment_dir(result.experiment_id)
    d.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "experiment_id": result.experiment_id,
        "label": result.label,
        "created": result.created,
        "config": result.config.to_dict(),
        "meta": result.meta,
    }
    (d / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result.fit_table.to_csv(d / "fit_table.csv", index=False, encoding="utf-8")
    (d / "km_curves.json").write_text(
        json.dumps(result.km_curves, ensure_ascii=False), encoding="utf-8"
    )
    for name, frame in (("fit_table_nowo", result.fit_table_nowo), ("bin_table", result.bin_table)):
        path = d / f"{name}.csv"
        if frame is not None:
            frame.to_csv(path, index=False, encoding="utf-8")
        elif path.exists():
            path.unlink()
    return d


def list_experiments() -> list[dict]:
    """Metadata for every saved experiment, newest first."""
    if not FIELD_SIM_ROOT.exists():
        return []
    out: list[dict] = []
    for d in FIELD_SIM_ROOT.iterdir():
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "experiment_id": payload.get("experiment_id", d.name),
                "label": payload.get("label", d.name),
                "created": payload.get("created", ""),
                "config": payload.get("config", {}),
                "path": str(d),
            }
        )
    out.sort(key=lambda r: r.get("created", ""), reverse=True)
    return out


def load_experiment(experiment_id: str) -> ExperimentResult:
    d = experiment_dir(experiment_id)
    payload = json.loads((d / "config.json").read_text(encoding="utf-8"))
    fit_table = pd.read_csv(d / "fit_table.csv")
    km_curves = json.loads((d / "km_curves.json").read_text(encoding="utf-8"))
    cfg = SimConfig.from_dict(payload["config"])
    nowo_path = d / "fit_table_nowo.csv"
    fit_table_nowo = pd.read_csv(nowo_path) if nowo_path.exists() else None
    bin_path = d / "bin_table.csv"
    bin_table = pd.read_csv(bin_path) if bin_path.exists() else None
    return ExperimentResult(
        config=cfg,
        fit_table=fit_table,
        km_curves=km_curves,
        label=payload.get("label", experiment_id),
        experiment_id=experiment_id,
        created=payload.get("created", ""),
        meta=payload.get("meta", {}),
        fit_table_nowo=fit_table_nowo,
        bin_table=bin_table,
    )
