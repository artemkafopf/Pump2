# Agent guide — Pump2

Instructions for AI agents (Claude Code, Codex, GPT, etc.) working in this repository.
Read this file before writing or modifying any code.

---

## Project in one sentence

Survival / reliability analysis of ESP (electric submersible pump) failures, backed by a
FastAPI + Vue web app and a series of Streamlit exploratory apps.

---

## Repository layout (target state — being migrated incrementally)

```
backend/
  analysis/               ← reusable computation library; all heavy logic lives here
    paths.py              ← SINGLE source of truth for every path; import from here
    common/               ← plotting, validation, shared helpers
    data/                 ← data loading / preparation utilities
    features/             ← derived features, stress transforms
    models/
      catboost/
      survival/
        weibull/
        latent_weibull/
        bayesian_latent_weibull/
        cox_ph/
        competing_risks/
    workflows/
      vt_failure/         ← 13-phase survival analysis (migrating from analysis/)
      vt_60hz/
      uvch/
  app/                    ← FastAPI application (routes, schemas, services)
  tests/                  ← pytest tests for backend/analysis

scripts/
  ingest/                 ← raw data ingestion (mirror xlsx, build SQLite)
  processing/             ← pipeline transforms (daily operating, TTF, etc.)
  marts/                  ← mart builds (aggregated modelling datasets)
  run/                    ← thin CLI wrappers that call workflows; no logic here
  reporting/              ← slide / PPTX generation scripts

streamlit_apps/           ← UI wrappers only; no heavy business logic

results/                  ← ALL analysis outputs (gitignored)
  <slug>/
    YYYY-MM-DD/
      tables/
      figures/
      reports/
      models/
      logs/
      manifest.json

docs/
  methodology/
  decisions/
  presentations/
  notes/

agents/                   ← this file and per-workflow task briefs
```

### Legacy locations (do not write new code here)

| Legacy path | Canonical replacement |
|---|---|
| `analysis/vt_failure/` | `backend/analysis/workflows/vt_failure/` |
| `analysis_outputs/` | `results/` |
| `temp/` | `results/<slug>/` or the OS temp dir |
| `catboost_info/` | `results/<slug>/models/` |
| `docs/uvch_codex/`, `docs/uvch_claude/` | `results/<slug>/reports/` |

---

## Where to put new code

| What you are writing | Where it goes |
|---|---|
| Reusable model, estimator, stat function | `backend/analysis/models/<family>/` |
| Workflow that runs a full analysis | `backend/analysis/workflows/<name>/` |
| Derived feature / stress transform | `backend/analysis/features/` |
| Data loading / cleaning helper | `backend/analysis/data/` |
| Plotting / reporting utility | `backend/analysis/common/` |
| CLI script that calls a workflow | `scripts/run/<name>.py` |
| Pipeline step (ingest / mart) | `scripts/ingest/` or `scripts/processing/` or `scripts/marts/` |
| Streamlit UI | `streamlit_apps/` |
| Test | `backend/tests/test_<module>.py` |

**Hard rule**: `backend/analysis/` contains only reusable, path-agnostic logic.
No hardcoded file paths, no `analysis_outputs/`, no `REPO_ROOT / "data"` literals
inside library code — use the resolvers from `analysis.paths`.

---

## Path convention — always use `analysis.paths`

```python
# At the top of any script or streamlit app:
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[N]   # N = depth to repo root
sys.path.insert(0, str(REPO_ROOT / "backend"))

from analysis.paths import (
    results_dir,
    resolve_v03_all_path,
    resolve_v03_failures_path,
    resolve_telemetry_db_path,
    resolve_techregime_db_path,
    resolve_lab_db_path,
    # also available: RAW_DIR, INTERIM_DIR, MARTS_DIR, WAREHOUSE_DIR
)
```

Never import from `analysis.input_paths` or `analysis.sqlite_paths` in new code —
those are backward-compatibility shims only.

---

## Writing results — always use `results_dir()`

```python
from analysis.paths import results_dir

out = results_dir("vt_failure_phase4_weibull")   # uses today's date automatically
# or pin the date explicitly:
out = results_dir("vt_failure_phase4_weibull", "2026-06-25")

# Standard subdirectories are created automatically:
out / "tables" / "weibull_params.csv"
out / "figures" / "survival_curve.png"
out / "reports" / "summary.md"
out / "models" / "weibull_fit.pkl"
out / "logs"   / "run.log"
out / "manifest.json"   # pre-filled with analysis name, date, git commit
```

**Slug naming convention**: `<workflow>_<phase_or_step>_<variant>` in snake_case,
e.g. `vt_failure_phase4_weibull`, `uvch_catboost_aft`, `vt_60hz_scenario`.

**Never write to**:
- `analysis_outputs/` (legacy, read-only for historical reference)
- `temp/` (transient scratch only; not tracked)
- `docs/` (only human-authored methodology and decision docs belong there)
- any absolute path outside the repo

---

## Import convention for `backend/analysis` library code

Inside `backend/analysis/` use **relative imports** between sibling modules:

```python
# Inside backend/analysis/models/survival/weibull.py
from ..common.plotting import plot_survival_curve   # relative
from analysis.paths import REPO_ROOT               # absolute ok for paths
```

In scripts and streamlit apps use **absolute imports** (after sys.path setup):

```python
from analysis.models.survival.weibull import fit_basic_weibull
from analysis.paths import results_dir
```

---

## Code style rules

- No hardcoded file paths anywhere except `analysis.paths`.
- No `print()` in library code — raise exceptions or return structured results.
- No `plt.show()` in library code — return figures or save to path arguments.
- Type-hint all public functions.
- Tests go in `backend/tests/`; name the test file `test_<module>.py`.
- Do not create `analysis_outputs/`, `catboost_info/`, or `temp/` from new code.

---

## Running tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## Data sources (read-only from scripts)

| Source | Access |
|---|---|
| V03 failures xlsx | `resolve_v03_all_path()` / `resolve_v03_failures_path()` |
| Telemetry SQLite | `resolve_telemetry_db_path()` |
| Tech-regime SQLite | `resolve_techregime_db_path()` |
| Lab chemistry SQLite | `resolve_lab_db_path()` |
| Warehouse marts | `WAREHOUSE_DIR / "mart_name.parquet"` (or `.csv`) |

All resolvers check `PUMP2_*_PATH` env vars first, then `data/inputs/` or
`data/sqlite/` local mirrors, then fall back to absolute external paths on the
developer machine.

---

## Analysis annotation requirement

**Every analysis — new or updated — must have an annotation file in `agents/analyses/`.**

Use `agents/ANALYSIS_TEMPLATE.md` as the starting point.  The annotation must be
created (or updated) as part of completing the analysis, not as an afterthought.

### Mandatory annotation fields

| Field | What to fill in |
|---|---|
| **Task / Research question** | One paragraph: what problem, what decision |
| **Status** | One of: planned / in-progress / complete / archived; last-run date; entry point |
| **Data** | Table: source name, how to access it, key columns/rows count |
| **Methods & models** | Bullet list: each method with one-line rationale |
| **Key findings** | Bullet list: only what the data shows, not narrative |
| **Known limitations** | What the analysis cannot answer |
| **Outputs** | Directory tree of what was written |
| **How to re-run** | Exact command |
| **Related analyses** | `[[slug]]` links to related annotation files |

### When to create/update the annotation

- **Before starting**: fill in Task, Data, Methods (planned) and mark `in-progress`
- **After first results**: fill in Key findings and Outputs
- **On completion**: mark `complete`, fill in last-run date and entry point
- **When the analysis is superseded**: mark `archived` and link to the replacement

### Annotation file naming

File: `agents/analyses/<slug>.md` where `<slug>` matches the `results_dir()` slug prefix.

Example: analysis writing to `results/vt_60hz_scenario/` → `agents/analyses/vt_60hz.md`

---

## Analysis index

| Slug prefix | Annotation | Status |
| --- | --- | --- |
| `vt_failure_*` | [agents/analyses/vt_failure.md](analyses/vt_failure.md) | complete |
| `vt_60hz_*` | [agents/analyses/vt_60hz.md](analyses/vt_60hz.md) | complete |
| `uvch_*` | [agents/analyses/uvch.md](analyses/uvch.md) | complete |
| `vt_freq_latent_*` | [agents/analyses/vt_freq_latent.md](analyses/vt_freq_latent.md) | complete |
| `bayesian_field_survival_*` | [agents/analyses/bayesian_field_survival.md](analyses/bayesian_field_survival.md) | complete |
| `vt_freq55_exposure_*` | [agents/analyses/vt_freq55_exposure.md](analyses/vt_freq55_exposure.md) | complete |
| `mature_ttf_*` | [agents/analyses/mature_ttf.md](analyses/mature_ttf.md) | complete |
| `esp_survival_*` | [agents/analyses/esp_survival.md](analyses/esp_survival.md) | in-progress |
