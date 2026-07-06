# Pump2 — Claude Code guide

> Full agent instructions: **[agents/AGENTS.md](agents/AGENTS.md)**
> Read it before writing any code.

---

## Critical rules (enforced every session)

### 1. All paths come from `analysis.paths`

```python
from analysis.paths import results_dir, resolve_v03_all_path, resolve_telemetry_db_path
```

Never hardcode `D:\Projects\...` or `analysis_outputs/` in new code.
`analysis.input_paths` and `analysis.sqlite_paths` are legacy shims — do not use in new code.

### 2. All results go through `results_dir()`

```python
out = results_dir("my_analysis_slug")   # creates results/<slug>/YYYY-MM-DD/
(out / "tables" / "summary.csv").write_text(...)
(out / "figures" / "plot.png")  # save matplotlib figure here
```

Slug format: `<workflow>_<step>_<variant>` — e.g. `vt_failure_phase4_weibull`.

### 3. New reusable logic → `backend/analysis/`

Scripts in `scripts/` are thin CLI wrappers.  
Heavy logic (models, estimators, data loaders) lives in `backend/analysis/`.

```
backend/analysis/
  paths.py         ← path registry (already exists)
  models/          ← put new model code here
  workflows/       ← put new end-to-end workflows here
  features/        ← derived features, stress transforms
  data/            ← data loading helpers
  common/          ← plotting, validation
```

### 4. sys.path bootstrap (scripts and streamlit apps)

```python
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[N]
sys.path.insert(0, str(REPO_ROOT / "backend"))
```

---

## Legacy locations — do not write here

| Location | Status |
|---|---|
| `analysis/vt_failure/` | migrating → `backend/analysis/workflows/vt_failure/` |
| `analysis_outputs/` | read-only legacy; new output → `results/` |
| `temp/`, `catboost_info/` | gitignored scratch; use `results/` instead |
| `docs/uvch_codex/`, `docs/uvch_claude/` | archived output; new → `results/` |

---

## Running tests

```bash
cd backend && python -m pytest tests/ -v
```

## Project memory

Auto-memory index: `C:\Users\alexe\.claude\projects\d--GitHub-Pump2\memory\MEMORY.md`
