# Analysis annotation template

Copy this file to `agents/analyses/<slug>.md` when starting or completing an analysis.
Fill in all sections. Partial annotations (in-progress) are fine — mark fields with `TBD`.

---

# `<slug>` — <Short title>

## Task / Research question

_One paragraph. What problem are we solving? What decision does this analysis inform?_

## Status

`[ ] planned` `[ ] in-progress` `[x] complete` `[ ] archived`

Last run: YYYY-MM-DD  
Entry point: `scripts/run/<slug>.py` or `python analysis/vt_failure/run.py`  
Results: `results/<slug>/YYYY-MM-DD/` (or `analysis_outputs/<legacy_dir>/`)

## Data

| Source | Access | Notes |
|--------|--------|-------|
| V03 failures xlsx | `resolve_v03_failures_path()` | N runs, K failures |
| Telemetry SQLite | `resolve_telemetry_db_path()` | daily freq, load, pressure |
| Warehouse mart | `WAREHOUSE_DIR / "mart__vt_freq55.db"` | pre-aggregated features |

## Methods & models

_List each method with a one-line description of why it was chosen._

- **Kaplan-Meier** — non-parametric survival curve; baseline comparison between groups
- **Weibull AFT** — parametric; β<1 infant mortality, β>1 wear-out
- ...

## Key findings

_Bullet list; only what the data actually shows, not interpretations._

- ...

## Known limitations / open questions

- ...

## Outputs

```
results/<slug>/YYYY-MM-DD/
  tables/   ← CSV summaries, parameter tables
  figures/  ← PNG/SVG survival curves, feature importance plots
  reports/  ← Markdown summary, PPTX slides
  models/   ← saved model files (.cbm, .pkl)
  logs/     ← run log
  manifest.json
```

## How to re-run

```bash
python scripts/run/<slug>.py
# or for legacy entry points:
python analysis/vt_failure/run.py all
```

## Related analyses

- [[<other-slug>]] — reason for relationship
