"""CLI: Weibull-скан c x k по стратам месторождения, отчёт RMST(0)/MRL(0).

Тонкая обёртка над ``analysis.workflows.production_risk.weibull_scan``; ячейки
(страта x c x k) считаются параллельно — k=2 на тысячах пробегов стоит минуты.

Примеры:
    # Ya, все страты, cause-specific (ГТМ + работающие цензурированы)
    python scripts/run/weibull_scan.py --field Ya --estimand cause_specific

    # то же, но МРП-эстиманд (любой подъём = событие)
    python scripts/run/weibull_scan.py --field Ya --estimand all_cause

    # только одна отсечка и k=1
    python scripts/run/weibull_scan.py --field Ya --cuts 30 --ks 1
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.paths import results_dir  # noqa: E402
from analysis.workflows.production_risk import esp_population as P  # noqa: E402
from analysis.workflows.production_risk import weibull_scan as W  # noqa: E402

# Страты по умолчанию: подрядчики по отдельности, объединение brt+slb (без «прочих»,
# которые на Ya — заметно другая популяция), и полный пул.
DEFAULT_STRATA: list[tuple[str, str | list[str] | None]] = [
    ("Pooled", None),
    ("brt", ["brt"]),
    ("slb", ["slb"]),
    ("oth", ["oth"]),
    ("brt+slb", ["brt", "slb"]),
]


def _cell(job):
    import warnings

    warnings.filterwarnings("ignore")
    label, c, k, estimand, t, e, num_starts = job
    from analysis.workflows.production_risk import weibull_scan as _W

    return asdict(_W.fit(np.asarray(t, float), np.asarray(e, int), k,
                         label, estimand, c, num_starts=num_starts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="Ya")
    ap.add_argument("--as-of", default="2026-07-01")
    ap.add_argument("--h2s", default="nonsour")
    ap.add_argument("--estimand", default=W.ESTIMAND_CAUSE_SPECIFIC,
                    choices=[W.ESTIMAND_ALL_CAUSE, W.ESTIMAND_CAUSE_SPECIFIC])
    ap.add_argument("--cuts", type=int, nargs="+", default=list(W.SUPPORTED_CUTS))
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--num-starts", type=int, default=60)
    ap.add_argument("--strata", nargs="+", default=None,
                    help="фильтр по меткам страт, напр. --strata brt+slb (по умолчанию все)")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    args = ap.parse_args()

    strata_def = DEFAULT_STRATA
    if args.strata:
        strata_def = [s for s in DEFAULT_STRATA if s[0] in args.strata]
        if not strata_def:
            ap.error(f"страты {args.strata} не найдены; доступны: {[s[0] for s in DEFAULT_STRATA]}")

    pop = P.build(args.as_of, gtm_is_failure=args.estimand == W.ESTIMAND_ALL_CAUSE)

    print(f"=== {args.field} — эстиманд={args.estimand}, клок=cal (календарь), as_of={args.as_of}")
    if args.estimand == W.ESTIMAND_CAUSE_SPECIFIC:
        print("    событие = ТОЛЬКО отказ; ГТМ/ППР и работающие насосы ЦЕНЗУРИРОВАНЫ")
    else:
        print("    событие = ЛЮБОЙ подъём (МРП); работающие насосы цензурированы")

    jobs, sets = [], {}
    for label, contractor in strata_def:
        tag = f"{args.field}_{label}"
        t, e, meta = W.build_stratum(args.as_of, args.field, h2s=args.h2s,
                                     contractor=contractor, estimand=args.estimand, pop=pop)
        sets[tag] = meta
        print("  %-14s пробегов=%4d  событий=%4d  цензура=%4d (из них в работе %d)"
              % (tag, meta["n"], meta["events"], meta["censored"], meta["running"]))
        for c in args.cuts:
            tc, ec = W.apply_cut(t, e, c)
            if ec.sum() < 5:
                print(f"     c={c}: событий {int(ec.sum())} < 5 — пропуск")
                continue
            for k in args.ks:
                jobs.append((tag, c, k, args.estimand, tc.tolist(), ec.tolist(), args.num_starts))

    print(f"\n=== {len(jobs)} ячеек на {args.workers} воркерах", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_cell, jobs):
            rows.append(r)
            print("   готово %-14s c=%-3d k=%d  beta=%.3f  RMST(0)=%6.0f  MRL(0)=%6.0f"
                  % (r["stratum"], r["c"], r["k"], r["beta"], r["rmst0"], r["mrl0"]), flush=True)

    df = pd.DataFrame(rows).sort_values(["stratum", "k", "c"])
    out = results_dir(f"production_risk_weibull_scan_{args.field.lower()}_{args.estimand}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "scan.csv", index=False, encoding="utf-8-sig")

    show = ["stratum", "c", "k", "n", "events", "beta", "eta", "rmst0", "mrl0",
            "rmst_km", "model_over_km", "tau", "b2_at_bound", "w1_degenerate"]
    pd.set_option("display.width", 250)
    print("\n=== РЕЗУЛЬТАТ (RMST(0)/MRL(0) в календарных сутках)")
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nТаблица: {out / 'tables' / 'scan.csv'}")
    print("\nЧтение: model_over_km ~1.00 подтверждает форму Вейбулла; >10% — искажает.")
    print("        b2_at_bound=True => beta2 прижат к ограничению, это регуляризатор, не износ.")
    print("        w1_degenerate=True => смесь схлопнулась, k=2 бессмысленна.")


if __name__ == "__main__":
    main()
