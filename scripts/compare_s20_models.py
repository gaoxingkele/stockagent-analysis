#!/usr/bin/env python
"""Same funnel, same days, different stage1 scorers: fold models, the frozen model, the unified recipe.

Lists are built with the frozen contract code (s20_pure.select for the offensive list, select_safe
for the safe list) from each scorer's scores, so only the scorer changes.

  folds    the published development scores (three walk-forward fold models, 50% row sample)
  frozen   the frozen stage1 model: re-scored for the development months, as saved for confirmation
  unified  train_s20_unified_v1.py, every block scored by a model trained before it

Descriptive: all of these days have been used before. Nothing after 2026-08-05 is read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_pure_r11_safe_up import load as load_folds  # noqa: E402
from analyze_s20_scissors import dn7_panel  # noqa: E402
from analyze_s20_trend_filters import trend_states  # noqa: E402
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, SafeConfig, select, select_safe  # noqa: E402

OUT = ROOT / "output/experiments/s20_model_compare_20261005"
SEGMENTS = [
    ("2025-03..08", "20250303", "20250831", "outside the frozen residual fit and tuning windows"),
    ("2025-09..10", "20250901", "20251031", "the frozen residual FIT window (in sample for frozen)"),
    ("2025-11..2026-01", "20251101", "20260126", "the frozen residual tuning window"),
    ("confirm", "20260127", "20260805", "after every frozen fit date"),
]


def lists_for(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = ["ts_code", "trade_date"]
    raw = scores[scores.groupby("trade_date").stage1_probability.rank(ascending=False, method="first") <= 20]
    return {"offensive": select(scores, PureConfig(), "U15D10")[cols], "safe": select_safe(scores, SafeConfig())[cols], "raw top 20": raw[cols]}


def summary(q: pd.DataFrame, kind: str) -> str:
    if q.empty:
        return "no days"
    col = "ret_safe" if kind == "safe" else "ret_v1"
    day = q.groupby("trade_date")[col].mean()
    month = day.groupby(day.index.str[:6]).mean()
    text = (f"days {len(day):3d} len {len(q) / len(day):4.1f} | per trade {day.mean():+5.2f} | pure_up {100 * (q.state == 'pure_up').mean():4.1f} "
            f"pure_down {100 * (q.state == 'pure_down').mean():4.1f} | band ok {100 * q.cls_a5_d10.isin([1, 3]).mean():4.1f} | "
            f"dn7 {100 * q.y_dn7.mean():4.1f} | MA20 rising {100 * q.ma20_up.mean():4.1f} | worst month {month.min():+5.2f} "
            f"months up {int((month > 0).sum())}/{len(month)}")
    return text


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scorers: dict[str, pd.DataFrame] = {}
    folds = load_folds()
    scorers["folds"] = folds.loc[folds.period == "dev", ["ts_code", "trade_date", "industry", "stage1_probability", "natr14"]]
    scorers["frozen"] = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    for name, folder in (("unified", "s20_unified_v1_20261005"), ("unified_v2", "s20_unified_v2_20261005")):
        path = ROOT / "output/experiments" / folder / "predictions.parquet"
        if path.exists():
            scorers[name] = pd.read_parquet(path).drop(columns="block")
    out = outcomes().merge(dn7_panel()[["ts_code", "trade_date", "y_dn7"]], on=["ts_code", "trade_date"], how="left")
    out = out.merge(trend_states()[["ts_code", "trade_date", "ma20_up"]], on=["ts_code", "trade_date"], how="left")
    picked = {name: {kind: q.merge(out, on=["ts_code", "trade_date"], how="inner") for kind, q in lists_for(s).items()}
              for name, s in scorers.items()}
    fold_days = set(scorers["folds"].trade_date)
    lines = [
        "S20 stage1 scorers compared through the frozen funnel. per trade: offensive and raw = +15%/-10% exit (list basis, cost in);",
        "safe = the take-profit band return. dn7 = touches -7% within 5 sessions. All days are used data.",
        "",
    ]
    for kind in ("offensive", "safe", "raw top 20"):
        lines.append(f"================ {kind} list ================")
        lines.append("-- the 145 days the fold models were scored on --")
        for name in scorers:
            q = picked[name][kind]
            lines.append(f"  {name:10s}: {summary(q[q.trade_date.isin(fold_days)], kind)}")
        for label, lo, hi, note in SEGMENTS:
            lines.append(f"-- {label} ({note}) --")
            for name in scorers:
                q = picked[name][kind]
                lines.append(f"  {name:10s}: {summary(q[q.trade_date.between(lo, hi)], kind)}")
            col = "ret_safe" if kind == "safe" else "ret_v1"
            for a_name, b_name in (("unified", "frozen"), ("unified_v2", "frozen"), ("unified_v2", "unified")):
                if a_name not in scorers:
                    continue
                a = picked[a_name][kind]
                b = picked[b_name][kind]
                d = (a[a.trade_date.between(lo, hi)].groupby("trade_date")[col].mean()
                     - b[b.trade_date.between(lo, hi)].groupby("trade_date")[col].mean()).dropna()
                if len(d) > 5:
                    month = d.groupby(d.index.str[:6]).mean()
                    lines.append(f"  {a_name} minus {b_name}: {d.mean():+.2f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)}, months up {int((month > 0).sum())}/{len(month)})")
        lines.append("-- by month, per trade --")
        col = "ret_safe" if kind == "safe" else "ret_v1"
        table = pd.DataFrame({name: picked[name][kind].groupby("trade_date")[col].mean().pipe(lambda d: d.groupby(d.index.str[:6]).mean())
                              for name in scorers}).round(2)
        lines.append(table.to_string())
        keys = {n: set(zip(picked[n][kind].ts_code, picked[n][kind].trade_date)) for n in scorers if n != "folds"}
        for a_name, b_name in (("unified", "frozen"), ("unified_v2", "frozen"), ("unified_v2", "unified")):
            if a_name in keys:
                lines.append(f"share of the {a_name} list also on the {b_name} list, all days: {100 * len(keys[a_name] & keys[b_name]) / max(len(keys[a_name]), 1):.1f}%")
        lines.append("")
    # facts for the model audit: quarterly blocks of the offensive list, with the list's and the market's share of rising MA20
    quarters = [("2025-03..05", "20250303", "20250531"), ("2025-06..08", "20250601", "20250831"), ("2025-09..11", "20250901", "20251130"),
                ("2025-12..2026-02", "20251201", "20260228"), ("2026-03..05", "20260301", "20260531"), ("2026-06..08", "20260601", "20260805")]
    universe = scorers["frozen"][["ts_code", "trade_date"]].merge(out[["ts_code", "trade_date", "ma20_up"]], on=["ts_code", "trade_date"], how="left")
    facts = {}
    for name in scorers:
        q = picked[name]["offensive"]
        s_q = picked[name]["safe"]
        facts[name] = {}
        for label, lo, hi in quarters:
            part = q[q.trade_date.between(lo, hi)]
            if part.empty:
                continue
            safe_part = s_q[s_q.trade_date.between(lo, hi)]
            mkt = universe[universe.trade_date.isin(set(part.trade_date))]
            facts[name][label] = {
                "days": int(part.trade_date.nunique()),
                "offensive": round(float(part.groupby("trade_date").ret_v1.mean().mean()), 2),
                "safe": round(float(safe_part.groupby("trade_date").ret_safe.mean().mean()), 2) if len(safe_part) else None,
                "ma20_up": round(float(part.ma20_up.mean()), 3), "ma20_up_market": round(float(mkt.ma20_up.mean()), 3),
            }
    (OUT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
