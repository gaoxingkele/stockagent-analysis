#!/usr/bin/env python
"""One stock under the production V12.31 / R20 scoring on one day.

  python scripts/score_v12_stock.py 688010.SH [--date YYYYMMDD]

Runs V12Scorer.score_market for the whole market (the production scorer, unchanged), then applies the
two published lists exactly as scripts/export_r20_pool_a_history.py does:
  pool_a        config/pool_a_r20_target_v1.json (predicted r20 or max gain >= 25%, max drawdown >= -15%)
  v12_31_top20  V7c main pool sorted by ratio = P(up)/(P(down)+0.01), industry cap 4, top 20
Needs the V12 feature pipeline up to the day (daily_review.update_data). Before scoring it checks that
every feature group has rows for the day itself: the loader otherwise silently uses the latest day of
that file. Not investment advice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from export_pump_history import stale_groups  # noqa: E402
from export_r20_pool_a_history import pool_a, v12_top20  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code")
    ap.add_argument("--date")
    a = ap.parse_args()
    code = a.code if "." in a.code else (a.code + (".SH" if a.code.startswith(("6", "9")) else ".SZ"))
    from stockagent_analysis.v12_scoring import V12Scorer
    day = a.date or max(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    scorer = V12Scorer.get(ROOT)
    stale = stale_groups(scorer, day)
    df = scorer.score_market(day)
    if df["pump_score"].isna().all():          # the 3-way model is loaded lazily; warm it and redo that step
        scorer.predict_one(df, "r5_pump_3way")
        df = scorer.apply_pump_3way(df)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[["ts_code", "name", "industry"]]
    for c in ("name", "industry"):
        if c not in df.columns:
            df = df.merge(basic[["ts_code", c]].drop_duplicates("ts_code"), on="ts_code", how="left")
    df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
    A = pool_a(df).reset_index(drop=True)
    V = v12_top20(df).reset_index(drop=True)
    me = df[df.ts_code == code]
    lines = [f"# {code} {me.name.iloc[0] if len(me) else ''}  V12.31 / R20 scoring on {day}", ""]
    lines.append(f"- feature groups without rows for {day}: {', '.join(stale) if stale else 'none (all fresh)'}"
                 + ("  -> those groups fell back to their latest day; treat the scores as approximate" if stale else ""))
    if me.empty:
        lines.append(f"- {code} not scored on {day} (suspended, ST or missing factors)")
        print("\n".join(lines))
        return 1
    r = me.iloc[0]
    n = len(df)

    def rank(col: str, asc: bool = False) -> str:
        v = df[col]
        return f"{int((v > r[col]).sum() + 1) if not asc else int((v < r[col]).sum() + 1)} of {int(v.notna().sum())}" if pd.notna(r[col]) else "n/a"
    lines += ["", "## R20 (20-day) models",
              f"- r20_pred {r.r20_pred:+.2f}% (rank {rank('r20_pred')}); buy_r20_score {r.buy_r20_score:.1f}",
              f"- predicted max gain 20d {r.pred_max_gain_20:+.2f}%, predicted max drawdown 20d {r.pred_max_dd_20:+.2f}%",
              f"- r5 {r.r5_pred:+.2f}%, r10 {r.r10_pred:+.2f}%; buy_score {r.buy_score:.1f}, sell_score {r.sell_score:.1f}, quadrant {r.get('quadrant', 'n/a')}",
              "", "## Start-up model (pump v3c)",
              f"- P(up) {r.pump_score:.3f}, P(down) {r.pump_down_score:.3f}, ratio {r.ratio:.2f} (rank {rank('ratio')})",
              "", "## Published lists"]
    in_a = code in set(A.ts_code)
    lines.append(f"- Pool A (R20 target contract): " + (f"YES, rank {int(A.index[A.ts_code == code][0]) + 1} of {len(A)}" if in_a
                 else f"no ({len(A)} names today; needs r20_pred or max gain >= 25% and max drawdown >= -15%)"))
    elig = bool(r.get("v7c_eligible", False))
    rec = bool(r.get("v7c_recommend", False))
    lines.append(f"- V7c main pool (6 iron rules): eligible {'yes' if elig else 'no'}, recommended {'yes' if rec else 'no'}")
    in_v = code in set(V.ts_code) if len(V) else False
    lines.append(f"- V12.31 Top20 (ratio sort, industry cap 4): " + (f"YES, rank {int(V.index[V.ts_code == code][0]) + 1}" if in_v
                 else f"no ({len(V)} names today)"))
    if len(V):
        lines.append(f"- today's V12.31 Top20: " + ", ".join(f"{x.ts_code}{x.get('name', '')}" for _, x in V.head(20).iterrows()))
    text = "\n".join(lines)
    out = ROOT / "output/experiments/stock_scores" / f"{code}_{day}_v12.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
