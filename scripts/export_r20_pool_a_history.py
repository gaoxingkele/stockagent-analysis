#!/usr/bin/env python
"""Export historical production R20 lists (Pool A + V12.31 Top20). RUN ON THE PRODUCTION MACHINE.

Needs what production uses: output/production/r20_v16_all_nost (+ the other V12
models), output/lgbm_maxgain/regressor_{gain,dd}.txt, output/factor_lab_3y and
output/tushare_cache. Re-scores every trading day with V12Scorer.score_market and
applies the frozen rules, exactly like daily_dashboard.build_pools:

  pool_a        config/pool_a_r20_target_v1.json: (pred r20 >= 25% OR pred max gain >= 25%)
                AND pred max drawdown >= -15%; SH/SZ; no cap, no TopN; ranked by
                max(r20_pred, pred_max_gain_20) desc, ratio desc
  v12_31_top20  V7c main pool, ratio = pump_up / (pump_down + 0.01) desc, industry cap 4, Top20

Usage:
    python scripts/export_r20_pool_a_history.py --start 20260414 --end 20260928
    (resumable: finished days are skipped; one parquet per day under the out dir)

Fairness: r20_v16_all_nost was trained on 2023-01-01..2026-04-13, so days before
2026-04-14 are in-sample for R20. Default start is therefore 20260414. Export
earlier days only for reference (--start 20250101); the comparison script flags them.

Send back the single file written at the end:
    output/r20_history/r20_lists_<start>_<end>.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "output/r20_history"
KEEP = ["ts_code", "name", "industry", "r20_pred", "pred_max_gain_20", "pred_max_dd_20", "buy_r20_score",
        "pump_score", "pump_down_score", "ratio", "v7c_recommend"]


def trading_days(start: str, end: str) -> list[str]:
    days = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    return [d for d in days if start <= d <= end]


def pool_a(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import daily_dashboard as dash
        pool, _ = dash._build_pool_a(df.copy())
        return pool
    except Exception as exc:  # noqa: BLE001  fall back to the frozen contract, verbatim
        print(f"  daily_dashboard unavailable ({exc}); applying the contract inline", flush=True)
        c = json.loads((ROOT / "config/pool_a_r20_target_v1.json").read_text(encoding="utf-8"))
        r20 = pd.to_numeric(df["r20_pred"], errors="coerce")
        g = pd.to_numeric(df["pred_max_gain_20"], errors="coerce")
        dd = pd.to_numeric(df["pred_max_dd_20"], errors="coerce")
        sel = (df.ts_code.str.endswith((".SH", ".SZ")) & r20.notna() & g.notna() & dd.notna()
               & ((r20 >= c["selection"]["any_of"]["predicted_t20_close_return_pct_gte"])
                  | (g >= c["selection"]["any_of"]["predicted_max_gain_within_20d_pct_gte"]))
               & (dd >= c["selection"]["and"]["predicted_max_adverse_excursion_20d_pct_gte"]))
        out = df[sel].copy()
        out["a_target_score"] = np.maximum(r20[sel], g[sel])
        return out.sort_values(["a_target_score", "ratio"], ascending=[False, False])


def v12_top20(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["v7c_recommend"].fillna(False).astype(bool)].sort_values("ratio", ascending=False)
    picks, ind = [], {}
    for _, row in main.iterrows():
        k = str(row.get("industry") or "unknown")
        if ind.get(k, 0) >= 4:
            continue
        ind[k] = ind.get(k, 0) + 1
        picks.append(row)
        if len(picks) >= 20:
            break
    return pd.DataFrame(picks)


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper() if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20260414")
    ap.add_argument("--end", default="20991231")
    a = ap.parse_args()
    from stockagent_analysis.v12_scoring import V12Scorer

    day_dir = OUT / "days"
    day_dir.mkdir(parents=True, exist_ok=True)
    scorer = V12Scorer.get(ROOT)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[["ts_code", "name", "industry"]]
    days = trading_days(a.start, a.end)
    for i, d in enumerate(days, 1):
        f = day_dir / f"{d}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        try:
            df = scorer.score_market(d)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(days)}] {d} FAILED: {exc}", flush=True)
            continue
        for c in ("name", "industry"):
            if c not in df.columns:
                df = df.merge(basic[["ts_code", c]].drop_duplicates("ts_code"), on="ts_code", how="left")
        df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
        A = pool_a(df)
        A = A.assign(list="pool_a", list_rank=np.arange(1, len(A) + 1))
        V = v12_top20(df)
        V = V.assign(list="v12_31_top20", list_rank=np.arange(1, len(V) + 1)) if len(V) else V
        cols = ["list", "list_rank", *[c for c in KEEP + ["a_target_score"] if c in A.columns or c in V.columns]]
        rows = pd.concat([x.reindex(columns=cols) for x in (A, V) if len(x)], ignore_index=True)
        rows.insert(0, "trade_date", d)
        rows.to_parquet(f, index=False)
        print(f"[{i}/{len(days)}] {d}: pool_a {len(A)}, v12_top20 {len(V)} ({time.time() - t0:.0f}s)", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(day_dir.glob("*.parquet")) if a.start <= p.stem <= a.end]
    if not parts:
        print("nothing exported")
        return 1
    allrows = pd.concat(parts, ignore_index=True)
    first, last = allrows.trade_date.min(), allrows.trade_date.max()
    dst = OUT / f"r20_lists_{first}_{last}.parquet"
    allrows.to_parquet(dst, index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta = {"file": dst.name, "days": int(allrows.trade_date.nunique()), "rows": len(allrows), "git_commit": commit,
            "models_sha256": {
                "r20_v16_all_nost": sha(ROOT / "output/production/r20_v16_all_nost/classifier.txt"),
                "regressor_gain": sha(ROOT / "output/lgbm_maxgain/regressor_gain.txt"),
                "regressor_dd": sha(ROOT / "output/lgbm_maxgain/regressor_dd.txt")},
            "r20_training_window": "20230101-20260413 (train_v16_full.py); earlier days are in-sample"}
    (OUT / f"r20_lists_{first}_{last}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {dst} and its .meta.json - send both files back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
