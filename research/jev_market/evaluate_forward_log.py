#!/usr/bin/env python
"""Score the forward Jev log once outcomes are known (phase 2).

Outcome for each logged evening: any broad sell-off session in the next 5
sessions (median return <= -2.5% or >= 100 limit-down), computed from
output/jev_market/live_market_days.csv. Compares Jev (A_real, D_numeric) with
the free logistic baseline fitted only on days before each logged evening.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_market_crash import FEATS, ROOT  # noqa: E402

OUT = ROOT / "output/jev_market"


def main() -> int:
    log = [json.loads(l) for l in (OUT / "forward_log.jsonl").read_text(encoding="utf-8").splitlines() if l]
    L = pd.DataFrame(log)
    m = pd.read_csv(OUT / "live_market_days.csv", dtype={"date": str}).sort_values("date").reset_index(drop=True)
    m["median_ret_5d"] = m.median_ret.rolling(5).sum()
    m["median_ret_20d"] = m.median_ret.rolling(20).sum()
    m["limit_down_5d"] = m.limit_down.rolling(5).sum()
    m["amount_z60"] = (m.amount - m.amount.rolling(60).mean()) / m.amount.rolling(60).std()
    m["crash"] = (m.median_ret <= -2.5) | (m.limit_down >= 100)
    idx = {d: i for i, d in enumerate(m.date)}
    rows = []
    for r in L.itertuples():
        i = idx.get(r.date)
        if i is None or i + 5 >= len(m):
            continue
        tr = m.iloc[: max(0, i - 5)].dropna(subset=FEATS)
        fut = m.crash.iloc[i + 1:i + 6].any()
        tr_y = [m.crash.iloc[j + 1:j + 6].any() for j in tr.index]
        base = LogisticRegression(max_iter=2000).fit(tr[FEATS], tr_y).predict_proba(m.loc[[i], FEATS])[0, 1]
        rows.append({"date": r.date, "A_real": r.A_real, "D_numeric": r.D_numeric, "baseline": base, "y": int(fut)})
    E = pd.DataFrame(rows)
    print(f"matured evenings: {len(E)} (positives {int(E.y.sum()) if len(E) else 0}) of {len(L)} logged")
    if len(E) and E.y.nunique() == 2:
        for c in ("A_real", "D_numeric", "baseline"):
            print(f"{c:10s} AUC {roc_auc_score(E.y, E[c]):.3f}  Brier {np.mean((E[c] - E.y) ** 2):.3f}")
    E.to_csv(OUT / "forward_eval.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
