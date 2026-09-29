#!/usr/bin/env python
"""Follow-up controls for the market sell-off probe (same 90 dates, no re-sampling).

E_news_swap : true snapshot + CCTV news of a hash-assigned OTHER date, no date.
              If AUC falls back to D_numeric, the lift in C comes from the news text.
Plus: per-year AUC (2026 dates are the least likely to be memorised) and
stronger free baselines on the same 8 numbers (single-feature rules, gradient boosting).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_market_crash import FEATS, OUT, api_key, ask, cctv, market_snapshot, snapshot_dict  # noqa: E402


def main() -> int:
    r = pd.read_csv(OUT / "probe_results.csv", dtype={"date": str})
    m = market_snapshot()
    m = m[m.label_ok & m.amount_z60.notna() & (m.date >= "20240401")].reset_index(drop=True)
    dates = sorted(r.date.unique())
    pool = [p.stem for p in (OUT / "cctv").glob("*.json")]

    def other(d):
        h = int(hashlib.sha256(("swap" + d).encode()).hexdigest(), 16)
        cand = [x for x in pool if abs(int(x[:6]) - int(d[:6])) >= 3]
        return sorted(cand)[h % len(cand)]

    key = api_key()
    jobs = {}
    with cf.ThreadPoolExecutor(4) as ex:
        for d in dates:
            row = m[m.date == d].iloc[0]
            state = {"market_after_close": snapshot_dict(row), "cctv_evening_news": cctv(other(d))}
            jobs[ex.submit(ask, state, key)] = d
        extra = [{"date": jobs[f], "cond": "E_news_swap", **f.result()} for f in cf.as_completed(jobs)]
    r = pd.concat([r, pd.DataFrame(extra)], ignore_index=True)
    r.to_csv(OUT / "probe_results.csv", index=False)

    # stronger free baselines: expanding window, refit monthly, predict next month
    m["gb_p"] = np.nan
    for mo in sorted(m.date.str[:6].unique())[6:]:
        tr = m[m.date.str[:6] < mo].iloc[:-5]
        te = m.date.str[:6] == mo
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05)
        m.loc[te, "gb_p"] = gb.fit(tr[FEATS], tr.crash_next5.astype(int)).predict_proba(m.loc[te, FEATS])[:, 1]
    m["rule_ld5"] = m.limit_down_5d
    m["rule_neg_ret5"] = -m.median_ret_5d
    m["rule_turnover"] = m.amount_z60

    t = r[r.ok].pivot(index="date", columns="cond", values="p").join(
        m.set_index("date")[["crash_next5", "gb_p", "rule_ld5", "rule_neg_ret5", "rule_turnover"]])
    t["year"] = t.index.str[:4]
    cols = ["A_real", "B_shifted", "C_no_date", "E_news_swap", "D_numeric", "gb_p", "rule_ld5", "rule_neg_ret5", "rule_turnover"]
    rows = []
    for grp, g in [("all", t), *t.groupby("year")]:
        y = g.crash_next5.astype(int)
        row = {"subset": grp, "n": len(g), "pos": int(y.sum())}
        for c in cols:
            ok = g[c].notna()
            row[c] = round(roc_auc_score(y[ok], g.loc[ok, c]), 3) if y[ok].nunique() == 2 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    # bootstrap CI (by month blocks) for A_real minus the best free baseline on all dates
    rng = np.random.default_rng(7)
    months = t.index.str[:6]
    diffs = []
    for _ in range(2000):
        pick = rng.choice(np.unique(months), len(np.unique(months)), replace=True)
        s = pd.concat([t[months == mo] for mo in pick])
        if s.crash_next5.nunique() < 2 or s.gb_p.isna().any():
            s = s.dropna(subset=["gb_p"])
        if s.crash_next5.nunique() < 2:
            continue
        y = s.crash_next5.astype(int)
        diffs.append(roc_auc_score(y, s.A_real) - roc_auc_score(y, s.rule_ld5))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    text = out.to_string(index=False) + f"\n\nA_real minus rule_ld5 AUC, month-block bootstrap 95% CI: [{lo:.3f}, {hi:.3f}]"
    (OUT / "probe_controls.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
