#!/usr/bin/env python
"""Round 5: reverse scoring as meta-labeling inside a high-amplitude pool.

stage1 is an amplitude meter (R01), so a reverse model trained on the whole
market mostly re-learns amplitude. Meta-labeling (Lopez de Prado 2018; Joubert
2022) trains the secondary model only where the primary says "candidate".
No stage1 OOF exists in the fit windows, so the training pool is a non-learned
amplitude proxy: daily top 20% by natr_14. Heads (chop handling differs):
  M_down : pure_down vs rest        (veto head, chop counted as "not down")
  M_dir  : pure_up  vs pure_down    (direction head, chop DROPPED)
Evaluated inside the stage1 daily Top100 on the test windows, against the
whole-market heads from R04 (C_down, A_dir).
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.s20 import purged_walk_forward_masks  # noqa: E402
from train_s20_v2_multitarget import S20_V2_FOLDS  # noqa: E402
from analyze_s20_pure_r01_states import OUT  # noqa: E402
from analyze_s20_pure_r04_direction import fit, load  # noqa: E402

U, D, POOL_Q = 15, 10, 0.8


def main() -> int:
    data, feats = load(U, D, 5000)
    data["amp_pool"] = data.groupby("trade_date")["natr_14"].rank(pct=True) >= POOL_Q
    r04 = pd.read_parquet(OUT / f"r04_preds_U{U}_D{D}.parquet")
    preds = []
    for fold in S20_V2_FOLDS:
        m = purged_walk_forward_masks(data["trade_date"], data["horizon_end_date"], fold)
        f, t = data[m["fit"] & data.amp_pool], data[m["tune"] & data.amp_pool]
        te = data[m["test"]].copy()
        Md = fit(f, t, feats, f.st.eq("pure_down"), t.st.eq("pure_down"))
        fm, tm = f[f.st.isin(["pure_up", "pure_down"])], t[t.st.isin(["pure_up", "pure_down"])]
        Mr = fit(fm, tm, feats, fm.st.eq("pure_up"), tm.st.eq("pure_up"))
        te["M_down"], te["M_dir"] = Md.predict(te[feats]), Mr.predict(te[feats])
        preds.append(te[["ts_code", "trade_date", "M_down", "M_dir"]])
        print(f"{fold.name}: pool fit {len(f):,} test {len(te):,} iters down/dir {Md.best_iteration}/{Mr.best_iteration}",
              flush=True)
    P = r04.merge(pd.concat(preds), on=["ts_code", "trade_date"], how="inner")
    uns = pd.read_parquet(ROOT / "output/experiments/s20_uns20_20260927/predictions.parquet",
                          columns=["ts_code", "trade_date", "uns20_p"])
    P = P.merge(uns, on=["ts_code", "trade_date"], how="left")
    P.to_parquet(OUT / "r05_preds.parquet", index=False)
    pool = P[P.s1_rank <= 100].copy()

    lines = [f"U={U} D={D}; training pool = daily top {int((1-POOL_Q)*100)}% natr_14; eval pool = stage1 top100 (50% sample)"]
    # 1. direction separation among movers inside the stage1 pool
    mv = pool[pool.st.isin(["pure_up", "pure_down"])]
    rows = []
    for name, col, sgn in (("stage1", "stage1_score", 1), ("A_dir (market, chop dropped)", "A_dir", 1),
                           ("-C_down (market)", "C_down", -1), ("-uns20_p (market)", "uns20_p", -1),
                           ("M_dir (pool, chop dropped)", "M_dir", 1), ("-M_down (pool)", "M_down", -1)):
        r = {"score": name}
        for fo, g in mv.groupby("fold"):
            g = g[g[col].notna()]
            r[fo] = round(roc_auc_score(g.st.eq("pure_up"), sgn * g[col]), 3) if g.st.nunique() == 2 else np.nan
        rows.append(r)
    lines.append("\n## direction AUC inside stage1 top100 (pure_up vs pure_down)\n" + pd.DataFrame(rows).to_string(index=False))

    # 2. veto: drop the worst v% of the pool by a reverse score, then take the stage1 top 20% of the pool size
    lines.append("\n## veto then stage1 re-take: per-day drop worst v% by score, keep stage1-best 20 of 100 equivalent")
    rows = []
    for name, col, sgn in (("none", None, 0), ("-C_down", "C_down", -1), ("-uns20_p", "uns20_p", -1),
                           ("-M_down", "M_down", -1), ("M_dir", "M_dir", 1)):
        for v in ((0.0,) if col is None else (0.2, 0.4, 0.6)):
            q = pool if col is None else pool[pool[col].notna()]
            if col is not None:
                bad = q.groupby("trade_date")[col].rank(pct=True, ascending=(sgn > 0)) <= v
                q = q[~bad]
            n_keep = (pool.groupby("trade_date").size() * 0.2).round().clip(lower=1)
            q = q.assign(r=q.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first"))
            q = q[q.r <= q.trade_date.map(n_keep)]
            sh = q.st.value_counts(normalize=True)
            rows.append({"veto": name, "v": v, "n": len(q),
                         "pure_up": round(100 * sh.get("pure_up", 0), 1), "pure_down": round(100 * sh.get("pure_down", 0), 1),
                         "chop": round(100 * sh.get("chop", 0), 1),
                         "exit_mean": round(float(np.nanmean(q.exit)), 2), "exit_win": round(100 * (q.exit > 0).mean(), 1)})
    lines.append(pd.DataFrame(rows).to_string(index=False))

    # 3. risk-coverage: order the pool by M_dir (within stage1 top100), keep top f
    lines.append("\n## risk-coverage inside stage1 top100: keep top f of pool by score")
    rows = []
    for name, col, sgn in (("stage1", "stage1_score", 1), ("M_dir", "M_dir", 1), ("-M_down", "M_down", -1),
                           ("A_dir", "A_dir", 1), ("-C_down", "C_down", -1)):
        q = pool.assign(r=pool.groupby("trade_date")[col].rank(pct=True, ascending=(sgn < 0)))
        for f in (1.0, 0.5, 0.2, 0.1, 0.05):
            s = q[q.r <= f]
            sh = s.st.value_counts(normalize=True)
            rows.append({"score": name, "keep": f, "n": len(s), "pure_up": round(100 * sh.get("pure_up", 0), 1),
                         "pure_down": round(100 * sh.get("pure_down", 0), 1),
                         "exit_mean": round(float(np.nanmean(s.exit)), 2), "exit_win": round(100 * (s.exit > 0).mean(), 1)})
    lines.append(pd.DataFrame(rows).to_string(index=False))
    text = "\n".join(lines)
    (OUT / "r05_meta_veto.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
