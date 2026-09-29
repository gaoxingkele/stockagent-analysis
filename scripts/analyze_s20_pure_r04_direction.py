#!/usr/bin/env python
"""Round 4: does dropping chop from the training labels buy direction?

Three LightGBM heads on the same 166 portable factors, same purged
walk-forward folds (S20_V2_FOLDS, test windows all inside the dev period):
  A dir_movers   : up_first(+U) vs pure_down(-D), chop rows DROPPED from training
  B up_vs_rest   : up_first vs everything else (chop kept, as stage1-like)
  C down_vs_rest : pure_down vs everything else (unS20-like, whole market)
Evaluation inside the stage1 daily Top-K candidate pool, where the question
"which of these movers goes up" actually lives.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from explore_r20_target_prob_v2 import _load_dataset  # noqa: E402
from stockagent_analysis.s20 import purged_walk_forward_masks  # noqa: E402
from train_s20_20r_residual import PORTABLE_EXCLUDED_FEATURES  # noqa: E402
from train_s20_v2_multitarget import COMPARABLE_SAMPLE_SEED, S20_V2_FOLDS  # noqa: E402
from analyze_s20_pure_r01_states import OUT, states  # noqa: E402
from analyze_s20_pure_r03_asymmetry import exit_return  # noqa: E402

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.04, num_leaves=31, min_data_in_leaf=250,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, seed=20260928)


def fit(fit_df, tune_df, feats, y_fit, y_tune):
    return lgb.train(PARAMS, lgb.Dataset(fit_df[feats], label=y_fit), num_boost_round=400,
                     valid_sets=[lgb.Dataset(tune_df[feats], label=y_tune)],
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])


def horizon_end(dates: pd.Series) -> pd.Series:
    cal = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    idx = {d: i for i, d in enumerate(cal)}
    return dates.map(lambda d: cal[min(idx[d] + 20, len(cal) - 1)] if d in idx else "99999999")


def load(U: int, D: int, sample_bps: int) -> tuple[pd.DataFrame, list[str]]:
    cache = OUT / f"feat_cache_bps{sample_bps}.parquet"
    if cache.exists():
        data = pd.read_parquet(cache)
        feats = [c for c in pd.read_csv(OUT / "feat_cols.csv")["f"]]
    else:
        data, feats, _ = _load_dataset(sample_bps, COMPARABLE_SAMPLE_SEED)
        feats = [c for c in feats if c not in PORTABLE_EXCLUDED_FEATURES]
        data = data[["ts_code", "trade_date", *feats]].copy()
        data.to_parquet(cache, index=False)
        pd.DataFrame({"f": feats}).to_csv(OUT / "feat_cols.csv", index=False)
    cols = ["ts_code", "trade_date", "ret20", f"up{U}_day", f"dn{D}_day", "dn5_day", "dn8_day",
            "up10_day", "up15_day", "up20_day", "dn10_day", "dn15_day"]
    panel = pd.read_parquet(OUT / "path_panel.parquet", columns=sorted(set(cols)))
    data = data.merge(panel, on=["ts_code", "trade_date"], how="inner")
    data["st"] = states(data, U, D).to_numpy()
    data = data[data["st"] != "ambiguous"].reset_index(drop=True)
    data["horizon_end_date"] = horizon_end(data["trade_date"])
    return data, feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--U", type=int, default=15)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--sample-bps", type=int, default=5000)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    if a.eval_only:
        return evaluate(pd.read_parquet(OUT / f"r04_preds_U{a.U}_D{a.D}.parquet"), a)
    data, feats = load(a.U, a.D, a.sample_bps)
    print(f"rows {len(data):,}, features {len(feats)}", flush=True)

    s1 = pd.read_parquet(ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
                         columns=["ts_code", "trade_date", "stage1_score"])
    s1["s1_rank"] = s1.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first")

    preds = []
    for fold in S20_V2_FOLDS:
        m = purged_walk_forward_masks(data["trade_date"], data["horizon_end_date"], fold)
        f, t, te = data[m["fit"]], data[m["tune"]], data[m["test"]].copy()
        movers_f, movers_t = f[f.st.isin(["pure_up", "pure_down"])], t[t.st.isin(["pure_up", "pure_down"])]
        A = fit(movers_f, movers_t, feats, movers_f.st.eq("pure_up"), movers_t.st.eq("pure_up"))
        B = fit(f, t, feats, f.st.eq("pure_up"), t.st.eq("pure_up"))
        C = fit(f, t, feats, f.st.eq("pure_down"), t.st.eq("pure_down"))
        te["A_dir"], te["B_up"], te["C_down"] = A.predict(te[feats]), B.predict(te[feats]), C.predict(te[feats])
        te["fold"] = fold.name
        preds.append(te[["ts_code", "trade_date", "fold", "st", "ret20", "A_dir", "B_up", "C_down",
                         *[c for c in te.columns if c.endswith("_day")]]])
        print(f"{fold.name}: fit {len(f):,} (movers {len(movers_f):,}) test {len(te):,} "
              f"iters A/B/C {A.best_iteration}/{B.best_iteration}/{C.best_iteration}", flush=True)
    P = pd.concat(preds, ignore_index=True).merge(s1, on=["ts_code", "trade_date"], how="left")
    P["exit"] = exit_return(P, a.U, a.D)
    P["A_x_B"] = P["A_dir"] * P["B_up"]                 # P(up|move) * P(up) proxy
    P["ratio"] = P["B_up"] / (P["C_down"] + 0.01)
    P.to_parquet(OUT / f"r04_preds_U{a.U}_D{a.D}.parquet", index=False)
    return evaluate(P, a)


def evaluate(P: pd.DataFrame, a) -> int:
    P = P[P["stage1_score"].notna()].copy()
    lines = [f"U={a.U} D={a.D} sample_bps={a.sample_bps} pool=stage1 top{a.pool}"]
    scores = {"stage1": "stage1_score", "A_dir": "A_dir", "B_up": "B_up", "-C_down": "C_down",
              "A_x_B": "A_x_B", "B/C ratio": "ratio"}
    for scope, Q in (("universe", P), ("pool", P[P.s1_rank <= a.pool])):
        lines.append(f"\n## {scope}: rows {len(Q):,}")
        mv = Q[Q.st.isin(["pure_up", "pure_down"])]
        rows = []
        for name, col in scores.items():
            sgn = -1 if name.startswith("-") else 1
            x = sgn * Q[col]
            auc_dir = np.nanmean([roc_auc_score(g.st.eq("pure_up"), sgn * g[col])
                                  for _, g in mv.groupby("trade_date") if g.st.nunique() == 2 and len(g) >= 10])
            q = Q.assign(x=x)
            q = q[q[col].notna()]
            q["r"] = q.groupby("trade_date")["x"].rank(ascending=False, pct=True)
            top = q[q.r <= 0.2]
            sh = top.st.value_counts(normalize=True)
            rows.append({"score": name, "dir_AUC_daily(movers)": round(auc_dir, 3),
                         "top20%_up_first": round(100 * sh.get("pure_up", 0), 1),
                         "top20%_pure_down": round(100 * sh.get("pure_down", 0), 1),
                         "top20%_chop": round(100 * sh.get("chop", 0), 1),
                         "top20%_exit_mean": round(float(np.nanmean(top.exit)), 2),
                         "top20%_exit_win": round(100 * (top.exit > 0).mean(), 1)})
        base = Q.st.value_counts(normalize=True)
        lines.append(pd.DataFrame(rows).to_string(index=False))
        lines.append(f"base: up_first {100*base.get('pure_up',0):.1f} pure_down {100*base.get('pure_down',0):.1f} "
                     f"chop {100*base.get('chop',0):.1f} exit mean {np.nanmean(Q.exit):.2f} win {100*(Q.exit>0).mean():.1f}")
        if scope == "pool":
            for fo, g in Q.groupby("fold"):
                gm = g[g.st.isin(["pure_up", "pure_down"])]
                lines.append(f"  {fo}: pool dir AUC  A_dir {roc_auc_score(gm.st.eq('pure_up'), gm.A_dir):.3f}  "
                             f"B_up {roc_auc_score(gm.st.eq('pure_up'), gm.B_up):.3f}  "
                             f"-C {roc_auc_score(gm.st.eq('pure_up'), -gm.C_down):.3f}  "
                             f"stage1 {roc_auc_score(gm.st.eq('pure_up'), gm.stage1_score):.3f}  "
                             f"base up/(up+down) {100*gm.st.eq('pure_up').mean():.1f}%")
    text = "\n".join(lines)
    (OUT / f"r04_direction_U{a.U}_D{a.D}.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
