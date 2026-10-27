#!/usr/bin/env python
"""CSI300 four-horizon path reward (I0009 data), volatility stripped (idea I0010).

  python scripts/analyze_horizon_reward_neutral.py

User request 2026-10-27: I0009 learned low volatility (corr(pred, NATR) -0.53..-0.75).  Strip volatility and run again.
PRE-REGISTERED.  Evidence tier reused_holdout / descriptive (same periods as I0009; nothing from 2026-08-06 on is read).
Reuses output/experiments/horizon_reward_csi300_20261026/{scores,features}.parquet unchanged.
Neutralisation: every day the constituents are cut into NATR terciles (within-day percentile of t_NATR); the score (and the forward return)
is re-ranked WITHIN day x tercile, so the target carries no NATR level information.  Features are unchanged (the model may still use
volatility features, but gains nothing from the level of volatility; the evaluation is also within tercile).
Part 2 univariate IC of every feature with the stratified score; Part 3 walk-forward LightGBM (same fixed parameters, folds as I0009);
pooled IC with stratified score and with stratified forward return (NW t); quintile gap of the prediction computed WITHIN tercile
(score, forward return, broken share); shuffle placebo within day x tercile (300); corr of the prediction with NATR / market cap.
Falsified (I0010) if for a horizon: pooled IC with the stratified score has NW t < 1.96, or the NATR-neutral top-minus-bottom forward return <= 0,
or the shuffle placebo reaches the observed gap in more than 5% of shuffles.
Output: output/experiments/horizon_reward_neutral_20261027/{report.txt, ic_univariate.csv, walkforward.csv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import analyze_horizon_reward_csi300 as B  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402

SRC = B.OUT
OUT = ROOT / "output/experiments/horizon_reward_neutral_20261027"
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def main() -> int:
    import lightgbm as lgb
    from scipy.stats import norm
    OUT.mkdir(parents=True, exist_ok=True)
    sc, fe = pd.read_parquet(SRC / "scores.parquet"), pd.read_parquet(SRC / "features.parquet")
    sc = sc[sc.end_date <= B.END]
    feats = [c for c in fe.columns if c not in ("ts_code", "trade_date")]
    fe[feats] = fe[feats].replace([np.inf, -np.inf], np.nan)
    feats = [c for c in feats if fe[c].notna().mean() > 0.7 and fe[c].nunique() > 5]
    pf = fe.groupby("trade_date")[feats].rank(pct=True).astype("float32")
    pf.insert(0, "trade_date", fe.trade_date.to_numpy())
    pf.insert(0, "ts_code", fe.ts_code.to_numpy())
    say(f"features {len(feats)}; stratifier t_NATR within-day percentile, terciles")
    rng = np.random.default_rng(20261027)
    bar = norm.isf(0.025 / (len(feats) * len(B.HORIZONS)))
    ic_rows, wf_rows = [], []
    for H in B.HORIZONS:
        s = sc[sc.H == H].merge(pf, on=["ts_code", "trade_date"])
        s = s[s.t_NATR.notna()].copy()
        s["broken"] = s.broken.astype(float)
        s["tier"] = np.minimum((s.t_NATR * 3).astype(int), 2)
        g = s.groupby(["trade_date", "tier"])
        s["ys"] = g.score.rank(pct=True)
        s["yf"] = g.fwd.rank(pct=True)
        say(f"\n{'=' * 98}\nHORIZON {H}: {len(s):,} constituent-days, {s.trade_date.nunique()} signal days")
        say("Part 1: by NATR tercile (low/mid/high) mean score " + "/".join(f"{s[s.tier == k].score.mean():+.2f}" for k in range(3)) +
            "; mean fwd % " + "/".join(f"{100 * s[s.tier == k].fwd.mean():+.1f}" for k in range(3)) +
            "; broken % " + "/".join(f"{100 * s[s.tier == k].broken.mean():.0f}" for k in range(3)))
        ic = B.daily_ic(s.trade_date, s[feats], s.ys)
        r = pd.DataFrame({"feature": feats, "ic": ic.mean().to_numpy(), "t": [tstat(ic[c].dropna()) for c in feats]})
        for bi, (a, b) in enumerate(B.BLOCKS):
            m = (ic.index >= a) & (ic.index <= b)
            r[f"ic_b{bi + 1}"] = ic[m].mean().to_numpy()
        r["H"] = H
        blocks = [f"ic_b{i + 1}" for i in range(len(B.BLOCKS))]
        r["same_sign"] = (np.sign(r[blocks]).nunique(axis=1) == 1)
        ic_rows.append(r)
        top = r.reindex(r.t.abs().sort_values(ascending=False).index).head(8)
        say(f"Part 2: IC with the stratified score; Bonferroni bar {bar:.2f}; over the bar {int((r.t.abs() >= bar).sum())}, |t|>=2 {int((r.t.abs() >= 2).sum())} of {len(r)}, same sign in all blocks {int(r.same_sign.sum())}")
        say(top[["feature", "ic", "t"] + blocks].round(3).to_string(index=False))
        preds = []
        for k in range(len(B.FOLDS) - 1):
            a, b = B.FOLDS[k], B.FOLDS[k + 1]
            te, tr = s[(s.trade_date >= a) & (s.trade_date < b)], s[s.end_date < a]
            if len(te) < 5000 or len(tr) < 20000:
                continue
            params = dict(objective="regression", num_leaves=15, learning_rate=0.03, min_data_in_leaf=500, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
            m = lgb.train(params, lgb.Dataset(tr[feats], tr.ys - 0.5), num_boost_round=300)
            te = te.assign(pred=m.predict(te[feats]), fold=a)
            preds.append(te[["ts_code", "trade_date", "fold", "tier", "pred", "ys", "yf", "score", "fwd", "broken"] + feats])
            fic = B.daily_ic(te.trade_date, te[["pred"]], te.ys).pred
            wf_rows.append({"H": H, "fold": a, "ic_score": fic.mean(), "t": tstat(fic.dropna()), "ic_fwd": B.daily_ic(te.trade_date, te[["pred"]], te.yf).pred.mean()})
        if not preds:
            continue
        P = pd.concat(preds, ignore_index=True)
        ics = B.daily_ic(P.trade_date, P[["pred"]], P.ys).pred.dropna()
        icf = B.daily_ic(P.trade_date, P[["pred"]], P.yf).pred.dropna()
        fold_ic = pd.DataFrame([w for w in wf_rows if w["H"] == H])
        say(f"Part 3: walk-forward {len(fold_ic)} folds, {len(P):,} test rows ({P.trade_date.min()}..{P.trade_date.max()})")
        say(f"  pooled IC with stratified score {ics.mean():+.4f} (NW t {tstat(ics):+.1f}); with stratified fwd return {icf.mean():+.4f} (t {tstat(icf):+.1f}); positive days {100 * (ics > 0).mean():.0f}%")
        say("  per fold IC: " + ", ".join(f"{r_.fold[:6]} {r_.ic_score:+.3f}(t{r_.t:+.1f})" for r_ in fold_ic.itertuples()) + f"; negative folds {int((fold_ic.ic_score < 0).sum())}")
        P["q"] = P.groupby(["trade_date", "tier"]).pred.rank(pct=True, method="first")
        hi_, lo_ = P[P.q > 0.8], P[P.q <= 0.2]
        gaps = {}
        for lab, k, mult in (("score", "score", 1), ("fwd return pp", "fwd", 100), ("broken pp", "broken", 100)):
            d = (hi_.groupby("trade_date")[k].mean() - lo_.groupby("trade_date")[k].mean()).dropna()
            gaps[k] = d.mean()
            say(f"  NATR-neutral top quintile minus bottom, {lab:14s}: {d.mean() * mult:+7.2f} (t {tstat(d):+.1f}); top {hi_[k].mean() * mult:+.2f}, bottom {lo_[k].mean() * mult:+.2f}")
        for k in range(3):
            h2, l2 = hi_[hi_.tier == k], lo_[lo_.tier == k]
            d = (h2.groupby("trade_date").fwd.mean() - l2.groupby("trade_date").fwd.mean()).dropna()
            say(f"    within tercile {k} (0 = low vol): fwd return gap {100 * d.mean():+.2f}pp (t {tstat(d):+.1f})")
        sh = []
        for _ in range(300):
            pr = P.groupby(["trade_date", "tier"]).pred.transform(lambda x: rng.permutation(x.to_numpy()))
            tt = P.assign(pred=pr)
            tt["q"] = tt.groupby(["trade_date", "tier"]).pred.rank(pct=True, method="first")
            sh.append((tt[tt.q > 0.8].groupby("trade_date").score.mean() - tt[tt.q <= 0.2].groupby("trade_date").score.mean()).mean())
        say(f"  shuffle placebo (300, within day x tercile): observed score gap {gaps['score']:+.3f}; shuffled 95th pct {np.percentile(sh, 95):+.3f}; shuffles >= observed {int((np.array(sh) >= gaps['score']).sum())}")
        base = {}
        for nm in ("t_NATR", "q_ROC20", "q_ROC60", "lmv", "pb", "boll_pctb"):
            if nm in feats:
                ic_b = B.daily_ic(P.trade_date, P[[nm]], P.ys)[nm].dropna()
                base[nm] = (ic_b.mean(), tstat(ic_b))
        say("  baseline IC with the stratified score: " + "; ".join(f"{k} {v[0]:+.3f}(t{v[1]:+.1f})" for k, v in base.items()))
        say(f"  corr(pred, NATR pct) {P.pred.corr(P.t_NATR):+.2f}; corr(pred, log mkt cap pct) {P.pred.corr(P.lmv):+.2f}")
        imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
        say("  last-fold top gain features: " + ", ".join(f"{k}" for k in imp.head(8).index))
    pd.concat(ic_rows).to_csv(OUT / "ic_univariate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(wf_rows).to_csv(OUT / "walkforward.csv", index=False, encoding="utf-8-sig")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
