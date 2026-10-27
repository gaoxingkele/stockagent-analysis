#!/usr/bin/env python
"""Path-reward score for every CSI300 stock on every day, four horizons (20 / 30 / 40 / 60 sessions); can features learn it?

  python scripts/analyze_horizon_reward_csi300.py

User request 2026-10-26 (idea I0009; follows I0008, the pool-only version that saturated): use the whole history rather than the S20 /
R20 pools to avoid over-fitting, score every stock at every close from what happens over the next 20 / 30 / 40 / 60 sessions (four groups
= four horizons), start with a base set of stocks = CSI300 constituents, then see whether anything can be learned.
PRE-REGISTERED before any result was looked at.  Evidence tier: reused_holdout / descriptive: every calendar period was already used by
earlier studies (a new population and a walk-forward fit, but not an unused window).  Nothing from 2026-08-06 on is read: a signal day is
used only when its horizon ends on or before 2026-08-05 (so the 60-session score stops around 2026-05).

Universe: CSI300 constituents, point in time (Tushare index_weight, latest monthly snapshot on or before the signal day).
Prices: split-adjusted daily bars, 2024-01-02..2026-08-05 (the cache starts 2024-01-02). Entry = next session open E.
Reward (fixed here; changed from I0008 so the score does not saturate - the I0008 floor held 61-81% of cases at -4):
  step 5%.  peak_prev(d) = max(0, hi_1..hi_{d-1}).   hi_d = high/E-1, lo_d = low/E-1.
  broken (any lo_d <= -10%; same-day new high still counts as broken)  ->  score = -2   (one point per 5% of loss, mirror of +1 per 5% of gain)
  otherwise  up = floor(max(hi)/5%)  (no cap),  giveback = max_d (peak_prev(d) - lo_d)/5%  (>= 0),  score = up - giveback
  Check: +40% straight up = 8; +5% then -4% from the peak = 1 - 0.8 = +0.2 ("only 1% left"); score >= -2 always.
Features at the close of t (nothing later): 282 Alpha158-style + TA-Lib + Bollinger %b factors (scripts/analyze_factor_zoo_in_pools.py code, run
on every day) and daily_basic pe_ttm, pb, ps_ttm, turnover_rate_f, log circ_mv - each turned into its within-day percentile among the day's constituents.
Everything is within-day (N0026 / N0027 lesson: a label learned across days learns "which day is good"): target = within-day percentile of the score.
Part 1 score distributions per horizon; correlation across horizons; score vs forward return and vs first-passage-style stops.
Part 2 univariate within-day Spearman IC with the score, whole history and four date blocks; Newey-West(19) t;
        Bonferroni bar over features x horizons; features whose IC sign is the same in all four blocks.
Part 3 walk-forward LightGBM (fixed parameters: num_leaves 15, lr 0.03, 300 trees, min_data_in_leaf 500, feature_fraction 0.5, bagging 0.8; no tuning),
        test folds = calendar quarters 2025Q1..2026Q2 (limited by label availability), train = earlier signal days whose horizon ended before the
        fold starts (purged). Reported: pooled daily IC with the score and with the forward return (NW t), per-fold IC, top quintile minus bottom
        quintile of the prediction (score, forward return, broken share), within-day shuffle placebo (300), baselines (best simple factor by
        training-period IC; NATR; 20-day momentum), correlation of the prediction with NATR and market-cap rank.
Falsified (I0009) if, for a horizon, the pooled walk-forward IC t < 1.96 or the sign differs in more than a quarter of the folds, or the top-quintile
forward return is not above the bottom quintile, or the prediction is explained by NATR (|corr| > 0.7) / beaten by the best simple factor.
Output: output/experiments/horizon_reward_csi300_20261026/{report.txt, scores.parquet, ic_univariate.csv, walkforward.csv}
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_factor_zoo_in_pools import one_stock  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402
from price_panel import adjusted_daily  # noqa: E402

OUT = ROOT / "output/experiments/horizon_reward_csi300_20261026"
IDXW = ROOT / "output/tushare_cache/index_weight"
HORIZONS = (20, 30, 40, 60)
STEP, STOP, END = 0.05, -0.10, "20260805"
FOLDS = ["20250102", "20250401", "20250701", "20251001", "20260102", "20260401", "20260806"]   # last = open end, not read
BLOCKS = [("20240301", "20241231"), ("20250101", "20250630"), ("20250701", "20251231"), ("20260101", "20260805")]
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------ universe
def members() -> pd.DataFrame:
    IDXW.mkdir(parents=True, exist_ok=True)
    f = IDXW / "000300.SH.parquet"
    if not f.exists():
        from dotenv import load_dotenv
        import tushare as ts
        load_dotenv(ROOT / ".env")
        pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
        parts = []
        for y in (2023, 2024, 2025, 2026):
            for m in range(1, 13):
                a, b = f"{y}{m:02d}01", f"{y}{m:02d}28"
                d = pro.index_weight(index_code="000300.SH", start_date=a, end_date=f"{y}{m:02d}31")
                if d is not None and len(d):
                    parts.append(d)
        pd.concat(parts).drop_duplicates(["con_code", "trade_date"]).to_parquet(f, index=False)
    w = pd.read_parquet(f)
    w = w[w.trade_date <= END]
    return w[["con_code", "trade_date"]].rename(columns={"con_code": "ts_code", "trade_date": "snap"})


def universe(days: list[str], w: pd.DataFrame) -> pd.DataFrame:
    snaps = sorted(w.snap.unique())
    rows = []
    for d in days:
        s = [x for x in snaps if x <= d]
        if s:
            rows.append(pd.DataFrame({"ts_code": w[w.snap == s[-1]].ts_code.to_numpy(), "trade_date": d}))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ rewards
def rewards(hi: np.ndarray, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak_prev = np.concatenate([np.zeros((len(hi), 1)), np.maximum.accumulate(np.maximum(hi, 0), axis=1)[:, :-1]], axis=1)
    giveback = np.maximum((peak_prev - lo) / STEP, 0).max(axis=1)
    up = np.floor(np.maximum(hi.max(axis=1), 0) / STEP)
    broken = (lo <= STOP).any(axis=1)
    return np.where(broken, -2.0, up - giveback), broken


def stock_scores(g: pd.DataFrame, sig_days: set[str]) -> pd.DataFrame:
    g = g.reset_index(drop=True)
    dates = g.trade_date.to_numpy()
    O, Hh, L, C = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sel = np.array([d in sig_days for d in dates])
    out = []
    for H in HORIZONS:
        i = np.where(sel)[0] + 1                       # entry = next session
        i = i[i + H - 1 < len(dates)]
        if not len(i):
            continue
        idx = i[:, None] + np.arange(H)[None, :]
        E = O[i]
        hi, lo = Hh[idx] / E[:, None] - 1, L[idx] / E[:, None] - 1
        sc, br = rewards(hi, lo)
        out.append(pd.DataFrame({"ts_code": g.ts_code.iloc[0], "trade_date": dates[i - 1], "H": H, "score": sc, "broken": br,
                                 "fwd": C[idx[:, -1]] / E - 1, "end_date": dates[idx[:, -1]]}))
    return pd.concat(out) if out else pd.DataFrame()


def _chunk(args):
    frame, keys = args
    sc, fe = [], []
    for code, g in frame.groupby("ts_code", sort=False):
        if code not in keys:
            continue
        g = g.reset_index(drop=True)
        sc.append(stock_scores(g, keys[code]))
        fe.append(one_stock(g, keys[code]))
    return (pd.concat([x for x in sc if len(x)], ignore_index=True), pd.concat(fe, ignore_index=True))


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    days = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet") if "20240102" <= p.stem <= END)
    U = universe(days[60:], members())
    keys: dict[str, set[str]] = {}
    for code, d in zip(U.ts_code, U.trade_date):
        keys.setdefault(code, set()).add(d)
    px = adjusted_daily("20240102", END, set(keys))
    codes = np.array(sorted(keys))
    chunks = [(px[px.ts_code.isin(set(codes[i::8]))], {k: keys[k] for k in codes[i::8]}) for i in range(8)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_chunk, chunks))
    sc = pd.concat([r[0] for r in res], ignore_index=True)
    fe = pd.concat([r[1] for r in res], ignore_index=True)
    db = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "pe_ttm", "pb", "ps_ttm", "turnover_rate_f", "circ_mv"])
                    for p in sorted((ROOT / "output/tushare_cache/daily_basic").glob("*.parquet")) if "20240102" <= p.stem <= END], ignore_index=True)
    db["trade_date"] = db.trade_date.astype(str)
    db["lmv"] = np.log(db.circ_mv.where(db.circ_mv > 0))
    fe = fe.merge(db.drop(columns="circ_mv"), on=["ts_code", "trade_date"], how="left")
    return sc, fe


# ------------------------------------------------------------------ stats
def daily_ic(day: pd.Series, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Spearman-style IC per day for every column of X (already within-day percentile) against y (within-day percentile)."""
    Xc = X - X.groupby(day).transform("mean")
    yc = y - y.groupby(day).transform("mean")
    num = Xc.mul(yc, axis=0).groupby(day).sum()
    den = np.sqrt((Xc ** 2).groupby(day).sum().mul((yc ** 2).groupby(day).sum(), axis=0))
    return num / den.replace(0, np.nan)


def main() -> int:
    import lightgbm as lgb
    from scipy.stats import norm
    OUT.mkdir(parents=True, exist_ok=True)
    cs, cf = OUT / "scores.parquet", OUT / "features.parquet"
    if cs.exists() and cf.exists():
        sc, fe = pd.read_parquet(cs), pd.read_parquet(cf)
    else:
        sc, fe = build()
        sc.to_parquet(cs, index=False)
        fe.to_parquet(cf, index=False)
    sc = sc[sc.end_date <= END]
    feats = [c for c in fe.columns if c not in ("ts_code", "trade_date")]
    fe[feats] = fe[feats].replace([np.inf, -np.inf], np.nan)
    feats = [c for c in feats if fe[c].notna().mean() > 0.7 and fe[c].nunique() > 5]
    pf = fe.groupby("trade_date")[feats].rank(pct=True).astype("float32")
    pf.insert(0, "trade_date", fe.trade_date.to_numpy())
    pf.insert(0, "ts_code", fe.ts_code.to_numpy())
    natr = fe.set_index(["ts_code", "trade_date"])["t_NATR"] if "t_NATR" in fe else None
    say(f"constituent-days with features {len(fe):,}; stocks {fe.ts_code.nunique()}; days {fe.trade_date.nunique()} ({fe.trade_date.min()}..{fe.trade_date.max()}); features {len(feats)}")
    rng = np.random.default_rng(20261026)
    bar = norm.isf(0.025 / (len(feats) * len(HORIZONS)))
    ic_rows, wf_rows = [], []
    for H in HORIZONS:
        s = sc[sc.H == H].merge(pf, on=["ts_code", "trade_date"])
        s["broken"] = s.broken.astype(float)
        s["ys"] = s.groupby("trade_date").score.rank(pct=True)
        s["yf"] = s.groupby("trade_date").fwd.rank(pct=True)
        say(f"\n{'=' * 98}\nHORIZON {H} sessions: {len(s):,} constituent-days, {s.trade_date.nunique()} signal days ({s.trade_date.min()}..{s.trade_date.max()})")
        q = s.score.quantile([.1, .25, .5, .75, .9]).round(2).tolist()
        say(f"Part 1: score mean {s.score.mean():+.2f}, quantiles 10/25/50/75/90% {q}, broken (-2) {100 * s.broken.mean():.1f}%, score>=4 {100 * (s.score >= 4).mean():.1f}%, "
            f">=8 {100 * (s.score >= 8).mean():.1f}%; mean fwd return {100 * s.fwd.mean():+.2f}%; corr(score, fwd) {s.score.corr(s.fwd):+.2f}")
        s["qgrp"] = pd.qcut(s.score.rank(method="first"), 4, labels=False)
        say("  four score groups (all days pooled): mean score / mean fwd return % / broken % -> " + "; ".join(
            f"G{k + 1}: {g.score.mean():+.2f}/{100 * g.fwd.mean():+.1f}/{100 * g.broken.mean():.0f}" for k, g in s.groupby("qgrp")))
        # ---- part 2
        ic = daily_ic(s.trade_date, s[feats], s.ys)
        r = pd.DataFrame({"feature": feats, "ic": ic.mean().to_numpy(), "t": [tstat(ic[c].dropna()) for c in feats]})
        for bi, (a, b) in enumerate(BLOCKS):
            m = (ic.index >= a) & (ic.index <= b)
            r[f"ic_b{bi + 1}"] = ic[m].mean().to_numpy()
        r["H"] = H
        blocks = [f"ic_b{i + 1}" for i in range(len(BLOCKS))]
        r["same_sign"] = (np.sign(r[blocks]).nunique(axis=1) == 1)
        ic_rows.append(r)
        top = r.reindex(r.t.abs().sort_values(ascending=False).index).head(10)
        say(f"Part 2: whole-history IC with the score; Bonferroni bar {bar:.2f}; features over the bar {int((r.t.abs() >= bar).sum())}, |t|>=2 {int((r.t.abs() >= 2).sum())} of {len(r)}, "
            f"same sign in all {len(BLOCKS)} blocks {int(r.same_sign.sum())}")
        say(top[["feature", "ic", "t"] + blocks].round(3).to_string(index=False))
        # ---- part 3
        preds = []
        for k in range(len(FOLDS) - 1):
            a, b = FOLDS[k], FOLDS[k + 1]
            te = s[(s.trade_date >= a) & (s.trade_date < b)]
            tr = s[(s.end_date < a)]
            if len(te) < 5000 or len(tr) < 20000:
                continue
            params = dict(objective="regression", num_leaves=15, learning_rate=0.03, min_data_in_leaf=500, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
            m = lgb.train(params, lgb.Dataset(tr[feats], tr.ys - 0.5), num_boost_round=300)
            te = te.assign(pred=m.predict(te[feats]), fold=a)
            preds.append(te[["ts_code", "trade_date", "fold", "pred", "ys", "yf", "score", "fwd", "broken"] + feats])
            fic = daily_ic(te.trade_date, te[["pred"]], te.ys).pred
            wf_rows.append({"H": H, "fold": a, "train_rows": len(tr), "test_rows": len(te), "ic_score": fic.mean(), "t": tstat(fic.dropna()),
                            "ic_fwd": daily_ic(te.trade_date, te[["pred"]], te.yf).pred.mean()})
        if not preds:
            continue
        P = pd.concat(preds, ignore_index=True)
        ics = daily_ic(P.trade_date, P[["pred"]], P.ys).pred.dropna()
        icf = daily_ic(P.trade_date, P[["pred"]], P.yf).pred.dropna()
        fold_ic = pd.DataFrame([w for w in wf_rows if w["H"] == H])
        say(f"Part 3: walk-forward, {len(fold_ic)} folds, {len(P):,} test constituent-days ({P.trade_date.min()}..{P.trade_date.max()})")
        say(f"  pooled daily IC with the score {ics.mean():+.4f} (NW t {tstat(ics):+.1f}); with the forward return {icf.mean():+.4f} (t {tstat(icf):+.1f}); positive days {100 * (ics > 0).mean():.0f}%")
        say("  per fold IC with the score: " + ", ".join(f"{r_.fold[:6]} {r_.ic_score:+.3f}(t{r_.t:+.1f})" for r_ in fold_ic.itertuples()) +
            f"; folds with negative IC: {int((fold_ic.ic_score < 0).sum())}")
        P["q"] = P.groupby("trade_date").pred.rank(pct=True, method="first")
        hi_, lo_ = P[P.q > 0.8], P[P.q <= 0.2]
        for lab, k, mult in (("score", "score", 1), ("fwd return pp", "fwd", 100), ("broken pp", "broken", 100)):
            d = (hi_.groupby("trade_date")[k].mean() - lo_.groupby("trade_date")[k].mean()).dropna()
            say(f"  top quintile minus bottom quintile, {lab:14s}: {d.mean() * mult:+7.2f} (t {tstat(d):+.1f}); top {hi_[k].mean() * mult:+.2f}, bottom {lo_[k].mean() * mult:+.2f}")
        obs = (hi_.groupby("trade_date").score.mean() - lo_.groupby("trade_date").score.mean()).mean()
        sh = []
        for _ in range(300):
            tt = P.assign(pred=P.groupby("trade_date").pred.transform(lambda x: rng.permutation(x.to_numpy())))
            tt["q"] = tt.groupby("trade_date").pred.rank(pct=True, method="first")
            sh.append((tt[tt.q > 0.8].groupby("trade_date").score.mean() - tt[tt.q <= 0.2].groupby("trade_date").score.mean()).mean())
        say(f"  shuffle placebo (300): observed score gap {obs:+.3f}; shuffled 95th pct {np.percentile(sh, 95):+.3f}; shuffles >= observed {int((np.array(sh) >= obs).sum())}")
        # baselines: best single factor by training-period (first fold's training-end) IC, NATR, momentum
        base = {}
        for nm in ("t_NATR", "q_ROC20", "q_ROC60", "lmv", "pb", "boll_pctb"):
            if nm in feats:
                ic_b = daily_ic(P.trade_date, P[[nm]], P.ys)[nm].dropna()
                base[nm] = (ic_b.mean(), tstat(ic_b))
        first = FOLDS[0]
        trn = s[s.end_date < first]
        tic = daily_ic(trn.trade_date, trn[feats], trn.ys).mean()
        best = tic.abs().idxmax()
        ic_b = daily_ic(P.trade_date, P[[best]], P.ys)[best].dropna() * np.sign(tic[best])
        base[f"best-by-train:{best}"] = (ic_b.mean(), tstat(ic_b))
        say("  baseline test IC with the score: " + "; ".join(f"{k} {v[0]:+.3f}(t{v[1]:+.1f})" for k, v in base.items()))
        say(f"  corr(pred, NATR pct) {P.pred.corr(P.t_NATR):+.2f}; corr(pred, log mkt cap pct) {P.pred.corr(P.lmv):+.2f}" if "lmv" in P else "")
    pd.concat(ic_rows).to_csv(OUT / "ic_univariate.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(wf_rows).to_csv(OUT / "walkforward.csv", index=False, encoding="utf-8-sig")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
