#!/usr/bin/env python
"""Launch-point reward score: label every historical S20 / R20 candidate with a path reward, then ask which features predict it.

  python scripts/analyze_launch_score.py

User idea 2026-10-26 (I0008): score each candidate buy point by how the price path develops after it - +points for each further 5% of
rise, big deductions for give-backs, negative when it breaks down - like a reinforcement-learning reward; then use features to predict
the score.   PRE-REGISTERED before any result was looked at.

Evidence tier: reused_holdout / descriptive. S1+S2 (S20) and R20 IS/OOS are periods already used; nothing here is validation.
Reserved window (signal days >= 2026-08-06) not read: only cases whose 20-session horizon ends on or before 2026-08-05 are used.

Candidates: S20 stage1 top 100 and R20 top 100 per signal day (the same pools as the earlier pool studies), signal days 2025-03-03..2026-08-05.
Path: entry = next-session open E; sessions 1..20 on split-adjusted daily bars; hi_d = high/E-1, lo_d = low/E-1.
Reward (all constants fixed here):
  step = 5%.  peak_prev(d) = max(0, hi_1..hi_{d-1}) (the day's own high-low swing is not counted as a give-back).
  stopped (any lo_d <= -10%, same day as a new high still counts)  ->  score = -4   (-2 points per 5% of loss)
  otherwise  up = min(floor(max(hi)/5%), 8)  (+1 per full 5% of rise, cap 8 = +40%)
             giveback = max_d (peak_prev(d) - lo_d)/5%  (>= 0; a dip below the entry price counts from peak_prev = 0)
             score = clip(up - LAMBDA * giveback, -4, 8),  LAMBDA = 2   (sensitivity 1 and 3 reported, not used further)
  Example: +5% (1 point) then -4% from the peak => 1 - 2*0.8 = -0.6.
Part 1  descriptive: distribution, relation to the first-passage success (+15% before -10%) and to ret20, per pool and period.
Part 2  univariate: 285 features (Alpha158-style + TA-Lib + Bollinger %b from the factor zoo, as within-day percentile inside the pool);
        per-day Spearman IC with the score; Newey-West(19) t; Bonferroni bar over features x periods; the pool's own ranking score
        (stage1 / r20_pred) is the baseline.   N0026 / N0027 lesson: label redesigns learned "which day is good", so everything is
        within-day: the target is demeaned by day and only within-day ranks are scored.
Part 3  model: LightGBM regression, FIXED parameters (num_leaves 15, lr 0.03, 300 trees, min_data_in_leaf 200, feature_fraction 0.5,
        bagging 0.8), target = score minus its day mean, features = within-day percentile. Train = first period of a pool, rows whose horizon ends
        before the test period starts (purge); test = second period. Reported on the test period: daily IC (NW t), top third minus bottom third
        of the prediction (score, success, stop-first), shuffle placebo (300 within-day shuffles), baseline IC of the pool's own ranking, S20 only:
        list level (prediction top third as an extra filter before the NATR cut, vs the contract list and vs random third).
Periods: S20 train 2025-03-03..2025-08-29 / test 2026-01-27..2026-08-05; R20 train 2026-01-27..2026-04-13 / test 2026-04-14..2026-08-05.
Falsified (I0008) if the test-period IC t < 1.96 or the sign differs from the train IC, or the top-third success is not above the bottom third,
or the list-level gain is not above the random-third control.
Output: output/experiments/launch_score_20261026/{report.txt, scores.parquet, ic_univariate.csv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_bottom_patterns import pools  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402
from price_panel import adjusted_daily  # noqa: E402
from validate_two_factors import list_compare, s20_list  # noqa: E402

OUT = ROOT / "output/experiments/launch_score_20261026"
STEP, CAP, FLOOR, STOP, H = 0.05, 8, -4.0, -0.10, 20
SEGS = {"S20": (("train", "20250303", "20250829"), ("test", "20260127", "20260805")),
        "R20": (("train", "20260127", "20260413"), ("test", "20260414", "20260805"))}
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def reward(hi: np.ndarray, lo: np.ndarray, lam: float) -> np.ndarray:
    """hi, lo: (n, 20) relative to the entry open."""
    peak_prev = np.concatenate([np.zeros((len(hi), 1)), np.maximum.accumulate(np.maximum(hi, 0), axis=1)[:, :-1]], axis=1)
    giveback = np.maximum((peak_prev - lo) / STEP, 0).max(axis=1)
    up = np.minimum(np.floor(np.maximum(hi.max(axis=1), 0) / STEP), CAP)
    score = np.clip(up - lam * giveback, FLOOR, CAP)
    return np.where((lo <= STOP).any(axis=1), FLOOR, score)


def build_scores(cases: pd.DataFrame) -> pd.DataFrame:
    px = adjusted_daily("20250201", "20260805", set(cases.ts_code))
    out = []
    for code, g in px.groupby("ts_code"):
        c = cases[cases.ts_code == code]
        dates = g.trade_date.to_numpy()
        pos = pd.Series(np.arange(len(dates)), index=dates)
        i = pos.reindex(c.trade_date).to_numpy()
        ok = ~np.isnan(i)
        i = i[ok].astype(int) + 1                    # entry session = signal day + 1
        c = c[ok]
        keep = i + H - 1 < len(dates)
        i, c = i[keep], c[keep]
        if not len(i):
            continue
        idx = i[:, None] + np.arange(H)[None, :]
        E = g.open.to_numpy()[i]
        hi, lo = g.high.to_numpy()[idx] / E[:, None] - 1, g.low.to_numpy()[idx] / E[:, None] - 1
        r = c[["ts_code", "trade_date"]].copy()
        r["end_date"] = dates[idx[:, -1]]
        r["peak"], r["trough"] = hi.max(axis=1), lo.min(axis=1)
        r["ret20"] = g.close.to_numpy()[idx[:, -1]] / E - 1
        for lam in (1, 2, 3):
            r[f"score{lam}"] = reward(hi, lo, lam)
        out.append(r)
    return pd.concat(out, ignore_index=True)


def ic_by_day(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    def one(g: pd.DataFrame) -> float:
        return g[x].rank().corr(g[y].rank()) if g[x].nunique() > 3 and g[y].nunique() > 3 else np.nan
    return df.groupby("trade_date").apply(one, include_groups=False).dropna()


def main() -> int:
    import lightgbm as lgb
    from scipy.stats import norm
    OUT.mkdir(parents=True, exist_ok=True)
    P = pools()
    for k in P:
        P[k] = P[k][P[k].trade_date.between("20250303", "20260805")]
    allc = pd.concat(P.values())[["ts_code", "trade_date"]].drop_duplicates()
    cache = OUT / "scores.parquet"
    sc = pd.read_parquet(cache) if cache.exists() else build_scores(allc)
    if not cache.exists():
        sc.to_parquet(cache, index=False)
    sc = sc[sc.end_date <= "20260805"]
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet", columns=["ts_code", "trade_date", "natr14", "ret_base", "up15_day", "dn10_day"])
    o["success"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    o["stopfirst"] = ((o.dn10_day > 0) & ((o.up15_day == 0) | (o.dn10_day <= o.up15_day))).astype(float)
    zoo = pd.read_parquet(ROOT / "output/experiments/factor_zoo_20261015/factors.parquet")
    feats = [c for c in zoo.columns if c not in ("ts_code", "trade_date")]
    rng = np.random.default_rng(20261026)
    rows = []
    for pname, segs in SEGS.items():
        base = P[pname].merge(sc, on=["ts_code", "trade_date"]).merge(o, on=["ts_code", "trade_date"], how="left")
        base["score_d"] = base.score2 - base.groupby("trade_date").score2.transform("mean")
        say(f"\n{'=' * 96}\n{pname} top 100: {len(base):,} matured cases (horizon ends <= 2026-08-05), {base.trade_date.nunique()} days")
        # ---------- part 1
        say("Part 1: score distribution (LAMBDA=2), and against the first-passage outcome")
        for sname, a, b in segs:
            g = base[base.trade_date.between(a, b)]
            if g.empty:
                continue
            q = g.score2.quantile([.1, .25, .5, .75, .9]).round(1).tolist()
            say(f"  {sname:5s} {a}..{b}: n {len(g):,}, mean {g.score2.mean():+.2f}, quantiles 10/25/50/75/90% {q}, stopped (-4) {100 * (g.score2 <= -4).mean():.1f}%, "
                f"score>=4 (peak>=+20% without big give-back) {100 * (g.score2 >= 4).mean():.1f}%, >=8 {100 * (g.score2 >= 8).mean():.1f}%")
            say(f"        success(+15% before -10%) {100 * g.success.mean():.1f}%; mean score success {g[g.success == 1].score2.mean():+.2f} vs fail {g[g.success == 0].score2.mean():+.2f}; "
                f"corr(score, ret20) {g.score2.corr(g.ret20):+.2f}, corr(score1, score3) {g.score1.corr(g.score3):+.2f}; mean score lam1/2/3 {g.score1.mean():+.2f}/{g.score2.mean():+.2f}/{g.score3.mean():+.2f}")
        # ---------- part 2
        F = base.merge(zoo, on=["ts_code", "trade_date"], how="left")
        keepf = [c for c in feats if F[c].notna().mean() > 0.7 and F[c].nunique() > 5]
        pr = F.groupby("trade_date")[keepf].rank(pct=True)
        pr.columns = [f"p_{c}" for c in keepf]
        F = pd.concat([F, pr], axis=1)
        bar = norm.isf(0.025 / (len(keepf) * 2))
        say(f"\nPart 2: univariate within-day Spearman IC with the score, {len(keepf)} features; Bonferroni |t| >= {bar:.2f} (features x 2 periods)")
        res = []
        for sname, a, b in segs:
            g = F[F.trade_date.between(a, b)]
            for c in keepf:
                d = ic_by_day(g.assign(x=g[f"p_{c}"]), "x", "score2")
                res.append({"pool": pname, "seg": sname, "feature": c, "ic": d.mean(), "t": tstat(d) if len(d) > 20 else np.nan})
        own = {}
        for sname, a, b in segs:
            g = base[base.trade_date.between(a, b)]
            d = ic_by_day(g.assign(x=-g["rank"]), "x", "score2")
            own[sname] = (d.mean(), tstat(d))
        say(f"  baseline - the pool's own ranking vs score: train IC {own['train'][0]:+.3f} (t {own['train'][1]:+.1f}), test IC {own['test'][0]:+.3f} (t {own['test'][1]:+.1f})")
        R = pd.DataFrame(res)
        R.to_csv(OUT / f"ic_univariate_{pname}.csv", index=False, encoding="utf-8-sig")
        W = R.pivot(index="feature", columns="seg", values=["ic", "t"])
        W.columns = [f"{a}_{b}" for a, b in W.columns]
        cons = W[(np.sign(W.ic_train) == np.sign(W.ic_test)) & (W.t_train.abs() >= 2) & (W.t_test.abs() >= 2)]
        cons = cons.assign(m=cons.t_test.abs()).sort_values("m", ascending=False)
        say(f"  features |t|>=2 with the same sign in both periods: {len(cons)} of {len(W)} (chance, both |t|>=2 and same sign: about {len(W) * 0.05 * 0.05 / 2 * 2:.1f}); "
            f"passing the Bonferroni bar in the test period: {int((W.t_test.abs() >= bar).sum())}; in both: {int(((W.t_test.abs() >= bar) & (W.t_train.abs() >= bar)).sum())}")
        say(cons.drop(columns="m").head(15).round(3).to_string())
        # ---------- part 3
        tr_s, tr_a, tr_b = segs[0]
        te_s, te_a, te_b = segs[1]
        tr = F[F.trade_date.between(tr_a, tr_b) & (F.end_date < te_a)]
        te = F[F.trade_date.between(te_a, te_b)].copy()
        cols = [f"p_{c}" for c in keepf]
        params = dict(objective="regression", num_leaves=15, learning_rate=0.03, min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
        m = lgb.train(params, lgb.Dataset(tr[cols], tr.score_d), num_boost_round=300)
        te["pred"] = m.predict(te[cols])
        tr = tr.assign(pred=m.predict(tr[cols]))
        say(f"\nPart 3: LightGBM (fixed parameters) trained on {len(tr):,} rows ({tr_a}..{tr_b}, horizon ends < {te_a}), tested on {len(te):,} rows ({te_a}..{te_b})")
        ic_tr, ic_te = ic_by_day(tr, "pred", "score2"), ic_by_day(te, "pred", "score2")
        say(f"  daily IC: train (in-sample) {ic_tr.mean():+.3f}; test {ic_te.mean():+.3f} (NW t {tstat(ic_te):+.1f}, {len(ic_te)} days, positive days {100 * (ic_te > 0).mean():.0f}%)")
        te["terc"] = te.groupby("trade_date").pred.rank(pct=True, method="first")
        hi_, lo_ = te[te.terc > 2 / 3], te[te.terc <= 1 / 3]
        for lab, k, mult in (("score", "score2", 1), ("success pp", "success", 100), ("stop-first pp", "stopfirst", 100), ("per trade", "ret_base", 1), ("ret20 pp", "ret20", 100)):
            d = (hi_.groupby("trade_date")[k].mean() - lo_.groupby("trade_date")[k].mean()).dropna()
            say(f"  top third minus bottom third, {lab:13s}: {d.mean() * mult:+7.2f} (t {tstat(d):+.1f}); top {hi_[k].mean() * mult:+.2f}, bottom {lo_[k].mean() * mult:+.2f}")
        obs = (hi_.groupby("trade_date").score2.mean() - lo_.groupby("trade_date").score2.mean()).mean()
        sh = []
        for _ in range(300):
            tt = te.assign(pred=te.groupby("trade_date").pred.transform(lambda s: rng.permutation(s.to_numpy())))
            tt["terc"] = tt.groupby("trade_date").pred.rank(pct=True, method="first")
            sh.append((tt[tt.terc > 2 / 3].groupby("trade_date").score2.mean() - tt[tt.terc <= 1 / 3].groupby("trade_date").score2.mean()).mean())
        say(f"  shuffle placebo (300): observed score gap {obs:+.3f}; shuffled mean {np.mean(sh):+.3f}, 95th pct {np.percentile(sh, 95):+.3f}; shuffles >= observed: {int((np.array(sh) >= obs).sum())}")
        imp = pd.Series(m.feature_importance("gain"), index=keepf).sort_values(ascending=False)
        say("  top 12 features by gain: " + ", ".join(f"{k} {100 * v / imp.sum():.0f}%" for k, v in imp.head(12).items()))
        te_rank_ic = ic_by_day(te.assign(x=-te["rank"]), "x", "score2")
        say(f"  baseline in the same test rows: pool's own ranking IC {te_rank_ic.mean():+.3f} (t {tstat(te_rank_ic):+.1f}); corr(pred, own ranking) {te['pred'].corr(-te['rank']):+.2f}, "
            f"corr(pred, natr14) {te.pred.corr(te.natr14):+.2f}")
        if pname == "S20":
            fr = te.rename(columns={"own": "stage1_probability"}).copy()
            fr["natr14"] = fr.natr14
            base_l = s20_list(fr)
            alt = s20_list(fr, keep=fr.terc > 2 / 3)
            rnd_keep = fr.groupby("trade_date", group_keys=False).apply(lambda g: pd.Series(rng.random(len(g)) < 1 / 3, index=g.index), include_groups=False)
            rnd = s20_list(fr, keep=rnd_keep.reindex(fr.index))
            say("  list level (contract funnel, extra filter before the NATR cut):")
            say("   " + list_compare("prediction top third", base_l, alt))
            say("   " + list_compare("control: random third", base_l, rnd))
            tail = fr.assign(terc=fr.terc)
        rows.append((pname, ic_te.mean(), tstat(ic_te)))
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
