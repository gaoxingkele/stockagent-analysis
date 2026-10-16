#!/usr/bin/env python
"""Deep check of the two factors picked from the factor zoo (Bollinger %b(20), CCI(14)) inside the S20 / R20 top-100 pools.

  python scripts/validate_two_factors.py

User request 2026-10-16. Evidence tier reused_holdout: every period here was already used, the factors were chosen
after looking at 282 candidates on these same periods, so nothing here is a validation. The point is to find out how
much of the effect survives harder questions. Signal days >= 2026-08-06 are not read.

Checks per pool x period x factor (selection = within-day percentile in the top 100, list basis U15D10):
  A cuts        tercile / half / quintile top-minus-bottom, daily rank IC, success rate by quintile
  B parameters  %b window 10/20/40, CCI window 10/14/20/30 (frozen choice is %b 20, CCI 14)
  C neutralise  rank residual after NATR + log market cap; inside industry; inside model-rank halves
  D stability   by month, share of positive days, leave-one-month-out, 10-day moving-block bootstrap CI
  E regime      CSI300 trailing 20-day return up vs down at the signal day
  F placebo     factor shuffled within day (300 draws): where the observed tercile difference falls
  G lists       baseline list vs the same list with the factor as an extra filter (R20 top 20 by r20_pred; S20 contract list)
Output: output/experiments/two_factor_validation_20261016/{report.txt, *.csv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_bottom_patterns import pools  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/two_factor_validation_20261016"
SEGS = {"S20": [("S1", "20250303", "20250829"), ("S2", "20260127", "20260805")],
        "R20": [("IS", "20260127", "20260413"), ("OOS", "20260414", "20260805")]}
FACTORS = {"pctb20": "boll_pctb", "cci14": "t_CCI"}
LINES: list[str] = []
RNG = np.random.default_rng(20261016)


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------ factor variants
def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
    c = g.close.to_numpy(float)
    tp = ((g.high + g.low + g.close) / 3).to_numpy(float)
    out = {"ts_code": g.ts_code.to_numpy(), "trade_date": g.trade_date.to_numpy()}
    cs = pd.Series(c)
    for n in (10, 20, 40):
        ma, sd = cs.rolling(n).mean(), cs.rolling(n).std(ddof=0)
        out[f"pctb{n}"] = ((cs - (ma - 2 * sd)) / (4 * sd).replace(0, np.nan)).to_numpy()
    for n in (10, 14, 20, 30):
        v = np.full(len(g), np.nan)
        if len(g) >= n:
            w = sliding_window_view(tp, n)
            m = w.mean(axis=1)
            md = np.abs(w - m[:, None]).mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                v[n - 1:] = (tp[n - 1:] - m) / (0.015 * md)
        out[f"cci{n}"] = v
    return pd.DataFrame(out)


def variants(codes: set[str]) -> pd.DataFrame:
    cache = OUT / "variants.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from price_panel import adjusted_daily
    px = adjusted_daily("20240801", "20260805", codes)
    v = pd.concat([_per_stock(g) for _, g in px.groupby("ts_code", sort=False)], ignore_index=True)
    v = v[v.trade_date >= "20250101"].replace([np.inf, -np.inf], np.nan)
    v.to_parquet(cache, index=False)
    return v


# ------------------------------------------------------------------ helpers
def pct(g: pd.DataFrame, col: str) -> pd.Series:
    return g.groupby("trade_date")[col].rank(pct=True)


def daily_diff(g: pd.DataFrame, top: pd.Series, bot: pd.Series, m: str) -> pd.Series:
    return (g[top].groupby(g.trade_date[top]).mean()[m] if False else
            g.loc[top].groupby("trade_date")[m].mean() - g.loc[bot].groupby("trade_date")[m].mean()).dropna()


def cut_stats(g: pd.DataFrame, score: pd.Series, hi: float, lo: float) -> dict:
    """top (score pct > hi) minus bottom (score pct <= lo), success (pp), per trade, stop-first (pp)."""
    p = score
    r = {}
    for m, lab, k in (("success", "succ", 100), ("ret_base", "ret", 1), ("stopfirst", "stop", 100)):
        d = daily_diff(g, p > hi, p <= lo, m)
        r[f"{lab}_d"], r[f"{lab}_t"] = d.mean() * k, (tstat(d) if len(d) > 20 else np.nan)
    return r


def fmt(r: dict) -> str:
    return (f"success {r['succ_d']:+5.1f} ({r['succ_t']:+4.1f}) | per trade {r['ret_d']:+5.2f} ({r['ret_t']:+4.1f}) "
            f"| stop-first {r['stop_d']:+5.1f} ({r['stop_t']:+4.1f})")


def rank_ic(g: pd.DataFrame, fcol: str, m: str) -> tuple[float, float]:
    x = g[["trade_date", fcol, m]].dropna()
    ic = x.groupby("trade_date").apply(lambda d: d[fcol].rank().corr(d[m].rank()) if d[m].nunique() > 1 else np.nan,
                                        include_groups=False).dropna()
    return ic.mean(), tstat(ic)


def block_boot(d: pd.Series, block: int = 10, reps: int = 2000) -> tuple[float, float]:
    x = d.dropna().to_numpy()
    n = len(x)
    if n < 2 * block:
        return np.nan, np.nan
    nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def residual_rank(g: pd.DataFrame, fcol: str, ctrl: list[str]) -> pd.Series:
    """within-day: rank(factor) minus its OLS fit on rank(controls); returned as within-day percentile."""
    out = pd.Series(np.nan, index=g.index)
    for _, d in g.groupby("trade_date"):
        y = d[fcol].rank(pct=True).to_numpy()
        X = np.column_stack([np.ones(len(d))] + [d[c].rank(pct=True).to_numpy() for c in ctrl])
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if ok.sum() < 30:
            continue
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        res = np.full(len(d), np.nan)
        res[ok] = y[ok] - X[ok] @ beta
        out.loc[d.index] = pd.Series(res, index=d.index).rank(pct=True)
    return out


# ------------------------------------------------------------------ list-level screens
def s20_list(frame: pd.DataFrame, keep: pd.Series | None = None, cfg: PureConfig = PureConfig()) -> pd.DataFrame:
    """The frozen funnel with an optional extra filter applied to the top-100 pool before the NATR cut."""
    f = frame[frame.stage1_probability.notna()].copy()
    f["pool_rank"] = f.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    pool = f[f.pool_rank <= cfg.pool_size].copy()
    if keep is not None:
        pool = pool[keep.reindex(pool.index).fillna(False).to_numpy()]
    pool["np"] = pool.groupby("trade_date").natr14.rank(pct=True)
    out = pool[pool.np.isna() | (pool.np <= 1 - cfg.rule("U15D10").amplitude_cap)].copy()
    out["lr"] = out.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    return out[out.lr <= cfg.top_k]


def list_compare(name: str, base: pd.DataFrame, alt: pd.DataFrame) -> str:
    rows = {}
    for m, k in (("success", 100), ("ret_base", 1), ("stopfirst", 100)):
        b, a = base.groupby("trade_date")[m].mean(), alt.groupby("trade_date")[m].mean()
        d = (a - b).dropna()
        rows[m] = (b.mean() * k, a.mean() * k, d.mean() * k, tstat(d), len(d))
    s, r, st = rows["success"], rows["ret_base"], rows["stopfirst"]
    return (f"{name:30s} | success {s[0]:5.1f} -> {s[1]:5.1f} ({s[2]:+4.1f}, t {s[3]:+4.1f}) | per trade {r[0]:+5.2f} -> {r[1]:+5.2f} "
            f"({r[2]:+5.2f}, t {r[3]:+4.1f}) | stop-first {st[0]:4.1f} -> {st[1]:4.1f} ({st[2]:+4.1f}) | names/day {len(alt) / alt.trade_date.nunique():4.1f}, days {s[4]}")


# ------------------------------------------------------------------ main
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pools()
    for k in P:
        P[k] = P[k][P[k].trade_date.between("20250303", "20260805")]
    codes = set().union(*(set(p.ts_code) for p in P.values()))
    var = variants(codes)
    zoo = pd.read_parquet(ROOT / "output/experiments/factor_zoo_20261015/factors.parquet",
                          columns=["ts_code", "trade_date", "boll_pctb", "t_CCI", "q_KUP", "t_NATR"])
    fac = zoo.merge(var, on=["ts_code", "trade_date"], how="left")
    say(f"recomputed vs zoo: corr(pctb20) {fac.pctb20.corr(fac.boll_pctb):.4f}, corr(cci14) {fac.cci14.corr(fac.t_CCI):.4f}")
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                        columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["success"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    o["stopfirst"] = ((o.dn10_day > 0) & ((o.up15_day == 0) | (o.dn10_day <= o.up15_day))).astype(float)
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    ind = fr[["ts_code", "trade_date", "industry"]].drop_duplicates(["ts_code", "trade_date"])
    mv = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "circ_mv"])
                    for p in sorted((ROOT / "output/tushare_cache/daily_basic").glob("*.parquet"))
                    if "20250101" <= p.stem <= "20260805"], ignore_index=True)
    mv["trade_date"] = mv.trade_date.astype(str)
    mv["lmv"] = np.log(mv.circ_mv.where(mv.circ_mv > 0))
    ix = pd.read_parquet(ROOT / "output/tushare_cache/index_daily/000300.SH.parquet", columns=["trade_date", "close"])
    ix["trade_date"] = ix.trade_date.astype(str)
    ix = ix.sort_values("trade_date")
    ix["mkt20"] = ix.close / ix.close.shift(20) - 1
    allv = ["pctb10", "pctb20", "pctb40", "cci10", "cci14", "cci20", "cci30"]
    csv_rows = []
    verdict: dict[tuple[str, str, str], dict] = {}

    for pname, segs in SEGS.items():
        base = (P[pname].merge(o, on=["ts_code", "trade_date"]).merge(fac, on=["ts_code", "trade_date"], how="left")
                .merge(ind, on=["ts_code", "trade_date"], how="left").merge(mv[["ts_code", "trade_date", "lmv"]], on=["ts_code", "trade_date"], how="left")
                .merge(ix[["trade_date", "mkt20"]], on="trade_date", how="left"))
        base["ind"] = base.industry.fillna("unknown")
        for sname, a, b in segs:
            g = base[base.trade_date.between(a, b)].reset_index(drop=True)
            say(f"\n{'=' * 100}\n{pname} top 100, {sname}: {len(g):,} cases, {g.trade_date.nunique()} days; success {100 * g.success.mean():.1f}%, "
                f"per trade {g.ret_base.mean():+.2f}, stop-first {100 * g.stopfirst.mean():.1f}%")
            for fname, fcol in FACTORS.items():
                say(f"\n--- {fname} ---")
                p = pct(g, fcol)
                # A. cuts
                say("A. cuts (top minus bottom, within-day pool percentile)")
                for lab, hi, lo in (("tercile 2/3 vs 1/3", 2 / 3, 1 / 3), ("half 1/2 vs 1/2", 0.5, 0.5), ("quintile 4/5 vs 1/5", 0.8, 0.2)):
                    r = cut_stats(g, p, hi, lo)
                    say(f"  {lab:20s} {fmt(r)}")
                    csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "A_" + lab, **r})
                    if lab.startswith("tercile"):
                        verdict[(pname, sname, fname)] = dict(r)
                ic_s, ic_st = rank_ic(g, fcol, "success")
                ic_r, ic_rt = rank_ic(g, fcol, "ret_base")
                say(f"  rank IC with success {ic_s:+.3f} (t {ic_st:+.1f}); with per-trade return {ic_r:+.3f} (t {ic_rt:+.1f})")
                q = np.ceil(p * 5).clip(1, 5)
                qs = g.groupby(["trade_date", q.rename("q")]).success.mean().groupby("q").mean() * 100
                qr = g.groupby(["trade_date", q.rename("q")]).ret_base.mean().groupby("q").mean()
                say("  by quintile (low->high) success: " + " ".join(f"{x:5.1f}" for x in qs) + " | per trade: " + " ".join(f"{x:+5.2f}" for x in qr))
                # B. parameters
                say("B. other windows (tercile success diff, t)")
                line = []
                for v in allv:
                    if v[:3] != fcol[:3] and not (v.startswith("pctb") and fname == "pctb20") and not (v.startswith("cci") and fname == "cci14"):
                        continue
                    r = cut_stats(g, pct(g, v), 2 / 3, 1 / 3)
                    line.append(f"{v} {r['succ_d']:+.1f}({r['succ_t']:+.1f}; trade {r['ret_d']:+.2f})")
                    csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "B_" + v, **r})
                say("  " + "  ".join(line))
                # C. neutralise
                say("C. neutralisation (tercile)")
                pn = residual_rank(g, fcol, ["t_NATR", "lmv"])
                r = cut_stats(g, pn, 2 / 3, 1 / 3)
                say(f"  after NATR + size       {fmt(r)}")
                csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "C_natr_size", **r})
                gi = g.assign(_x=g[fcol])
                pin = gi.groupby(["trade_date", "ind"])["_x"].rank(pct=True)
                cnt = gi.groupby(["trade_date", "ind"])["_x"].transform("count")
                pin = pin.where(cnt >= 3)
                r = cut_stats(g.loc[pin.notna()], pin[pin.notna()], 2 / 3, 1 / 3)
                say(f"  inside industry (n>=3)  {fmt(r)}   [{int(pin.notna().sum()):,} cases]")
                csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "C_industry", **r})
                for half, mask in (("model rank 1-50", g["rank"] <= 50), ("model rank 51-100", g["rank"] > 50)):
                    gg = g[mask]
                    r = cut_stats(gg, pct(gg, fcol), 2 / 3, 1 / 3)
                    say(f"  {half:22s}  {fmt(r)}")
                    csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "C_" + half, **r})
                # D. stability
                say("D. stability (tercile success diff)")
                d = daily_diff(g, p > 2 / 3, p <= 1 / 3, "success") * 100
                dr = daily_diff(g, p > 2 / 3, p <= 1 / 3, "ret_base")
                mon = d.groupby(d.index.str[:6]).agg(["mean", "count"])
                monr = dr.groupby(dr.index.str[:6]).mean()
                say("  by month: " + "  ".join(f"{m} {r['mean']:+5.1f}pp/{monr[m]:+5.2f} (n{int(r['count'])})" for m, r in mon.iterrows()))
                loo = [tstat(d[d.index.str[:6] != m]) for m in mon.index]
                lo_, hi_ = block_boot(d)
                say(f"  positive days {100 * (d > 0).mean():.0f}% of {len(d)}; leave-one-month-out t min {np.nanmin(loo):+.1f} max {np.nanmax(loo):+.1f}; "
                    f"10-day-block bootstrap 95% CI of mean diff [{lo_:+.1f}, {hi_:+.1f}] pp")
                csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "D_stability", "pos_days": (d > 0).mean(),
                                 "loo_t_min": np.nanmin(loo), "ci_lo": lo_, "ci_hi": hi_})
                # E. regime
                say("E. market regime (CSI300 trailing 20d return at signal day)")
                for lab, mask in (("market up  ", g.mkt20 > 0), ("market down", g.mkt20 <= 0)):
                    gg = g[mask]
                    if gg.trade_date.nunique() < 15:
                        say(f"  {lab} too few days ({gg.trade_date.nunique()})")
                        continue
                    r = cut_stats(gg, pct(gg, fcol), 2 / 3, 1 / 3)
                    say(f"  {lab} ({gg.trade_date.nunique():3d} days) {fmt(r)}")
                    csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "E_" + lab.strip(), **r})
                # F. placebo
                obs = verdict[(pname, sname, fname)]["succ_d"]
                null = []
                days = {k: v.index.to_numpy() for k, v in g.groupby("trade_date")}
                for _ in range(300):
                    sh = np.empty(len(g))
                    for idx in days.values():
                        sh[idx] = RNG.permutation(len(idx))
                    ps = pd.Series(sh, index=g.index).groupby(g.trade_date).rank(pct=True)
                    dd = daily_diff(g, ps > 2 / 3, ps <= 1 / 3, "success")
                    null.append(dd.mean() * 100)
                null = np.array(null)
                say(f"F. placebo: observed {obs:+.1f}pp; shuffled within day mean {null.mean():+.2f}, sd {null.std():.2f}, max |draw| {np.abs(null).max():.2f}; "
                    f"draws at least as large as observed: {int((np.abs(null) >= abs(obs)).sum())}/300")
                csv_rows.append({"pool": pname, "seg": sname, "factor": fname, "check": "F_placebo", "obs": obs, "null_sd": null.std()})

            # G. lists
            say(f"\n--- G. lists, {pname} {sname}: baseline vs extra factor filter (same days, paired; list basis) ---")
            if pname == "R20":
                gl = g.copy()
                base_l = gl[gl["rank"] <= 20]
                for fname, fcol in FACTORS.items():
                    pf = pct(gl, fcol)
                    gl[f"p_{fname}"] = pf
                    say(list_compare(f"{fname}: top tercile -> top 20", base_l, gl[pf > 2 / 3].sort_values("rank").groupby("trade_date").head(20)))
                    say(list_compare(f"{fname}: drop bottom tercile", base_l, gl[pf > 1 / 3].sort_values("rank").groupby("trade_date").head(20)))
                    sc = gl["rank"].rank(pct=True) * 0 + (100 - gl["rank"]) / 100 + pf
                    gl["blend"] = sc
                    say(list_compare(f"{fname}: blend rank + factor", base_l, gl.sort_values("blend", ascending=False).groupby("trade_date").head(20)))
                ctrl = gl[gl["rank"] <= 20]
                rnd = gl.groupby("trade_date", group_keys=False).sample(frac=0.33, random_state=7).sort_values("rank").groupby("trade_date").head(20)
                say(list_compare("control: random third -> top 20", base_l, rnd))
                say(f"  (baseline list = r20_pred top 20; the top-100 mean is {100 * g.success.mean():.1f}% success, {g.ret_base.mean():+.2f} per trade)")
            else:
                f2 = fr[fr.trade_date.between(a, b)].copy()
                f2 = f2.merge(fac[["ts_code", "trade_date", "boll_pctb", "t_CCI"]], on=["ts_code", "trade_date"], how="left").reset_index(drop=True)
                chk = select(f2.drop(columns=["boll_pctb", "t_CCI"]), PureConfig(), "U15D10")
                base_l = s20_list(f2)
                same = set(map(tuple, chk[["ts_code", "trade_date"]].to_numpy())) == set(map(tuple, base_l[["ts_code", "trade_date"]].to_numpy()))
                say(f"  baseline funnel reproduces select(): {same} ({len(base_l)} names)")
                f2["pool_rank"] = f2.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
                inpool = f2.pool_rank <= 100
                bl = base_l.merge(o, on=["ts_code", "trade_date"])
                for fname, fcol in FACTORS.items():
                    pf = f2[fcol].where(inpool).groupby(f2.trade_date).rank(pct=True)
                    for lab, thr in (("top tercile", 2 / 3), ("drop bottom tercile", 1 / 3)):
                        keep = pf > thr
                        al = s20_list(f2, keep).merge(o, on=["ts_code", "trade_date"])
                        say(list_compare(f"{fname}: {lab} before NATR cut", bl, al))
                rnd = f2[inpool].groupby("trade_date", group_keys=False).sample(frac=0.33, random_state=7)
                al = s20_list(f2, pd.Series(f2.index.isin(rnd.index), index=f2.index)).merge(o, on=["ts_code", "trade_date"])
                say(list_compare("control: random third in pool", bl, al))
                say(f"  (baseline = contract list; top-100 pool mean {100 * g.success.mean():.1f}% success, {g.ret_base.mean():+.2f} per trade)")

    pd.DataFrame(csv_rows).to_csv(OUT / "checks.csv", index=False, encoding="utf-8-sig")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
