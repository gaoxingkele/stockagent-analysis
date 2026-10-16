#!/usr/bin/env python
"""TA-Lib and Qlib-Alpha158 style factors inside the S20 / R20 top-100 pools: which separate winners from losers?

  python scripts/analyze_factor_zoo_in_pools.py

User request 2026-10-15. Same protocol as scripts/analyze_list_indicators.py (descriptive on used periods,
evidence tier reused_holdout, nothing here is a validated rule):
  - within-day percentile rank inside the pool; top tercile minus bottom tercile; success = +15% before -10%
    within 20 sessions from the next open (list basis, U15D10), plus per-trade return; same-day paired
    Newey-West(19) t;
  - pick on the first period of a pool (S1 / R20 in-sample, |t| >= 2 on success), confirm on the second
    (same sign, |t| >= 2); the number expected by chance is printed next to the number found;
  - the Bonferroni bar counts every factor x 4 periods x 2 metrics;
  - factors that confirm are also correlated with Bollinger %b (the shadow candidate I0005) so one
    underlying "recent strength" factor is not counted as many.
Factor sources (Qlib is not installed; Alpha158 is re-implemented from its published expression list):
  qlib  KBAR (9), PRICE (3), and for windows 5/10/20/30/60: ROC MA STD BETA RSQR RESI MAX MIN QTLU QTLD
        RANK RSV IMAX IMIN IMXD CORR CORD CNTP CNTN CNTD SUMP SUMN SUMD VMA VSTD WVMA VSUMP VSUMN VSUMD
  talib every non-pattern function with default parameters (overlap studies as close-relative, price-scaled
        momentum as /close, volume AD/OBV/ADOSC as 20-day change over 20-day volume)
Signal days >= 2026-08-06 are not read. Output: output/experiments/factor_zoo_20261015/
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from numpy.lib.stride_tricks import sliding_window_view as swv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

OUT = ROOT / "output/experiments/factor_zoo_20261015"
SEGS = {"S20": [("S1", "20250303", "20250829"), ("S2", "20260127", "20260805")],
        "R20": [("IS", "20260127", "20260413"), ("OOS", "20260414", "20260805")]}
WINS = (5, 10, 20, 30, 60)
DIFF_SCALE = {"AO", "AC", "DPO", "ERI", "QSTICK", "ADR", "APO", "MACD", "MACDEXT", "MACDFIX", "MINUS_DM", "PLUS_DM", "MOM", "ATR", "TRANGE", "STDDEV", "LINEARREG_SLOPE", "AVGDEV"}
PRICE_LEVEL = {"LINEARREG", "LINEARREG_INTERCEPT", "TSF", "VWAP"}
SKIP = {"FRACTAL", "MARKETFI", "MAVP", "BETA", "CORREL", "VAR", "AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"}
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def win(x: np.ndarray, d: int) -> np.ndarray:
    out = np.full((len(x), d), np.nan)
    if len(x) >= d:
        out[d - 1:] = swv(x, d)
    return out


def rowcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    am, bm = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    return np.where(den > 0, (am * bm).sum(axis=1) / np.where(den > 0, den, 1), np.nan)


def qlib_factors(o, h, l, c, v) -> dict[str, np.ndarray]:
    e = 1e-12
    f: dict[str, np.ndarray] = {}
    f["q_KMID"] = (c - o) / o
    f["q_KLEN"] = (h - l) / o
    f["q_KMID2"] = (c - o) / (h - l + e)
    f["q_KUP"] = (h - np.maximum(o, c)) / o
    f["q_KUP2"] = (h - np.maximum(o, c)) / (h - l + e)
    f["q_KLOW"] = (np.minimum(o, c) - l) / o
    f["q_KLOW2"] = (np.minimum(o, c) - l) / (h - l + e)
    f["q_KSFT"] = (2 * c - h - l) / o
    f["q_KSFT2"] = (2 * c - h - l) / (h - l + e)
    f["q_OPEN0"], f["q_HIGH0"], f["q_LOW0"] = o / c, h / c, l / c
    cs = pd.Series(c)
    ret = (cs / cs.shift()).to_numpy()
    lv = np.log(v + 1)
    vr = np.log(pd.Series(v).div(pd.Series(v).shift()).to_numpy() + 1)
    absret_v = np.abs(ret - 1) * v
    dc = np.r_[np.nan, np.diff(c)]
    dv = np.r_[np.nan, np.diff(v)]
    for d in WINS:
        wc, wh, wl, wv_ = win(c, d), win(h, d), win(l, d), win(v, d)
        t = np.arange(d) - (d - 1) / 2
        ym = wc.mean(axis=1)
        sxy = ((wc - ym[:, None]) * t).sum(axis=1)
        sxx = (t ** 2).sum()
        slope = sxy / sxx
        syy = ((wc - ym[:, None]) ** 2).sum(axis=1)
        f[f"q_ROC{d}"] = pd.Series(c).shift(d).to_numpy() / c
        f[f"q_MA{d}"] = ym / c
        f[f"q_STD{d}"] = wc.std(axis=1, ddof=1) / c if d > 1 else np.zeros_like(c)
        f[f"q_BETA{d}"] = slope / c
        f[f"q_RSQR{d}"] = np.where(syy > 0, sxy ** 2 / (sxx * np.where(syy > 0, syy, 1)), np.nan)
        f[f"q_RESI{d}"] = (wc[:, -1] - (ym + slope * t[-1])) / c
        f[f"q_MAX{d}"] = wh.max(axis=1) / c
        f[f"q_MIN{d}"] = wl.min(axis=1) / c
        f[f"q_QTLU{d}"] = np.quantile(wc, 0.8, axis=1) / c
        f[f"q_QTLD{d}"] = np.quantile(wc, 0.2, axis=1) / c
        f[f"q_RANK{d}"] = (wc <= wc[:, -1:]).mean(axis=1)
        f[f"q_RSV{d}"] = (c - wl.min(axis=1)) / (wh.max(axis=1) - wl.min(axis=1) + e)
        imax, imin = np.argmax(wh, axis=1), np.argmin(wl, axis=1)
        f[f"q_IMAX{d}"] = imax / d
        f[f"q_IMIN{d}"] = imin / d
        f[f"q_IMXD{d}"] = (imax - imin) / d
        f[f"q_CORR{d}"] = rowcorr(wc, win(lv, d))
        f[f"q_CORD{d}"] = rowcorr(win(ret, d), win(vr, d))
        up, dn = win((dc > 0).astype(float), d), win((dc < 0).astype(float), d)
        f[f"q_CNTP{d}"], f[f"q_CNTN{d}"] = up.mean(axis=1), dn.mean(axis=1)
        f[f"q_CNTD{d}"] = f[f"q_CNTP{d}"] - f[f"q_CNTN{d}"]
        sp, sn, sa = win(np.clip(dc, 0, None), d).sum(axis=1), win(np.clip(-dc, 0, None), d).sum(axis=1), win(np.abs(dc), d).sum(axis=1)
        f[f"q_SUMP{d}"], f[f"q_SUMN{d}"] = sp / (sa + e), sn / (sa + e)
        f[f"q_SUMD{d}"] = f[f"q_SUMP{d}"] - f[f"q_SUMN{d}"]
        f[f"q_VMA{d}"] = wv_.mean(axis=1) / (v + e)
        f[f"q_VSTD{d}"] = wv_.std(axis=1, ddof=1) / (v + e)
        wa = win(absret_v, d)
        f[f"q_WVMA{d}"] = wa.std(axis=1, ddof=1) / (wa.mean(axis=1) + e)
        vp, vn, va = win(np.clip(dv, 0, None), d).sum(axis=1), win(np.clip(-dv, 0, None), d).sum(axis=1), win(np.abs(dv), d).sum(axis=1)
        f[f"q_VSUMP{d}"], f[f"q_VSUMN{d}"] = vp / (va + e), vn / (va + e)
        f[f"q_VSUMD{d}"] = f[f"q_VSUMP{d}"] - f[f"q_VSUMN{d}"]
    return f


def talib_factors(o, h, l, c, v) -> dict[str, np.ndarray]:
    import talib
    from talib import abstract
    inputs = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    f: dict[str, np.ndarray] = {}
    groups = talib.get_function_groups()
    for grp, names in groups.items():
        if grp in ("Pattern Recognition", "Math Operators", "Math Transform", "Price Transform"):
            continue
        for name in names:
            if name in SKIP:
                continue
            try:
                fn = abstract.Function(name)
                res = fn(inputs)
            except Exception:  # noqa: BLE001
                continue
            res = res if isinstance(res, list) else [res]
            for i, r in enumerate(res):
                r = np.asarray(r, float)
                key = f"t_{name}" + (f"_{i}" if len(res) > 1 else "")
                if name in ("NVI", "PVI"):
                    r = r / pd.Series(r).shift(20).to_numpy() - 1
                elif name in ("WAD", "PVT"):
                    r = (r - pd.Series(r).shift(20).to_numpy()) / (pd.Series(np.abs(r)).rolling(20).mean().to_numpy() + 1e-12)
                elif name == "EFI":
                    r = r / (pd.Series(c * v).rolling(20).mean().to_numpy() + 1e-12)
                elif name in ("AD", "OBV", "ADOSC"):
                    vs = pd.Series(v).rolling(20).sum().to_numpy()
                    r = (r - pd.Series(r).shift(20).to_numpy()) / vs if name != "ADOSC" else r / pd.Series(v).rolling(20).mean().to_numpy()
                elif grp == "Overlap Studies" or name in PRICE_LEVEL:
                    r = r / c - 1
                elif name in DIFF_SCALE:
                    r = r / c
                f[key] = r
    return f


def one_stock(g: pd.DataFrame, keep_dates: set[str]) -> pd.DataFrame:
    o, h, l, c, v = (g[k].to_numpy(float) for k in ("open", "high", "low", "close", "vol"))
    f = {**qlib_factors(o, h, l, c, v), **talib_factors(o, h, l, c, v)}
    m = c.copy()
    sd = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    f["boll_pctb"] = (c - (pd.Series(c).rolling(20).mean().to_numpy() - 2 * sd)) / (4 * sd)
    df = pd.DataFrame(f).astype("float32")
    df["ts_code"], df["trade_date"] = g.ts_code.to_numpy(), g.trade_date.to_numpy()
    return df[df.trade_date.isin(keep_dates)]


def _chunk(args) -> pd.DataFrame:
    frame, keys = args
    parts = []
    for code, g in frame.groupby("ts_code", sort=False):
        if code in keys:
            parts.append(one_stock(g.reset_index(drop=True), keys[code]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build(P: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from price_panel import adjusted_daily
    keys: dict[str, set[str]] = {}
    for p in P.values():
        for code, d in zip(p.ts_code, p.trade_date):
            keys.setdefault(code, set()).add(d)
    px = adjusted_daily("20240801", "20260805", set(keys))
    codes = np.array(sorted(keys))
    chunks = [(px[px.ts_code.isin(set(codes[i::8]))], {k: keys[k] for k in codes[i::8]}) for i in range(8)]
    with ProcessPoolExecutor(8) as ex:
        out = pd.concat([x for x in ex.map(_chunk, chunks) if len(x)], ignore_index=True)
    return out


def main() -> int:
    from analyze_bottom_patterns import pools
    from analyze_r20_s20_nested import tstat
    from scipy.stats import norm
    OUT.mkdir(parents=True, exist_ok=True)
    P = pools()
    for k in P:
        P[k] = P[k][P[k].trade_date.between("20250303", "20260805")]
    cache = OUT / "factors.parquet"
    zoo = pd.read_parquet(cache) if cache.exists() else build(P)
    if not cache.exists():
        zoo.to_parquet(cache, index=False)
    names = [c for c in zoo.columns if c not in ("ts_code", "trade_date")]
    zoo[names] = zoo[names].replace([np.inf, -np.inf], np.nan)
    names = [c for c in names if zoo[c].notna().mean() > 0.5 and zoo[c].nunique() > 5]
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                        columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["success"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    n_tests = len(names) * 8
    bar = norm.isf(0.025 / n_tests)
    say(f"factors {len(names)} (qlib {sum(n.startswith('q_') for n in names)}, talib {sum(n.startswith('t_') for n in names)}); "
        f"Bonferroni over {n_tests} tests: |t| >= {bar:.2f}")
    res: dict[tuple[str, str], pd.DataFrame] = {}
    for pname, segs in SEGS.items():
        base = P[pname].merge(o, on=["ts_code", "trade_date"]).merge(zoo, on=["ts_code", "trade_date"], how="left")
        for sname, a, b in segs:
            g = base[base.trade_date.between(a, b)]
            pct = g.groupby("trade_date")[names].rank(pct=True)
            rows = []
            for k in names:
                top, bot = g[pct[k] > 2 / 3], g[pct[k] <= 1 / 3]
                r = {"f": k}
                for m in ("success", "ret_base"):
                    dd = (top.groupby("trade_date")[m].mean() - bot.groupby("trade_date")[m].mean()).dropna()
                    r[f"{m}_d"] = dd.mean() * (100 if m == "success" else 1)
                    r[f"{m}_t"] = tstat(dd) if len(dd) > 5 else np.nan
                rows.append(r)
            res[(pname, sname)] = pd.DataFrame(rows).set_index("f")
            x = res[(pname, sname)]
            say(f"\n==== {pname} top 100, {sname}: {len(g):,} cases; largest |t| on success {x.success_t.abs().max():.1f}; "
                f"factors above the corrected bar: {int((x.success_t.abs() >= bar).sum())}; above |t| 2: {int((x.success_t.abs() >= 2).sum())} of {len(x)}")
    allres = pd.concat({f"{p}-{s}": v for (p, s), v in res.items()}, axis=1)
    allres.to_csv(OUT / "factor_results.csv", encoding="utf-8-sig")
    pctb = zoo.set_index(["ts_code", "trade_date"])["boll_pctb"]
    say("\n==== pick on period 1 (|t| >= 2), confirm on period 2 (same sign, |t| >= 2) ====")
    conf_sets = {}
    for pname, segs in SEGS.items():
        r1, r2 = res[(pname, segs[0][0])], res[(pname, segs[1][0])]
        pick = r1[r1.success_t.abs() >= 2].index
        conf = [k for k in pick if np.sign(r2.loc[k, "success_d"]) == np.sign(r1.loc[k, "success_d"]) and abs(r2.loc[k, "success_t"]) >= 2]
        conf_sets[pname] = conf
        say(f"\n{pname}: picked {len(pick)} of {len(names)} on {segs[0][0]}; confirmed {len(conf)} on {segs[1][0]}; "
            f"expected by chance with no signal about {len(names) * 0.05 * 0.025:.1f}")
        base = P[pname].merge(zoo, on=["ts_code", "trade_date"], how="left")
        base = base[base.trade_date.between(segs[1][1], segs[1][2])]
        pr = base.groupby("trade_date")[names].rank(pct=True)
        cb = pr["boll_pctb"] if "boll_pctb" in pr else base.groupby("trade_date").boll_pctb.rank(pct=True)
        rows = []
        for k in conf:
            rows.append((k, r1.loc[k, "success_d"], r1.loc[k, "success_t"], r2.loc[k, "success_d"], r2.loc[k, "success_t"],
                         r1.loc[k, "ret_base_d"], r2.loc[k, "ret_base_d"], pr[k].corr(cb, method="spearman")))
        rows.sort(key=lambda x: -min(abs(x[2]), abs(x[4])))
        say(f"  {'factor':14s} | {segs[0][0]} success diff (t) | {segs[1][0]} success diff (t) | per trade diff | rank corr with %b")
        for k, d1, t1, d2, t2, p1, p2, cr in rows[:30]:
            say(f"  {k:14s} | {d1:+6.1f} ({t1:+5.1f}) | {d2:+6.1f} ({t2:+5.1f}) | {p1:+.2f} / {p2:+.2f} | {cr:+.2f}")
        if len(rows) > 30:
            say(f"  ... {len(rows) - 30} more in factor_results.csv")
        low = [r for r in rows if abs(r[7]) < 0.5]
        say(f"  confirmed with |rank corr with %b| < 0.5 (not just the same strength factor): {len(low)}: "
            + ", ".join(r[0] for r in low[:15]))
    both = sorted(set(conf_sets["S20"]) & set(conf_sets["R20"]))
    say(f"\nconfirmed in both pools: {len(both)} {', '.join(both[:25])}")
    over = [(c, allres.loc[:, [f'{p}-{s}' for p, segs in SEGS.items() for s, _, _ in segs]].xs('success_t', axis=1, level=1).loc[c]) for c in names]
    tab = allres.xs("success_t", axis=1, level=1)
    big = tab[(tab.abs() >= bar).sum(axis=1) >= 1]
    say(f"\nfactors with |t| above the corrected bar ({bar:.2f}) in at least one period: {len(big)}")
    for k, r in big.sort_values(tab.columns[3], key=lambda s: -s.abs()).head(25).iterrows():
        say(f"  {k:14s} " + "  ".join(f"{c} {r[c]:+.1f}" for c in tab.columns))
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
