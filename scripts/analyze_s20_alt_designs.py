#!/usr/bin/env python
"""2026-10-11 alternative-design trial for S20, as pre-registered in wiki/2026-10-11_s20-alt-designs-trial.md.

  python scripts/analyze_s20_alt_designs.py [--parts 1,2,3,4,5,6]

Segments S1 2025-03-03..2025-08-29 (outside the frozen model's fitting window) and S2 2026-01-27..2026-08-05
(confirmation). Yardstick: list-basis per-trade U15D10 return (cost 0.3), same-day paired differences with
Newey-West(19) t. Baseline: the frozen S20 contract list (frozen_scores -> top 100 -> drop top-40% NATR -> 20).
Inputs: build_s20_alt_paths.py (paths.parquet), train_s20_alt_labels.py (pred_atr / pred_exc), the unified v1
predictions, Tushare event caches, 15-minute bars, broad index closes. Output: report_<parts>.txt in OUT.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_r20_s20_nested import tstat  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/s20_alt_designs_20261011"
SEG = {"S1": ("20250303", "20250829"), "S2": ("20260127", "20260805")}
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def seg_of(d: pd.Series) -> pd.Series:
    out = pd.Series("", index=d.index)
    for k, (a, b) in SEG.items():
        out[d.between(a, b)] = k
    return out


def load_paths() -> pd.DataFrame:
    p = pd.read_parquet(OUT / "paths.parquet")
    p["seg"] = seg_of(p.trade_date)
    return p[p.seg != ""]


def frozen_list(paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fz = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    fz = fz[seg_of(fz.trade_date) != ""]
    L = select(fz, PureConfig(), "U15D10").merge(paths.drop(columns=["natr14"]), on=["ts_code", "trade_date"])
    return fz, L


def summary(name: str, q: pd.DataFrame, col: str = "ret_base") -> dict:
    day = q.groupby("trade_date")[col].mean()
    return {"list": name, "names": len(q), "per_trade": q[col].mean(), "daily": day.mean(), "t": tstat(day),
            "stop_first%": 100 * ((q.dn10_day > 0) & ((q.up15_day == 0) | (q.dn10_day <= q.up15_day))).mean(),
            "up_first%": 100 * ((q.up15_day > 0) & ((q.dn10_day == 0) | (q.up15_day < q.dn10_day))).mean()}


def table(rows: list[dict]) -> None:
    say(f"{'list':44s} | names | per trade | stop-first | up-first | daily mean (t)")
    for r in rows:
        say(f"{r['list']:44s} | {r['names']:5d} | {r['per_trade']:+8.2f}  | {r['stop_first%']:8.1f}%  | {r['up_first%']:6.1f}%  | "
            f"{r['daily']:+.2f} ({r['t']:+.1f})")


def paired(name: str, a: pd.DataFrame, b: pd.DataFrame, col_a: str = "ret_base", col_b: str = "ret_base") -> None:
    da, db = a.groupby("trade_date")[col_a].mean(), b.groupby("trade_date")[col_b].mean()
    d = (da - db).dropna()
    say(f"    {name:60s}: {d.mean():+.2f} (t {tstat(d):+.1f}, days {len(d)})")


def by_segment(fn) -> None:
    for s in SEG:
        say(f"\n-- {s} {SEG[s][0]}..{SEG[s][1]}")
        fn(s)


def top(frame: pd.DataFrame, col: str, k: int = 20) -> pd.DataFrame:
    r = frame.groupby("trade_date")[col].rank(ascending=False, method="first")
    return frame[r <= k]


# ---------------------------------------------------------------- directions 1, 2, 6 (new labels)
def scored(paths: pd.DataFrame, name: str) -> pd.DataFrame:
    p = pd.read_parquet(OUT / f"pred_{name}.parquet").dropna()
    return p.merge(paths, on=["ts_code", "trade_date"])


def market_pct(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame.groupby("trade_date")[col].rank(pct=True)


def part_labels(paths: pd.DataFrame, fz: pd.DataFrame, L: pd.DataFrame) -> None:
    v1 = pd.read_parquet(ROOT / "output/experiments/s20_unified_v1_20261005/predictions.parquet",
                         columns=["ts_code", "trade_date", "stage1_probability"]).rename(columns={"stage1_probability": "score"})
    v1 = v1.merge(paths, on=["ts_code", "trade_date"])
    natr_mkt = paths.assign(natr_pct=market_pct(paths, "natr14"))[["ts_code", "trade_date", "natr_pct"]]
    gate = style_gate()
    for name, title in (("atr", "direction 1 - ATR-unit label"), ("exc", "direction 2 - industry-excess label")):
        say(f"\n==== {title}")
        S = scored(paths, name)
        lists = {f"{name} top20 (1a/2a)": top(S, "score"),
                 f"{name} funnel (1b/2b)": select(S, PureConfig(), "U15D10", score_col="score"),
                 "unified v1 (positive20) funnel, same recipe": select(v1, PureConfig(), "U15D10", score_col="score"),
                 "frozen contract list": L}

        def seg(s: str) -> None:
            rows = []
            for k, q in lists.items():
                q = q[q.seg == s]
                rows.append(summary(k, q))
            table(rows)
            for k, q in lists.items():
                q = q[q.seg == s].merge(natr_mkt, on=["ts_code", "trade_date"])
                say(f"    {k:44s}: market NATR pct median {q.natr_pct.median():.2f}, 20d industry excess {q.exc20.mean():+.2f}, "
                    f"gate-closed-day per trade {q[~q.trade_date.map(gate).fillna(False)].ret_base.mean():+.2f}")
            Lq = L[L.seg == s]
            paired(f"{name} top20 - frozen contract", lists[f'{name} top20 (1a/2a)'][lambda x: x.seg == s], Lq)
            paired(f"{name} funnel - frozen contract", lists[f'{name} funnel (1b/2b)'][lambda x: x.seg == s], Lq)
            paired(f"{name} funnel - unified v1 funnel (label effect)", lists[f'{name} funnel (1b/2b)'][lambda x: x.seg == s],
                   lists["unified v1 (positive20) funnel, same recipe"][lambda x: x.seg == s])
        by_segment(seg)


def style_gate() -> pd.Series:
    from score_s20_stock import index_close
    hs, cy = index_close("000300.SH"), index_close("399006.SZ")
    return (cy / cy.shift(20) - 1) > (hs / hs.shift(20) - 1)


def ev_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EV (%) of U15D10 for daily drift mu (%) and daily vol sigma (%): Gaussian, 4 sub-steps a day."""
    rng = np.random.default_rng(20261011)
    Z = rng.standard_normal((4000, 80)).astype(np.float32)
    mus, sigs = np.round(np.arange(-1.5, 1.5001, 0.05), 3), np.round(np.arange(0.5, 10.001, 0.25), 3)
    ev = np.zeros((len(mus), len(sigs)))
    for i, m in enumerate(mus):
        for j, s in enumerate(sigs):
            path = np.cumsum(m / 4 + s / 2 * Z, axis=1)
            up, dn = path >= 15, path <= -10
            fu = np.where(up.any(1), up.argmax(1), 999)
            fd = np.where(dn.any(1), dn.argmax(1), 999)
            r = np.where(fu < fd, 15.0, np.where(fd < 999, -10.0, path[:, -1]))
            ev[i, j] = r.mean() - 0.3
    return mus, sigs, ev


def part_dual(paths: pd.DataFrame, fz: pd.DataFrame, L: pd.DataFrame) -> None:
    from scipy.interpolate import RegularGridInterpolator
    from train_s20_unified_v1 import BLOCKS
    say("\n==== direction 6 - direction x volatility")
    S = scored(paths, "atr")
    S["natr_pct"] = market_pct(S, "natr14")
    S["dir_rank"] = market_pct(S, "score")
    pool = fz.assign(pool_rank=fz.groupby("trade_date").stage1_probability.rank(ascending=False, method="first"))
    pool = pool[pool.pool_rank <= 100][["ts_code", "trade_date"]]
    a = top(S.merge(pool, on=["ts_code", "trade_date"]), "score")
    b = top(S[S.natr_pct.between(0.4, 0.8)], "score")
    # 6c: walk-forward calibration of z = ret20 / (rv20 sqrt 20) on the direction-score decile
    S["z"] = S.ret20 / (S.rv20 * np.sqrt(20))
    S["dec"] = np.minimum((S.dir_rank * 10).astype(int), 9)
    mus, sigs, ev = ev_table()
    f = RegularGridInterpolator((mus, sigs), ev, bounds_error=False, fill_value=None)
    parts = []
    # calibration rows: every out-of-sample scored day (not only the two segments) whose horizon ended before the block
    allp = pd.read_parquet(OUT / "paths.parquet", columns=["ts_code", "trade_date", "horizon_end_date", "ret20", "rv20"])
    C = pd.read_parquet(OUT / "pred_atr.parquet").dropna().merge(allp, on=["ts_code", "trade_date"])
    C["z"] = C.ret20 / (C.rv20 * np.sqrt(20))
    C["dec"] = np.minimum((C.groupby("trade_date").score.rank(pct=True) * 10).astype(int), 9)
    for start, end in BLOCKS:
        hist = C[(C.horizon_end_date < start) & C.z.notna()]
        blk = S[S.trade_date.between(start, end)].copy()
        if hist.empty or blk.empty:
            continue
        zbar = hist.groupby("dec").z.mean()
        blk["mu"] = blk.dec.map(zbar) * blk.rv20 / np.sqrt(20)
        blk = blk.dropna(subset=["mu", "rv20"])
        blk["ev"] = f(np.c_[blk.mu.clip(-1.5, 1.5), blk.rv20.clip(0.5, 10)])
        parts.append(blk)
        say(f"6c calibration for block {start}: rows {len(hist):,}, z by decile " + " ".join(f"{v:+.3f}" for v in zbar.values))
    c = top(pd.concat(parts, ignore_index=True), "ev")

    def seg(s: str) -> None:
        Lq = L[L.seg == s]
        rows = [summary("6a frozen top100 -> direction top20", a[a.seg == s]),
                summary("6b NATR 40-80% band -> direction top20", b[b.seg == s]),
                summary("6c EV(mu, sigma) top20", c[c.seg == s]), summary("frozen contract list", Lq)]
        table(rows)
        for nm, q in (("6a", a), ("6b", b), ("6c", c)):
            paired(f"{nm} - frozen contract", q[q.seg == s], Lq)
        g = S[S.seg == s].copy()
        g["vol_t"] = np.minimum((g.natr_pct * 3).astype(int), 2)
        g["dir_q"] = np.minimum((g.dir_rank * 5).astype(int), 4)
        grid = g.pivot_table(index="vol_t", columns="dir_q", values="ret_base", aggfunc="mean").round(2)
        grid.index = ["NATR low", "NATR mid", "NATR high"]
        grid.columns = [f"dir Q{i + 1}" for i in grid.columns]
        say("    descriptive grid, per trade U15D10 (all stocks):\n" + grid.to_string())
    by_segment(seg)


# ---------------------------------------------------------------- direction 3 (event flags)
def calendar() -> list[str]:
    return sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))


def part_events(paths: pd.DataFrame, fz: pd.DataFrame, L: pd.DataFrame) -> None:
    say("\n==== direction 3 - event flags inside the frozen contract list")
    cal = calendar()
    pos = {d: i for i, d in enumerate(cal)}
    ev = ROOT / "output/tushare_cache/events"
    sf = pd.concat([pd.read_parquet(p) for p in sorted((ev / "share_float").glob("*.parquet"))], ignore_index=True)
    sf = sf[pd.to_numeric(sf.float_ratio, errors="coerce") >= 0]
    sf = sf.groupby(["ts_code", "ann_date", "float_date"], as_index=False).float_ratio.sum()
    ht = pd.concat([pd.read_parquet(p) for p in sorted((ev / "stk_holdertrade").glob("*.parquet"))], ignore_index=True)
    ht = ht[ht.in_de == "DE"][["ts_code", "ann_date"]].drop_duplicates()
    q = L[["ts_code", "trade_date", "seg", "ret_base", "up15_day", "dn10_day", "list_rank"]].copy()
    q["i"] = q.trade_date.map(pos)
    end20 = q.i.map(lambda i: cal[min(i + 20, len(cal) - 1)])
    start20 = q.i.map(lambda i: cal[max(i - 20, 0)])
    start5 = q.i.map(lambda i: cal[max(i - 4, 0)])
    m = q[["ts_code", "trade_date"]].assign(e=end20).merge(sf[sf.float_ratio >= 1], on="ts_code")
    m = m[(m.ann_date <= m.trade_date) & (m.float_date > m.trade_date) & (m.float_date <= m.e)]
    q["F1_unlock"] = q.set_index(["ts_code", "trade_date"]).index.isin(m.set_index(["ts_code", "trade_date"]).index)
    m = q[["ts_code", "trade_date"]].assign(s=start20).merge(ht, on="ts_code")
    m = m[(m.ann_date <= m.trade_date) & (m.ann_date >= m.s)]
    q["F2_reduce"] = q.set_index(["ts_code", "trade_date"]).index.isin(m.set_index(["ts_code", "trade_date"]).index)
    mg_files = sorted((ev / "margin_detail").glob("*.parquet"))
    if mg_files:
        mg = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "rzye"]) for p in mg_files if p.stat().st_size > 0],
                       ignore_index=True)
        mg["trade_date"] = mg.trade_date.astype(str)
        mg = mg.sort_values(["ts_code", "trade_date"])
        mg["g5"] = mg.groupby("ts_code").rzye.pct_change(5)
        mg["g5_pct"] = mg.groupby("trade_date").g5.rank(pct=True)
        q = q.merge(mg[["ts_code", "trade_date", "g5_pct"]], on=["ts_code", "trade_date"], how="left")
        q["F3_margin"] = q.g5_pct >= 0.9
        say(f"margin coverage of list rows {q.g5_pct.notna().mean():.0%} (margin cache to {mg.trade_date.max()})")
    tl_files = [p for p in sorted((ev / "top_list").glob("*.parquet")) if p.stat().st_size > 0]
    if tl_files:
        tl = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date"]) for p in tl_files], ignore_index=True).drop_duplicates()
        tl["trade_date"] = tl.trade_date.astype(str)
        m = q[["ts_code", "trade_date"]].assign(s=start5).merge(tl.rename(columns={"trade_date": "d"}), on="ts_code")
        m = m[(m.d <= m.trade_date) & (m.d >= m.s)]
        q["F4_toplist"] = q.set_index(["ts_code", "trade_date"]).index.isin(m.set_index(["ts_code", "trade_date"]).index)
        say(f"top_list cache to {tl.trade_date.max()}")
    flags = [c for c in ("F1_unlock", "F2_reduce", "F3_margin", "F4_toplist") if c in q]

    def seg(s: str) -> None:
        g = q[q.seg == s]
        for fl in flags:
            on, off = g[g[fl]], g[~g[fl]]
            so = lambda x: 100 * ((x.dn10_day > 0) & ((x.up15_day == 0) | (x.dn10_day <= x.up15_day))).mean()
            say(f"  {fl:11s}: flagged {len(on):4d} ({100 * len(on) / len(g):4.1f}%) per trade {on.ret_base.mean():+.2f} stop-first {so(on):.1f}% "
                f"| unflagged {off.ret_base.mean():+.2f} stop-first {so(off):.1f}%")
            paired(f"{fl}: list without flagged names - full list", off, g)
    by_segment(seg)


# ---------------------------------------------------------------- direction 4 (execution)
def part_execution(paths: pd.DataFrame, fz: pd.DataFrame, L: pd.DataFrame) -> None:
    say("\n==== direction 4 - execution layer (frozen contract list)")
    t3 = L[L.list_rank <= 3]
    lo3 = L[L.groupby("trade_date").natr14.rank(method="first") <= 3]

    def seg(s: str) -> None:
        Lq = L[L.seg == s]
        table([summary("4a stage1 top 3", t3[t3.seg == s]), summary("4a lowest-NATR 3", lo3[lo3.seg == s]),
               summary("full list (= expected random 3)", Lq)])
        paired("4a stage1 top 3 - list mean", t3[t3.seg == s], Lq)
        paired("4a lowest-NATR 3 - list mean", lo3[lo3.seg == s], Lq)
    by_segment(seg)
    entry_timing(L)
    sizing(L)


def entry_timing(L: pd.DataFrame) -> None:
    say("\n-- 4b entry timing (names with 15-minute bars on the entry day)")
    cal = calendar()
    nxt = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}
    q = L[["ts_code", "trade_date", "seg", "ret_base"]].copy()
    q["entry_day"] = q.trade_date.map(nxt)
    codes = set(q.ts_code)
    mdir = ROOT / "output/tushare_cache/min15"
    rows = []
    for code in sorted(codes):
        f = mdir / f"{code}.parquet"
        if not f.exists():
            continue
        m = pd.read_parquet(f)
        m["day"] = m.trade_time.astype(str).str[:10].str.replace("-", "")
        m["hm"] = m.trade_time.astype(str).str[11:16]
        want = set(q.loc[q.ts_code == code, "entry_day"])
        for d, g in m[m.day.isin(want)].groupby("day"):
            if len(g) < 12:
                continue
            g = g.sort_values("hm")
            ten = g[g.hm == "10:00"]
            after = g[g.hm > "10:00"]
            if ten.empty or after.empty or g.vol.sum() <= 0:
                continue
            rows.append({"ts_code": code, "entry_day": d, "p_open": g.open.iloc[0], "p_1000": ten.close.iloc[0],
                         "hi_after_1000": after.high.max(), "lo_after_1000": after.low.min(),
                         "p_vwap": g.amount.sum() / g.vol.sum()})
    E = pd.DataFrame(rows)
    q = q.merge(E, on=["ts_code", "entry_day"])
    # daily bars for the 20 sessions from the entry day
    days = sorted(set(q.entry_day))
    idx = {d: i for i, d in enumerate(cal)}
    need = sorted({cal[j] for d in days for j in range(idx[d], min(idx[d] + 20, len(cal)))})
    D = pd.concat([pd.read_parquet(ROOT / "output/tushare_cache/daily" / f"{d}.parquet",
                                   columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close"]) for d in need])
    D = D[D.ts_code.isin(codes)]
    D["trade_date"] = D.trade_date.astype(str)
    D = D.set_index(["ts_code", "trade_date"]).sort_index()
    # vwap scale: 15-minute amount/vol may be in different units than price; rescale to the bar price level
    out = []
    for r in q.itertuples(index=False):
        i = idx[r.entry_day]
        seq = cal[i:i + 20]
        try:
            g = D.loc[r.ts_code].reindex(seq)
        except KeyError:
            continue
        if g.close.isna().sum() > 2:
            continue
        vwap = r.p_vwap
        if not (0.5 * r.p_open < vwap < 2 * r.p_open):
            vwap = vwap / 100 if 0.5 * r.p_open < vwap / 100 < 2 * r.p_open else np.nan
        res = {"ts_code": r.ts_code, "trade_date": r.trade_date, "seg": r.seg,
               "gap": g.open.iloc[0] / g.pre_close.iloc[0] - 1, "ret_base": r.ret_base}
        for nm, p, h1, l1 in (("open", g.open.iloc[0], g.high.iloc[0], g.low.iloc[0]),
                              ("1000", r.p_1000, r.hi_after_1000, r.lo_after_1000),
                              ("vwap", vwap, np.nan, np.nan)):
            if not np.isfinite(p) or p <= 0:
                res[f"r_{nm}"] = np.nan
                continue
            hi = np.r_[h1, g.high.iloc[1:].to_numpy()] / p - 1
            lo = np.r_[l1, g.low.iloc[1:].to_numpy()] / p - 1
            up = np.where(hi >= 0.15)[0]
            dn = np.where(lo <= -0.10)[0]
            fu, fd = (up[0] if len(up) else 99), (dn[0] if len(dn) else 99)
            last = g.close.ffill().iloc[-1] / p - 1
            res[f"r_{nm}"] = (15.0 if fu < fd else (-10.0 if fd < 99 else 100 * last)) - 0.3
        out.append(res)
    T = pd.DataFrame(out)
    T["r_skipgap"] = np.where(T.gap > 0.05, 0.0, T.r_open)

    def seg(s: str) -> None:
        g = T[T.seg == s]
        say(f"  names {len(g)} (of {len(L[L.seg == s])} list rows); open-entry check vs U15D10 panel: mean |diff| "
            f"{(g.r_open - g.ret_base).abs().mean():.2f}")
        for c in ("r_open", "r_1000", "r_vwap", "r_skipgap"):
            say(f"    {c:10s} per trade {g[c].mean():+.2f}")
        for c in ("r_1000", "r_vwap", "r_skipgap"):
            d = (g.groupby("trade_date")[c].mean() - g.groupby("trade_date").r_open.mean()).dropna()
            say(f"    {c} - open: {d.mean():+.2f} (t {tstat(d):+.1f}, days {len(d)})")
        say(f"    gap >5% names {100 * (g.gap > 0.05).mean():.1f}%, their open-entry per trade {g[g.gap > 0.05].r_open.mean():+.2f}")
    by_segment(seg)


def sizing(L: pd.DataFrame) -> None:
    say("\n-- 4c sizing by the style-gate gap")
    from score_s20_stock import index_close
    hs, cy = index_close("000300.SH"), index_close("399006.SZ")
    gap = (cy / cy.shift(20) - 1) - (hs / hs.shift(20) - 1)
    day = L.groupby(["seg", "trade_date"]).ret_base.mean().reset_index()
    day["gap"] = day.trade_date.map(gap)
    day["w_bin"] = (day.gap > 0).astype(float)
    day["w_scaled"] = (day.gap / 0.05).clip(0, 1)

    def seg(s: str) -> None:
        g = day[day.seg == s].dropna()
        for w in ("w_bin", "w_scaled"):
            say(f"    {w:9s}: exposure {g[w].mean():.2f}, return per unit exposure {(g[w] * g.ret_base).sum() / g[w].sum():+.2f}, "
                f"book mean per day {(g[w] * g.ret_base).mean():+.2f}")
        on = g[g.gap > 0]
        if len(on) > 25:
            x = on.gap - on.gap.mean()
            beta = (x * on.ret_base).sum() / (x * x).sum()
            resid = on.ret_base - on.ret_base.mean() - beta * x
            from experiment_cross_filter_v2 import nw_se
            se = nw_se((x * resid).to_numpy()) * len(x) / (x * x).sum()
            say(f"    dose-response on gate-open days: slope {beta:+.1f} per unit gap (t {beta / se:+.1f}, days {len(on)})")
    by_segment(seg)


# ---------------------------------------------------------------- direction 5 (exits)
def part_exits(paths: pd.DataFrame, fz: pd.DataFrame, L: pd.DataFrame) -> None:
    say("\n==== direction 5 - exit variants (frozen contract list)")

    def seg(s: str) -> None:
        g = L[L.seg == s]
        for c, nm in (("ret_base", "U15D10"), ("ret_x1", "X1 day-10 time stop"), ("ret_x2", "X2 half at +10%"),
                      ("ret_x3", "X3 breakeven after +10%")):
            say(f"    {nm:28s} per trade {g[c].mean():+.2f}  worst month {g.groupby(g.trade_date.str[:6])[c].mean().min():+.2f}")
        for c in ("ret_x1", "ret_x2", "ret_x3"):
            paired(f"{c} - U15D10", g, g, c, "ret_base")
    by_segment(seg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="1,2,3,4,5,6")
    parts = ap.parse_args().parts.split(",")
    paths = load_paths()
    fz, L = frozen_list(paths)
    say(f"segments {SEG}; frozen contract list rows {len(L)}")
    say("baseline frozen contract list: " + ", ".join(f"{s} per trade {L[L.seg == s].ret_base.mean():+.2f}" for s in SEG))
    if "1" in parts or "2" in parts:
        part_labels(paths, fz, L)
    if "6" in parts:
        part_dual(paths, fz, L)
    if "3" in parts:
        part_events(paths, fz, L)
    if "4" in parts:
        part_execution(paths, fz, L)
    if "5" in parts:
        part_exits(paths, fz, L)
    (OUT / f"report_{'_'.join(parts)}.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
