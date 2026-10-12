#!/usr/bin/env python
"""Compare the WorkBuddy / TongDaXin double-bottom screens (docs/1012分析参考) with R20 and S20.

  python scripts/compare_workbuddy_double_bottom.py [--day 20261012] [--skip-history]

Part 1 (the day): score every name in the two reference lists with the frozen S20 stage1 (market rank,
  top-100 pool, contract list, safe list) and the production R20 (r20_pred market rank, pool A), add our
  bottom-pattern tags, and compare with the market.
Part 2 (history): re-implement the published W-bottom rule (22-session window; two pivot lows >= 4
  sessions apart, within 10%; rebound >= 5% between them; both lows in the lower 45% of the window range;
  close > 1.01 x second low) and the strict screen (close > MA20, close < 1.10 x MA60, PE(TTM) > 0, no ST,
  close >= 97% of the neckline). First check the re-implementation against the day's lists, then on the
  S20 periods S1 2025-03-03..08-29 / S2 2026-01-27..08-05 and the R20 period 01-27..08-05: list-basis
  U15D10 (cost 0.3) of the hits vs the market, and whether S20 or R20 ranks the hits (top 10 by score vs
  all hits, same day, Newey-West(19) t). The rule is a third-party spec, not searched here, but the
  periods are already used: evidence tier reused_holdout. Signal days >= 2026-08-06 are not read.
Output: output/experiments/workbuddy_compare_20261012/{report.txt, day_scores.csv, wb_hits.parquet}
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_r20_s20_nested import tstat  # noqa: E402

REF = ROOT / "docs/1012分析参考"
OUT = ROOT / "output/experiments/workbuddy_compare_20261012"
SEGS = {"S1 2025-03-03..08-29": ("20250303", "20250829"), "S2 2026-01-27..08-05": ("20260127", "20260805")}
R20_SEGS = {"R20 in-sample 01-27..04-13": ("20260127", "20260413"), "R20 OOS 04-14..08-05": ("20260414", "20260805")}
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def ts_code(x: str) -> str:
    return f"{x[2:]}.{x[:2].upper()}"


def load_refs() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(REF / "双底形态选股结果_20261012.csv", encoding="utf-8-sig", dtype={"代码": str})
    b = pd.read_csv(REF / "双底形态+RSI超卖回升_20261012.csv", encoding="utf-8-sig", dtype={"代码": str})
    for x in (a, b):
        x["ts_code"] = x["代码"].map(ts_code)
    return a, b


# ---------------------------------------------------------------- W-bottom rule as published
def wb_stock(l: np.ndarray, h: np.ndarray, c: np.ndarray, first: int) -> dict[str, np.ndarray]:
    n = len(c)
    out = {k: np.full(n, np.nan) for k in ("neck", "l1", "l2", "i1", "i2")}
    hit = np.zeros(n, bool)
    for D in range(max(first, 25), n):
        w0 = D - 21
        lw = l[w0:D + 1]
        piv = [i for i in range(w0 + 2, D) if l[i] <= l[max(w0, i - 2):min(D, i + 2) + 1].min()]
        if len(piv) < 2:
            continue
        a = min(piv, key=lambda i: l[i])
        far = [i for i in piv if abs(i - a) >= 4]
        if not far:
            continue
        b = min(far, key=lambda i: l[i])
        i1, i2 = min(a, b), max(a, b)
        l1, l2 = l[i1], l[i2]
        if abs(l2 / l1 - 1) > 0.10:
            continue
        neck = h[i1 + 1:i2].max()
        if neck < 1.05 * min(l1, l2):
            continue
        lo, hi = lw.min(), h[w0:D + 1].max()
        if max(l1, l2) > lo + 0.45 * (hi - lo) or c[D] <= 1.01 * l2:
            continue
        hit[D] = True
        out["neck"][D], out["l1"][D], out["l2"][D], out["i1"][D], out["i2"][D] = neck, l1, l2, D - i1, D - i2
    out["wb"] = hit
    return out


def _chunk(args: tuple[pd.DataFrame, str]) -> pd.DataFrame:
    frame, out_from = args
    parts = []
    for code, g in frame.groupby("ts_code", sort=False):
        td = g.trade_date.to_numpy()
        first = int(np.searchsorted(td, out_from))
        c = g.close.to_numpy(float)
        r = wb_stock(g.low.to_numpy(float), g.high.to_numpy(float), c, first)
        ma20 = pd.Series(c).rolling(20).mean().to_numpy()
        ma60 = pd.Series(c).rolling(60).mean().to_numpy()
        df = pd.DataFrame({"ts_code": code, "trade_date": td, "close": c, "ma20": ma20, "ma60": ma60, **r})
        parts.append(df[df.wb & (df.trade_date >= out_from)])
    return pd.concat(parts, ignore_index=True)


def wb_panel(start: str, end: str, out_from: str, workers: int = 8) -> pd.DataFrame:
    from price_panel import adjusted_daily
    px = adjusted_daily(start, end)
    codes = px.ts_code.unique()
    chunks = [(px[px.ts_code.isin(set(codes[i::workers]))], out_from) for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        wb = pd.concat(ex.map(_chunk, chunks), ignore_index=True)
    pe = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "pe_ttm"])
                    for f in sorted((ROOT / "output/tushare_cache/daily_basic").glob("*.parquet")) if out_from <= f.stem <= end])
    wb = wb.merge(pe, on=["ts_code", "trade_date"], how="left")
    names = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name"]).drop_duplicates("ts_code")
    wb = wb.merge(names, on="ts_code", how="left")
    wb["dist_neck"] = wb.close / wb.neck
    wb["strict"] = ((wb.close > wb.ma20) & (wb.close < 1.10 * wb.ma60) & (wb.pe_ttm > 0)
                    & ~wb.name.fillna("").str.contains("ST") & (wb.dist_neck >= 0.97))
    return add_score(wb)


def add_score(wb: pd.DataFrame) -> pd.DataFrame:
    """Approximate pattern score (their sub-score formulas are not published): similarity 25, rebound 25,
    breakout 25 on the same caps they describe; position and volume are left out. Used only to take a
    top 30 per day from the strict screen, as their published list does."""
    diff = (wb.l2 / wb.l1 - 1).abs()
    reb = wb.neck / wb[["l1", "l2"]].min(axis=1) - 1
    wb["approx_score"] = (25 * (1 - diff / 0.10).clip(0, 1) + 25 * (reb / 0.10).clip(0, 1)
                          + 25 * ((wb.dist_neck - 0.90) / 0.10).clip(0, 1))
    r = wb[wb.strict].groupby("trade_date").approx_score.rank(ascending=False, method="first")
    wb["top30"] = False
    wb.loc[r.index, "top30"] = r <= 30
    return wb


# ---------------------------------------------------------------- part 1: the day
def day_scores(day: str) -> pd.DataFrame:
    from export_r20_pool_a_history import pool_a
    from run_s20_pure_v1_shadow import SHADOW, daily_natr, load_factors
    from stockagent_analysis.s20_pure import PureConfig, SafeConfig, _frozen_models, select, select_safe, stage1_probability
    from daily_best_lists import v12_market
    _, _, features = _frozen_models()
    f = load_factors(SHADOW / "factor_groups", features, "20260901")
    f = f[f.trade_date == day].copy()
    f["stage1_probability"] = stage1_probability(f)
    f = f.merge(daily_natr("20260901"), on=["ts_code", "trade_date"], how="left")
    f["s20_rank"] = f.stage1_probability.rank(ascending=False, method="first")
    f["s20_pct"] = f.stage1_probability.rank(pct=True)
    off, safe = select(f, PureConfig(), "U15D10"), select_safe(f, SafeConfig())
    v12, _ = v12_market(day)
    v12["r20_rank"] = v12.r20_pred.rank(ascending=False, method="first")
    v12["r20_pct"] = v12.r20_pred.rank(pct=True)
    A = pool_a(v12)
    m = f[["ts_code", "stage1_probability", "s20_rank", "s20_pct", "natr14"]].merge(
        v12[["ts_code", "r20_pred", "buy_r20_score", "r20_rank", "r20_pct", "pred_max_gain_20", "pred_max_dd_20"]], on="ts_code", how="outer")
    m["in_s20_top100"] = m.s20_rank <= 100
    m["in_s20_list"] = m.ts_code.isin(off.ts_code)
    m["in_safe_list"] = m.ts_code.isin(safe.ts_code)
    m["in_r20_top100"] = m.r20_rank <= 100
    m["in_pool_a"] = m.ts_code.isin(A.ts_code)
    return m


def part1(day: str) -> pd.DataFrame:
    from bottom_pattern_pools import bottom_tags
    a, b = load_refs()
    m = day_scores(day)
    bt = bottom_tags(set(b.ts_code), day)
    say(f"==== Part 1: {day}, WorkBuddy lists: strict {len(a)} names (subset of the base), base {len(b)} names")
    say(f"market scored: S20 {m.s20_rank.notna().sum()}, R20 {m.r20_rank.notna().sum()}")
    rows = []
    for nm, ref in (("strict 30", a), ("base 180", b)):
        q = ref[["ts_code", "名称", "形态得分", "综合评分"]].merge(m, on="ts_code", how="left")
        rows.append(q.assign(list=nm))
        say(f"\n-- {nm}: matched S20 {q.s20_pct.notna().sum()}, R20 {q.r20_pct.notna().sum()}")
        for k, lab in (("s20_pct", "S20 market percentile (1 = best)"), ("r20_pct", "R20 market percentile")):
            x = q[k].dropna()
            say(f"{lab}: mean {x.mean():.2f}, median {x.median():.2f}, top 10% {100 * (x >= .9).mean():.0f}%, "
                f"top 30% {100 * (x >= .7).mean():.0f}%, bottom 30% {100 * (x <= .3).mean():.0f}%  (market: 10% / 30% / 30%)")
        say("in S20 top 100: " + ", ".join(q[q.in_s20_top100 == True]["名称"]) + f" ({int((q.in_s20_top100 == True).sum())})")
        say("in S20 contract list: " + ", ".join(q[q.in_s20_list == True]["名称"]) + f" ({int((q.in_s20_list == True).sum())})")
        say("in S20 safe list: " + ", ".join(q[q.in_safe_list == True]["名称"]) + f" ({int((q.in_safe_list == True).sum())})")
        say("in R20 top 100: " + ", ".join(q[q.in_r20_top100 == True]["名称"]) + f" ({int((q.in_r20_top100 == True).sum())})")
        say("in pool A: " + ", ".join(q[q.in_pool_a == True]["名称"]) + f" ({int((q.in_pool_a == True).sum())})")
        rho_s = q[["形态得分", "s20_pct"]].corr("spearman").iloc[0, 1]
        rho_r = q[["形态得分", "r20_pct"]].corr("spearman").iloc[0, 1]
        rho_c = q[["综合评分", "s20_pct"]].corr("spearman").iloc[0, 1]
        rho_cr = q[["综合评分", "r20_pct"]].corr("spearman").iloc[0, 1]
        say(f"Spearman: pattern score vs S20 {rho_s:+.2f}, vs R20 {rho_r:+.2f}; composite vs S20 {rho_c:+.2f}, vs R20 {rho_cr:+.2f}")
        say(f"NATR(14) median: list {q.natr14.median():.3f}, market {m.natr14.median():.3f}; "
            f"R20 predicted max gain median {q.pred_max_gain_20.median():+.1f}% vs market {m.pred_max_gain_20.median():+.1f}%")
    d = pd.concat(rows, ignore_index=True).merge(bt, on="ts_code", how="left")
    base = d[d.list == "base 180"]
    say(f"\nour bottom tags on the base 180: W bottom (breakout or forming) {base.bottom_tags.fillna('').str.contains('W底').sum()}, "
        f"any tag {(base.bottom_score.fillna(0) > 0).sum()}, in our bottom zone {int(base.zone.fillna(False).sum())}")
    return d


# ---------------------------------------------------------------- part 2: history
def part2(day: str, wb: pd.DataFrame) -> None:
    a, b = load_refs()
    today = wb[wb.trade_date == day]
    for nm, ref, ours in (("base 180", b, today), ("strict 30 vs our strict", a, today[today.strict]),
                          ("strict 30 vs our strict top 30", a, today[today.top30])):
        inter = set(ref.ts_code) & set(ours.ts_code)
        say(f"re-implementation check {day}, {nm}: ours {len(ours)}, theirs {len(ref)}, overlap {len(inter)} "
            f"(recall {100 * len(inter) / len(ref):.0f}%, precision {100 * len(inter) / max(len(ours), 1):.0f}%)")
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                        columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    fz = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet",
                         columns=["ts_code", "trade_date", "stage1_probability"])
    r20 = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "r20_pred"])
                     for p in sorted((ROOT / "output/r20_history/market").glob("*.parquet"))], ignore_index=True)
    r20["trade_date"] = r20.trade_date.astype(str)
    mk = o.merge(fz, on=["ts_code", "trade_date"], how="left").merge(r20, on=["ts_code", "trade_date"], how="left")
    mk = mk[mk.trade_date <= "20260805"]
    mk["s20_rank"] = mk.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    mk["r20_rank"] = mk.groupby("trade_date").r20_pred.rank(ascending=False, method="first")
    h = wb[wb.trade_date <= "20260805"].merge(mk, on=["ts_code", "trade_date"], how="inner")

    def stop_first(q: pd.DataFrame) -> float:
        return 100 * ((q.dn10_day > 0) & ((q.up15_day == 0) | (q.dn10_day <= q.up15_day))).mean()

    def block(sname: str, a0: str, b0: str, score: str) -> None:
        g = h[h.trade_date.between(a0, b0)]
        mday = mk[mk.trade_date.between(a0, b0)].groupby("trade_date").ret_base.mean()
        say(f"\n==== {sname}: market per trade {mk[mk.trade_date.between(a0, b0)].ret_base.mean():+.2f}, "
            f"stop-first {stop_first(mk[mk.trade_date.between(a0, b0)]):.1f}%")
        for lab, q in (("WB base", g), ("WB strict", g[g.strict]), ("WB top 30", g[g.top30])):
            if q.empty:
                continue
            dd = (q.groupby("trade_date").ret_base.mean() - mday).dropna()
            say(f"{lab:9s}: {len(q):6d} names on {q.trade_date.nunique()} days ({len(q) / q.trade_date.nunique():.0f}/day), "
                f"per trade {q.ret_base.mean():+.2f}, stop-first {stop_first(q):.1f}% | vs market same day {dd.mean():+.2f} (t {tstat(dd):+.1f})")
            s = q.dropna(subset=[score])
            if s.empty:
                continue
            day_all = s.groupby("trade_date").ret_base.mean()
            top = s.sort_values(score, ascending=False).groupby("trade_date").head(10)
            bot = s.sort_values(score, ascending=True).groupby("trade_date").head(10)
            dt = (top.groupby("trade_date").ret_base.mean() - day_all).dropna()
            db = (bot.groupby("trade_date").ret_base.mean() - day_all).dropna()
            ic = s.groupby("trade_date").filter(lambda x: len(x) >= 10).groupby("trade_date").apply(
                lambda x: x[score].corr(x.ret_base, method="spearman"))
            nm = "S20" if score == "stage1_probability" else "R20"
            rk = "s20_rank" if nm == "S20" else "r20_rank"
            say(f"   ranked by {nm}: top 10 per trade {top.ret_base.mean():+.2f} (vs all hits {dt.mean():+.2f}, t {tstat(dt):+.1f}); "
                f"bottom 10 {bot.ret_base.mean():+.2f} (vs all {db.mean():+.2f}, t {tstat(db):+.1f}); daily IC {ic.mean():+.3f} (t {tstat(ic):+.1f}, {len(ic)} days)")
            inpool = s[s[rk] <= 100]
            say(f"   hits also in the {nm} market top 100: {len(inpool)} names on {inpool.trade_date.nunique()} days, per trade "
                f"{inpool.ret_base.mean() if len(inpool) else float('nan'):+.2f}")

    for sname, (a0, b0) in SEGS.items():
        block(sname, a0, b0, "stage1_probability")
    for sname, (a0, b0) in R20_SEGS.items():
        block(sname, a0, b0, "r20_pred")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", default="20261012")
    ap.add_argument("--skip-history", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = part1(a.day)
    d.to_csv(OUT / "day_scores.csv", index=False, encoding="utf-8-sig")
    if not a.skip_history:
        cache = OUT / "wb_hits.parquet"
        if cache.exists():
            wb = add_score(pd.read_parquet(cache))
        else:
            # history ends at 08-05 (reserved window not read); the day itself is list preparation only
            wb = pd.concat([wb_panel("20241101", "20260805", out_from="20250303"),
                            wb_panel("20260501", a.day, out_from=a.day)], ignore_index=True)
            wb.to_parquet(cache, index=False)
        say("")
        part2(a.day, wb)
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
