#!/usr/bin/env python
"""Pump start-up / start-down scores on the S20 funnel, and the daily lists of shadow rule S0012.

Inputs
  output/pump_history/pump_scores_*.parquet   from scripts/export_pump_history.py (rows with usable=True only)
  frozen stage1 scores (s20_frozen_rescore_20261005) for signal days up to 2026-08-05
  output/experiments/s20_pure_v1_shadow/funnel_pool.parquet for the reserved days (the full funnel
  pool, from scripts/s20_shadow_funnel_survivors.py)

Part A, used windows only (signal days <= 2026-08-05), descriptive (reused_holdout):
  1. does pump mean in the S20 population what it means market-wide? 5-day start-up / start-down
     rates (the pump label itself) by pump_score and pump_down_score tercile inside the survivors
  2. ratio bands inside the survivors: 5-day labels and the 20-day +15/-10 outcome
  3. S0012 against the frozen list, same day: stop-first share, per-trade return, up share; the
     outcome of the names it removes. Split at 2026-05-22 (end of the pump model's validation).
Part B, reserved days: S0012 lists only. No outcome of a reserved day is read; scoring waits for
  the shadow's maturity.

Output: output/experiments/s20_pump_filter_20261006/{report.txt, shadow_lists_S0012.csv (closed), shadow_lists_S0014.csv}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/s20_pump_filter_20261006"
RESERVED_FROM = "20260806"
PUMP_VALID_END = "20260522"
RATIO_CUT = 5.0
TOP_K = 20


def pump_scores() -> pd.DataFrame:
    files = sorted((ROOT / "output/pump_history").glob("pump_scores_*.parquet"))
    if not files:
        raise SystemExit("no pump history yet: run scripts/export_pump_history.py first")
    p = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    p = p[p.usable].drop_duplicates(["ts_code", "trade_date"], keep="last")
    return p[["ts_code", "trade_date", "pump_score", "pump_down_score", "ratio"]]


def five_day_labels(codes: set[str], start: str, end: str) -> pd.DataFrame:
    """The pump label, recomputed: next open as base, max high / min low over the next 5 sessions."""
    parts = []
    for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")):
        if start <= f.stem <= end:
            x = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
            parts.append(x[x.ts_code.isin(codes)])
    px = pd.concat(parts)
    px["trade_date"] = px.trade_date.astype(str)
    px = px.sort_values(["ts_code", "trade_date"])
    g = px.groupby("ts_code")
    nxt = g.open.shift(-1)
    hi = g.high.transform(lambda s: s.rolling(5, min_periods=5).max().shift(-5))
    lo = g.low.transform(lambda s: s.rolling(5, min_periods=5).min().shift(-5))
    up, dn = hi / nxt - 1, lo / nxt - 1
    past = g.close.pct_change(5)
    px["start_up"] = ((up >= 0.10) & (dn >= -0.05)).astype(float)
    px["start_dn"] = ((dn <= -0.10) & (up <= 0.05) & (past <= 0.08)).astype(float)
    px.loc[hi.isna(), ["start_up", "start_dn"]] = np.nan   # prices stop before the reserved window: last 5 days unlabelled
    return px[["ts_code", "trade_date", "start_up", "start_dn"]]


def apply_rule(survivors: pd.DataFrame) -> pd.DataFrame:
    """S0012: drop ratio > 5 among the funnel survivors, top 20 by stage1, refill from the dropped if short."""
    rows = []
    for _, g in survivors.groupby("trade_date"):
        g = g.sort_values("stage1_probability", ascending=False)
        drop = g.ratio > RATIO_CUT                     # missing ratio -> not dropped
        kept = g[~drop].head(TOP_K)
        if len(kept) < TOP_K:
            kept = pd.concat([kept, g[drop].head(TOP_K - len(kept))])
        rows.append(kept.assign(s0012_rank=np.arange(1, len(kept) + 1)))
    return pd.concat(rows, ignore_index=True)


def apply_pump_up(survivors: pd.DataFrame) -> pd.DataFrame:
    """S0014: drop the lowest third of P(up-start) among the funnel survivors, top 20 by stage1, refill if short."""
    rows = []
    for _, g in survivors.groupby("trade_date"):
        g = g.sort_values("stage1_probability", ascending=False)
        drop = g.pump_score.rank(pct=True) <= 1 / 3          # missing score -> NaN rank -> not dropped
        kept = g[~drop].head(TOP_K)
        if len(kept) < TOP_K:
            kept = pd.concat([kept, g[drop].head(TOP_K - len(kept))])
        rows.append(kept.assign(s0014_rank=np.arange(1, len(kept) + 1)))
    return pd.concat(rows, ignore_index=True)


def line(x: pd.DataFrame, label: str) -> str:
    return (f"{label:38s} n {len(x):5d} | 5d start-up {100 * x.start_up.mean():5.1f}% start-down {100 * x.start_dn.mean():5.1f}% "
            f"| 20d up {100 * x.up.mean():5.1f}% stop-first {100 * x.stop.mean():5.1f}% per trade {x.ret_v1.mean():+5.2f}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pump = pump_scores()
    first = pump.trade_date.min()
    lines = [f"pump scores usable: {pump.trade_date.nunique()} days ({first}..{pump.trade_date.max()})",
             "list basis, +15%/-10%/20d exit; descriptive on used windows (reused_holdout)", ""]

    # ---------------- Part A: used windows ----------------
    frozen = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    frozen = frozen[(frozen.trade_date >= first) & (frozen.trade_date < RESERVED_FROM)]
    survivors = select(frozen, PureConfig(top_k=100), "U15D10").merge(pump, on=["ts_code", "trade_date"], how="left")
    out = outcomes()
    out = out[out.trade_date < RESERVED_FROM][["ts_code", "trade_date", "ret_v1", "state"]]
    out["up"] = out.state.isin(["pure_up", "dirty_up"])
    out["stop"] = out.state == "pure_down"
    lab = five_day_labels(set(survivors.ts_code), first, RESERVED_FROM)
    s = survivors.merge(out, on=["ts_code", "trade_date"]).merge(lab, on=["ts_code", "trade_date"], how="left")
    s["seg"] = np.where(s.trade_date <= PUMP_VALID_END, "pump validation", "pump out-of-sample")
    lines.append(f"S20 funnel survivors with a pump score: {s.ratio.notna().mean():.1%} of {len(s)} rows, {s.trade_date.nunique()} days")
    lines.append(f"ratio in survivors: quantiles {s.ratio.quantile([.1, .5, .9, .99]).round(2).to_dict()}; share > {RATIO_CUT:g}: {(s.ratio > RATIO_CUT).mean():.1%}")
    lines += ["", "== A1. pump meaning inside the S20 survivors (terciles within the day) =="]
    for col in ("pump_score", "pump_down_score"):
        s["t"] = s.groupby("trade_date")[col].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["low", "mid", "high"]))
        for t, g in s.groupby("t", observed=True):
            lines.append(line(g, f"{col} {t}"))
    lines += ["", "== A2. ratio bands inside the survivors =="]
    s["band"] = pd.cut(s.ratio, [-np.inf, 1, 2, 5, np.inf], labels=["<=1", "1-2", "2-5", ">5"])
    for (seg, b), g in s.groupby(["seg", "band"], observed=True):
        lines.append(line(g, f"{seg:18s} ratio {b}"))
    lines += ["", f"== A3. S0012 (drop ratio>{RATIO_CUT:g}, refill to {TOP_K}) against the frozen list, same day =="]
    base = s[s.list_rank <= TOP_K]
    rule = apply_rule(s)
    for seg in ("pump validation", "pump out-of-sample", "all"):
        b = base if seg == "all" else base[base.seg == seg]
        r = rule if seg == "all" else rule[rule.seg == seg]
        lines.append(line(b, f"[{seg}] frozen list"))
        lines.append(line(r, f"[{seg}] S0012 list"))
        removed = b[~b.set_index(["ts_code", "trade_date"]).index.isin(r.set_index(["ts_code", "trade_date"]).index)]
        lines.append(line(removed, f"[{seg}] names S0012 removes"))
        for metric in ("stop", "ret_v1"):
            d = (r.groupby("trade_date")[metric].mean() - b.groupby("trade_date")[metric].mean()).dropna()
            if len(d) > 5:
                lines.append(f"    same-day diff {metric}: {d.mean():+.4f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)}, days with a change {(d != 0).sum()})")
        lines.append("")

    lines += [f"== A4. S0014 (drop the lowest third of P(up) among survivors, refill to {TOP_K}) against the frozen list, same day =="]
    rule14 = apply_pump_up(s)
    for seg in ("pump validation", "pump out-of-sample", "all"):
        b = base if seg == "all" else base[base.seg == seg]
        r = rule14 if seg == "all" else rule14[rule14.seg == seg]
        lines.append(line(b, f"[{seg}] frozen list"))
        lines.append(line(r, f"[{seg}] S0014 list"))
        key_b, key_r = b.set_index(["ts_code", "trade_date"]).index, r.set_index(["ts_code", "trade_date"]).index
        lines.append(line(b[~key_b.isin(key_r)], f"[{seg}] names S0014 removes"))
        lines.append(line(r[~key_r.isin(key_b)], f"[{seg}] names S0014 adds"))
        for metric in ("ret_v1", "stop", "up"):
            d = (r.groupby("trade_date")[metric].mean() - b.groupby("trade_date")[metric].mean()).dropna()
            if len(d) > 5:
                lines.append(f"    same-day diff {metric}: {d.mean():+.4f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)})")
        lines.append("")

    # ---------------- Part B: reserved days, lists only ----------------
    # full funnel pool from s20_shadow_funnel_survivors.py (daily_lists_raw.csv keeps only the top 20)
    raw = pd.read_parquet(ROOT / "output/experiments/s20_pure_v1_shadow/funnel_pool.parquet")
    raw = raw[(raw.trade_date >= RESERVED_FROM) & raw.survives_cut40].copy()
    raw["list_rank"] = raw.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    res = raw.merge(pump, on=["ts_code", "trade_date"], how="left")
    lists = apply_rule(res)
    cols = ["trade_date", "s0012_rank", "ts_code", "name", "industry", "stage1_probability", "list_rank", "pump_score", "pump_down_score", "ratio"]
    lists[cols].to_csv(OUT / "shadow_lists_S0012.csv", index=False, encoding="utf-8-sig")
    changed = lists.groupby("trade_date").apply(lambda g: (g.list_rank > TOP_K).sum())
    lines += ["== B. reserved days: S0012 lists written (no outcomes read) ==",
              f"days {lists.trade_date.nunique()} ({lists.trade_date.min()}..{lists.trade_date.max()}); pump coverage {res.ratio.notna().mean():.1%}; "
              f"names swapped versus the frozen list per day: mean {changed.mean():.1f}, days with any swap {(changed > 0).sum()}"]
    lists14 = apply_pump_up(res)
    cols14 = ["trade_date", "s0014_rank", "ts_code", "name", "industry", "stage1_probability", "list_rank", "pump_score", "pump_down_score", "ratio"]
    lists14[cols14].to_csv(OUT / "shadow_lists_S0014.csv", index=False, encoding="utf-8-sig")
    ch14 = lists14.groupby("trade_date").apply(lambda g: (g.list_rank > TOP_K).sum())
    lines.append(f"S0014 lists: days {lists14.trade_date.nunique()}; names swapped versus the frozen list per day: "
                 f"mean {ch14.mean():.1f}, min {ch14.min()}, max {ch14.max()}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
