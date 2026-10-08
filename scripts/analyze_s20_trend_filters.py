#!/usr/bin/env python
"""Trend-state scissors on the frozen S20 list: moving-average direction and MACD golden-cross age.

Asked for by the user on 2026-10-05. For each name on the signal day:
  * MA(10/20/30/60/99) of the adjusted close is "rising" when it is above its
    own value five sessions earlier, otherwise "falling";
  * MACD(12, 26, 9): when DIF is above DEA the name is in a golden state and its
    age is the number of sessions since DIF crossed above; otherwise it is in a
    dead state.
Everything is known at the signal-day close. Closes are adjusted with the
cumulative daily percentage change so dividends do not bend the averages.

Descriptive only: both windows are used data and this family of questions has
been compared many times there. The reserved window is not read.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_scissors import FROZEN, build, funnel, stat  # noqa: E402
from experiment_cross_filter_v2 import RESERVED_FROM, nw_se  # noqa: E402

OUT = ROOT / "output/experiments/s20_trend_filters_20261005"
MAS = (10, 20, 30, 60, 99)
SLOPE_LAG = 5
AGE_BINS = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 9999)]


def trend_states() -> pd.DataFrame:
    daily = ROOT / "output/tushare_cache/daily"
    px = pd.concat([pd.read_parquet(path, columns=["ts_code", "trade_date", "pct_chg"])
                    for path in sorted(daily.glob("*.parquet")) if path.stem < RESERVED_FROM])
    px["trade_date"] = px["trade_date"].astype(str)
    px["ts_code"] = px["ts_code"].astype(str)
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    by = px.ts_code
    px["adj"] = np.exp(np.log1p(px.pct_chg.fillna(0.0) / 100.0).groupby(by).cumsum())
    for n in MAS:
        ma = px.adj.groupby(by).transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
        px[f"ma{n}_up"] = (ma > ma.groupby(by).shift(SLOPE_LAG)).astype(float).where(ma.groupby(by).shift(SLOPE_LAG).notna())
    ema = lambda s, span: s.groupby(by).transform(lambda x: x.ewm(span=span, adjust=False).mean())  # noqa: E731
    dif = ema(px.adj, 12) - ema(px.adj, 26)
    dea = ema(dif, 9)
    gold = dif > dea
    flip = gold != gold.groupby(by).shift()
    run = flip.groupby(by).cumsum()                       # id of the current golden or dead stretch
    px["macd_age"] = px.groupby([by, run]).cumcount().astype(float)     # 0 on the day of the cross
    px["macd_gold"] = gold.astype(float)                                # 1.0 golden, 0.0 dead
    seen = px.groupby(by).cumcount()
    px.loc[seen < 60, ["macd_age", "macd_gold"]] = np.nan  # let the averages settle
    return px[["ts_code", "trade_date", *[f"ma{n}_up" for n in MAS], "macd_gold", "macd_age"]]


def rule_pick(cand: pd.DataFrame, ok: pd.Series, keep: int = 20) -> pd.DataFrame:
    """From the frozen list's first 50 places, take names that pass first, in stage1 order, then fill."""
    q = cand.assign(_bad=~ok.fillna(False).astype(bool))
    return q.sort_values(["trade_date", "_bad", "list_rank"]).groupby("trade_date").head(keep)


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str = "ret_v1") -> tuple[float, float, int]:
    d = (a.groupby("trade_date")[col].mean() - b.groupby("trade_date")[col].mean()).dropna()
    se = nw_se(d.to_numpy())
    return float(d.mean()), float(d.mean() / se) if se > 0 else float("nan"), len(d)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = build().merge(trend_states(), on=["ts_code", "trade_date"], how="left")
    lines = [
        "Trend-state scissors on the frozen S20 list (pool 100, cut 40% by NATR, keep 20).",
        f"MA rising = above its value {SLOPE_LAG} sessions earlier. MACD(12,26,9); age 0 = the day DIF crossed above DEA.",
        "ret = +15%/-10% exit return per listed name, list basis. dn7 = low touches -7% within 5 sessions of entry.",
        "Descriptive: both windows are used data.",
        "",
    ]
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for period in ("dev", "confirm"):
        part = f[f.period == period]
        frozen = funnel(part, FROZEN[0], FROZEN[1])
        base20 = frozen[frozen.list_rank <= 20]
        first50 = frozen[frozen.list_rank <= 50]
        base_day = base20.groupby("trade_date").ret_v1.mean()
        lines.append(f"===== {period}: {part.trade_date.nunique()} days; frozen list {stat(base20)} =====")

        lines.append("-- 1. inside the frozen first 50 places: rising against falling, one average at a time --")
        for n in MAS:
            up, dn = first50[first50[f"ma{n}_up"] == True], first50[first50[f"ma{n}_up"] == False]  # noqa: E712
            d, t, days = paired(up, dn)
            d7, t7, _ = paired(up, dn, "y_dn7")
            lines.append(f"MA{n:<2d} rising {100 * len(up) / len(first50):4.1f}% of names | rising: {stat(up)}")
            lines.append(f"     falling                    | falling: {stat(dn)}")
            lines.append(f"     same-day rising minus falling: ret {d:+.2f} (t {t:+.2f}, days {days}); dn7 {100 * d7:+.1f} pts (t {t7:+.2f})")
        lines.append("")

        lines.append("-- 2. as a rule: from the first 50 places keep 20, names that pass first (stage1 order), then fill --")
        rules: dict[str, pd.Series] = {}
        for n in MAS:
            rules[f"MA{n} rising"] = first50[f"ma{n}_up"] == True    # noqa: E712
            rules[f"MA{n} falling"] = first50[f"ma{n}_up"] == False  # noqa: E712
        for a, b in itertools.combinations(MAS, 2):
            for sa, sb in itertools.product((True, False), repeat=2):
                name = f"MA{a} {'rising' if sa else 'falling'} & MA{b} {'rising' if sb else 'falling'}"
                rules[name] = (first50[f"ma{a}_up"] == sa) & (first50[f"ma{b}_up"] == sb)
        for lo, hi in AGE_BINS:
            label = f"MACD golden, age {lo}-{hi}" if hi < 9999 else f"MACD golden, age {lo}+"
            rules[label] = (first50.macd_gold == True) & first50.macd_age.between(lo, hi)  # noqa: E712
        rules["MACD golden (any age)"] = first50.macd_gold == True   # noqa: E712
        rules["MACD dead"] = first50.macd_gold == False              # noqa: E712
        for name, ok in rules.items():
            picked = rule_pick(first50, ok)
            passed = ok.fillna(False).astype(bool)
            share = 100 * passed.loc[picked.index].mean()
            day = picked.groupby("trade_date").ret_v1.mean()
            d = (day - base_day.reindex(day.index)).dropna()
            t = d.mean() / nw_se(d.to_numpy())
            summary.setdefault(name, {})[period] = (float(d.mean()), float(t))
            lines.append(f"{name:36s} pass {100 * passed.mean():4.1f}% of first 50, {share:5.1f}% of kept | {stat(picked, base_day)}")
        lines.append("")

        lines.append("-- 3. MACD state inside the frozen first 50 places --")
        for lo, hi in AGE_BINS:
            q = first50[(first50.macd_gold == True) & first50.macd_age.between(lo, hi)]  # noqa: E712
            lines.append(f"golden, {lo:2d}-{hi if hi < 9999 else '..':>4} sessions since the cross: n/day {len(q) / part.trade_date.nunique():4.1f} | {stat(q)}")
        dead = first50[first50.macd_gold == False]  # noqa: E712
        lines.append(f"dead (DIF below DEA)                   : n/day {len(dead) / part.trade_date.nunique():4.1f} | {stat(dead)}")
        for lo, hi in AGE_BINS:
            q = dead[dead.macd_age.between(lo, hi)]
            lines.append(f"  dead, {lo:2d}-{hi if hi < 9999 else '..':>4} sessions since the dead cross: n/day {len(q) / part.trade_date.nunique():4.1f} | {stat(q)}")
        lines.append("")

    lines.append("== rules whose difference against the frozen list has the same sign in both windows ==")
    both = [(n, v) for n, v in summary.items() if len(v) == 2 and v["dev"][0] * v["confirm"][0] > 0]
    for name, v in sorted(both, key=lambda x: -min(abs(x[1]["dev"][1]), abs(x[1]["confirm"][1]))):
        lines.append(f"{name:36s} dev {v['dev'][0]:+.2f} (t {v['dev'][1]:+.2f}) | confirm {v['confirm'][0]:+.2f} (t {v['confirm'][1]:+.2f})")
    lines.append(f"rules looked at per window: {len(summary)}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
