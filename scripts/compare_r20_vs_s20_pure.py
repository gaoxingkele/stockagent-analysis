#!/usr/bin/env python
"""Head-to-head: production R20 lists vs S20-Pure lists, same days, same yardstick.

Input: the parquet written by scripts/export_r20_pool_a_history.py on the
production machine (copy it to output/r20_history/).

    python scripts/compare_r20_vs_s20_pure.py output/r20_history/r20_lists_20260414_20260928.parquet
    python scripts/compare_r20_vs_s20_pure.py --published      # lists production actually published
                                                             # (output/daily_pick/dashboard_*/poolA_system.csv)

Lists compared on the common matured signal days:
  R20 Pool A (all, variable length) | R20 Pool A Top20 | V12.31 Top20
  S20-Pure v1 | v1 + valve B | v1.1 safe | universe
Windows:
  fair        >= 20260414  both R20 and stage1 models are out of sample
  r20_insample < 20260414  R20 was trained on these days: reference only
Yardsticks (identical to s20_pure_history.score): band exit +5/+15/-10 (success,
bad, 15% drawdown, per-trade mean), the +15/-10 rule, pure-up share, book max DD.
Also: paired daily difference in per-trade return vs Pool A Top20 with a
month-block bootstrap 95% interval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s20_pure_history import OUT, history_lists, outcomes, score, universe_rows  # noqa: E402

FAIR_FROM = "20260414"


def paired_ci(a: pd.Series, b: pd.Series, n: int = 2000) -> tuple[float, float, float]:
    d = (a - b).dropna()
    if d.empty:
        return (np.nan, np.nan, np.nan)
    months = d.index.str[:6]
    uniq = np.unique(months)
    rng = np.random.default_rng(20260930)
    boots = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        boots.append(pd.concat([d[months == m] for m in pick]).mean())
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def published_lists() -> pd.DataFrame:
    """Pool A exactly as production published it each day (file order = rank)."""
    parts = []
    for d in sorted((ROOT / "output/daily_pick").glob("dashboard_*/poolA_system.csv")):
        f = pd.read_csv(d, dtype={"ts_code": str})
        if f.empty:
            continue
        f["trade_date"] = d.parent.name.split("_")[1]
        f["list"] = "pool_a"
        f["list_rank"] = range(1, len(f) + 1)
        parts.append(f[["trade_date", "ts_code", "list", "list_rank"]])
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    r20 = published_lists() if sys.argv[1] == "--published" else pd.read_parquet(sys.argv[1])
    r20["trade_date"] = r20.trade_date.astype(str)
    ours = history_lists()
    o = outcomes()
    days = sorted(set(r20.trade_date) & set(ours.trade_date) & set(o.trade_date))
    if not days:
        raise SystemExit("no common matured days")
    label = "生产池A(已发布)" if sys.argv[1] == "--published" else "R20 池A 全部"
    sets = {
        label: r20[r20.list == "pool_a"],
        "R20 池A 前20": r20[(r20.list == "pool_a") & (r20.list_rank <= 20)],
        "V12.31 前20": r20[r20.list == "v12_31_top20"],
        "S20 进攻版 v1": ours[ours.rule == "U15D10"],
        "S20 进攻版+阀门B": ours[ours.rule == "U15D10+B"],
        "S20 稳健版 v1.1": ours[ours.rule == "safe_v1_1"],
        "全市场": universe_rows(days),
    }
    rows, daily = [], {}
    for name, L in sets.items():
        q = L[L.trade_date.isin(days)][["trade_date", "ts_code"]].merge(o, on=["ts_code", "trade_date"])
        daily[name] = q.groupby("trade_date").ret_safe.mean()
        for win, g in (("fair", q[q.trade_date >= FAIR_FROM]), ("r20_insample", q[q.trade_date < FAIR_FROM]),
                       ("all", q)):
            if len(g):
                rows.append({"window": win, "list": name, **score(g)})
    T = pd.DataFrame(rows)
    lines = [f"common matured signal days: {len(days)} ({days[0]}..{days[-1]}); fair window from {FAIR_FROM}",
             "", T.to_string(index=False), "",
             "## paired per-trade difference vs the R20 reference list (band exit, fair window), month-block bootstrap 95% CI"]
    ref = "R20 池A 前20" if "R20 池A 前20" in daily and len(daily["R20 池A 前20"]) else label
    base = daily[ref]
    base = base[base.index >= FAIR_FROM]
    for name, s in daily.items():
        if name == ref or name not in daily or daily[name].empty:
            continue
        m, lo, hi = paired_ci(s[s.index >= FAIR_FROM], base)
        lines.append(f"{name:14s} mean diff {m:+.2f}pp  95% CI [{lo:+.2f}, {hi:+.2f}]")
    Mo = pd.DataFrame({k: v.groupby(v.index.str[:6]).mean() for k, v in daily.items()}).round(2)
    lines += ["", "## monthly per-trade mean (band exit, %)", Mo.to_string()]
    text = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "published" if sys.argv[1] == "--published" else "replay"
    (OUT / f"compare_r20_{tag}.txt").write_text(text, encoding="utf-8")
    T.to_csv(OUT / f"compare_r20_{tag}.csv", index=False, encoding="utf-8-sig")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
