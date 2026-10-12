#!/usr/bin/env python
"""Bottom patterns and the 0-100 bottom score for the day's S20 top 100 and R20 top 100 (latest day by default).

  python scripts/bottom_pattern_pools.py [--date YYYYMMDD]

S20 top 100: frozen stage1 on the shadow factor store (as scripts/daily_best_lists.py). R20 top 100: production
V12Scorer.score_market r20_pred. Patterns from scripts/bottom_patterns.py (definitions and weights frozen in
wiki/2026-10-12_bottom-pattern-lab.md). Research display only: the backtest found no pattern that works in
both periods inside these pools, so the score is context, not a buy signal. Not investment advice.
Output: output/experiments/bottom_patterns_20261012/latest_<date>.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bottom_patterns import FLAGS, pattern_panel  # noqa: E402

OUT = ROOT / "output/experiments/bottom_patterns_20261012"
NAMES = {"B1": "缠论底背离", "B2c": "W底突破", "B2f": "W底形成", "B3": "双插针+MA5拐头", "B4": "RSI底背离",
         "B5": "KDJ超卖金叉", "B6": "吞没/晨星", "B7": "缩量后放量"}


def s20_top100(day: str | None) -> tuple[str, pd.DataFrame]:
    from run_s20_pure_v1_shadow import SHADOW, load_factors
    from stockagent_analysis.s20_pure import _frozen_models, stage1_probability
    _, _, features = _frozen_models()
    f = load_factors(SHADOW / "factor_groups", features, "20260901")
    day = day or f.trade_date.max()
    f = f[f.trade_date == day].copy()
    f["own"] = stage1_probability(f)
    f["rank"] = f.own.rank(ascending=False, method="first")
    return day, f[f["rank"] <= 100][["ts_code", "own", "rank"]]


def r20_top100(day: str) -> pd.DataFrame:
    from export_pump_history import stale_groups
    from stockagent_analysis.v12_scoring import V12Scorer
    scorer = V12Scorer.get(ROOT)
    stale = stale_groups(scorer, day)
    if stale:
        print(f"warning: V12 feature groups without rows for {day}: {stale}")
    df = scorer.score_market(day)
    df["own"] = df.r20_pred
    df["rank"] = df.own.rank(ascending=False, method="first")
    return df[df["rank"] <= 100][["ts_code", "own", "rank"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    a = ap.parse_args()
    day, S = s20_top100(a.date)
    R = r20_top100(day)
    start = (pd.Timestamp(day) - pd.Timedelta(days=420)).strftime("%Y%m%d")
    pp = pattern_panel(start, day, out_from=day, codes=set(S.ts_code) | set(R.ts_code))
    pp = pp[pp.trade_date == day]
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name", "industry"]).drop_duplicates("ts_code")
    md = [f"# 底部形态：S20 前 100 / R20 前 100（{day}）", "",
          "形态定义与权重见 wiki/2026-10-12_bottom-pattern-lab.md。回测里没有一个形态在两段都有效，分数只作背景参考，不是买入信号。非投资建议。", ""]
    for nm, P, own in (("S20 前 100（冻结 stage1）", S, "stage1"), ("R20 前 100（r20 预测%）", R, "r20%")):
        q = P.merge(pp, on="ts_code", how="left").merge(basic, on="ts_code", how="left")
        q["形态"] = q.apply(lambda r: "、".join(NAMES[k] + ("（回踩再上，非首次突破）" if k == "B2c" and not bool(r.get("B2c_first", False)) else "")
                                              for k in FLAGS if bool(r.get(k, False))), axis=1)
        hit = q[q.bottom_score.fillna(0) > 0].sort_values(["bottom_score", "rank"], ascending=[False, True])
        md += [f"## {nm}：{len(hit)} 只命中至少一个形态；在底部区（距 60 日高点 ≥15%）{int(q.zone.fillna(False).sum())} 只", ""]
        md += ["| 池内名次 | 代码 | 名称 | 行业 | " + own + " | 底部分 | 底部区 | 形态 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in hit.iterrows():
            ov = f"{r.own:.3f}" if own == "stage1" else f"{r.own:+.2f}"
            md.append(f"| {r['rank']:.0f} | {r.ts_code} | {r['name']} | {r.industry} | {ov} | {r.bottom_score:.0f} | "
                      f"{'是' if r.zone else '否'} | {r['形态']} |")
        md += ["", "各形态命中数：" + "，".join(f"{NAMES[k]} {int(q[k].fillna(False).sum())}" for k in FLAGS), ""]
    both = sorted(set(S.ts_code) & set(R.ts_code))
    md += [f"两池重合：{', '.join(both) if both else '无'}"]
    text = "\n".join(md) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"latest_{day}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
