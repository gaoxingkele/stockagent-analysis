#!/usr/bin/env python
"""The day's best recommendations of R20 (V12.31) and S20 (S20-Pure v1), with scores and pump ratio.

  python scripts/daily_best_lists.py [--date YYYYMMDD] [--no-flags]

R20: V12Scorer.score_market (production, unchanged) -> pool A (config/pool_a_r20_target_v1.json), the R20
     representative list (user 2026-10-10), and the V12.31 Top20 (V7c main pool by ratio, industry cap 4)
     shown for reference, as in export_r20_pool_a_history.py.
S20: frozen stage1 + funnel (config/s20_pure_v1.json) -> offensive list U15D10, and the safe list v1.1;
     valve (config/s20_pure_valve_v1.json) action B on red days. Scores on 0-100: market score (V12
     anchors) and pool score (rank in the stage1 top 100).
ratio = pump v3c P(up) / (P(down) + 0.01), the same number for both systems.
Shadow layers on the day (not live): style gate S0010, CSI300 phase S0007, risk flags S0016.
Needs the caches, regimes, S20 factor store and the V12 feature pipeline up to the day.
Output: output/experiments/daily_best/best_<date>.md. Not investment advice.
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
from export_pump_history import stale_groups  # noqa: E402
from export_r20_pool_a_history import pool_a, v12_top20  # noqa: E402
from run_s20_pure_v1_shadow import SHADOW, daily_natr, load_factors, valve_table  # noqa: E402
from score_s20_stock import DEFAULT_SEMAS, index_close, last_hour, semas_factors  # noqa: E402
from stockagent_analysis.market_valve import LEVEL_CN  # noqa: E402
from stockagent_analysis.s20_display import s20_pool_score, s20_score_100  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, SafeConfig, _frozen_models, select, select_safe, stage1_probability  # noqa: E402


def v12_market(day: str) -> tuple[pd.DataFrame, list[str]]:
    from stockagent_analysis.v12_scoring import V12Scorer
    scorer = V12Scorer.get(ROOT)
    stale = stale_groups(scorer, day)
    df = scorer.score_market(day)
    if df["pump_score"].isna().all():
        scorer.predict_one(df, "r5_pump_3way")
        df = scorer.apply_pump_3way(df)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[["ts_code", "name", "industry"]]
    for c in ("name", "industry"):
        if c not in df.columns:
            df = df.merge(basic[["ts_code", c]].drop_duplicates("ts_code"), on="ts_code", how="left")
    df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
    df["ratio_rank"] = df["ratio"].rank(ascending=False, method="min")
    return df, stale


def table(rows: pd.DataFrame, cols: list[tuple[str, str, str]]) -> list[str]:
    out = ["| " + " | ".join(h for h, _, _ in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in rows.iterrows():
        cells = []
        for _, c, fmt in cols:
            v = r.get(c, np.nan)
            cells.append("—" if (isinstance(v, float) and np.isnan(v)) or v is None else (fmt.format(v) if fmt else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--no-flags", action="store_true", help="skip the S0016 risk flags (they fetch 15-minute bars)")
    a = ap.parse_args()
    cfg, safe_cfg = PureConfig(), SafeConfig()
    _, _, features = _frozen_models()
    f = load_factors(SHADOW / "factor_groups", features, "20260901")
    day = a.date or f.trade_date.max()
    f = f[f.trade_date == day].copy()
    f["stage1_probability"] = stage1_probability(f)
    f["score_100"] = s20_score_100(f)
    f = f.merge(daily_natr("20260901"), on=["ts_code", "trade_date"], how="left")
    f["pool_rank_all"] = f.stage1_probability.rank(ascending=False, method="first")
    f["score_pool"] = s20_pool_score(f.pool_rank_all, cfg.pool_size)
    v12, stale = v12_market(day)
    pump = v12[["ts_code", "pump_score", "pump_down_score", "ratio", "ratio_rank"]]
    off = select(f, cfg, "U15D10").merge(pump, on="ts_code", how="left").sort_values("list_rank")
    safe = select_safe(f, safe_cfg).merge(pump, on="ts_code", how="left")
    A = pool_a(v12).reset_index(drop=True)
    A["rank"] = np.arange(1, len(A) + 1)
    V = v12_top20(v12).reset_index(drop=True)
    V["rank"] = np.arange(1, len(V) + 1)

    hs, cy = index_close("000300.SH"), index_close("399006.SZ")
    r20h, r60h = hs.loc[day] / hs.shift(20).loc[day] - 1, hs.loc[day] / hs.shift(60).loc[day] - 1
    r20c = cy.loc[day] / cy.shift(20).loc[day] - 1
    phase = "上涨" if (r20h > 0 and r60h > 0) else ("下行" if (r20h < 0 and r60h < 0) else "转折")
    vt = valve_table().set_index("date")
    level = vt.loc[day, "level"] if day in vt.index else "unknown"

    flags_note = ""
    if not a.no_flags:
        surv = select(f, PureConfig(top_k=cfg.pool_size), "U15D10")
        try:
            lh = last_hour(list(surv.ts_code), day).rank(pct=True)
            sf = semas_factors(day, DEFAULT_SEMAS)
            sf = sf[sf.ts_code.isin(surv.ts_code)].set_index("ts_code")
            off["flag_lh"] = off.ts_code.map(lh) > 0.8
            off["flag_f01"] = off.ts_code.map(sf.F01.rank(pct=True)) > 2 / 3
            off["flag_f20"] = off.ts_code.map(sf.F20.rank(pct=True)) <= 1 / 3
            off["flags"] = off[["flag_lh", "flag_f01", "flag_f20"]].sum(axis=1).astype(int)
            off["risk"] = np.where(off["flags"] >= 2, "减半·后买", "")
        except Exception as err:  # noqa: BLE001
            flags_note = f"（风险旗未算出：{err}）"

    md = [f"# R20 / S20 最佳推荐 {day}", "",
          "收盘后计算，次日开盘可执行。冻结合约与生产规则未改；影子层只作参考。非投资建议。", "",
          "## 当天盘面", "",
          f"- 阀门：**{LEVEL_CN.get(level, level)}**" + ("（动作 B：进攻版改用稳健版）" if level == "red" else ""),
          f"- S0010 风格门（主影子）：创业板 20 日 {100 * r20c:+.2f}% 对 沪深300 {100 * r20h:+.2f}% → **{'开' if r20c > r20h else '关（不建新仓）'}**",
          f"- S0007 三态：沪深300 20 日 {100 * r20h:+.2f}%、60 日 {100 * r60h:+.2f}% → **{phase}**",
          f"- V12 特征当天是否齐全：{'是' if not stale else '否，缺 ' + ', '.join(stale)}",
          "",
          "ratio = pump v3c P(涨)/(P(跌)+0.01)；括号内为全市场 ratio 名次。2～5 为池 A 历史上较好的一段，>5 为 P(跌)≈0 的分母放大段。", ""]
    md += [f"## R20（代表）· 池 A（{len(A)} 只；r20 预测或最大涨幅 ≥25% 且最大回撤 ≥ −15%）", ""]
    md += table(A.head(30), [("#", "rank", "{:.0f}"), ("代码", "ts_code", ""), ("名称", "name", ""), ("行业", "industry", ""),
                             ("买分", "buy_r20_score", "{:.1f}"), ("r20 预测%", "r20_pred", "{:+.2f}"),
                             ("最大涨%", "pred_max_gain_20", "{:+.1f}"), ("最大回撤%", "pred_max_dd_20", "{:+.1f}"),
                             ("ratio", "ratio", "{:.2f}"), ("ratio 名次", "ratio_rank", "{:.0f}")])
    md += ["", f"## 参考 · V12.31 Top20（{len(V)} 只；V7c 主推池按 ratio 排序，单行业 ≤4；历史上弱于池 A，仅作参考）", ""]
    md += table(V, [("#", "rank", "{:.0f}"), ("代码", "ts_code", ""), ("名称", "name", ""), ("行业", "industry", ""),
                    ("ratio", "ratio", "{:.2f}"), ("P(涨)", "pump_score", "{:.3f}"), ("P(跌)", "pump_down_score", "{:.3f}"),
                    ("买分", "buy_r20_score", "{:.1f}"), ("r20 预测%", "r20_pred", "{:+.2f}")])
    cols = [("#", "list_rank", "{:.0f}"), ("代码", "ts_code", ""), ("名称", "name", ""), ("行业", "industry", ""),
            ("池内分", "score_pool", "{:.0f}"), ("全市场分", "score_100", "{:.1f}"), ("stage1", "stage1_probability", "{:.3f}"),
            ("ratio", "ratio", "{:.2f}"), ("ratio 名次", "ratio_rank", "{:.0f}"), ("NATR", "natr14", "{:.3f}")]
    if "flags" in off:
        cols += [("风险旗", "flags", "{:.0f}"), ("S0016", "risk", "")]
    md += ["", f"## S20 · 进攻版 U15D10（前 100 → 剪 NATR 最高 40% → 前 20；+15%/−10%/20 日）{flags_note}", ""]
    md += table(off, cols)
    safe = safe.reset_index(drop=True)
    safe["n"] = np.arange(1, len(safe) + 1)
    md += ["", f"## S20 · 稳健版 v1.1（全市场波动最低 {int(safe_cfg.natr_pct_max * 100)}% → stage1 → 前 {safe_cfg.top_k}，单行业 ≤{safe_cfg.industry_cap}）", ""]
    md += table(safe, [("#", "n", "{:.0f}"), ("代码", "ts_code", ""), ("名称", "name", ""), ("行业", "industry", ""),
                       ("全市场分", "score_100", "{:.1f}"), ("stage1", "stage1_probability", "{:.3f}"),
                       ("ratio", "ratio", "{:.2f}"), ("ratio 名次", "ratio_rank", "{:.0f}")])
    both = sorted(set(off.ts_code) & set(A.ts_code))
    ref = sorted(set(off.ts_code) & set(V.ts_code))
    md += ["", "## 交集", "", f"- S20 进攻版 ∩ R20 池 A：{', '.join(both) if both else '无'}；∩ V12.31 Top20（参考）：{', '.join(ref) if ref else '无'}",
           f"- ratio 中位数：S20 进攻版 {off.ratio.median():.2f}，池 A {A.ratio.median() if len(A) else float('nan'):.2f}，"
           f"V12.31 Top20 {V.ratio.median() if len(V) else float('nan'):.2f}，全市场 {v12.ratio.median():.2f}"]
    text = "\n".join(md) + "\n"
    out = ROOT / "output/experiments/daily_best" / f"best_{day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
