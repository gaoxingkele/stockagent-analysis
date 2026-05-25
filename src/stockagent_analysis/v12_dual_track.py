"""V12 双轨架构 (P3 调整版).

原计划 (plans/v12_consensus_integration_2026_05_20.md):
  A 60% V7c + R5 中期 / B 30% r1 T+1 短期 / C 10% 现金

调整版 (2026-05-21, r1 模型 ST 偏见证伪后):
  A 70% V7c + R5 反向严过滤  (Bot 15%, 集中 5-15 股, 中期 20 日)
  B 20% V7c + R5 反向宽过滤  (Bot 15-35%, 补足分散, 15-25 股, 中期 20 日)
  C 10% 现金缓冲

Why r1 被砍:
  - 长 OOS 142 日: r1_next_open_v3_long Top10 含 ST 月化+21% Sharpe 18.5
  - **但排除 ST 后** 月化-4.6% Sharpe -5.0
  - dist<3 过滤后, 多数日子 Top10 里 8-10 只 ST 股
  - 模型对低价小波动股 (ST 普遍低价) 有过拟合, alpha 不可实战

Why 同一 α 拆两轨:
  - 严过滤集中 → 弹性高 (单仓影响大), 适合高信念
  - 宽过滤分散 → 净值稳定 (Sharpe 高), 适合保底
  - 整体 Sharpe = √(70%² × Sharpe_A² + 20%² × Sharpe_B² + 相关系数项)
    若两轨高度相关 (同一信号), 集中度风险下降
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


def _apply_industry_cap(pool: pd.DataFrame, track_alloc: float,
                          cap_pct: float,
                          prior_industry_alloc: dict | None = None) -> pd.DataFrame:
    """行业 cap (支持跨轨累计): 行业累计 alloc (含 prior_industry_alloc) ≤ cap_pct.

    prior_industry_alloc: {industry: alloc} 字典, 表示之前已用掉的行业 alloc
                            (典型用法: A 轨先建, 算 A 行业 alloc; B 轨建时传入这个字典)
    """
    if pool.empty or cap_pct >= 1.0:
        return pool
    n = len(pool)
    per_stock = track_alloc / n
    pool = pool.sort_values("r5_long_rank_in_pool")
    industry_alloc = dict(prior_industry_alloc or {})
    keep_idx = []
    for idx, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        new_alloc = industry_alloc.get(ind, 0) + per_stock
        if new_alloc > cap_pct + 1e-9:
            continue
        industry_alloc[ind] = new_alloc
        keep_idx.append(idx)
    return pool.loc[keep_idx]


def _industry_alloc_dict(pool: pd.DataFrame, track_alloc: float) -> dict:
    """计算 pool 各行业累计 alloc (用最终 n 算 per_stock)."""
    if pool.empty: return {}
    n = len(pool)
    per_stock = track_alloc / n
    counts: dict[str, int] = {}
    for _, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        counts[ind] = counts.get(ind, 0) + 1
    return {ind: cnt * per_stock for ind, cnt in counts.items()}


def build_dual_track(df: pd.DataFrame,
                       track_a_pct: float = 0.70,
                       track_b_pct: float = 0.20,
                       cash_pct: float = 0.10,
                       a_bottom_pct: float = 1.0,   # v3: 默认不过滤 (用 max_a_stocks 控制)
                       b_bottom_pct: float = 1.0,
                       max_a_stocks: int = 8,    # v3: 调小 (V7c v3 池约 10-30 股)
                       max_b_stocks: int = 15,
                       industry_cap_per_track: float = 0.20,  # v5: 单轨内行业 cap
                       cross_track_industry_cap: float = 0.20,  # v6: 跨轨累计行业 cap (= cap_in 一致最佳)
                       sort_signal: str = "pump_score",  # v10: 'pump_score'(默认)/'r5_long_rank'
                       pump_down_excl_threshold: float = 0.60,  # v11: pump_down 绝对阈值 (v1 模型)
                       pump_down_excl_top_pct: float = 0.20  # v12: 排除 V7c 池内 P_down Top N% (抗模型分布漂移, 默认 20%)
                       ) -> dict:
    """构建双轨持仓清单 + 仓位.

    输入 df 必须含以下字段 (由 v12_scoring.score_market 提供):
      - ts_code, name, industry, buy_score, r5_long_pred,
        r5_long_rank_in_pool, v7c_recommend, v7c_recommend_diversified,
        r20_pred (用于 tie-break)

    返回:
      {
        "track_a": pd.DataFrame  (严过滤 Bot 15%, capped max_a_stocks),
        "track_b": pd.DataFrame  (宽过滤 Bot 15-35%, capped max_b_stocks),
        "summary": {
            "n_a": int, "n_b": int, "n_v7c": int,
            "alloc": {"a": pct, "b": pct, "cash": pct},
            "per_stock_a_pct": float, "per_stock_b_pct": float,
        }
      }
    """
    if abs(track_a_pct + track_b_pct + cash_pct - 1.0) > 1e-6:
        raise ValueError(f"仓位和必须 = 1.0 (a={track_a_pct} b={track_b_pct} c={cash_pct})")

    df = df.copy()
    n_v7c = int(df["v7c_recommend"].sum())

    # v11+v12: 跌启动子硬过滤
    # 优先用 v12 pct rank (推荐, 抗概率漂移); 兼容 v11 绝对阈值
    n_filtered_pump_down = 0
    if pump_down_excl_top_pct > 0 and "pump_down_rank_in_pool" in df.columns and \
       df["pump_down_rank_in_pool"].notna().any():
        # v12: V7c 池内 P_down Top N% 排除
        before_filter = df["v7c_recommend"].astype(bool).sum()
        cutoff = 1 - pump_down_excl_top_pct   # rank > cutoff = Top N%
        df.loc[(df["v7c_recommend"].astype(bool)) &
                 (df["pump_down_rank_in_pool"] > cutoff), "v7c_recommend"] = False
        after_filter = df["v7c_recommend"].astype(bool).sum()
        n_filtered_pump_down = before_filter - after_filter
    elif pump_down_excl_threshold < 1.0 and "pump_down_score" in df.columns:
        # v11 兼容: 绝对阈值
        before_filter = df["v7c_recommend"].astype(bool).sum()
        df.loc[df["pump_down_score"] >= pump_down_excl_threshold, "v7c_recommend"] = False
        after_filter = df["v7c_recommend"].astype(bool).sum()
        n_filtered_pump_down = before_filter - after_filter

    # v10: 按 sort_signal 选择排序键
    # pump_score: 高=好 (启动子概率 0-1), 降序选 Top
    # r5_long_rank: 低=好 (短期超跌反向), 升序选 Bot
    if sort_signal == "pump_score" and "pump_score_rank_in_pool" in df.columns:
        rank_col = "pump_score_rank_in_pool"
        ascending = False   # pump 高的优先
    else:
        rank_col = "r5_long_rank_in_pool"
        ascending = True    # r5 低的优先 (反向)

    # V7c 主推荐 + 有排序键的股
    pool = df[
        df["v7c_recommend"].astype(bool)
        & df[rank_col].notna()
    ].sort_values(rank_col, ascending=ascending).copy()
    # 同时填 r5_long_rank_in_pool 作为辅助列 (兼容 _apply_industry_cap 用)
    if "r5_long_rank_in_pool" not in pool.columns:
        pool["r5_long_rank_in_pool"] = pool[rank_col]
    elif sort_signal == "pump_score":
        # 用 pump_score_rank 重新作为 cap 排序键 (越高越优先保留)
        pool["r5_long_rank_in_pool"] = 1 - pool["pump_score_rank_in_pool"]  # 反向, cap 内仍用升序逻辑

    if len(pool) == 0:
        empty = pd.DataFrame(columns=df.columns)
        return {"track_a": empty, "track_b": empty,
                 "summary": {"n_a": 0, "n_b": 0, "n_v7c": n_v7c,
                              "alloc": {"a": track_a_pct, "b": track_b_pct, "cash": cash_pct},
                              "per_stock_a_pct": 0.0, "per_stock_b_pct": 0.0,
                              "sort_signal": sort_signal}}

    # Track A: 排序后前 max_a_stocks 个 (pump 高 或 r5 低)
    a_pool = pool.head(max_a_stocks).copy()
    # 注: a_bottom_pct/b_bottom_pct 参数保留作为可选硬上限 (默认不强制, 仅 r5 模式生效)
    if a_bottom_pct < 1.0 and sort_signal != "pump_score":
        a_pool = a_pool[a_pool["r5_long_rank_in_pool"] < a_bottom_pct]

    # v5: A 轨内行业 cap (默认 ≤ 20% 资金, 避免单行业集中)
    # 长 OOS 验证: cap=0.20 让最差月从 -0.89pp 改善到 -0.07pp, α +0.20pp Sharpe +0.24
    if industry_cap_per_track is not None and industry_cap_per_track < 1.0 and len(a_pool) > 0:
        a_pool = _apply_industry_cap(a_pool, track_a_pct, industry_cap_per_track)

    # v6: 算 A 最终各行业 alloc, 用于 B 轨跨轨 cap
    a_industry_alloc = _industry_alloc_dict(a_pool, track_a_pct)

    # Track B: 从 pool 中 A 已选剩余的股 (与 A 不重叠)
    b_candidates = pool[~pool["ts_code"].isin(a_pool["ts_code"])]
    b_pool = b_candidates.head(max_b_stocks).copy()
    if b_bottom_pct < 1.0 and sort_signal != "pump_score":
        b_pool = b_pool[b_pool["r5_long_rank_in_pool"] < b_bottom_pct]

    # v5+v6: B 轨先轨内 cap, 然后跨轨 cap (用 A 已占的行业 alloc)
    if industry_cap_per_track is not None and industry_cap_per_track < 1.0 and len(b_pool) > 0:
        b_pool = _apply_industry_cap(b_pool, track_b_pct, industry_cap_per_track)
    # v6 跨轨: A_industry_alloc + B_increment ≤ cross_track_industry_cap
    # 5/15 实测: 化工原料 A 0.23 + B 0.11 = 0.35 > 0.30 → 触发 cap, 砍 B 化工原料股
    if cross_track_industry_cap is not None and cross_track_industry_cap < 1.0 and len(b_pool) > 0:
        b_pool = _apply_industry_cap(b_pool, track_b_pct, cross_track_industry_cap,
                                       prior_industry_alloc=a_industry_alloc)

    n_a = len(a_pool)
    n_b = len(b_pool)
    per_a = track_a_pct / n_a if n_a > 0 else 0.0
    per_b = track_b_pct / n_b if n_b > 0 else 0.0

    a_pool["track"] = "A_concentrated"
    a_pool["alloc_pct"] = per_a
    b_pool["track"] = "B_diversified"
    b_pool["alloc_pct"] = per_b

    return {
        "track_a": a_pool,
        "track_b": b_pool,
        "summary": {
            "n_a": n_a, "n_b": n_b, "n_v7c": n_v7c,
            "alloc": {"a": track_a_pct, "b": track_b_pct, "cash": cash_pct},
            "per_stock_a_pct": per_a, "per_stock_b_pct": per_b,
            "a_bottom_pct": a_bottom_pct, "b_bottom_pct": b_bottom_pct,
            "sort_signal": sort_signal,
            "pump_down_excl_threshold": pump_down_excl_threshold,
            "pump_down_excl_top_pct": pump_down_excl_top_pct,
            "n_filtered_pump_down": int(n_filtered_pump_down),
        },
    }


def render_dual_track_md(result: dict, trade_date: str) -> str:
    """渲染 markdown 报告."""
    s = result["summary"]
    a = result["track_a"]
    b = result["track_b"]
    lines = [
        f"# V12 双轨持仓清单 ({trade_date})\n\n",
        f"V7c 主推荐总数: {s['n_v7c']}\n",
        f"轨 A 集中 (Bot {s['a_bottom_pct']*100:.0f}%): {s['n_a']} 股, 资金 {s['alloc']['a']*100:.0f}%, "
        f"单仓 {s['per_stock_a_pct']*100:.2f}%\n",
        f"轨 B 分散 (Bot {s['a_bottom_pct']*100:.0f}-{s['b_bottom_pct']*100:.0f}%): {s['n_b']} 股, "
        f"资金 {s['alloc']['b']*100:.0f}%, 单仓 {s['per_stock_b_pct']*100:.2f}%\n",
        f"现金缓冲: {s['alloc']['cash']*100:.0f}%\n\n",
        "## 轨 A 集中持仓 (高信念, 重仓 R5 极低分)\n\n",
        "| # | 代码 | 名称 | 行业 | buy | r5_long_rank | 单仓 % |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for i, (_, r) in enumerate(a.iterrows()):
        nm = r.get("name", "") if pd.notna(r.get("name", "")) else ""
        ind = r.get("industry", "") if pd.notna(r.get("industry", "")) else ""
        lines.append(f"| {i+1} | {r['ts_code']} | {nm} | {ind} | "
                      f"{r['buy_score']:.1f} | {r['r5_long_rank_in_pool']:.3f} | "
                      f"{r['alloc_pct']*100:.2f} |\n")

    lines.append("\n## 轨 B 分散持仓 (稳健补足)\n\n")
    lines.append("| # | 代码 | 名称 | 行业 | buy | r5_long_rank | 单仓 % |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for i, (_, r) in enumerate(b.iterrows()):
        nm = r.get("name", "") if pd.notna(r.get("name", "")) else ""
        ind = r.get("industry", "") if pd.notna(r.get("industry", "")) else ""
        lines.append(f"| {i+1} | {r['ts_code']} | {nm} | {ind} | "
                      f"{r['buy_score']:.1f} | {r['r5_long_rank_in_pool']:.3f} | "
                      f"{r['alloc_pct']*100:.2f} |\n")

    lines.append(f"\n## 风险提示\n\n")
    lines.append(f"- 长 OOS 142 日: R5 反向 d5_bot30 (排除 ST) Sharpe 2.67 月化 +2.43%\n")
    lines.append(f"- V7c 主推荐 (长 OOS, 已上线) 月化 α +4.82pp\n")
    lines.append(f"- 双轨预期月化: 70% × (4.82 + 2.4) + 20% × (4.82 + 1.5) ≈ +6.3pp/月\n")
    lines.append(f"- 实盘 25% 折扣后: 月化 ~+4.7pp (保守估计)\n")
    return "".join(lines)
