"""6 实战池 + 1 反思池 分类器 (Sprint 1.4, V12.11).

每股按优先级分配到唯一池子:
  优先级: 池2 > 池1 > 池4 > 池5 > 池6 > 池3 > None

实战池:
  池1 V7c 主推 (核心 50%) — 整理后突破
  池2 三引擎共识 (重点 15%) — V7c + 政策受益
  池3 超跌反弹 (短线 5%) — RSI6<30 + MA60 偏离 + 量放
  池4 底部突破 (左侧 10%) — 历史 zombie + MA60↑ + 突破新高
  池5 政策风口 (主题 10%) — policy_heat≥70 + 板块涨幅
  池6 强势股回调 (接力 10%) — 60日涨 + 5日缩量回调

反思池 (与实战池正交):
  池0 看走眼 — 历史 V12 误判, 由 review_pool_builder 单独维护
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# 池定义
POOL_CONFIG = {
    "pool1_v7c_main": {
        "name": "V7c 主推", "desc": "整理后突破", "target_share": 0.50,
        "max_pos_per_stock": 0.025, "holding_days": 20,
    },
    "pool2_triple_consensus": {
        "name": "三引擎共识", "desc": "V7c 主推 + 政策受益", "target_share": 0.15,
        "max_pos_per_stock": 0.05, "holding_days": 45,
    },
    "pool3_oversold_rebound": {
        "name": "超跌反弹", "desc": "短期反弹", "target_share": 0.05,
        "max_pos_per_stock": 0.015, "holding_days": 8,
    },
    "pool4_bottom_breakout": {
        "name": "底部突破", "desc": "W/U 底突破", "target_share": 0.10,
        "max_pos_per_stock": 0.025, "holding_days": 30,
    },
    "pool5_policy_wave": {
        "name": "政策风口", "desc": "板块爆发", "target_share": 0.10,
        "max_pos_per_stock": 0.015, "holding_days": 15,
    },
    "pool6_strong_pullback": {
        "name": "强势股回调", "desc": "龙头休整", "target_share": 0.10,
        "max_pos_per_stock": 0.02, "holding_days": 12,
    },
}


def _safe_get(row: pd.Series, key: str, default=0.0):
    v = row.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def assign_pool(row: pd.Series, percentiles: Optional[dict] = None) -> Optional[str]:
    """单股分配池子, 按优先级返回首个匹配池.

    Args:
      row: V12Scorer.score_market() df 的一行
      percentiles: 分位阈值 (如 pyr_velocity_20_60 的 p35), 可选
    """
    # 池 2: 三引擎共识 (优先级最高)
    if row.get("v7c_recommend") and _safe_get(row, "policy_heat_score") >= 70:
        return "pool2_triple_consensus"

    # 池 1: V7c 主推
    if row.get("v7c_recommend"):
        return "pool1_v7c_main"

    # 池 4: 底部突破 (W/U 底)
    # 判据: 过去 60 日 zombie_days_pct > 0.6 (近 20 日内曾横盘) + MA60 上行 + 突破前高
    # 简化: 当前 zombie_days_pct > 0.5 (横盘中) + ma60_slope_short > 0.003 (开始上行)
    if (_safe_get(row, "zombie_days_pct") > 0.5
        and _safe_get(row, "ma60_slope_short") > 0.003
        and _safe_get(row, "buy_score") > 60
        and not row.get("is_zombie", False)):
        return "pool4_bottom_breakout"

    # 池 5: 政策风口
    if (_safe_get(row, "policy_heat_score") >= 70
        and _safe_get(row, "buy_score") > 60):
        return "pool5_policy_wave"

    # 池 6: 强势股回调 (买分≥85 表示强势, 排除 ST/极端 100 + 排除矛盾段)
    name = str(row.get("name") or "")
    is_st = "ST" in name.upper() or "*ST" in name
    buy = _safe_get(row, "buy_score")
    if (85 <= buy < 95
        and not is_st
        and row.get("quadrant") not in ("矛盾段", "主流空")
        and not row.get("is_zombie", False)):
        return "pool6_strong_pullback"

    # 池 3: 超跌反弹 (短线, 严格判据)
    # r10_pred 高 + r20_pred 不超强 (短期机会 vs 长期突破)
    # buy ∈ [50, 75] (不极端) + sell < 50 (避雷干净)
    if (_safe_get(row, "r10_pred") > 2.0
        and 50 <= buy < 75
        and _safe_get(row, "sell_score") < 50
        and not row.get("is_zombie", False)
        and not is_st):
        return "pool3_oversold_rebound"

    return None


def assign_all(df: pd.DataFrame) -> pd.DataFrame:
    """对全市场 df 加 pool 字段."""
    df = df.copy()
    df["pool"] = df.apply(assign_pool, axis=1)
    return df


def pool_summary(df: pd.DataFrame) -> dict:
    """池子分布统计."""
    cnt = df["pool"].value_counts(dropna=True).to_dict()
    return {
        "total_in_pools": int(df["pool"].notna().sum()),
        "by_pool": {k: int(v) for k, v in cnt.items()},
        "configs": POOL_CONFIG,
    }


def filter_by_pool(df: pd.DataFrame, pool: str, top_n: Optional[int] = None) -> pd.DataFrame:
    """按池筛选, 可选取 top_n (按 r20_pred 降序)."""
    sub = df[df["pool"] == pool].copy()
    sub = sub.sort_values("r20_pred", ascending=False)
    if top_n: sub = sub.head(top_n)
    return sub
