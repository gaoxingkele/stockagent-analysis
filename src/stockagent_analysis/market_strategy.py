# -*- coding: utf-8 -*-
"""A股市场策略框架 — 三阶段策略映射。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MarketStrategy:
    """市场策略建议。"""
    phase: str          # "offensive" / "balanced" / "defensive"
    phase_cn: str       # "进攻" / "平衡" / "防守"
    position_cap: float  # 建议最大仓位 (0.0-1.0)
    sector_bias: str    # 行业偏好建议
    risk_note: str      # 风险提示


STRATEGY_MAP = {
    "offensive": MarketStrategy(
        phase="offensive", phase_cn="进攻",
        position_cap=0.9,
        sector_bias="偏好成长股、科技、新能源，可适当追涨强势板块",
        risk_note="注意阶段性顶部信号，设好止盈线",
    ),
    "balanced": MarketStrategy(
        phase="balanced", phase_cn="平衡",
        position_cap=0.6,
        sector_bias="均衡配置，兼顾价值与成长，关注低位补涨板块",
        risk_note="控制仓位，避免单一板块过度集中",
    ),
    "defensive": MarketStrategy(
        phase="defensive", phase_cn="防守",
        position_cap=0.3,
        sector_bias="偏好高股息、消费、公用事业等防御板块",
        risk_note="严格止损，轻仓观望为主",
    ),
}


def determine_strategy(regime: dict[str, Any]) -> MarketStrategy:
    """基于市场状态判定策略阶段。"""
    regime_name = regime.get("regime", "unknown")
    vol_20d = float(regime.get("index_vol_20d", 2))

    if regime_name == "bull" and vol_20d < 2.5:
        return STRATEGY_MAP["offensive"]
    elif regime_name == "bear" or vol_20d > 3.5:
        return STRATEGY_MAP["defensive"]
    else:
        return STRATEGY_MAP["balanced"]


def strategy_to_dict(strategy: MarketStrategy) -> dict[str, Any]:
    """序列化为JSON可存储格式。"""
    return {
        "phase": strategy.phase,
        "phase_cn": strategy.phase_cn,
        "position_cap": strategy.position_cap,
        "sector_bias": strategy.sector_bias,
        "risk_note": strategy.risk_note,
    }
