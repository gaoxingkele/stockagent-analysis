"""仓位管理 (Sprint 1.2 + 1.3, V12.12).

两块功能:
  1. 仓位计算 (Kelly Criterion 动态): calc_position_size()
  2. 止盈止损规则: check_exit_signal()

入场判据 (Kelly Criterion 简化版):
  base_size = pool 单股上限 (POOL_CONFIG[pool]['max_pos_per_stock'])
  confidence = (r20_pred / 20) × (1 - sell_v6_prob)  # 0~1
  pos_size = base_size × confidence_multiplier  (clamp 到 [0.005, max])
  其中 multiplier = 0.5 + confidence (即 0.5~1.5x)

出场信号:
  1. 持有期满 (≥ pool 配置的 holding_days) → 全清
  2. 实际涨幅 ≥ r20_pred × 0.80 → 减半 (止盈)
  3. 实际跌幅 ≥ -8% → 清仓 (止损)
  4. 进入 zombie 状态 → 减半
  5. regime 触发减仓且仓位 > regime 建议 → 按比例减
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from .pool_classifier import POOL_CONFIG


# ──── 仓位计算 (Kelly) ────

def calc_position_size(row: pd.Series, total_portfolio: float = 1.0,
                       regime_position_ratio: float = 1.0) -> dict:
    """单股建议仓位 (0~max_pos_per_stock).

    Args:
      row: V12Scorer 输出的一行 (含 pool / r20_pred / sell_score / sell_20_v6_prob 等)
      total_portfolio: 总仓位比例 (默认 1.0, 满仓)
      regime_position_ratio: regime 减仓系数 (来自 regime_monitor)

    Returns:
      {"pool": pool_name, "base_size": x, "confidence": x, "raw_size": x,
       "final_size": x, "reason": "..."}
    """
    pool = row.get("pool")
    if not pool or pool not in POOL_CONFIG:
        return {"pool": pool, "final_size": 0.0,
                 "reason": "not in any pool"}

    cfg = POOL_CONFIG[pool]
    base = cfg["max_pos_per_stock"]

    # 置信度 = (r20_pred 归一化到 0-1, 上限 +15%) × (1 - sell_v6_prob)
    r20 = float(row.get("r20_pred", 0))
    r20_norm = max(0.0, min(1.0, r20 / 15.0))   # 15% 视为最高置信
    sell_prob = float(row.get("sell_20_v6_prob", 0.5))
    confidence = r20_norm * (1.0 - sell_prob)
    # multiplier: 0.5 + confidence (0.5~1.5x base)
    multiplier = 0.5 + confidence

    raw_size = base * multiplier * regime_position_ratio
    # clamp 到 [0.005, base * 1.5]
    final_size = max(0.005, min(base * 1.5, raw_size))

    return {
        "pool": pool,
        "pool_name": cfg["name"],
        "base_size": round(base, 4),
        "confidence": round(confidence, 3),
        "multiplier": round(multiplier, 3),
        "regime_adj": round(regime_position_ratio, 2),
        "raw_size": round(raw_size, 4),
        "final_size": round(final_size, 4),
        "reason": (f"{pool} base={base*100:.2f}% × "
                    f"conf={confidence:.2f} × regime={regime_position_ratio:.2f}"),
    }


def calc_positions_batch(df: pd.DataFrame, total_portfolio: float = 1.0,
                          regime_position_ratio: float = 1.0) -> pd.DataFrame:
    """对全 df 批量计算 position_size."""
    df = df.copy()
    sizes = df.apply(lambda r: calc_position_size(r, total_portfolio, regime_position_ratio), axis=1)
    df["position_size"] = sizes.apply(lambda s: s.get("final_size", 0.0))
    df["position_confidence"] = sizes.apply(lambda s: s.get("confidence", 0.0))
    return df


def build_portfolio(df: pd.DataFrame, total_capital: float = 1.0,
                     regime_position_ratio: float = 1.0,
                     per_pool_caps: Optional[dict] = None) -> pd.DataFrame:
    """组合层面分配仓位, 严格按 pool 占比和单股上限.

    Args:
      df: V12 score 含 pool 字段
      total_capital: 1.0 = 100% 可用资金
      regime_position_ratio: regime 减仓系数 (0.30 ~ 1.00)
      per_pool_caps: 池占比 override (默认按 POOL_CONFIG)

    Returns 选中的股票 + 实际仓位 (按 r20_pred 降序填到池满)
    """
    df = df[df["pool"].notna()].copy()
    if df.empty: return df

    out_rows = []
    for pool_id, cfg in POOL_CONFIG.items():
        pool_share = (per_pool_caps or {}).get(pool_id, cfg["target_share"])
        pool_budget = total_capital * pool_share * regime_position_ratio
        max_pos = cfg["max_pos_per_stock"] * regime_position_ratio
        sub = df[df["pool"] == pool_id].sort_values("r20_pred", ascending=False)
        budget_left = pool_budget
        for _, row in sub.iterrows():
            info = calc_position_size(row, total_capital, regime_position_ratio)
            size = min(info["final_size"], max_pos, budget_left)
            if size < 0.005:
                continue   # 太小不入仓
            r = row.to_dict()
            r["position_size"] = round(size, 4)
            r["pool_share"] = pool_share
            r["confidence"] = info["confidence"]
            out_rows.append(r)
            budget_left -= size
            if budget_left < 0.005: break
    return pd.DataFrame(out_rows)


# ──── 止盈止损 ────

def check_exit_signal(holding: dict, current: dict) -> dict:
    """单股出场判断.

    Args:
      holding: {entry_date, entry_price, pool, r20_pred_at_entry, position_size}
      current: {date, close, is_zombie, regime_position_ratio}

    Returns:
      {action: 'hold'/'reduce_half'/'exit_all', reason: str, new_size: x}
    """
    pool = holding.get("pool")
    if not pool or pool not in POOL_CONFIG:
        return {"action": "hold", "reason": "unknown pool"}
    cfg = POOL_CONFIG[pool]

    entry_price = holding.get("entry_price")
    current_price = current.get("close")
    if entry_price is None or current_price is None:
        return {"action": "hold", "reason": "no price"}
    ret_pct = (current_price / entry_price - 1) * 100

    # 1. 持有期满
    days_held = current.get("days_held", 0)
    if days_held >= cfg["holding_days"]:
        return {"action": "exit_all",
                 "reason": f"持有 {days_held} 日 ≥ {cfg['holding_days']}, 期满",
                 "new_size": 0.0, "exit_ret_pct": round(ret_pct, 2)}

    # 2. 止损 -8%
    if ret_pct <= -8.0:
        return {"action": "exit_all",
                 "reason": f"实际跌幅 {ret_pct:.2f}% ≤ -8%, 止损",
                 "new_size": 0.0, "exit_ret_pct": round(ret_pct, 2)}

    # 3. 止盈: 涨幅 ≥ r20_pred × 80% 减半
    r20_pred = holding.get("r20_pred_at_entry", 0)
    if r20_pred > 0 and ret_pct >= r20_pred * 0.80:
        return {"action": "reduce_half",
                 "reason": f"涨幅 {ret_pct:.2f}% ≥ r20_pred {r20_pred:.2f}% × 80%, 止盈减半",
                 "new_size": round(holding.get("position_size", 0) * 0.5, 4)}

    # 4. zombie 触发减半
    if current.get("is_zombie"):
        return {"action": "reduce_half",
                 "reason": "进入 zombie 区, 减半",
                 "new_size": round(holding.get("position_size", 0) * 0.5, 4)}

    # 5. regime 触发减仓 (持仓比例 > regime 建议时按比例减)
    regime_ratio = current.get("regime_position_ratio", 1.0)
    current_size = holding.get("position_size", 0)
    cfg_max = cfg["max_pos_per_stock"]
    target_size = cfg_max * regime_ratio
    if current_size > target_size * 1.2:
        return {"action": "reduce_to_regime",
                 "reason": f"regime 建议仓位 {regime_ratio*100:.0f}%, 按比例减",
                 "new_size": round(target_size, 4)}

    return {"action": "hold", "reason": "未触发出场",
             "ret_pct": round(ret_pct, 2)}


def evaluate_holdings(holdings_df: pd.DataFrame, current_df: pd.DataFrame,
                      current_date: str) -> pd.DataFrame:
    """批量评估持仓清单 (回测/实盘用).

    Args:
      holdings_df: 持仓快照 (entry_date, ts_code, entry_price, pool, r20_pred, position_size)
      current_df: 当日 V12 score (含 close, is_zombie)
      current_date: 当前日期

    Returns 持仓 + 出场动作
    """
    cur_map = current_df.set_index("ts_code").to_dict("index")
    out = []
    for _, h in holdings_df.iterrows():
        ts = h["ts_code"]
        cur = cur_map.get(ts, {})
        days_held = _calc_days_between(h.get("entry_date", current_date), current_date)
        # close 来源 (daily cache); 简化暂用 V12 输出
        sig = check_exit_signal(
            holding={**h.to_dict(), "days_held": days_held},
            current={**cur, "days_held": days_held},
        )
        out.append({**h.to_dict(), **sig, "current_date": current_date})
    return pd.DataFrame(out)


def _calc_days_between(d1: str, d2: str) -> int:
    """两个 YYYYMMDD 日期之间的交易日数 (简化: 用日历日 / 1.4)."""
    from datetime import datetime
    try:
        dt1 = datetime.strptime(d1, "%Y%m%d")
        dt2 = datetime.strptime(d2, "%Y%m%d")
        return max(0, int((dt2 - dt1).days / 1.4))
    except Exception:
        return 0
