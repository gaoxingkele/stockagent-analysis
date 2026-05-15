"""Regime 触发减仓监控 (Sprint 1.5, V12.9).

灾难月救命: V7c OOS 202508 -4.95pp 系统失效, V12 无法识别 regime 切换.
本模块基于 daily_regime.parquet 输出建议仓位比例.

Regime 状态映射:
  bull_policy / bull_fast      → 100% (强势爆发)
  bull_slow_diverge            → 100% (慢牛)
  mixed                        → 80%  (中性, 减仓 20%)
  sideways                     → 60%  (横盘, 减仓 40%)
  bear                         → 30%  (熊市, 重度减仓)

触发逻辑 (避免噪声):
  - 看过去 5 日 regime 序列
  - 取最近 3 日的"主导 regime" (3 日内出现 ≥2 次的 regime)
  - 按主导 regime 映射仓位
  - 单日抖动不影响 (例如 0507 突然 bull_fast 一日不算)
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent.parent
REGIME_PATH = ROOT / "output" / "regimes" / "daily_regime.parquet"

# 仓位映射
POSITION_MAP = {
    "bull_policy": 1.00,
    "bull_fast": 1.00,
    "bull_slow_diverge": 1.00,
    "mixed": 0.80,
    "sideways": 0.60,
    "bear": 0.30,
}


class RegimeMonitor:
    _cache: Optional[pd.DataFrame] = None

    @classmethod
    def _load(cls) -> pd.DataFrame:
        if cls._cache is None:
            df = pd.read_parquet(REGIME_PATH)
            df["trade_date"] = df["trade_date"].astype(str)
            cls._cache = df.sort_values("trade_date").reset_index(drop=True)
        return cls._cache

    @classmethod
    def get_recent_regimes(cls, date: str, n: int = 5) -> list[str]:
        """返回 date 当日及前 n-1 日 regime 序列 (按日期升序)."""
        df = cls._load()
        sub = df[df["trade_date"] <= date].tail(n)
        return sub["regime"].tolist()

    @classmethod
    def get_dominant_regime(cls, date: str, window: int = 3) -> str:
        """取最近 window 日的主导 regime (出现次数最多, 若并列取最近一日)."""
        recent = cls.get_recent_regimes(date, n=window)
        if not recent:
            return "mixed"
        from collections import Counter
        cnt = Counter(recent)
        # 主导: 出现 ≥ window//2 + 1 次, 否则用最近一日
        most_common, count = cnt.most_common(1)[0]
        if count >= (window // 2 + 1):
            return most_common
        return recent[-1]   # 没有明显主导, 用最近一日

    @classmethod
    def get_position_ratio(cls, date: str, window: int = 3) -> dict:
        """返回当日建议仓位比例 + 上下文.

        多因子综合:
          1. base = regime 主导仓位 (POSITION_MAP)
          2. 顶部预警 (RSI/量能/价格): 强制减仓
          3. 恐慌底部 (RSI<25 + 放量大跌): 允许加仓 (但不超过 1.0)
        """
        df = cls._load()
        if df.empty or date < df["trade_date"].min():
            return {"position_ratio": 1.0, "regime": "unknown",
                    "reason": "no data", "recent_regimes": []}
        recent = cls.get_recent_regimes(date, n=5)
        dominant = cls.get_dominant_regime(date, window=window)
        base_ratio = POSITION_MAP.get(dominant, 1.00)

        cur = df[df["trade_date"] == date]
        if cur.empty: cur = df[df["trade_date"] <= date].tail(1)
        cur_row = cur.iloc[0]

        rsi = float(cur_row["rsi14"])
        ret_5d = float(cur_row["ret_5d"])
        vol_ratio = float(cur_row["vol_ratio"])
        vol_z60 = float(cur_row.get("vol_z60", 0)) if pd.notna(cur_row.get("vol_z60")) else 0

        # 触发器列表 (从最严到最松, 取最低仓位)
        triggers = []
        ratio = base_ratio
        # 触发 1: RSI 极端超买 (顶部预警)
        if rsi >= 85:
            triggers.append(f"RSI={rsi:.0f} 极端超买")
            ratio = min(ratio, 0.40)
        elif rsi >= 75 and ret_5d > 0.05:
            triggers.append(f"RSI={rsi:.0f} + 5日涨{ret_5d*100:.1f}% 高位连涨")
            ratio = min(ratio, 0.60)
        # 触发 2: 恐慌量 + 大跌
        if vol_z60 >= 2.5 and ret_5d < -0.03:
            triggers.append(f"vol_z60={vol_z60:.1f} + 5日跌{ret_5d*100:.1f}% 恐慌出货")
            ratio = min(ratio, 0.30)
        # 触发 3: 高位异常放量 (派发信号)
        if rsi >= 80 and vol_z60 >= 2.0 and vol_ratio >= 1.4:
            triggers.append(f"RSI={rsi:.0f} + vol_z60={vol_z60:.1f} 高位派发")
            ratio = min(ratio, 0.50)

        return {
            "date": date,
            "current_regime": cur_row["regime"],
            "current_regime_id": int(cur_row["regime_id"]),
            "dominant_regime_3d": dominant,
            "base_ratio": base_ratio,
            "position_ratio": ratio,
            "triggers": triggers,
            "recent_regimes_5d": recent,
            "context": {
                "ret_5d_pct": round(ret_5d * 100, 2),
                "ret_20d_pct": round(float(cur_row["ret_20d"]) * 100, 2),
                "rsi14": round(rsi, 1),
                "vol_ratio": round(vol_ratio, 2),
                "vol_z60": round(vol_z60, 2),
            },
            "reason": _explain(dominant, recent) + (
                f"; 触发减仓 ({len(triggers)} 项)" if triggers else ""
            ),
        }


def _explain(dominant: str, recent: list[str]) -> str:
    desc = {
        "bull_policy": "强势爆发 (政策驱动)",
        "bull_fast": "快速上涨",
        "bull_slow_diverge": "慢牛分化",
        "mixed": "中性混合",
        "sideways": "横盘震荡",
        "bear": "熊市",
    }
    main = desc.get(dominant, dominant)
    seq = " → ".join(recent[-3:]) if recent else "?"
    return f"近 3 日主导 [{main}], 序列: {seq}"


def historical_position_curve(start: str, end: str) -> pd.DataFrame:
    """跑出 [start, end] 期间每日建议仓位曲线 (含触发器)."""
    rows = []
    df = RegimeMonitor._load()
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
    for d in df["trade_date"]:
        info = RegimeMonitor.get_position_ratio(d)
        rows.append({
            "trade_date": d,
            "regime": info["current_regime"],
            "dominant": info["dominant_regime_3d"],
            "base_ratio": info["base_ratio"],
            "position_ratio": info["position_ratio"],
            "n_triggers": len(info["triggers"]),
            "trigger_summary": "; ".join(info["triggers"])[:80],
            "ret_5d_pct": info["context"]["ret_5d_pct"],
            "rsi14": info["context"]["rsi14"],
            "vol_z60": info["context"]["vol_z60"],
        })
    return pd.DataFrame(rows)
