"""波浪理论报告 - ZigZag 波浪位 + 斐波那契。

首版: 基于已有的 key_levels (fib 回撤) + 动量/回撤判断当前波浪阶段。
深度实现(ZigZag + Elliott 5 推动 3 调整)可后续迭代。
"""
from __future__ import annotations

from typing import Any

from ._helpers import fmt_num, fmt_pct


def _infer_wave_stage(mom_20: float | None, dd_60: float | None, trend_strength: float | None) -> str:
    """基于动量与回撤粗略推测当前波浪阶段。"""
    try:
        m = float(mom_20) if mom_20 is not None else 0.0
        d = float(dd_60) if dd_60 is not None else 0.0
        t = float(trend_strength) if trend_strength is not None else 0.0
    except (ValueError, TypeError):
        return "未知"

    if m > 8 and t > 3:
        return "推动浪 3/5 进行中(主升)"
    if m > 3 and t > 0:
        return "推动浪启动或延续(2-3 浪间)"
    if -3 < m <= 3 and -5 < d <= -1:
        return "调整浪 2 或 4 进行中"
    if d <= -15 and m < -5:
        return "C 浪下跌或 5 浪见顶后回落"
    if m < -3 and t < 0:
        return "下跌浪 A 或延续的调整"
    return "震荡整理(无明确主浪方向)"


def build_wave_report(symbol: str, name: str, ctx: dict[str, Any]) -> str:
    features = ctx.get("features", {}) if isinstance(ctx, dict) else {}
    snap = ctx.get("snapshot", {}) if isinstance(ctx, dict) else {}

    mom_20 = features.get("momentum_20")
    dd_60 = features.get("drawdown_60")
    vol_20 = features.get("volatility_20")
    trend = features.get("trend_strength")

    stage = _infer_wave_stage(mom_20, dd_60, trend)

    lines = [
        f"## 波浪结构 · {symbol} {name}",
        "",
        "### 当前波浪阶段推断(启发式)",
        f"- 推断阶段: **{stage}**",
        f"- 依据: 20 日动量={fmt_pct(mom_20)}, 60 日回撤={fmt_pct(dd_60)}, 趋势强度={fmt_pct(trend)}",
        f"- 波动率(20d): {fmt_pct(vol_20)}",
        "",
        "### Fibonacci 回撤/扩展位",
    ]

    key_levels = features.get("key_levels", {}) if isinstance(features, dict) else {}
    if isinstance(key_levels, dict) and key_levels:
        # 常见 key: swing_high, swing_low, fib_236, fib_382, fib_500, fib_618, fib_786
        for k, v in key_levels.items():
            lines.append(f"- {k}: {fmt_num(v, 2)}")
    else:
        lines.append("- (未计算)")

    lines.append("")
    lines.append("### 当前价格位置 (相对 fib)")
    close = snap.get("close")
    if close and isinstance(key_levels, dict) and key_levels:
        try:
            close_f = float(close)
            hi = key_levels.get("swing_high") or key_levels.get("high")
            lo = key_levels.get("swing_low") or key_levels.get("low")
            if hi and lo:
                hi_f = float(hi); lo_f = float(lo)
                if hi_f > lo_f:
                    pos = (close_f - lo_f) / (hi_f - lo_f) * 100
                    lines.append(f"- 当前价 {fmt_num(close, 2)} 处于 swing {fmt_num(lo_f, 2)}-{fmt_num(hi_f, 2)} 区间的 {fmt_num(pos, 1)}% 位置")
        except (ValueError, TypeError):
            pass

    lines.append("")
    lines.append("### 波浪失效参考条件")
    lines.append("- 若推动浪假设: 跌破 2 浪低点 (fib 0.618 下方) 则失效")
    lines.append("- 若调整浪假设: 突破前高则可能提前进入新推动浪")
    lines.append("")
    lines.append("> 注: 本报告为启发式波浪推断, 精确波浪标注需人工结合 ZigZag 摆动点分析。")

    return "\n".join(lines)
