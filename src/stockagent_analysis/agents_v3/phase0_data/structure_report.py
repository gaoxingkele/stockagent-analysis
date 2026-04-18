"""K线结构报告 - 缠论/Donchian/Ichimoku/K线形态。"""
from __future__ import annotations

from typing import Any

from ._helpers import fmt_num, fmt_pct, g, tf_label


def build_structure_report(symbol: str, name: str, ctx: dict[str, Any]) -> str:
    """输出 K 线结构 markdown 报告。"""
    features = ctx.get("features", {}) if isinstance(ctx, dict) else {}
    kli = features.get("kline_indicators", {}) if isinstance(features, dict) else {}

    lines: list[str] = [f"## K 线结构 · {symbol} {name}", ""]

    # --- 缠论 ---
    lines.append("### 缠论结构")
    any_chan = False
    for tf in ("day", "week", "month"):
        td = kli.get(tf, {}) if isinstance(kli, dict) else {}
        chan = td.get("chanlun") if isinstance(td, dict) else None
        if not isinstance(chan, dict) or not chan:
            continue
        any_chan = True
        lines.append(f"- **{tf_label(tf)}**:")
        for k in ("current_phase", "current_bi", "current_duan", "latest_pivot", "trade_signal",
                  "bi_count", "duan_count", "pivot_low", "pivot_high"):
            v = chan.get(k)
            if v is not None:
                lines.append(f"  - {k}: {v}")
    if not any_chan:
        lines.append("- (无缠论数据)")

    # --- Ichimoku ---
    lines.append("")
    lines.append("### Ichimoku 一目均衡图")
    any_ichi = False
    for tf in ("day", "week"):
        td = kli.get(tf, {}) if isinstance(kli, dict) else {}
        ichi = td.get("ichimoku") if isinstance(td, dict) else None
        if not isinstance(ichi, dict) or not ichi:
            continue
        any_ichi = True
        lines.append(f"- **{tf_label(tf)}**:")
        for k in ("tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
                  "cloud_color", "price_vs_cloud", "signal"):
            v = ichi.get(k)
            if v is not None:
                if isinstance(v, (int, float)):
                    lines.append(f"  - {k}: {fmt_num(v, 2)}")
                else:
                    lines.append(f"  - {k}: {v}")
    if not any_ichi:
        lines.append("- (无 Ichimoku 数据)")

    # --- Donchian 通道 (如果 features.channel_reversal 有) ---
    lines.append("")
    lines.append("### Donchian 通道 8 阶段状态")
    cr = features.get("channel_reversal") if isinstance(features, dict) else None
    if isinstance(cr, dict) and cr:
        for k in ("phase", "phase_cn", "days_in_phase",
                  "upper", "middle", "lower",
                  "rsi", "rsi_slope", "volume_ratio",
                  "signal"):
            v = cr.get(k)
            if v is not None:
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {fmt_num(v, 2)}")
                else:
                    lines.append(f"- {k}: {v}")
    else:
        lines.append("- (无 Donchian 通道数据, 可能日线不足 160 根)")

    # --- 背离 ---
    lines.append("")
    lines.append("### 多周期背离")
    any_div = False
    for tf in ("day", "week", "month"):
        td = kli.get(tf, {}) if isinstance(kli, dict) else {}
        dv = td.get("divergence") if isinstance(td, dict) else None
        if not isinstance(dv, dict) or not dv:
            continue
        any_div = True
        signals = []
        if dv.get("macd_bot_div"):
            signals.append(f"MACD 底背离(强度 {dv.get('macd_bot_strength', '?')})")
        if dv.get("macd_top_div"):
            signals.append(f"MACD 顶背离(强度 {dv.get('macd_top_strength', '?')})")
        if dv.get("rsi_bot_div"):
            signals.append("RSI 底背离")
        if dv.get("rsi_top_div"):
            signals.append("RSI 顶背离")
        if signals:
            lines.append(f"- **{tf_label(tf)}**: {'; '.join(signals)}")
        else:
            lines.append(f"- **{tf_label(tf)}**: 无")
    if not any_div:
        lines.append("- (未检测到背离信号)")

    # --- K线形态组合 ---
    lines.append("")
    lines.append("### K 线形态组合")
    any_pat = False
    for tf in ("day", "week"):
        td = kli.get(tf, {}) if isinstance(kli, dict) else {}
        if not isinstance(td, dict) or not td.get("ok"):
            continue
        combo = td.get("kline_combo_5") or td.get("kline_patterns")
        if combo:
            any_pat = True
            lines.append(f"- **{tf_label(tf)}**: {combo}")
    if not any_pat:
        lines.append("- (无显著形态)")

    return "\n".join(lines)
