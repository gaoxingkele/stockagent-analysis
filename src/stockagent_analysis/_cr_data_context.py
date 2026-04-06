# -*- coding: utf-8 -*-
"""ChannelReversalAgent._build_data_context 新实现 — 注入完整系统提示词。

此文件被 patch_agents.py 读取后注入到 agents.py 中。
也可作为 mixin 直接使用。
"""
from __future__ import annotations
from pathlib import Path
from typing import Any


_SYSTEM_PROMPT_CACHE: str | None = None


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is None:
        prompt_path = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts" / "channel_reversal_system.txt"
        if prompt_path.exists():
            _SYSTEM_PROMPT_CACHE = prompt_path.read_text(encoding="utf-8")
        else:
            _SYSTEM_PROMPT_CACHE = ""
    return _SYSTEM_PROMPT_CACHE


def build_data_context(self, ctx: dict[str, Any]) -> str:
    """为LLM构建通道反转完整分析上下文：系统提示词 + 实时数据摘要。"""
    parts: list[str] = []

    sys_prompt = _load_system_prompt()
    if sys_prompt:
        parts.append(sys_prompt)
        parts.append("")
        parts.append("======== 六、当前股票实时数据 ========")
        parts.append("")

    snap = ctx.get("snapshot", {})
    if snap:
        parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

    try:
        from .channel_reversal import compute_channel, detect_phases
        import math
        df = self._load_day_kline()
        if df is not None and len(df) >= 160:
            df = compute_channel(df)
            df = detect_phases(df)
            tail = df.tail(20)
            last = df.iloc[-1]

            phase_labels = {
                "0U": "上半区(多头)", "0D": "下半区(空头)",
                "1": "破下轨(超卖)", "2": "回到轨道(反弹启动)",
                "3A": "弱反弹(大概率破新低)", "3B": "过中轨(看多确认)",
                "4A": "有效反转(突破上轨)", "4B": "上轨整理(蓄力)",
            }

            parts.append("")
            parts.append("[通道轨道]")
            parts.append(
                f"上轨={last['ch_upper']:.2f} | 中轨={last['ch_middle']:.2f} | "
                f"下轨={last['ch_lower']:.2f}"
            )
            ch_width = last["ch_upper"] - last["ch_lower"]
            if last["ch_middle"] > 0:
                parts.append(f"通道宽度: {ch_width:.2f} ({ch_width/last['ch_middle']*100:.1f}%)")

            parts.append("")
            parts.append("[阶段判定]")
            parts.append(
                f"当前阶段: {phase_labels.get(last['phase'], last['phase'])} "
                f"(持续{int(last['phase_days'])}日)"
            )
            parts.append(f"本地通道评分: {last['cr_score']:.0f}")

            rsi = last.get("rsi")
            rsi_slope = last.get("rsi_slope")
            vol_r = last.get("vol_ratio")
            rsi_str = f"{rsi:.1f}" if rsi is not None and not math.isnan(rsi) else "N/A"
            slope_str = f"{rsi_slope:+.2f}" if rsi_slope is not None and not math.isnan(rsi_slope) else "N/A"
            vol_str = f"{vol_r:.2f}" if vol_r is not None and not math.isnan(vol_r) else "N/A"
            parts.append("")
            parts.append("[确认信号]")
            parts.append(f"RSI(14)={rsi_str} | RSI斜率(5日)={slope_str} | 量比(vs 20日均量)={vol_str}")

            rsi_ok = rsi is not None and not math.isnan(rsi) and rsi > 40
            slope_ok = rsi_slope is not None and not math.isnan(rsi_slope) and rsi_slope > 0.5
            vol_ok = vol_r is not None and not math.isnan(vol_r) and vol_r > 1.2
            confirms = sum([rsi_ok, slope_ok, vol_ok])
            parts.append(
                f"三重确认: RSI>40={'Y' if rsi_ok else 'N'} | "
                f"RSI斜率>0.5={'Y' if slope_ok else 'N'} | "
                f"量比>1.2={'Y' if vol_ok else 'N'} -> {confirms}/3"
            )

            phase_changes = []
            prev_p = None
            for _, row in tail.iterrows():
                p = row.get("phase")
                if p != prev_p and p is not None:
                    phase_changes.append(f"{phase_labels.get(p, p)}({int(row['phase_days'])}日)")
                    prev_p = p
            if phase_changes:
                parts.append("")
                parts.append("[阶段演进(近20日)]")
                parts.append("\u2192".join(phase_changes))

            close = last["close"]
            upper = last["ch_upper"]
            lower = last["ch_lower"]
            mid = last["ch_middle"]
            if upper > lower:
                pos_pct = (close - lower) / (upper - lower) * 100
                parts.append("")
                parts.append("[价格位置]")
                parts.append(f"通道位置: {pos_pct:.1f}% (0%=下轨, 100%=上轨)")
                parts.append(
                    f"距上轨: {(close/upper-1)*100:+.2f}% | "
                    f"距中轨: {(close/mid-1)*100:+.2f}% | "
                    f"距下轨: {(close/lower-1)*100:+.2f}%"
                )

        else:
            parts.append("[动态通道] 日线数据不足160根，无法计算")
    except Exception as e:
        parts.append(f"[动态通道] 计算异常: {e}")

    return "\n".join(parts)
