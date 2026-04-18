"""资金/筹码报告 - 量比/OBV/换手/筹码/主力/融资融券/北向。"""
from __future__ import annotations

from typing import Any

from ._helpers import fmt_num, fmt_pct, g


def build_capital_report(symbol: str, name: str, ctx: dict[str, Any]) -> str:
    features = ctx.get("features", {}) if isinstance(ctx, dict) else {}
    snap = ctx.get("snapshot", {}) if isinstance(ctx, dict) else {}

    lines: list[str] = [f"## 资金与筹码 · {symbol} {name}", ""]

    # --- 量能 ---
    lines.append("### 量价量能")
    lines.append(f"- 涨跌幅: {fmt_pct(snap.get('pct_chg'))}")
    lines.append(f"- 换手率: {fmt_pct(snap.get('turnover_rate'))}")
    lines.append(f"- 量比 5/20: {fmt_num(features.get('volume_ratio_5_20'))}")

    # --- 筹码分布 ---
    chip = features.get("chip_distribution", {}) if isinstance(features, dict) else {}
    lines.append("")
    lines.append("### 筹码分布")
    if isinstance(chip, dict) and chip:
        lines.append(f"- 获利盘占比: {fmt_num(chip.get('profit_ratio'))}%")
        lines.append(f"- 套牢盘占比: {fmt_num(chip.get('trapped_ratio'))}%")
        lines.append(f"- 筹码集中度: {fmt_num(chip.get('concentration'))}%")
        lines.append(f"- 平均成本: {fmt_num(chip.get('avg_cost'), 2)}")
        lines.append(f"- 筹码健康度: {fmt_num(chip.get('health_score'))}")
    else:
        lines.append("- (无筹码数据)")

    # --- 主力/资金流 ---
    lines.append("")
    lines.append("### 主力/资金流向")
    # orchestrator 中的 capital_flow 数据可能在 features.capital_flow 或 margin_data
    cf = features.get("capital_flow", {}) if isinstance(features, dict) else {}
    if isinstance(cf, dict) and cf:
        for k in ("main_net", "super_net", "large_net", "medium_net", "small_net",
                  "sector_net", "net_5d"):
            v = cf.get(k)
            if v is not None:
                lines.append(f"- {k}: {fmt_num(v, 2)}")
    else:
        lines.append("- (无资金流数据)")

    # --- 融资融券 ---
    margin = features.get("margin_data", {}) if isinstance(features, dict) else {}
    lines.append("")
    lines.append("### 融资融券")
    if isinstance(margin, dict) and margin:
        for k in ("margin_balance", "short_balance", "net_buy_5d", "margin_change_5d_pct"):
            v = margin.get(k)
            if v is not None:
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {fmt_num(v, 2)}")
                else:
                    lines.append(f"- {k}: {v}")
    else:
        lines.append("- (无融资融券数据)")

    # --- 北向 ---
    hsgt = features.get("hsgt_data", {}) if isinstance(features, dict) else {}
    lines.append("")
    lines.append("### 北向资金(陆股通持仓)")
    if isinstance(hsgt, dict) and hsgt:
        for k in ("hold_ratio", "hold_change_pct", "hold_change_5d", "net_flow_5d"):
            v = hsgt.get(k)
            if v is not None:
                if isinstance(v, (int, float)):
                    lines.append(f"- {k}: {fmt_num(v, 2)}")
                else:
                    lines.append(f"- {k}: {v}")
    else:
        lines.append("- (无北向数据)")

    return "\n".join(lines)
