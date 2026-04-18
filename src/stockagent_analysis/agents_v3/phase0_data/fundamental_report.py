"""基本面报告 - PE/PB/ROE/营收/净利/负债/同业对比。"""
from __future__ import annotations

from typing import Any

from ._helpers import fmt_num, fmt_pct


def build_fundamental_report(symbol: str, name: str, ctx: dict[str, Any]) -> str:
    features = ctx.get("features", {}) if isinstance(ctx, dict) else {}
    fund = ctx.get("fundamentals", {}) if isinstance(ctx, dict) else {}

    lines: list[str] = [f"## 基本面 · {symbol} {name}", ""]

    # --- 估值 ---
    lines.append("### 估值指标")
    lines.append(f"- PE(TTM): {fmt_num(features.get('pe_ttm'))}")
    lines.append(f"- PB: {fmt_num(features.get('pb'))}")
    total_mv = features.get("total_mv")
    if total_mv is not None:
        try:
            mv = float(total_mv)
            if mv > 1e8:
                lines.append(f"- 总市值: {fmt_num(mv / 1e8)} 亿")
            else:
                lines.append(f"- 总市值: {fmt_num(mv)}")
        except (ValueError, TypeError):
            lines.append(f"- 总市值: {total_mv}")

    # --- 盈利能力 ---
    lines.append("")
    lines.append("### 盈利能力")
    lines.append(f"- ROE: {fmt_pct(features.get('roe'))}")
    lines.append(f"- 毛利率: {fmt_pct(features.get('grossprofit_margin'))}")
    lines.append(f"- 净利率: {fmt_pct(features.get('netprofit_margin'))}")
    lines.append(f"- EPS: {fmt_num(features.get('eps'), 3)}")
    lines.append(f"- 每股现金流: {fmt_num(features.get('cfps'), 3)}")

    # --- 成长性 ---
    lines.append("")
    lines.append("### 成长性")
    lines.append(f"- 营收同比: {fmt_pct(features.get('revenue_yoy'))}")
    lines.append(f"- 净利同比: {fmt_pct(features.get('netprofit_yoy'))}")

    # --- 财务健康 ---
    lines.append("")
    lines.append("### 财务健康")
    lines.append(f"- 资产负债率: {fmt_pct(features.get('debt_to_assets'))}")
    cr = features.get("current_ratio")
    if cr is not None:
        lines.append(f"- 流动比率: {fmt_num(cr)}")
    qr = features.get("quick_ratio")
    if qr is not None:
        lines.append(f"- 速动比率: {fmt_num(qr)}")

    # --- 同业对比 ---
    lines.append("")
    lines.append("### 同业对比")
    peer = features.get("peer_comparison", {}) if isinstance(features, dict) else {}
    if isinstance(peer, dict) and peer.get("peers"):
        ind = peer.get("industry", "")
        ind_avg = peer.get("industry_avg", {}) or {}
        lines.append(f"- 所属行业: {ind}")
        if ind_avg:
            parts = []
            if ind_avg.get("pe_ttm") is not None:
                parts.append(f"PE={fmt_num(ind_avg['pe_ttm'])}")
            if ind_avg.get("roe") is not None:
                parts.append(f"ROE={fmt_pct(ind_avg['roe'])}")
            if ind_avg.get("revenue_yoy") is not None:
                parts.append(f"营收增速={fmt_pct(ind_avg['revenue_yoy'])}")
            if parts:
                lines.append(f"- 行业均值: {' | '.join(parts)}")
        peers = peer.get("peers", [])
        if peers:
            lines.append("- 同业 TOP:")
            for p in peers[:5]:
                if isinstance(p, dict):
                    nm = p.get("name", "")
                    code = p.get("symbol", p.get("code", ""))
                    pe_v = p.get("pe_ttm")
                    roe_v = p.get("roe")
                    line = f"  - {code} {nm}"
                    extra = []
                    if pe_v is not None:
                        extra.append(f"PE={fmt_num(pe_v)}")
                    if roe_v is not None:
                        extra.append(f"ROE={fmt_pct(roe_v)}")
                    if extra:
                        line += " (" + ", ".join(extra) + ")"
                    lines.append(line)
    else:
        lines.append("- (无同业对比数据)")

    # --- 财报摘要 ---
    if isinstance(fund, dict) and fund:
        summary = fund.get("summary") or fund.get("latest_report_summary")
        if summary:
            lines.append("")
            lines.append("### 财报摘要")
            lines.append(f"> {summary}")

    return "\n".join(lines)
