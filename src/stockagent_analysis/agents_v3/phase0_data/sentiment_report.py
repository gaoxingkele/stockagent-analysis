"""舆情环境报告 - 新闻/板块/市场感知/融资融券情绪。"""
from __future__ import annotations

from typing import Any

from ._helpers import fmt_num, fmt_pct


def build_sentiment_report(symbol: str, name: str, ctx: dict[str, Any]) -> str:
    features = ctx.get("features", {}) if isinstance(ctx, dict) else {}
    news = ctx.get("news", []) if isinstance(ctx, dict) else []

    lines: list[str] = [f"## 舆情与市场环境 · {symbol} {name}", ""]

    # --- 新闻情绪 ---
    lines.append("### 新闻情绪")
    ns = features.get("news_sentiment")
    nc = features.get("news_count")
    lines.append(f"- 情绪分: {fmt_num(ns)} (新闻条数={nc})")
    if isinstance(news, list) and news:
        lines.append("- 近期新闻:")
        for item in news[:6]:
            if isinstance(item, dict):
                title = item.get("title", item.get("headline", ""))
                date = item.get("date") or item.get("time") or ""
                src = item.get("source", "")
                tag = f"[{date}]" if date else ""
                tag2 = f"({src})" if src else ""
                if title:
                    lines.append(f"  - {tag} {title} {tag2}".strip())

    # --- 市场环境 ---
    mc = features.get("market_context", {}) if isinstance(features, dict) else {}
    lines.append("")
    lines.append("### 市场环境感知")
    if isinstance(mc, dict) and mc:
        ms = mc.get("market_score")
        mp = mc.get("market_phase_cn") or mc.get("market_phase")
        lines.append(f"- 大盘评分: {fmt_num(ms)} | 阶段: {mp}")
        indices = mc.get("index_states") or []
        if indices:
            lines.append("- 主要指数:")
            for idx in indices[:5]:
                if isinstance(idx, dict):
                    cn = idx.get("name") or idx.get("code")
                    st = idx.get("state_cn") or idx.get("state")
                    ret20 = idx.get("ret_20d")
                    lines.append(f"  - {cn}: {st} (20d={fmt_pct(ret20)})")
        sectors = mc.get("sector_heats") or []
        if sectors:
            lines.append("- 板块热度 TOP:")
            for sec in sectors[:5]:
                if isinstance(sec, dict):
                    n = sec.get("sector_name")
                    rk = sec.get("rank")
                    p5 = sec.get("pct_chg_5d")
                    lead = sec.get("lead_stock")
                    lines.append(f"  - {n} (排名={rk}, 5d={fmt_pct(p5)}, 龙头={lead})")
        etfs = mc.get("etf_states") or []
        if etfs:
            lines.append("- 关联 ETF:")
            for e in etfs[:3]:
                if isinstance(e, dict):
                    n = e.get("name") or e.get("code")
                    st = e.get("state_cn") or e.get("state")
                    lines.append(f"  - {n}: {st}")
        vs = mc.get("vision_summary")
        if vs:
            lines.append(f"- 视觉摘要: {vs[:300]}")
    else:
        lines.append("- (无市场环境数据)")

    # --- 市场策略阶段 ---
    strategy = features.get("market_strategy", {}) if isinstance(features, dict) else {}
    if isinstance(strategy, dict) and strategy:
        lines.append("")
        lines.append("### 策略建议(市场阶段)")
        lines.append(f"- 阶段: {strategy.get('phase_cn', strategy.get('phase', '未知'))}")
        pos_cap = strategy.get("position_cap")
        if pos_cap is not None:
            try:
                lines.append(f"- 最大建议仓位: {fmt_num(float(pos_cap) * 100)}%")
            except (ValueError, TypeError):
                pass
        sb = strategy.get("sector_bias")
        if sb:
            lines.append(f"- 板块偏向: {sb}")

    # --- 数据完整性 ---
    di = ctx.get("data_integrity", {}) if isinstance(ctx, dict) else {}
    if isinstance(di, dict) and di:
        lines.append("")
        lines.append("### 数据完整性")
        for k, v in di.items():
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)
