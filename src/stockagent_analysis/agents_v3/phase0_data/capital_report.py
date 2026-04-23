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

    # --- 筹码分布 (优先 Tushare 真实 cyq_perf) ---
    tushare_cyq = features.get("tushare_cyq") or {}
    chip = features.get("chip_distribution", {}) if isinstance(features, dict) else {}
    lines.append("")
    lines.append("### 筹码分布")
    if tushare_cyq:
        lines.append(f"- **数据源**: Tushare cyq_perf (真实筹码绩效)")
        lines.append(f"- 获利盘比例: {fmt_num(tushare_cyq.get('winner_rate'))}%")
        lines.append(f"- 加权平均成本: {fmt_num(tushare_cyq.get('weight_avg_cost'), 2)}")
        lines.append(f"- 成本分位: 5%={fmt_num(tushare_cyq.get('cost_5pct'), 2)} | "
                     f"15%={fmt_num(tushare_cyq.get('cost_15pct'), 2)} | "
                     f"50%={fmt_num(tushare_cyq.get('cost_50pct'), 2)} | "
                     f"85%={fmt_num(tushare_cyq.get('cost_85pct'), 2)} | "
                     f"95%={fmt_num(tushare_cyq.get('cost_95pct'), 2)}")
        lines.append(f"- 筹码分散度: {fmt_num(tushare_cyq.get('dispersion'), 3)} (越大越分散)")
        lines.append(f"- 历史高/低: {fmt_num(tushare_cyq.get('his_high'), 2)} / {fmt_num(tushare_cyq.get('his_low'), 2)}")
        # Tushare 筹码分布集中度
        chips_sum = features.get("tushare_cyq_chips") or {}
        if chips_sum:
            lines.append(f"- Top20% 价位集中度: {fmt_num(chips_sum.get('top20_concentration'))}% "
                         f"(覆盖价位数 {chips_sum.get('price_count')} 个)")
    elif isinstance(chip, dict) and chip:
        lines.append(f"- 数据源: 本地估算")
        lines.append(f"- 获利盘占比: {fmt_num(chip.get('profit_ratio'))}%")
        lines.append(f"- 套牢盘占比: {fmt_num(chip.get('trapped_ratio'))}%")
        lines.append(f"- 筹码集中度: {fmt_num(chip.get('concentration'))}%")
        lines.append(f"- 平均成本: {fmt_num(chip.get('avg_cost'), 2)}")
        lines.append(f"- 筹码健康度: {fmt_num(chip.get('health_score'))}")
    else:
        lines.append("- (无筹码数据)")

    # --- 股东户数趋势(Tushare) ---
    holders = features.get("tushare_holders") or []
    if holders and len(holders) >= 2:
        lines.append("")
        lines.append("### 股东户数趋势")
        first_count = holders[0].get("holder_num")
        last_count = holders[-1].get("holder_num")
        if first_count and last_count:
            try:
                first_v = float(first_count); last_v = float(last_count)
                change_pct = (last_v - first_v) / first_v * 100 if first_v else 0
                trend = "↑ 筹码分散(利空)" if change_pct > 2 else "↓ 筹码集中(利好)" if change_pct < -2 else "基本持平"
                lines.append(f"- 首期({holders[0].get('end_date')}): {int(first_v):,} 户")
                lines.append(f"- 末期({holders[-1].get('end_date')}): {int(last_v):,} 户")
                lines.append(f"- 变化: {change_pct:+.2f}%  {trend}")
            except Exception:
                pass

    # --- 主力/资金流 (优先 Tushare moneyflow 4 档分层) ---
    tushare_mf = features.get("tushare_moneyflow") or {}
    lines.append("")
    lines.append("### 主力/资金流向")
    if tushare_mf:
        lines.append(f"- **数据源**: Tushare moneyflow (4 档分层)")
        lines.append(f"- 最新 ({tushare_mf.get('trade_date_latest')}) 净流入:")
        lines.append(f"  - 超大单净: {fmt_num(tushare_mf.get('latest_super_large_net'), 1)} 万元")
        lines.append(f"  - 大单净: {fmt_num(tushare_mf.get('latest_large_net'), 1)} 万元")
        lines.append(f"  - 中单净: {fmt_num(tushare_mf.get('latest_medium_net'), 1)} 万元")
        lines.append(f"  - 小单净: {fmt_num(tushare_mf.get('latest_small_net'), 1)} 万元")
        lines.append(f"  - 主力合计(大+超大): {fmt_num(tushare_mf.get('latest_main_net'), 1)} 万元")
        days = tushare_mf.get("days", 10)
        lines.append(f"- {days} 日累计:")
        lines.append(f"  - 总净流入: {fmt_num(tushare_mf.get(f'sum_{days}d_net_total'), 1)} 万元")
        lines.append(f"  - 主力净流入: {fmt_num(tushare_mf.get(f'sum_{days}d_main_net'), 1)} 万元")
        lines.append(f"  - 超大单累计: {fmt_num(tushare_mf.get(f'sum_{days}d_super_large_net'), 1)} 万元")
        lines.append(f"  - 大单累计: {fmt_num(tushare_mf.get(f'sum_{days}d_large_net'), 1)} 万元")
    else:
        cf = features.get("capital_flow", {}) if isinstance(features, dict) else {}
        if isinstance(cf, dict) and cf:
            lines.append(f"- 数据源: 本地资金流")
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
