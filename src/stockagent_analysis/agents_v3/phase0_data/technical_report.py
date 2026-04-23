"""技术面报告 - MA/MACD/RSI/KDJ/布林/动量/波动 三周期。"""
from __future__ import annotations

from typing import Any

from ._helpers import fmt_num, fmt_pct, g, tf_label


def build_technical_report(symbol: str, name: str, ctx: dict[str, Any]) -> str:
    """输出技术面 markdown 报告。"""
    snap = ctx.get("snapshot", {})
    features = ctx.get("features", {})
    kli = features.get("kline_indicators", {}) if isinstance(features, dict) else {}

    lines: list[str] = [
        f"## 技术面 · {symbol} {name}",
        "",
        "### 实时行情",
        f"- 收盘: {fmt_num(snap.get('close'))}",
        f"- 涨跌幅: {fmt_pct(snap.get('pct_chg'))}",
        f"- 换手率: {fmt_pct(snap.get('turnover_rate'))}",
        f"- 数据源: {snap.get('source', 'N/A')}",
        "",
        "### 核心动量与波动",
        f"- 20 日动量: {fmt_pct(features.get('momentum_20'))}",
        f"- 20 日波动率: {fmt_pct(features.get('volatility_20'))}",
        f"- 60 日最大回撤: {fmt_pct(features.get('drawdown_60'))}",
        f"- 量比(5/20): {fmt_num(features.get('volume_ratio_5_20'))}",
        f"- 趋势强度: {fmt_pct(features.get('trend_strength'))}",
    ]

    # --- Tushare 预计算技术指标(qfq, 更权威) ---
    tsf = features.get("tushare_factors") or {}
    if tsf:
        lines.append("")
        lines.append("### Tushare 预计算指标 (qfq 前复权)")
        lines.append(f"- 交易日: {tsf.get('trade_date')}  收盘: {fmt_num(tsf.get('close_qfq'))} ")
        lines.append(
            f"- 均线: MA5={fmt_num(tsf.get('ma5'))} / MA10={fmt_num(tsf.get('ma10'))} / "
            f"MA20={fmt_num(tsf.get('ma20'))} / MA60={fmt_num(tsf.get('ma60'))} / "
            f"MA90={fmt_num(tsf.get('ma90'))} / MA250={fmt_num(tsf.get('ma250'))}"
        )
        lines.append(
            f"- EMA: 5={fmt_num(tsf.get('ema5'))} / 20={fmt_num(tsf.get('ema20'))} / "
            f"60={fmt_num(tsf.get('ema60'))}"
        )
        lines.append(
            f"- 布林带: 上={fmt_num(tsf.get('boll_upper'))} / 中={fmt_num(tsf.get('boll_mid'))} / "
            f"下={fmt_num(tsf.get('boll_lower'))}"
        )
        lines.append(
            f"- MACD: DIF={fmt_num(tsf.get('macd_dif'), 3)} DEA={fmt_num(tsf.get('macd_dea'), 3)} "
            f"Hist={fmt_num(tsf.get('macd_hist'), 3)}"
        )
        lines.append(
            f"- KDJ: K={fmt_num(tsf.get('kdj_k'), 1)} D={fmt_num(tsf.get('kdj_d'), 1)} "
            f"J={fmt_num(tsf.get('kdj_j'), 1)}"
        )
        lines.append(
            f"- RSI: 6日={fmt_num(tsf.get('rsi6'), 1)} / 12日={fmt_num(tsf.get('rsi12'), 1)} / "
            f"24日={fmt_num(tsf.get('rsi24'), 1)}"
        )
        lines.append(
            f"- DMI 趋势强度: ADX={fmt_num(tsf.get('dmi_adx'), 1)} (>25 趋势明显)  "
            f"+DI={fmt_num(tsf.get('dmi_pdi'), 1)} / -DI={fmt_num(tsf.get('dmi_mdi'), 1)}"
        )
        lines.append(
            f"- BIAS: 1={fmt_num(tsf.get('bias1'))} / 2={fmt_num(tsf.get('bias2'))} / "
            f"3={fmt_num(tsf.get('bias3'))}"
        )
        lines.append(
            f"- 其他: CCI={fmt_num(tsf.get('cci'))}  MFI={fmt_num(tsf.get('mfi'))}  "
            f"WR={fmt_num(tsf.get('wr'))}  ATR={fmt_num(tsf.get('atr'), 3)}  "
            f"TRIX={fmt_num(tsf.get('trix'), 3)}"
        )
        lines.append(
            f"- 连续: 涨 {tsf.get('updays', 0):.0f} 日 / 跌 {tsf.get('downdays', 0):.0f} 日  "
            f"(历史高 {tsf.get('topdays', 0):.0f} 日 / 低 {tsf.get('lowdays', 0):.0f} 日)"
        )

    lines.append("")
    lines.append("### 多周期技术指标 (本地计算 · 日/周/月对比)")

    for tf in ("day", "week", "month"):
        td = kli.get(tf, {}) if isinstance(kli, dict) else {}
        if not isinstance(td, dict) or not td.get("ok"):
            lines.append(f"- **{tf_label(tf)}**: 数据不足或未计算")
            continue
        rsi = td.get("rsi")
        dif = td.get("macd_dif")
        dea = td.get("macd_dea")
        hist = td.get("macd_hist")
        k = td.get("kdj_k"); d_ = td.get("kdj_d"); j = td.get("kdj_j")
        bu = td.get("boll_upper"); bm = td.get("boll_mid"); bl = td.get("boll_lower")
        mom = td.get("momentum_10")
        slope = td.get("trend_slope_pct")
        ma_sys = td.get("ma_system") or {}

        lines.append(f"- **{tf_label(tf)}** ({td.get('rows', 0)} 根):")
        lines.append(
            f"  - RSI={fmt_num(rsi, 1)} | "
            f"MACD DIF={fmt_num(dif, 3)} DEA={fmt_num(dea, 3)} Hist={fmt_num(hist, 3)}"
        )
        lines.append(
            f"  - KDJ: K={fmt_num(k, 1)} D={fmt_num(d_, 1)} J={fmt_num(j, 1)}"
        )
        lines.append(
            f"  - 布林: 上={fmt_num(bu, 2)} 中={fmt_num(bm, 2)} 下={fmt_num(bl, 2)}"
        )
        lines.append(
            f"  - 10 日动量={fmt_pct(mom)} | 斜率={fmt_pct(slope)}"
        )
        if isinstance(ma_sys, dict) and ma_sys:
            ma_parts = []
            for key in ("ma5", "ma10", "ma20", "ma60", "ma120", "ma250"):
                val = ma_sys.get(key)
                if val is not None:
                    ma_parts.append(f"{key.upper()}={fmt_num(val, 2)}")
            if ma_parts:
                lines.append(f"  - 均线: {' '.join(ma_parts)}")
            arrangement = ma_sys.get("arrangement") or ma_sys.get("排列")
            if arrangement:
                lines.append(f"  - 均线排列: {arrangement}")

    lines.append("")
    lines.append("### 关键价位 (Fibonacci)")
    key_levels = features.get("key_levels", {}) if isinstance(features, dict) else {}
    if isinstance(key_levels, dict) and key_levels:
        for k, v in key_levels.items():
            lines.append(f"- {k}: {fmt_num(v, 2)}")
    else:
        lines.append("- (未计算)")

    lines.append("")
    lines.append("### 相对强度 (RS)")
    rs_ind = features.get("rs_vs_industry")
    rs_etf = features.get("rs_vs_etf")
    rs_lead = features.get("rs_vs_leaders")
    lines.append(f"- 对行业: {fmt_num(rs_ind, 2) if rs_ind is not None else 'N/A'}")
    lines.append(f"- 对 ETF: {fmt_num(rs_etf, 2) if rs_etf is not None else 'N/A'}")
    lines.append(f"- 对龙头: {fmt_num(rs_lead, 2) if rs_lead is not None else 'N/A'}")

    return "\n".join(lines)
