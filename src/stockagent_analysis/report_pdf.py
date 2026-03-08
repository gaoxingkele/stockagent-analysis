# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _score_to_decision_level_cn(score: float) -> str:
    """根据评分映射到五级决策中文。强烈买入(≥85)、弱买入(70-85)、观望(50-70)、弱卖出(40-50)、强烈卖出(<40)。"""
    if score >= 85:
        return "强烈买入"
    if score >= 70:
        return "弱买入"
    if score >= 50:
        return "观望"
    if score >= 40:
        return "弱卖出"
    return "强烈卖出"


def _safe_filename(name: str) -> str:
    invalid = '\\/:*?"<>|'
    out = "".join("_" if c in invalid else c for c in name).strip()
    return out or "report"


def _register_fonts() -> tuple[str, str]:
    simsun_candidates = [
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simsun.ttf",
    ]
    bold_candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc",
    ]
    body_font = "STSong-Light"
    bold_font = "STSong-Light"
    try:
        for p in simsun_candidates:
            if Path(p).exists():
                pdfmetrics.registerFont(TTFont("SimSun", p))
                body_font = "SimSun"
                break
    except Exception:
        pass
    try:
        for p in bold_candidates:
            if Path(p).exists():
                pdfmetrics.registerFont(TTFont("CNBold", p))
                bold_font = "CNBold"
                break
    except Exception:
        pass
    if body_font == "STSong-Light":
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    if bold_font == "STSong-Light":
        bold_font = body_font
    return body_font, bold_font


# ─────────────────────────────────────────────────────────────────
#  框架说明表
# ─────────────────────────────────────────────────────────────────
def _add_framework_intro_table(
    flow: list,
    result: dict[str, Any],
    body_font: str,
    bold_font: str,
    st_h: Any,
    st_body: Any,
) -> None:
    """多智能体分析框架说明：六大维度模块一览。"""
    votes = result.get("agent_votes", [])
    n_agents = len(votes)
    flow.append(Paragraph("多智能体分析框架说明", st_h))
    rows = [["模块", "核心智能体", "分析维度", "数据来源"]]
    modules = [
        ("基本面", "基本面分析师", "PE/PB/市值/估值/盈利质量", "Tushare / AKShare"),
        ("技术面", "趋势/技术指标/K线视觉/结构", "多周期K线、MA均线、MACD/RSI/KDJ/布林", "历史行情 / 图像识别"),
        ("资金面", "资金流向/精细资金流/主力行为", "主力净流入、超大单、北向资金、换手率", "东方财富 / AKShare"),
        ("情绪面", "情绪/深度舆情/NLP情绪", "新闻热度、社媒情绪、公告舆情", "新闻爬取 / NLP分析"),
        ("宏观政策", "宏观关联/板块与政策/大盘联动", "利率汇率、板块政策、指数联动", "财经新闻 / 宏观数据"),
        ("综合决策", "量化/风险/顶底结构/融资融券", "综合加权评分、Bull-Bear辩论、五级决策", "多维数据融合"),
    ]
    for m in modules:
        rows.append(list(m))
    tbl = Table(rows, colWidths=[24 * mm, 38 * mm, 62 * mm, 36 * mm])
    style = [
        ("FONTNAME", (0, 0), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F8FBFF"), colors.white]),
    ]
    tbl.setStyle(TableStyle(style))
    flow.append(tbl)
    flow.append(
        Paragraph(
            f"本报告由 <b>{n_agents} 个专业智能体</b>并行分析，覆盖技术面、基本面、资金面、情绪面、宏观政策五大维度，"
            "经 Bull-Bear 辩论与 Judge 仲裁后，输出五级加权综合决策。",
            st_body,
        )
    )
    flow.append(Spacer(1, 8))


# ─────────────────────────────────────────────────────────────────
#  多模型权重汇总表
# ─────────────────────────────────────────────────────────────────
def _add_multi_model_table(
    flow: list,
    result: dict[str, Any],
    body_font: str,
    bold_font: str,
    st_h: Any,
    st_body: Any,
) -> None:
    """多模型模式下，各模型独立评分+权重，在文档前部添加汇总表。"""
    votes = result.get("agent_votes", [])
    model_weights = result.get("model_weights", {})
    model_scores = result.get("model_scores", {})
    model_totals = result.get("model_totals", {})
    providers = list(model_weights.keys())
    if not providers or not votes:
        return

    n = len(providers)
    content_w = 166.0
    agent_w, agg_w = 36.0, 22.0
    col_model = max(24.0, (content_w - agent_w - agg_w) / n) if n else 28.0
    col_widths = [agent_w * mm] + [col_model * mm] * n + [agg_w * mm]

    headers = ["Agent"] + [str(p) for p in providers] + ["汇总均分"]
    table_data = [headers]

    for v in votes:
        aid = v["agent_id"]
        role = (v.get("role", "") or aid)[:8]
        row = [role]
        contrib_sum = 0.0
        for p in providers:
            s = model_scores.get(p, {}).get(aid, float(v.get("score_0_100", 50)))
            w = model_weights.get(p, {}).get(aid, 0)
            row.append(f"{s:.1f}/{w:.0%}")
            contrib_sum += s * w
        contrib_sum /= len(providers) if providers else 1
        row.append(f"{contrib_sum:.1f}")
        table_data.append(row)

    total_row = ["合计"]
    for p in providers:
        total_row.append(f"{model_totals.get(p, 0):.1f}")
    final_avg = result.get("final_score", 0.0)
    total_row.append(f"{final_avg:.1f}")
    table_data.append(total_row)

    tbl = Table(table_data, colWidths=col_widths[: 1 + n + 1])
    style = [
        ("FONTNAME", (0, 0), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0D3B66")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F0F4F8")),
    ]
    tbl.setStyle(TableStyle(style))
    flow.append(Paragraph("多模型权重与评分汇总（前部）", st_h))
    flow.append(tbl)
    flow.append(
        Paragraph(
            "<b>说明：</b>总调度在 Agent 执行前将 Agent 定义传给各模型，各模型独立分配权重（和为100%）；"
            "Agent 执行后，各模型独立对每项研判打分。表中「M(分/权)」为各模型对该 Agent 的分数与权重。",
            st_body,
        )
    )
    flow.append(Spacer(1, 8))


# ─────────────────────────────────────────────────────────────────
#  公司概览
# ─────────────────────────────────────────────────────────────────
def _add_company_overview(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
) -> None:
    """公司概览：代码、名称、市值、PE/PB、换手率等。"""
    symbol = result.get("symbol", "")
    name = result.get("name", "")
    feats = result.get("analysis_features") or {}
    if not isinstance(feats, dict):
        flow.append(Paragraph(f"标的：{symbol} {name}。", st_body))
        return
    pe = feats.get("pe_ttm")
    pb = feats.get("pb")
    total_mv = feats.get("total_mv")
    turn = feats.get("turnover_rate")
    close_price = feats.get("close")
    flow.append(Paragraph("公司概览", st_h))
    rows = [["项目", "内容"]]
    rows.append(["股票代码 / 名称", f"{symbol}  {name}"])
    if close_price is not None:
        rows.append(["最新收盘价", f"{float(close_price):.2f} 元"])
    if total_mv is not None:
        mv_yi = total_mv / 1e8 if total_mv > 1e8 else total_mv / 1e4
        rows.append(["总市值", f"{mv_yi:.2f}亿元" if total_mv > 1e8 else f"{mv_yi:.2f}万元"])
    if pe is not None:
        rows.append(["市盈率 PE-TTM", f"{float(pe):.2f}"])
    if pb is not None:
        rows.append(["市净率 PB", f"{float(pb):.2f}"])
    if turn is not None:
        rows.append(["换手率", f"{float(turn):.2f}%"])
    if len(rows) <= 1:
        flow.append(Paragraph(f"标的：{symbol} {name}。", st_body))
        return
    tbl = Table(rows, colWidths=[40 * mm, 85 * mm])
    tbl.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    flow.append(tbl)
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  关键价位与 Fibonacci
# ─────────────────────────────────────────────────────────────────
def _add_key_levels_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
) -> None:
    """关键价位与 Fibonacci 回撤位表。"""
    feats = result.get("analysis_features") or {}
    key_levels = feats.get("key_levels") if isinstance(feats, dict) else {}
    if not key_levels or not key_levels.get("ok"):
        return
    flow.append(Paragraph("关键价位与 Fibonacci 回撤", st_h))
    rows = [["价位类型", "价格(元)", "说明"]]
    rows.append(["波段高点", str(key_levels.get("band_high", "")), "近期高点"])
    rows.append(["波段低点", str(key_levels.get("band_low", "")), "近期低点"])
    rows.append(["当前价", str(key_levels.get("current", "")), "最新收盘"])
    rows.append(["23.6% 回撤", str(key_levels.get("retrace_236", "")), "弱支撑/阻力"])
    rows.append(["38.2% 回撤", str(key_levels.get("retrace_382", "")), "中性支撑/阻力"])
    rows.append(["50.0% 回撤", str(key_levels.get("retrace_50", "")), "心理关口"])
    rows.append(["61.8% 回撤", str(key_levels.get("retrace_618", "")), "强支撑/阻力（黄金比例）"])
    tbl = Table(rows, colWidths=[38 * mm, 32 * mm, 65 * mm])
    tbl.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    flow.append(tbl)
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  多周期趋势状态
# ─────────────────────────────────────────────────────────────────
def _add_multi_timeframe_trend_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
) -> None:
    """多周期趋势状态表：月线/周线/日线 趋势状态、动量、RSI 等。"""
    feats = result.get("analysis_features") or {}
    kli = feats.get("kline_indicators") if isinstance(feats, dict) else {}
    if not isinstance(kli, dict):
        return

    def _val(v: Any, fmt: str = ".2f") -> str:
        return (fmt.format(float(v)) if v is not None else "-") if fmt else (str(v) if v is not None else "-")

    tf_order = ["month", "week", "day"]
    tf_labels = {"month": "月线", "week": "周线", "day": "日线"}
    rows = [["周期", "趋势状态", "动量(10根)%", "RSI(14)", "趋势斜率%", "波动率%", "K线组合"]]
    has_any = False
    for tf in tf_order:
        ind = kli.get(tf)
        if not isinstance(ind, dict) or not ind.get("ok"):
            continue
        label = ind.get("timeframe_label", tf_labels.get(tf, tf))
        mom = ind.get("momentum_10")
        rsi = ind.get("rsi")
        slope = ind.get("trend_slope_pct")
        vol_tf = ind.get("volatility_20")
        combo = ind.get("kline_combo_5") or "-"
        if mom is not None:
            state = "偏多↑" if float(mom) > 2 else ("偏空↓" if float(mom) < -2 else "震荡→")
        else:
            state = "-"
        rows.append([label, state, _val(mom), _val(rsi), _val(slope), _val(vol_tf), str(combo)[:10]])
        has_any = True
    if not has_any:
        return
    flow.append(Paragraph("多周期趋势状态（月线/周线/日线）", st_h))
    tbl = Table(rows, colWidths=[18 * mm, 18 * mm, 24 * mm, 20 * mm, 22 * mm, 22 * mm, 36 * mm])
    tbl.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    flow.append(tbl)
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  均线系统（MA5-250）
# ─────────────────────────────────────────────────────────────────
def _add_ma_system_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
) -> None:
    """均线系统表：MA5/MA10/MA20/MA60/MA120/MA250 当前值与价格偏离度、多头/空头信号。"""
    feats = result.get("analysis_features") or {}
    kli = feats.get("kline_indicators") if isinstance(feats, dict) else {}
    day_ind = kli.get("day") if isinstance(kli, dict) else {}
    if not isinstance(day_ind, dict) or not day_ind.get("ok"):
        return
    ma_system = day_ind.get("ma_system")
    if not ma_system:
        return
    curr = day_ind.get("close") or feats.get("close")
    flow.append(Paragraph("均线系统（日线 MA5 – MA250）", st_h))
    rows = [["均线", "当前值(元)", "偏离度%", "信号"]]
    ma_names = {
        "ma5": "MA5 (1周)", "ma10": "MA10 (2周)", "ma20": "MA20 (月线)",
        "ma60": "MA60 (季线)", "ma120": "MA120 (半年线)", "ma250": "MA250 (年线)",
    }
    for key, label in ma_names.items():
        entry = ma_system.get(key, {})
        val = entry.get("value")
        pct = entry.get("pct_above")
        if val is None:
            signal = "-（数据不足）"
            rows.append([label, "-", "-", signal])
        else:
            if pct is not None:
                signal = "价格在均线上方▲" if pct > 0 else "价格在均线下方▼"
            else:
                signal = "-"
            rows.append([label, f"{val:.2f}", f"{pct:+.2f}%" if pct is not None else "-", signal])

    # 判断多空排列
    above_cnt = sum(
        1 for k in ["ma5", "ma10", "ma20", "ma60"]
        if ma_system.get(k, {}).get("pct_above") is not None and ma_system[k]["pct_above"] > 0
    )
    arrange = "多头排列（短期均线均在价格下方）" if above_cnt == 4 else (
        "空头排列（短期均线均在价格上方）" if above_cnt == 0 else f"混合排列（{above_cnt}/4 短均线低于价格）"
    )

    tbl = Table(rows, colWidths=[38 * mm, 30 * mm, 24 * mm, 63 * mm])
    tbl.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    flow.append(tbl)
    flow.append(Paragraph(f"<b>均线排列研判：</b>{arrange}", st_body))
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  技术指标读数（多周期完整版）
# ─────────────────────────────────────────────────────────────────
def _add_tech_indicators_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
) -> None:
    """技术指标完整读数表（三周期：日线/周线/月线）。"""
    feats = result.get("analysis_features") or {}
    kli = feats.get("kline_indicators") if isinstance(feats, dict) else {}
    if not isinstance(kli, dict):
        return

    flow.append(Paragraph("技术指标量化读数（日线/周线/月线）", st_h))

    tf_order = [("day", "日线"), ("week", "周线"), ("month", "月线")]
    for tf, label in tf_order:
        ind = kli.get(tf)
        if not isinstance(ind, dict) or not ind.get("ok"):
            continue

        def _v(v, fmt=".1f"):
            return (fmt.format(v) if v is not None else "-") if fmt else str(v) if v is not None else "-"

        rows = [["指标", "读数", "指标", "读数"]]
        rows.append(["RSI(14)", _v(ind.get("rsi")), "MACD DIF", _v(ind.get("macd_dif"), ".4f")])
        rows.append(["MACD DEA", _v(ind.get("macd_dea"), ".4f"), "MACD 柱", _v(ind.get("macd_hist"), ".4f")])
        rows.append(["KDJ K", _v(ind.get("kdj_k")), "KDJ D", _v(ind.get("kdj_d"))])
        rows.append(["KDJ J", _v(ind.get("kdj_j")), "StochRSI", _v(ind.get("stoch_rsi"))])
        rows.append(["布林上轨", _v(ind.get("boll_upper"), ".2f"), "布林中轨", _v(ind.get("boll_mid"), ".2f")])
        rows.append(["布林下轨", _v(ind.get("boll_lower"), ".2f"), "趋势斜率%", _v(ind.get("trend_slope_pct"))])
        rows.append(["动量(10根)%", _v(ind.get("momentum_10")), "波动率%", _v(ind.get("volatility_20"))])

        flow.append(Paragraph(f"<b>▎{label}</b>", st_body))
        tbl = Table(rows, colWidths=[32 * mm, 28 * mm, 32 * mm, 28 * mm])
        tbl.setStyle(
            TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        flow.append(tbl)
        flow.append(Spacer(1, 4))
    flow.append(Spacer(1, 4))


# ─────────────────────────────────────────────────────────────────
#  多模型权重汇总表（前置部分）
# ─────────────────────────────────────────────────────────────────
def _add_weighted_score_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
    level_color_map: dict[str, str],
    final_score: float,
) -> None:
    """全智能体加权评分汇总表：列出所有Agent，9核心在前，扩展在后。"""
    votes = result.get("agent_votes", [])
    if not votes:
        return

    CORE_DIMS = ["TREND", "TECH", "DERIV_MARGIN", "LIQ", "CAPITAL_FLOW", "BETA",
                 "SECTOR_POLICY", "SENTIMENT", "FUNDAMENTAL"]
    core_votes = [v for v in votes if v.get("dim_code", "") in CORE_DIMS]
    other_votes = [v for v in votes if v.get("dim_code", "") not in CORE_DIMS]
    # 核心维度按 CORE_DIMS 顺序排列
    core_order = {d: i for i, d in enumerate(CORE_DIMS)}
    core_votes.sort(key=lambda v: core_order.get(v.get("dim_code", ""), 99))
    ordered_votes = core_votes + other_votes

    flow.append(Paragraph("多智能体加权评分体系", st_h))
    total_weight = sum(float(v.get("weight", 0)) for v in votes) or 1.0

    # 表头：序号 | 智能体 | 评分 | 五级建议 | 权重
    table_data = [["#", "智能体", "评分", "五级建议", "权重"]]

    for idx, v in enumerate(ordered_votes, 1):
        role = v.get("role", v.get("dim_code", ""))
        # 截断到12个字符（约6中文字）
        if len(role) > 12:
            role = role[:12]
        score = float(v.get("score_0_100", 50))
        level_cn = _score_to_decision_level_cn(score)
        weight = float(v.get("weight", 0))
        weight_pct = weight / total_weight if total_weight else 0
        # 核心维度标记 *
        is_core = v.get("dim_code", "") in CORE_DIMS
        num_str = f"{idx}" if not is_core else f"{idx}*"
        table_data.append([num_str, role, f"{score:.1f}", level_cn, f"{weight_pct:.1%}"])

    # 合计行
    table_data.append(["", "综合评分", f"{final_score:.1f}",
                        _score_to_decision_level_cn(final_score), "100%"])

    col_w = [10 * mm, 48 * mm, 18 * mm, 26 * mm, 18 * mm]
    table = Table(table_data, colWidths=col_w)

    n_core = len(core_votes)
    tbl_style = [
        ("FONTNAME", (0, 0), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0D3B66")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        # 合计行底色
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F0F4F8")),
        ("FONTNAME", (0, -1), (-1, -1), bold_font),
        # 核心维度行浅底色
        ("BACKGROUND", (0, 1), (-1, n_core), colors.HexColor("#FAFCFF")),
    ]
    # 核心与扩展分隔线
    if n_core < len(ordered_votes):
        tbl_style.append(("LINEBELOW", (0, n_core), (-1, n_core), 1.0, colors.HexColor("#0D3B66")))

    # 五级建议列着色
    level_col = 3
    for i, row in enumerate(table_data[1:], 1):
        if len(row) > level_col:
            lvl = row[level_col]
            lc = level_color_map.get(lvl, "#222222")
            tbl_style.append(("TEXTCOLOR", (level_col, i), (level_col, i), colors.HexColor(lc)))
            tbl_style.append(("FONTNAME", (level_col, i), (level_col, i), bold_font))
    table.setStyle(TableStyle(tbl_style))
    flow.append(table)
    flow.append(
        Paragraph(
            "<b>说明：</b>带*为9核心维度。最终评分 = Σ(各Agent评分×LLM权重) / Σ(权重)。"
            "五级映射：强烈买入(≥85)、弱买入(70-85)、观望(50-70)、弱卖出(40-50)、强烈卖出(&lt;40)。",
            st_body,
        )
    )
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  行业竞争格局（来自 industry agent 研判）
# ─────────────────────────────────────────────────────────────────
def _add_industry_competition(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
) -> None:
    """行业竞争格局：尝试从 industry/sector_policy agent 的 reason 中提取竞争信息。"""
    votes = result.get("agent_votes", [])
    industry_vote = next(
        (v for v in votes if v.get("dim_code", "") in {"INDUSTRY", "SECTOR_POLICY"}), None
    )
    flow.append(Paragraph("行业与竞争格局", st_h))
    if industry_vote:
        role = industry_vote.get("role", "板块/行业分析师")
        score = float(industry_vote.get("score_0_100", 50))
        reason = str(industry_vote.get("reason", ""))
        flow.append(
            Paragraph(
                f"<b>{role}</b>（评分 {score:.1f}）：{reason[:300]}{'…' if len(reason) > 300 else ''}",
                st_body,
            )
        )
    else:
        flow.append(
            Paragraph(
                "行业 CR3/CR5、主要竞争对手及核心竞争优势（需行业分析智能体数据；"
                "可结合情景分析与板块政策研判综合判断）。",
                st_body,
            )
        )
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  情景分析三情景表
# ─────────────────────────────────────────────────────────────────
def _add_scenario_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
    final_score: float,
    decision_level_cn: str,
) -> None:
    """情景分析：乐观/中性/悲观三情景概率表 + 分批建仓策略。"""
    scenario_text = result.get("scenario_analysis", "")
    position_text = result.get("position_strategy", "")

    flow.append(Paragraph("情景分析与可执行策略", st_h))

    # 三情景概率表（结构化）
    feats = result.get("analysis_features") or {}
    close_price = feats.get("close") or feats.get("kline_indicators", {}).get("day", {}).get("close")

    if close_price:
        cp = float(close_price)
        bull_prob, base_prob, bear_prob = 30, 50, 20
        # 根据评分微调概率
        if final_score >= 80:
            bull_prob, base_prob, bear_prob = 45, 40, 15
        elif final_score >= 65:
            bull_prob, base_prob, bear_prob = 35, 45, 20
        elif final_score < 50:
            bull_prob, base_prob, bear_prob = 15, 40, 45
        elif final_score < 40:
            bull_prob, base_prob, bear_prob = 10, 30, 60

        rows = [["情景", "概率", "触发条件", "价格目标(元)", "止损参考"]]
        rows.append([
            "乐观", f"{bull_prob}%",
            "放量突破关键阻力 / 业绩超预期 / 板块政策催化",
            f"{cp * 1.15:.2f}（+15%）",
            f"{cp * 0.93:.2f}（-7%）",
        ])
        rows.append([
            "中性", f"{base_prob}%",
            "维持当前区间震荡 / 无明显催化剂",
            f"{cp * 1.05:.2f}（+5%）",
            f"{cp * 0.95:.2f}（-5%）",
        ])
        rows.append([
            "悲观", f"{bear_prob}%",
            "放量跌破支撑 / 业绩低于预期 / 板块政策收紧",
            f"{cp * 0.88:.2f}（-12%）",
            f"{cp * 0.90:.2f}（-10%）",
        ])
        tbl = Table(rows, colWidths=[16 * mm, 14 * mm, 56 * mm, 36 * mm, 33 * mm])
        tbl_style = [
            ("FONTNAME", (0, 0), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFF0F0")),
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F0F8FF")),
            ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F0FFF0")),
        ]
        tbl.setStyle(TableStyle(tbl_style))
        flow.append(tbl)
        flow.append(Spacer(1, 4))

        # 分批建仓策略（3批）
        if decision_level_cn in {"强烈买入", "弱买入"}:
            flow.append(Paragraph("<b>分批建仓策略（仅供参考）</b>", st_body))
            pos_rows = [["批次", "建仓时机", "建议仓位", "介入价格参考", "止损位"]]
            pos_rows.append([
                "第一批", "当前价附近 / 量能温和放大",
                "30%", f"{cp:.2f}（当前）", f"{cp * 0.95:.2f}（-5%）"
            ])
            pos_rows.append([
                "第二批", "突破近期阻力并回踩确认",
                "30%", f"{cp * 1.03:.2f}（+3%）", f"{cp * 0.97:.2f}（-3%）"
            ])
            pos_rows.append([
                "第三批", "放量创新高 / 催化剂落地",
                "40%", f"{cp * 1.06:.2f}（+6%）", f"{cp * 1.00:.2f}（保本）"
            ])
            pos_tbl = Table(pos_rows, colWidths=[16 * mm, 44 * mm, 20 * mm, 34 * mm, 30 * mm])
            pos_tbl.setStyle(
                TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), body_font),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
                    ("FONTNAME", (0, 0), (-1, 0), bold_font),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ])
            )
            flow.append(pos_tbl)
            flow.append(Spacer(1, 4))

    # LLM 生成的情景分析文本（若有）
    if scenario_text:
        flow.append(Paragraph(f"<b>情景分析（AI生成）：</b>{scenario_text}", st_body))
    if position_text:
        flow.append(Paragraph(f"<b>止损与仓位建议（AI生成）：</b>{position_text}", st_body))
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  主构建函数
# ─────────────────────────────────────────────────────────────────
def build_investor_pdf(run_dir: Path, result: dict[str, Any]) -> Path:
    body_font, bold_font = _register_fonts()
    symbol = str(result.get("symbol", "")).strip()
    name = str(result.get("name", "")).strip()
    pdf_name = _safe_filename(f"{symbol}_{name}_投资者报告.pdf")
    pdf_path = run_dir / pdf_name

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="个股决策报告",
    )

    styles = getSampleStyleSheet()
    st_title = ParagraphStyle(
        "TitleCN", parent=styles["Title"],
        fontName=bold_font, fontSize=16, leading=20,
        textColor=colors.HexColor("#111111"),
    )
    st_h = ParagraphStyle(
        "H2CN", parent=styles["Heading2"],
        fontName=bold_font, fontSize=12, leading=16,
        textColor=colors.HexColor("#0D3B66"),
    )
    st_body = ParagraphStyle(
        "BodyCN", parent=styles["BodyText"],
        fontName=body_font, fontSize=11, leading=17,
        textColor=colors.black,
    )
    st_bold_inline = ParagraphStyle(
        "BodyBoldInline", parent=st_body,
        fontName=body_font, fontSize=11, leading=17,
    )

    symbol = result.get("symbol", "")
    name = result.get("name", "")
    final_decision = result.get("final_decision", "hold")
    final_score = float(result.get("final_score", 0.0))
    provider = result.get("provider", "grok")

    decision_level_map = {
        "strong_buy": "强烈买入",
        "weak_buy": "弱买入",
        "hold": "观望",
        "weak_sell": "弱卖出",
        "strong_sell": "强烈卖出",
    }
    decision_level = result.get("decision_level", "")
    decision_level_cn = decision_level_map.get(decision_level, decision_level or "观望")
    decision_map = {"buy": "买入", "hold": "观望", "sell": "卖出"}
    decision_cn = decision_map.get(final_decision, final_decision)

    level_color_map = {
        "强烈买入": "#C41E3A",
        "弱买入": "#E74C3C",
        "观望": "#0066CC",
        "弱卖出": "#27AE60",
        "强烈卖出": "#228B22",
    }
    decision_color = level_color_map.get(decision_level_cn, "#222222")

    votes = result.get("agent_votes", [])
    top_items = sorted(
        votes,
        key=lambda x: (abs(float(x.get("score_0_100", 50)) - 50) * float(x.get("weight", 0))),
        reverse=True,
    )[:5]

    flow = []

    # ── 标题 ──
    flow.append(Paragraph("A股个股多智能体决策报告（投资者版）", st_title))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(
        f"标的：<font name='{bold_font}'>{symbol}  {name}</font>"
        f"　　生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        st_body,
    ))
    multi_mode_desc = result.get("multi_model_weight_mode")
    engine_text = (
        "多模型自分配权重 + 加权评分均化（已屏蔽辩论）"
        if multi_mode_desc
        else f"{len(votes)}智能体辩论 + 加权评分"
    )
    flow.append(Paragraph(
        f"分析引擎：{provider}　　评估方式：{engine_text}",
        st_body,
    ))
    flow.append(Spacer(1, 8))

    # ── 多智能体框架说明 ──
    _add_framework_intro_table(flow, result, body_font, bold_font, st_h, st_body)

    # ── 核心结论 ──
    flow.append(Paragraph("核心结论（重点）", st_h))
    flow.append(
        Paragraph(
            f"最终建议（五级）：<b><font name='{bold_font}' color='{decision_color}'>"
            f"{decision_level_cn}</font></b>　　"
            f"综合评分：<b><font name='{bold_font}'>{final_score:.2f}</font></b>（0-100分制）",
            st_bold_inline,
        )
    )
    flow.append(
        Paragraph(
            f"<b>决策解释：</b>当前评分 {final_score:.2f} 落在五级「<b>"
            f"<font color='{decision_color}'>{decision_level_cn}</font></b>」。"
            "评级标准：强烈买入(≥85)、弱买入(70-85)、观望(50-70)、弱卖出(40-50)、强烈卖出(&lt;40)。",
            st_body,
        )
    )
    multi_providers = result.get("multi_eval_providers", [])
    if multi_providers:
        flow.append(
            Paragraph(
                f"<b>大模型投票：</b>本报告由 {', '.join(multi_providers)} 等模型对 {len(votes)} 个智能体结论进行综合评估后形成。",
                st_body,
            )
        )
    flow.append(
        Paragraph(
            "该结论由多维数据（K线历史、趋势、技术、资金、基本面、政策与情绪）共同投票形成，"
            "适合作为决策参考，不替代个人风险承受评估。",
            st_body,
        )
    )
    short_hold = result.get("short_term_hold", "")
    medium_hold = result.get("medium_long_term_hold", "")
    if short_hold or medium_hold:
        flow.append(Paragraph(f"<b>短线建议：</b>{short_hold}；<b>中长线建议：</b>{medium_hold}。", st_body))
    flow.append(Spacer(1, 8))

    # ── 全智能体评分总表（首页核心） ──
    _add_weighted_score_table(flow, result, st_h, st_body, body_font, bold_font,
                              level_color_map, final_score)

    # ── 多模型权重表（若启用多模型模式）──
    if result.get("multi_model_weight_mode") and result.get("model_weights") and result.get("model_totals"):
        _add_multi_model_table(flow, result, body_font, bold_font, st_h, st_body)

    # ── 公司概览 ──
    _add_company_overview(flow, result, st_h, st_body, body_font, bold_font)

    # ── 关键价位与 Fibonacci ──
    _add_key_levels_table(flow, result, st_h, st_body, body_font)

    # ── 多周期趋势状态 ──
    _add_multi_timeframe_trend_table(flow, result, st_h, st_body, body_font)

    # ── 均线系统（MA5-250）──
    _add_ma_system_table(flow, result, st_h, st_body, body_font, bold_font)

    # ── 技术指标完整读数（三周期）──
    _add_tech_indicators_table(flow, result, st_h, st_body, body_font)

    # ── 各 Agent 简要研判 ──
    TECH_STRUCTURE_DIMS = {"TREND", "TECH", "TOP_STRUCTURE", "BOTTOM_STRUCTURE", "LIQ",
                           "CAPITAL_FLOW", "QUANT", "BETA", "DERIV_MARGIN"}
    flow.append(Paragraph("各 Agent 简要研判分析", st_h))
    for v in votes:
        role = v.get("role", "")
        dim_code = v.get("dim_code", "")
        score = float(v.get("score_0_100", 50))
        level_cn = _score_to_decision_level_cn(score)
        reason = str(v.get("reason", ""))
        one_line = (reason[:90] + "…") if len(reason) > 90 else reason
        level_tag = "〔日/周/月线〕" if dim_code in TECH_STRUCTURE_DIMS else ""
        lc = level_color_map.get(level_cn, "#222222")
        flow.append(
            Paragraph(
                f"• <font name='{bold_font}'>{role}</font>：评分 <b>{score:.1f}</b>，"
                f"建议 <b><font color='{lc}'>{level_cn}</font></b>。{one_line} {level_tag}",
                st_body,
            )
        )
    flow.append(Spacer(1, 8))

    # ── 关键依据（Top5）──
    flow.append(Paragraph("关键依据（Top5 高权重智能体）", st_h))
    for item in top_items:
        role = item.get("role", "")
        score = float(item.get("score_0_100", 50))
        level_cn = _score_to_decision_level_cn(score)
        conf = float(item.get("confidence_0_1", 0.5))
        reason = str(item.get("reason", ""))
        reason_display = reason[:200] if reason else ""
        lc = level_color_map.get(level_cn, "#222222")
        flow.append(
            Paragraph(
                f"• <font name='{bold_font}'>{role}</font>：建议 <b><font color='{lc}'>{level_cn}</font></b>，"
                f"评分 <b>{score:.1f}</b>，置信度 {conf:.2f}。{reason_display}",
                st_body,
            )
        )
    flow.append(Spacer(1, 8))

    # ── Bull vs Bear 辩论 ──
    debate = result.get("debate_bull_bear") or {}
    if debate and (debate.get("bull_reason") or debate.get("bear_reason")):
        flow.append(Paragraph("Bull vs Bear 辩论", st_h))
        bull_role = debate.get("bull_role", debate.get("bull_agent_id", "看多"))
        bear_role = debate.get("bear_role", debate.get("bear_agent_id", "看空"))
        bull_r = str(debate.get("bull_reason", ""))
        bear_r = str(debate.get("bear_reason", ""))
        flow.append(Paragraph(
            f"<b>看多（Bull）</b> — {bull_role} 评分 {debate.get('bull_score', 0)}："
            f"{bull_r[:220]}{'…' if len(bull_r) > 220 else ''}",
            st_body,
        ))
        flow.append(Paragraph(
            f"<b>看空（Bear）</b> — {bear_role} 评分 {debate.get('bear_score', 0)}："
            f"{bear_r[:220]}{'…' if len(bear_r) > 220 else ''}",
            st_body,
        ))
        flow.append(Paragraph(f"<b>Judge 仲裁：</b>{debate.get('judge_msg', '')}", st_body))
        flow.append(Spacer(1, 6))

    # ── 行业竞争格局 ──
    _add_industry_competition(flow, result, st_h, st_body, body_font)

    # ── 风险提示矩阵 ──
    flow.append(Paragraph("风险提示矩阵", st_h))
    risk_items = []
    for v in votes:
        if not isinstance(v, dict):
            continue
        score = float(v.get("score_0_100", 50))
        role = v.get("role", "")
        if score < 40 and role:
            risk_items.append(f"{role}评分偏低({score:.0f})，关注该维度风险")
        elif score < 50 and role and "基本面" in role:
            risk_items.append(f"{role}偏弱，关注经营与估值风险")
    if risk_items:
        for r in risk_items[:5]:
            flow.append(Paragraph(f"• {r}", st_body))
    else:
        flow.append(Paragraph(
            "市场风险、流动性风险、估值波动等详见各 Agent 研判；"
            "综合评分中性及以上时无单项强风险标注。",
            st_body,
        ))
    flow.append(Spacer(1, 6))

    # ── 情景分析与策略 ──
    _add_scenario_table(flow, result, st_h, st_body, body_font, bold_font,
                        final_score, decision_level_cn)

    # ── 结果摘要 ──
    flow.append(Paragraph("结果摘要", st_h))
    summary_data = [["指标", "结果"]]
    summary_data.append(["股票", f"{symbol}  {name}"])
    summary_data.append([
        "最终建议（五级）",
        Paragraph(f"<b><font color='{decision_color}'>{decision_level_cn}</font></b>", st_body),
    ])
    summary_data.append(["综合评分", f"{final_score:.2f}"])
    if short_hold or medium_hold:
        summary_data.append(["短线建议", short_hold])
        summary_data.append(["中长线建议", medium_hold])
    summary_data.append(["辩论轮次", str(result.get("debate_rounds", 0))])
    summary_data.append(["智能体数量", str(len(votes))])
    summary_table = Table(summary_data, colWidths=[38 * mm, 120 * mm])
    summary_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), body_font),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0D3B66")),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    flow.append(summary_table)
    flow.append(Spacer(1, 10))
    flow.append(Paragraph(
        "风险提示：市场有风险，投资需谨慎。本报告仅供研究与参考，不构成投资建议。",
        st_body,
    ))

    doc.build(flow)
    return pdf_path
