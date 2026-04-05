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


# ─────────────────────────────────────────────────────────────────
#  表格单元格自动换行辅助
# ─────────────────────────────────────────────────────────────────
_CELL_STYLES: dict[str, ParagraphStyle] = {}


def _esc(text) -> str:
    """Escape XML special chars for reportlab Paragraph."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cell(text: str, font: str = "", size: float = 9, bold: bool = False,
          color: str = "#222222", align: int = 0) -> Paragraph:
    """将文本包裹为 Paragraph 以支持表格单元格内自动换行。

    align: 0=左对齐 1=居中 2=右对齐
    """
    key = f"{font}_{size}_{bold}_{color}_{align}"
    if key not in _CELL_STYLES:
        _CELL_STYLES[key] = ParagraphStyle(
            f"Cell_{key}",
            fontName=font or "STSong-Light",
            fontSize=size,
            leading=size + 3,
            textColor=colors.HexColor(color),
            wordWrap="CJK",
            alignment=align,
        )
    safe = _esc(text)
    if bold:
        safe = f"<b>{safe}</b>"
    return Paragraph(safe, _CELL_STYLES[key])


def _cells(row: list, font: str = "", size: float = 9, bold: bool = False) -> list:
    """批量转换一行的所有单元格。"""
    return [_cell(str(c), font, size, bold) for c in row]


# ─────────────────────────────────────────────────────────────────
#  顶底结构信号辅助（独立评估，交叉印证）
# ─────────────────────────────────────────────────────────────────

def _structure_signal_label(dim_code: str, score: float) -> str:
    """顶/底结构专用标签（不使用通用五级映射）。

    BOTTOM_STRUCTURE: 高分=底部信号强(买入参考)，低分=无底部信号(中性)
    TOP_STRUCTURE: 高分=顶部信号强(卖出警示)，低分=无顶部信号(中性)
    """
    if dim_code == "BOTTOM_STRUCTURE":
        if score >= 70:
            return "底部显著(买入参考)"
        if score >= 55:
            return "底部偏弱"
        return "无底部信号(中性)"
    if dim_code == "TOP_STRUCTURE":
        if score >= 70:
            return "顶部显著(卖出警示)"
        if score >= 55:
            return "顶部偏弱"
        return "无顶部信号(中性)"
    return ""


def _structure_tf_breakdown(kline_indicators: dict) -> dict[str, list[str]]:
    """从 kline_indicators 计算顶/底结构各周期信号级别。"""
    tf_label_map = {"day": "日线", "week": "周线", "month": "月线"}
    top_tfs: list[str] = []
    bot_tfs: list[str] = []
    for tf in ("day", "week", "month"):
        ind = kline_indicators.get(tf)
        if not isinstance(ind, dict) or not ind.get("ok"):
            continue
        upper = float(ind.get("upper_shadow_ratio", 0))
        lower = float(ind.get("lower_shadow_ratio", 0))
        mom = float(ind.get("momentum_10", 0))
        amp = float(ind.get("amplitude_20", 0))
        label = tf_label_map[tf]
        if (upper > 40 and mom < 0) or upper > 35 or (mom < -5 and amp > 15):
            top_tfs.append(label)
        if (lower > 40 and mom > 0) or lower > 35 or mom > 5:
            bot_tfs.append(label)
    return {"top": top_tfs, "bottom": bot_tfs}


def _structure_color(dim_code: str, score: float) -> str:
    """顶底结构专用颜色。"""
    if dim_code == "BOTTOM_STRUCTURE":
        if score >= 70:
            return "#C41E3A"  # 红(买入信号)
        return "#666666"
    if dim_code == "TOP_STRUCTURE":
        if score >= 70:
            return "#228B22"  # 绿(卖出信号)
        return "#666666"
    return "#222222"


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
    rows = [_cells(["模块", "核心智能体", "分析维度", "数据来源"], body_font, 9, True)]
    modules = [
        ("基本面", "基本面分析师", "PE/PB/市值/估值/盈利质量", "Tushare / AKShare"),
        ("技术面", "趋势/技术指标/K线视觉/结构", "多周期K线、MA均线、MACD/RSI/KDJ/布林", "历史行情 / 图像识别"),
        ("资金面", "资金流向/精细资金流/主力行为", "主力净流入、超大单、北向资金、换手率", "东方财富 / AKShare"),
        ("情绪面", "情绪/深度舆情/NLP情绪", "新闻热度、社媒情绪、公告舆情", "新闻爬取 / NLP分析"),
        ("宏观政策", "宏观关联/板块与政策/大盘联动", "利率汇率、板块政策、指数联动", "财经新闻 / 宏观数据"),
        ("综合决策", "量化/风险/顶底结构/融资融券", "综合加权评分、Bull-Bear辩论、五级决策", "多维数据融合"),
    ]
    for m in modules:
        rows.append(_cells(list(m), body_font, 9))
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

    headers = _cells(["Agent"] + [str(p) for p in providers] + ["汇总均分"], bold_font, 8, True)
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
        table_data.append(_cells(row, body_font, 8))

    total_row = ["合计"]
    for p in providers:
        total_row.append(f"{model_totals.get(p, 0):.1f}")
    final_avg = result.get("final_score", 0.0)
    total_row.append(f"{final_avg:.1f}")
    table_data.append(_cells(total_row, bold_font, 8, True))

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
    rows = [_cells(["项目", "内容"], bold_font, 10, True)]
    rows.append(_cells(["股票代码 / 名称", f"{symbol}  {name}"], body_font, 10))
    if close_price is not None:
        rows.append(_cells(["最新收盘价", f"{float(close_price):.2f} 元"], body_font, 10))
    if total_mv is not None:
        mv_yi = total_mv / 1e8 if total_mv > 1e8 else total_mv / 1e4
        rows.append(_cells(["总市值", f"{mv_yi:.2f}亿元" if total_mv > 1e8 else f"{mv_yi:.2f}万元"], body_font, 10))
    if pe is not None:
        rows.append(_cells(["市盈率 PE-TTM", f"{float(pe):.2f}"], body_font, 10))
    if pb is not None:
        rows.append(_cells(["市净率 PB", f"{float(pb):.2f}"], body_font, 10))
    if turn is not None:
        rows.append(_cells(["换手率", f"{float(turn):.2f}%"], body_font, 10))
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
    rows = [_cells(["价位类型", "价格(元)", "说明"], body_font, 9, True)]
    rows.append(_cells(["波段高点", str(key_levels.get("band_high", "")), "近期高点"], body_font, 9))
    rows.append(_cells(["波段低点", str(key_levels.get("band_low", "")), "近期低点"], body_font, 9))
    rows.append(_cells(["当前价", str(key_levels.get("current", "")), "最新收盘"], body_font, 9))
    rows.append(_cells(["23.6% 回撤", str(key_levels.get("retrace_236", "")), "弱支撑/阻力"], body_font, 9))
    rows.append(_cells(["38.2% 回撤", str(key_levels.get("retrace_382", "")), "中性支撑/阻力"], body_font, 9))
    rows.append(_cells(["50.0% 回撤", str(key_levels.get("retrace_50", "")), "心理关口"], body_font, 9))
    rows.append(_cells(["61.8% 回撤", str(key_levels.get("retrace_618", "")), "强支撑/阻力（黄金比例）"], body_font, 9))
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
    rows = [_cells(["周期", "趋势状态", "动量(10根)%", "RSI(14)", "趋势斜率%", "波动率%", "K线组合"], body_font, 8, True)]
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
        rows.append(_cells([label, state, _val(mom), _val(rsi), _val(slope), _val(vol_tf), str(combo)[:10]], body_font, 8))
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
    rows = [_cells(["均线", "当前值(元)", "偏离度%", "信号"], bold_font, 9, True)]
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
            rows.append(_cells([label, "-", "-", signal], body_font, 9))
        else:
            if pct is not None:
                signal = "价格在均线上方▲" if pct > 0 else "价格在均线下方▼"
            else:
                signal = "-"
            rows.append(_cells([label, f"{val:.2f}", f"{pct:+.2f}%" if pct is not None else "-", signal], body_font, 9))

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

        rows = [_cells(["指标", "读数", "指标", "读数"], body_font, 9, True)]
        rows.append(_cells(["RSI(14)", _v(ind.get("rsi")), "MACD DIF", _v(ind.get("macd_dif"), ".4f")], body_font, 9))
        rows.append(_cells(["MACD DEA", _v(ind.get("macd_dea"), ".4f"), "MACD 柱", _v(ind.get("macd_hist"), ".4f")], body_font, 9))
        rows.append(_cells(["KDJ K", _v(ind.get("kdj_k")), "KDJ D", _v(ind.get("kdj_d"))], body_font, 9))
        rows.append(_cells(["KDJ J", _v(ind.get("kdj_j")), "StochRSI", _v(ind.get("stoch_rsi"))], body_font, 9))
        rows.append(_cells(["布林上轨", _v(ind.get("boll_upper"), ".2f"), "布林中轨", _v(ind.get("boll_mid"), ".2f")], body_font, 9))
        rows.append(_cells(["布林下轨", _v(ind.get("boll_lower"), ".2f"), "趋势斜率%", _v(ind.get("trend_slope_pct"))], body_font, 9))
        rows.append(_cells(["动量(10根)%", _v(ind.get("momentum_10")), "波动率%", _v(ind.get("volatility_20"))], body_font, 9))

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
    """全智能体加权评分汇总表：按短线/中长期分组显示。"""
    votes = result.get("agent_votes", [])
    if not votes:
        return

    # ── 短线参考（5日内）──
    SHORT_TERM_DIMS = [
        "KLINE_1H", "KLINE_DAY", "KLINE_PATTERN", "VOLUME_PRICE",
        "SUPPORT_RESISTANCE", "TRENDLINE",
        "SENTIMENT", "NLP_SENTIMENT", "CAPITAL_FLOW", "FLOW_DETAIL",
        "MM_BEHAVIOR", "LIQ",
    ]
    # ── 中长期参考（1月以上）──
    MID_LONG_DIMS = [
        "KLINE_WEEK", "KLINE_MONTH", "TREND", "TECH",
        "DIVERGENCE", "CHANLUN", "CHART_PATTERN", "TIMEFRAME_RESONANCE",
        "TOP_STRUCTURE", "BOTTOM_STRUCTURE",
        "FUNDAMENTAL", "DERIV_MARGIN", "BETA", "SECTOR_POLICY",
        "MACRO", "INDUSTRY", "QUANT",
    ]
    short_order = {d: i for i, d in enumerate(SHORT_TERM_DIMS)}
    mid_order = {d: i for i, d in enumerate(MID_LONG_DIMS)}

    short_votes = sorted(
        [v for v in votes if v.get("dim_code", "") in SHORT_TERM_DIMS],
        key=lambda v: short_order.get(v.get("dim_code", ""), 99),
    )
    mid_votes = sorted(
        [v for v in votes if v.get("dim_code", "") in MID_LONG_DIMS],
        key=lambda v: mid_order.get(v.get("dim_code", ""), 99),
    )
    # 未归类的放到中长期末尾
    classified = set(SHORT_TERM_DIMS) | set(MID_LONG_DIMS)
    other_votes = [v for v in votes if v.get("dim_code", "") not in classified]
    mid_votes.extend(other_votes)

    flow.append(Paragraph("多智能体加权评分体系", st_h))
    total_weight = sum(float(v.get("weight", 0)) for v in votes) or 1.0

    # 获取 kline_indicators 用于顶底结构周期解读
    feats = result.get("analysis_features") or {}
    kli = feats.get("kline_indicators") if isinstance(feats, dict) else {}
    tf_breakdown = _structure_tf_breakdown(kli) if isinstance(kli, dict) else {"top": [], "bottom": []}
    _STRUCTURE_DIMS = {"TOP_STRUCTURE", "BOTTOM_STRUCTURE"}

    def _vote_row(idx: int, v: dict) -> list:
        """构建单个 Agent 行，顶底结构使用专用标签。"""
        dim_code = v.get("dim_code", "")
        role = v.get("role", v.get("dim_code", ""))
        if len(role) > 12:
            role = role[:12]
        score = float(v.get("score_0_100", 50))
        weight_pct = float(v.get("weight", 0)) / total_weight
        if dim_code in _STRUCTURE_DIMS:
            label = _structure_signal_label(dim_code, score)
            lc = _structure_color(dim_code, score)
            # 追加周期级别
            tfs = tf_breakdown.get("top" if dim_code == "TOP_STRUCTURE" else "bottom", [])
            if tfs:
                role_display = f"{role}({'/'.join(tfs)})"
            else:
                role_display = role
        else:
            label = _score_to_decision_level_cn(score)
            lc = level_color_map.get(label, "#222222")
            role_display = role
        return [
            _cell(str(idx), body_font, 8), _cell(role_display, body_font, 8),
            _cell(f"{score:.1f}", body_font, 8), _cell(label, bold_font, 8, True, lc),
            _cell(f"{weight_pct:.1%}", body_font, 8),
        ]

    # 表头
    table_data = [_cells(["#", "智能体", "评分", "信号/建议", "权重"], bold_font, 8, True)]

    # ── 短线分组标题行 ──
    table_data.append([_cell("", body_font, 8), _cell("短线参考（5日内）", bold_font, 8, True, "#E65100"),
                        _cell("", body_font, 8), _cell("", body_font, 8), _cell("", body_font, 8)])
    n_short_header = len(table_data) - 1

    for idx, v in enumerate(short_votes, 1):
        table_data.append(_vote_row(idx, v))

    # 短线小计
    if short_votes:
        sw_total = sum(float(v.get("weight", 0)) for v in short_votes)
        sw_scores = [float(v.get("score_0_100", 50)) for v in short_votes]
        sw_weights = [float(v.get("weight", 0)) for v in short_votes]
        sw_avg = sum(s * w for s, w in zip(sw_scores, sw_weights)) / sw_total if sw_total else 50.0
        sw_level = _score_to_decision_level_cn(sw_avg)
        sw_lc = level_color_map.get(sw_level, "#222222")
        table_data.append([
            _cell("", bold_font, 8), _cell("短线小计", bold_font, 8, True),
            _cell(f"{sw_avg:.1f}", bold_font, 8, True), _cell(sw_level, bold_font, 8, True, sw_lc),
            _cell(f"{sw_total / total_weight:.1%}", bold_font, 8, True),
        ])
    n_short_subtotal = len(table_data) - 1

    # ── 中长期分组标题行 ──
    table_data.append([_cell("", body_font, 8), _cell("中长期参考（1月以上）", bold_font, 8, True, "#0D47A1"),
                        _cell("", body_font, 8), _cell("", body_font, 8), _cell("", body_font, 8)])
    n_mid_header = len(table_data) - 1

    for idx, v in enumerate(mid_votes, 1):
        table_data.append(_vote_row(idx, v))

    # 中长期小计
    if mid_votes:
        mw_total = sum(float(v.get("weight", 0)) for v in mid_votes)
        mw_scores = [float(v.get("score_0_100", 50)) for v in mid_votes]
        mw_weights = [float(v.get("weight", 0)) for v in mid_votes]
        mw_avg = sum(s * w for s, w in zip(mw_scores, mw_weights)) / mw_total if mw_total else 50.0
        mw_level = _score_to_decision_level_cn(mw_avg)
        mw_lc = level_color_map.get(mw_level, "#222222")
        table_data.append([
            _cell("", bold_font, 8), _cell("中长期小计", bold_font, 8, True),
            _cell(f"{mw_avg:.1f}", bold_font, 8, True), _cell(mw_level, bold_font, 8, True, mw_lc),
            _cell(f"{mw_total / total_weight:.1%}", bold_font, 8, True),
        ])
    n_mid_subtotal = len(table_data) - 1

    # ── 综合评分行 ──
    final_level = _score_to_decision_level_cn(final_score)
    final_lc = level_color_map.get(final_level, "#222222")
    table_data.append([
        _cell("", bold_font, 9), _cell("综合评分", bold_font, 9, True),
        _cell(f"{final_score:.1f}", bold_font, 9, True), _cell(final_level, bold_font, 9, True, final_lc),
        _cell("100%", bold_font, 9, True),
    ])

    col_w = [10 * mm, 48 * mm, 18 * mm, 26 * mm, 18 * mm]
    table = Table(table_data, colWidths=col_w)

    tbl_style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        # 表头底色
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
        # 综合评分行底色
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F0F4F8")),
        # 短线分组标题行
        ("BACKGROUND", (0, n_short_header), (-1, n_short_header), colors.HexColor("#FFF3E0")),
        ("SPAN", (1, n_short_header), (4, n_short_header)),
        # 短线小计行
        ("BACKGROUND", (0, n_short_subtotal), (-1, n_short_subtotal), colors.HexColor("#FFF8E1")),
        # 中长期分组标题行
        ("BACKGROUND", (0, n_mid_header), (-1, n_mid_header), colors.HexColor("#E3F2FD")),
        ("SPAN", (1, n_mid_header), (4, n_mid_header)),
        # 中长期小计行
        ("BACKGROUND", (0, n_mid_subtotal), (-1, n_mid_subtotal), colors.HexColor("#E8EAF6")),
    ]
    table.setStyle(TableStyle(tbl_style))
    flow.append(table)
    flow.append(
        Paragraph(
            "<b>说明：</b>短线参考以未来5个交易日为视角，中长期参考以1个月以上为视角。"
            "最终评分 = Σ(各Agent评分×LLM权重) / Σ(权重)。"
            "五级映射：强烈买入(≥85)、弱买入(70-85)、观望(50-70)、弱卖出(40-50)、强烈卖出(&lt;40)。"
            "<br/><b>顶/底结构说明：</b>底部结构高分=底部信号显著(买入参考)，低分=无底部信号(中性，非卖出)；"
            "顶部结构高分=顶部信号显著(卖出警示)，低分=无顶部信号(中性，非买入)。"
            "两者独立评估、交叉印证，通常仅一个结构显著。括号内标注信号所在周期级别。",
            st_body,
        )
    )
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  行业竞争格局（来自 industry agent 研判）
# ─────────────────────────────────────────────────────────────────
def _add_financial_health_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
) -> None:
    """财务健康度 + 成长性表格。"""
    features = result.get("analysis_features", {})
    debt = features.get("debt_to_assets")
    cr = features.get("current_ratio")
    qr = features.get("quick_ratio")
    roe = features.get("roe")
    gm = features.get("grossprofit_margin")
    rev_yoy = features.get("revenue_yoy")
    np_yoy = features.get("netprofit_yoy")
    # 如果全部缺失则跳过
    if all(v is None for v in [debt, cr, qr, roe, rev_yoy, np_yoy]):
        return

    flow.append(Paragraph("财务健康度与成长性", st_h))

    def _fmt(v, suffix="%", decimals=1):
        if v is None:
            return "—"
        return f"{float(v):.{decimals}f}{suffix}"

    def _eval(v, good_range, warn_range, label_good="✅", label_warn="⚠️", label_bad="❌"):
        if v is None:
            return "—"
        fv = float(v)
        if good_range[0] <= fv <= good_range[1]:
            return label_good
        if warn_range[0] <= fv <= warn_range[1]:
            return label_warn
        return label_bad

    hdr = [_cell("指标", body_font, bold=True), _cell("数值", body_font, bold=True),
           _cell("安全线", body_font, bold=True), _cell("评价", body_font, bold=True)]
    rows = [hdr]
    if debt is not None:
        rows.append([_cell("资产负债率", body_font), _cell(_fmt(debt), body_font),
                      _cell("<60% 优 / >70% 危险", body_font),
                      _cell(_eval(debt, (0, 50), (50, 70)), body_font)])
    if cr is not None:
        rows.append([_cell("流动比率", body_font), _cell(_fmt(cr, "", 2), body_font),
                      _cell(">1.5 安全", body_font),
                      _cell(_eval(cr, (1.5, 99), (1.0, 1.5)), body_font)])
    if qr is not None:
        rows.append([_cell("速动比率", body_font), _cell(_fmt(qr, "", 2), body_font),
                      _cell(">1.0 安全", body_font),
                      _cell(_eval(qr, (1.0, 99), (0.8, 1.0)), body_font)])
    if roe is not None:
        rows.append([_cell("ROE", body_font), _cell(_fmt(roe), body_font),
                      _cell(">15% 优秀", body_font),
                      _cell(_eval(roe, (15, 99), (8, 15)), body_font)])
    if gm is not None:
        rows.append([_cell("毛利率", body_font), _cell(_fmt(gm), body_font),
                      _cell("行业相关", body_font), _cell("—", body_font)])
    if rev_yoy is not None:
        rows.append([_cell("营收增速(YoY)", body_font), _cell(_fmt(rev_yoy), body_font),
                      _cell(">15% 良好", body_font),
                      _cell(_eval(rev_yoy, (15, 999), (0, 15)), body_font)])
    if np_yoy is not None:
        rows.append([_cell("净利润增速(YoY)", body_font), _cell(_fmt(np_yoy), body_font),
                      _cell(">20% 优秀", body_font),
                      _cell(_eval(np_yoy, (20, 999), (0, 20)), body_font)])

    col_w = [90, 70, 100, 50]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 6))


def _add_peer_comparison_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
) -> None:
    """同业估值对比表（TOP5 + 目标股 + 行业均值）。"""
    features = result.get("analysis_features", {})
    peer = features.get("peer_comparison", {})
    peers = peer.get("peers", [])
    if not peers:
        return

    flow.append(Paragraph(f"同业估值对比（{peer.get('industry', '')}）", st_h))

    def _f(v, d=1):
        if v is None:
            return "—"
        return f"{float(v):.{d}f}"

    hdr = [_cell(h, body_font, bold=True) for h in
           ["公司", "PE(TTM)", "PB", "ROE%", "营收增速%", "利润增速%", "市值(亿)"]]
    rows = [hdr]

    # 目标股自身（高亮）
    symbol_name = result.get("name", "目标股")
    rows.append([
        _cell(f"<b>{_esc(symbol_name)}</b>", body_font),
        _cell(_f(features.get("pe_ttm")), body_font),
        _cell(_f(features.get("pb")), body_font),
        _cell(_f(features.get("roe")), body_font),
        _cell(_f(features.get("revenue_yoy")), body_font),
        _cell(_f(features.get("netprofit_yoy")), body_font),
        _cell(_f(features.get("total_mv"), 0), body_font),
    ])

    # TOP5 同业
    for p in peers[:5]:
        rows.append([
            _cell(_esc(p.get("name", "")), body_font),
            _cell(_f(p.get("pe_ttm")), body_font),
            _cell(_f(p.get("pb")), body_font),
            _cell(_f(p.get("roe")), body_font),
            _cell(_f(p.get("revenue_yoy")), body_font),
            _cell(_f(p.get("netprofit_yoy")), body_font),
            _cell(_f(p.get("total_mv"), 0), body_font),
        ])

    # 行业均值
    ind_avg = peer.get("industry_avg", {})
    if ind_avg:
        rows.append([
            _cell("<b>行业均值</b>", body_font),
            _cell(_f(ind_avg.get("pe_ttm")), body_font),
            _cell("—", body_font),
            _cell(_f(ind_avg.get("roe")), body_font),
            _cell(_f(ind_avg.get("revenue_yoy")), body_font),
            _cell("—", body_font),
            _cell("—", body_font),
        ])

    col_w = [70, 50, 40, 45, 55, 55, 55]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # 目标股行高亮
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFF3CD")),
    ]
    t.setStyle(TableStyle(style_cmds))
    flow.append(t)
    flow.append(Spacer(1, 6))


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

        rows = [_cells(["情景", "概率", "触发条件", "价格目标(元)", "止损参考"], bold_font, 9, True)]
        rows.append(_cells([
            "乐观", f"{bull_prob}%",
            "放量突破关键阻力 / 业绩超预期 / 板块政策催化",
            f"{cp * 1.15:.2f}（+15%）",
            f"{cp * 0.93:.2f}（-7%）",
        ], body_font, 9))
        rows.append(_cells([
            "中性", f"{base_prob}%",
            "维持当前区间震荡 / 无明显催化剂",
            f"{cp * 1.05:.2f}（+5%）",
            f"{cp * 0.95:.2f}（-5%）",
        ], body_font, 9))
        rows.append(_cells([
            "悲观", f"{bear_prob}%",
            "放量跌破支撑 / 业绩低于预期 / 板块政策收紧",
            f"{cp * 0.88:.2f}（-12%）",
            f"{cp * 0.90:.2f}（-10%）",
        ], body_font, 9))
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
            pos_rows = [_cells(["批次", "建仓时机", "建议仓位", "介入价格参考", "止损位"], bold_font, 9, True)]
            pos_rows.append(_cells([
                "第一批", "当前价附近 / 量能温和放大",
                "30%", f"{cp:.2f}（当前）", f"{cp * 0.95:.2f}（-5%）"
            ], body_font, 9))
            pos_rows.append(_cells([
                "第二批", "突破近期阻力并回踩确认",
                "30%", f"{cp * 1.03:.2f}（+3%）", f"{cp * 0.97:.2f}（-3%）"
            ], body_font, 9))
            pos_rows.append(_cells([
                "第三批", "放量创新高 / 催化剂落地",
                "40%", f"{cp * 1.06:.2f}（+6%）", f"{cp * 1.00:.2f}（保本）"
            ], body_font, 9))
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
        flow.append(Paragraph(f"<b>情景分析（AI生成）：</b>{_esc(scenario_text)}", st_body))
    if position_text:
        flow.append(Paragraph(f"<b>止损与仓位建议（AI生成）：</b>{_esc(position_text)}", st_body))
    flow.append(Spacer(1, 6))


def _add_entry_plans_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
) -> None:
    """A/B/C三种入场方案对比表 + 斐波那契关键位。"""
    plans = result.get("entry_plans") or []
    fib = result.get("fibonacci") or {}
    feats = result.get("analysis_features") or {}
    cp = feats.get("close") or feats.get("kline_indicators", {}).get("day", {}).get("close")
    if not cp:
        return
    cp = float(cp)

    # 斐波那契关键位
    if fib and fib.get("ok"):
        direction = "上涨回调" if fib.get("uptrend") else "下跌反弹"
        flow.append(Paragraph(f"斐波那契关键位（{direction}）", st_h))
        fib_rows = [_cells(["类型", "价格", "距当前价"], bold_font, 9, True)]
        fib_fields = [
            ("swing_high", "近期高点"), ("swing_low", "近期低点"),
            ("retrace_236", "23.6%回调"), ("retrace_382", "38.2%回调"),
            ("retrace_500", "50%回调"), ("retrace_618", "61.8%回调"),
            ("extend_1272", "127.2%延伸"), ("extend_1618", "161.8%延伸"),
        ]
        for key, label in fib_fields:
            val = fib.get(key)
            if val is not None:
                try:
                    val = float(val)
                    diff_pct = (val - cp) / cp * 100
                    fib_rows.append(_cells([label, f"{val:.2f}", f"{diff_pct:+.1f}%"], body_font, 9))
                except (ValueError, TypeError):
                    pass
        if len(fib_rows) > 1:
            tbl = Table(fib_rows, colWidths=[36 * mm, 30 * mm, 30 * mm])
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]))
            flow.append(tbl)
            flow.append(Spacer(1, 4))

    # 三方案入场对比表
    if plans and isinstance(plans, list) and len(plans) > 0:
        flow.append(Paragraph("入场方案对比（AI生成）", st_h))
        plan_rows = [_cells(["方案", "入场价", "目标价", "止损价", "RR比", "距当前"], bold_font, 9, True)]
        for p in plans:
            if not isinstance(p, dict):
                continue
            try:
                entry = float(p.get("entry", 0))
                target = float(p.get("target", 0))
                stop = float(p.get("stop", 0))
                rr = p.get("rr", "")
                if isinstance(rr, (int, float)):
                    rr_str = f"1:{rr:.1f}"
                else:
                    rr_str = str(rr)
                entry_diff = (entry - cp) / cp * 100 if cp > 0 else 0
                plan_rows.append(_cells([
                    str(p.get("name", "")),
                    f"{entry:.2f}",
                    f"{target:.2f}",
                    f"{stop:.2f}",
                    rr_str,
                    f"{entry_diff:+.1f}%",
                ], body_font, 9))
            except (ValueError, TypeError):
                continue
        if len(plan_rows) > 1:
            tbl = Table(plan_rows, colWidths=[20 * mm, 24 * mm, 24 * mm, 24 * mm, 20 * mm, 22 * mm])
            _bg_colors = [
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FFF0F0")),  # 追涨-红
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F0FFF0")),  # 回踩-绿
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F0F8FF")),  # 确认-蓝
            ]
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ] + _bg_colors[:len(plan_rows) - 1]))
            flow.append(tbl)
            flow.append(Spacer(1, 4))


def _add_sniper_points_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
) -> None:
    """狙击点位表格：首选/次选买入、止损、目标价。"""
    sp = result.get("sniper_points") or {}
    if not sp:
        return
    feats = result.get("analysis_features") or {}
    cp = feats.get("close")
    if not cp:
        return
    cp = float(cp)

    flow.append(Paragraph("狙击点位（AI生成）", st_h))
    fields = [
        ("ideal_buy", "首选买入", "支撑位附近低吸"),
        ("secondary_buy", "次选买入", "二次确认位"),
        ("stop_loss", "止损价", "跌破即离场"),
        ("take_profit_1", "目标价1", "第一止盈目标"),
        ("take_profit_2", "目标价2", "第二止盈目标"),
    ]
    rows = [_cells(["点位类型", "价格", "距当前价", "说明"], bold_font, 9, True)]
    for key, label, desc in fields:
        price = sp.get(key)
        if price is not None:
            try:
                price = float(price)
                diff_pct = (price - cp) / cp * 100
                rows.append(_cells([label, f"{price:.2f}", f"{diff_pct:+.1f}%", desc], body_font, 9))
            except (ValueError, TypeError):
                pass
    if len(rows) <= 1:
        return
    tbl = Table(rows, colWidths=[28 * mm, 28 * mm, 28 * mm, 72 * mm])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 6))


def _add_position_advice_table(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
) -> None:
    """空仓/持仓分别建议表格。"""
    pa = result.get("position_advice") or {}
    if not pa:
        return
    flow.append(Paragraph("空仓/持仓操作建议（AI生成）", st_h))
    rows = [_cells(["场景", "操作建议"], bold_font, 9, True)]
    rows.append(_cells(["空仓（未持有）", pa.get("no_position", "暂无建议")], body_font, 9))
    rows.append(_cells(["持仓（已持有）", pa.get("has_position", "暂无建议")], body_font, 9))
    ratio = pa.get("position_ratio", "")
    if ratio:
        rows.append(_cells(["建议仓位", str(ratio)], body_font, 9))
    tbl = Table(rows, colWidths=[38 * mm, 120 * mm])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), body_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 6))


# ─────────────────────────────────────────────────────────────────
#  报告总结卡片
# ─────────────────────────────────────────────────────────────────

_SUMMARY_HEADER_BG = "#1A3C6D"
_SUMMARY_HEADER_FG = "#FFFFFF"
_SUMMARY_SECTION_BG = "#EDF2FA"
_SUMMARY_BUY_BG = "#E8F5E9"
_SUMMARY_SELL_BG = "#FFEBEE"
_SUMMARY_TP_BG = "#FFF8E1"
_SUMMARY_MA_BG = "#F3F6FB"
_SUMMARY_GRID = "#C5CDD8"


def _summary_cell(text: str, font: str, size: float = 10, bold: bool = False,
                  color: str = "#222222", align: int = 0) -> Paragraph:
    """Summary card cell helper."""
    key = f"{font}_{size}_{color}_{align}"
    if key not in _CELL_STYLES:
        _CELL_STYLES[key] = ParagraphStyle(
            f"sc_{key}", fontName=font, fontSize=size, leading=size + 3,
            textColor=colors.HexColor(color), wordWrap="CJK", alignment=align,
        )
    safe = _esc(text)
    if bold:
        safe = f"<b>{safe}</b>"
    return Paragraph(safe, _CELL_STYLES[key])


def _pct_diff(current: float, target: float) -> str:
    """Calculate percentage difference string."""
    if current <= 0:
        return "—"
    pct = (target - current) / current * 100
    return f"{pct:+.1f}%"


def _add_summary_card(
    flow: list,
    result: dict[str, Any],
    st_h: Any,
    st_body: Any,
    body_font: str,
    bold_font: str,
) -> None:
    """在PDF末尾添加报告总结卡片：评分 + 点位 + 策略 + 解读。"""

    feats = result.get("analysis_features") or {}
    if not isinstance(feats, dict):
        feats = {}
    ki_day = feats.get("kline_indicators", {}).get("day", {})
    ki_week = feats.get("kline_indicators", {}).get("week", {})
    ma_day = ki_day.get("ma_system", {})
    ma_week = ki_week.get("ma_system", {})
    sp = result.get("sniper_points") or {}
    pa = result.get("position_advice") or {}

    # --- 推算现价 ---
    ma5_data = ma_day.get("ma5", {})
    ma5_val = float(ma5_data.get("value", 0) or 0)
    ma5_pct = float(ma5_data.get("pct_above", 0) or 0)
    current_price = ma5_val * (1 + ma5_pct / 100) if ma5_val > 0 else 0

    # ================================================================
    #  Section Title
    # ================================================================
    flow.append(Spacer(1, 12))
    st_card_title = ParagraphStyle(
        "card_title", fontName=bold_font, fontSize=16, leading=20,
        textColor=colors.HexColor("#1A3C6D"), spaceAfter=8,
    )
    flow.append(Paragraph("报告总结", st_card_title))

    # ================================================================
    #  1. 综合评分
    # ================================================================
    flow.append(Paragraph(
        f"<font name='{bold_font}' color='#1A3C6D'>一、综合评分</font>", st_body))
    flow.append(Spacer(1, 4))

    model_totals = result.get("model_totals") or {}
    final_score = float(result.get("final_score", 0))
    decision_level_cn = _score_to_decision_level_cn(final_score)
    level_colors = {
        "强烈买入": "#C41E3A", "弱买入": "#E74C3C",
        "观望": "#0066CC", "弱卖出": "#27AE60", "强烈卖出": "#228B22",
    }
    dec_color = level_colors.get(decision_level_cn, "#222222")

    # Build score rows: header + providers + summary
    score_rows = []
    score_rows.append([
        _summary_cell("项目", bold_font, 10, True, _SUMMARY_HEADER_FG),
        _summary_cell("数值", bold_font, 10, True, _SUMMARY_HEADER_FG),
        _summary_cell("说明", bold_font, 10, True, _SUMMARY_HEADER_FG),
    ])
    for p_name, p_score in model_totals.items():
        score_rows.append([
            _summary_cell(p_name.upper(), body_font, 10),
            _summary_cell(f"{float(p_score):.1f}", body_font, 10, False, "#333333", 1),
            _summary_cell("Provider加权评分", body_font, 9, False, "#888888"),
        ])
    # Average row
    score_rows.append([
        _summary_cell("均分", bold_font, 11, True, "#1A3C6D"),
        _summary_cell(f"{final_score:.1f}", bold_font, 12, True, dec_color, 1),
        _summary_cell(f"决策: {decision_level_cn}", bold_font, 10, True, dec_color),
    ])
    # MA5 bias row
    bias_text = f"{ma5_pct:+.1f}%"
    if abs(ma5_pct) > 8:
        bias_color, bias_note = "#C41E3A", "严重超买，不宜追高" if ma5_pct > 0 else "严重超卖，关注反弹"
    elif abs(ma5_pct) > 5:
        bias_color, bias_note = "#E67E22", "偏离较大，注意风险" if ma5_pct > 0 else "偏离较大，关注企稳"
    else:
        bias_color, bias_note = "#27AE60", "正常范围"
    score_rows.append([
        _summary_cell("MA5乖离率", body_font, 10),
        _summary_cell(bias_text, bold_font, 11, True, bias_color, 1),
        _summary_cell(bias_note, body_font, 9, False, bias_color),
    ])

    avg_row_idx = len(model_totals) + 1  # header=0, providers, then avg
    bias_row_idx = avg_row_idx + 1
    score_tbl = Table(score_rows, colWidths=[40 * mm, 32 * mm, 90 * mm])
    score_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_SUMMARY_HEADER_BG)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_SUMMARY_HEADER_FG)),
        ("BACKGROUND", (0, avg_row_idx), (-1, avg_row_idx), colors.HexColor("#EDF2FA")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_SUMMARY_GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    # Alternate row backgrounds for providers
    for i in range(1, avg_row_idx):
        bg = "#F8FBFF" if i % 2 == 1 else "#FFFFFF"
        score_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(bg)))
    score_tbl.setStyle(TableStyle(score_style))
    flow.append(score_tbl)
    flow.append(Spacer(1, 10))

    # ================================================================
    #  2. 关键点位与均线体系
    # ================================================================
    flow.append(Paragraph(
        f"<font name='{bold_font}' color='#1A3C6D'>二、关键点位与均线体系</font>", st_body))
    flow.append(Spacer(1, 4))

    price_rows = []
    row_colors = []  # (row_idx, bg_color)

    price_rows.append([
        _summary_cell("类型", bold_font, 10, True, _SUMMARY_HEADER_FG),
        _summary_cell("价格", bold_font, 10, True, _SUMMARY_HEADER_FG),
        _summary_cell("距当前价", bold_font, 10, True, _SUMMARY_HEADER_FG),
        _summary_cell("来源/性质", bold_font, 10, True, _SUMMARY_HEADER_FG),
    ])

    def _add_price_row(label, price_val, bg, source, text_color="#333333"):
        if price_val is None or price_val == 0:
            return
        try:
            pv = float(price_val)
        except (ValueError, TypeError):
            return
        idx = len(price_rows)
        diff = _pct_diff(current_price, pv)
        price_rows.append([
            _summary_cell(label, bold_font if bg != _SUMMARY_MA_BG else body_font, 10,
                          bg != _SUMMARY_MA_BG, text_color),
            _summary_cell(f"{pv:.2f}", body_font, 10, False, "#333333", 1),
            _summary_cell(diff, body_font, 10, False, "#666666", 1),
            _summary_cell(source, body_font, 9, False, "#888888"),
        ])
        row_colors.append((idx, bg))

    # Current price
    if current_price > 0:
        idx = len(price_rows)
        price_rows.append([
            _summary_cell("当前价", bold_font, 11, True, "#1A3C6D"),
            _summary_cell(f"{current_price:.2f}", bold_font, 11, True, "#1A3C6D", 1),
            _summary_cell("—", body_font, 10, False, "#999999", 1),
            _summary_cell("最新收盘", body_font, 9, False, "#888888"),
        ])
        row_colors.append((idx, "#E8EDF5"))

    # Take profit 2 (highest)
    _add_price_row("止盈目标②", sp.get("take_profit_2"), _SUMMARY_TP_BG,
                   "AI情景分析", "#B8860B")
    # Take profit 1
    _add_price_row("止盈目标①", sp.get("take_profit_1"), _SUMMARY_TP_BG,
                   "AI情景分析", "#B8860B")
    # Ideal buy
    _add_price_row("首选买入价", sp.get("ideal_buy"), _SUMMARY_BUY_BG,
                   "AI狙击点位", "#2E7D32")
    # Secondary buy
    _add_price_row("次选买入价", sp.get("secondary_buy"), _SUMMARY_BUY_BG,
                   "AI狙击点位", "#2E7D32")
    # Stop loss
    _add_price_row("止损价", sp.get("stop_loss"), _SUMMARY_SELL_BG,
                   "AI风控线", "#C62828")

    # MA levels - daily
    for ma_key, ma_label in [("ma5", "日MA5"), ("ma10", "日MA10"),
                              ("ma20", "日MA20"), ("ma60", "日MA60")]:
        ma_info = ma_day.get(ma_key, {})
        val = ma_info.get("value")
        if val:
            _add_price_row(ma_label, val, _SUMMARY_MA_BG, "均线系统")

    # MA levels - weekly
    for ma_key, ma_label in [("ma5", "周MA5"), ("ma20", "周MA20")]:
        ma_info = ma_week.get(ma_key, {})
        val = ma_info.get("value")
        if val:
            _add_price_row(ma_label, val, _SUMMARY_MA_BG, "均线系统")

    if len(price_rows) > 1:
        price_tbl = Table(price_rows, colWidths=[35 * mm, 28 * mm, 28 * mm, 71 * mm])
        price_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_SUMMARY_HEADER_BG)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_SUMMARY_HEADER_FG)),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_SUMMARY_GRID)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]
        for r_idx, r_bg in row_colors:
            price_style.append(("BACKGROUND", (0, r_idx), (-1, r_idx),
                                colors.HexColor(r_bg)))
        price_tbl.setStyle(TableStyle(price_style))
        flow.append(price_tbl)
    flow.append(Spacer(1, 10))

    # ================================================================
    #  3. 操作策略
    # ================================================================
    if pa or sp:
        flow.append(Paragraph(
            f"<font name='{bold_font}' color='#1A3C6D'>三、操作策略</font>", st_body))
        flow.append(Spacer(1, 4))

        strat_rows = []
        strat_rows.append([
            _summary_cell("场景", bold_font, 10, True, _SUMMARY_HEADER_FG),
            _summary_cell("策略建议", bold_font, 10, True, _SUMMARY_HEADER_FG),
            _summary_cell("关键价位", bold_font, 10, True, _SUMMARY_HEADER_FG),
        ])
        strat_bgs = []

        # No position
        no_pos = pa.get("no_position", "")
        if no_pos:
            idx = len(strat_rows)
            strat_rows.append([
                _summary_cell("空仓操作", bold_font, 10, True, "#2E7D32"),
                _summary_cell(no_pos, body_font, 9),
                _summary_cell(
                    f"买入: {sp.get('ideal_buy', '—')} / {sp.get('secondary_buy', '—')}"
                    if sp else "—", body_font, 9, False, "#2E7D32"),
            ])
            strat_bgs.append((idx, _SUMMARY_BUY_BG))

        # Has position
        has_pos = pa.get("has_position", "")
        if has_pos:
            idx = len(strat_rows)
            strat_rows.append([
                _summary_cell("持仓操作", bold_font, 10, True, "#1565C0"),
                _summary_cell(has_pos, body_font, 9),
                _summary_cell(f"仓位: {pa.get('position_ratio', '—')}", body_font, 9,
                              False, "#1565C0"),
            ])
            strat_bgs.append((idx, "#E3F2FD"))

        # Stop loss
        sl = sp.get("stop_loss")
        if sl:
            idx = len(strat_rows)
            sl_diff = _pct_diff(current_price, float(sl)) if current_price > 0 else "—"
            strat_rows.append([
                _summary_cell("止损设置", bold_font, 10, True, "#C62828"),
                _summary_cell(f"跌破 {sl} 果断离场", body_font, 9, False, "#C62828"),
                _summary_cell(f"止损位: {sl} ({sl_diff})", body_font, 9, True, "#C62828"),
            ])
            strat_bgs.append((idx, _SUMMARY_SELL_BG))

        # Take profit
        tp1, tp2 = sp.get("take_profit_1"), sp.get("take_profit_2")
        if tp1 or tp2:
            idx = len(strat_rows)
            parts = []
            if tp1:
                parts.append(f"目标① {tp1}")
            if tp2:
                parts.append(f"目标② {tp2}")
            strat_rows.append([
                _summary_cell("止盈参考", bold_font, 10, True, "#B8860B"),
                _summary_cell("分批止盈，到达目标位逐步减仓", body_font, 9),
                _summary_cell(" / ".join(parts), body_font, 9, True, "#B8860B"),
            ])
            strat_bgs.append((idx, _SUMMARY_TP_BG))

        if len(strat_rows) > 1:
            strat_tbl = Table(strat_rows, colWidths=[30 * mm, 88 * mm, 44 * mm])
            strat_style = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(_SUMMARY_HEADER_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(_SUMMARY_HEADER_FG)),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(_SUMMARY_GRID)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
            for r_idx, r_bg in strat_bgs:
                strat_style.append(("BACKGROUND", (0, r_idx), (-1, r_idx),
                                    colors.HexColor(r_bg)))
            strat_tbl.setStyle(TableStyle(strat_style))
            flow.append(strat_tbl)
        flow.append(Spacer(1, 10))

    # ================================================================
    #  4. 乖离率状态 + 关键解读
    # ================================================================
    flow.append(Paragraph(
        f"<font name='{bold_font}' color='#1A3C6D'>四、关键解读</font>", st_body))
    flow.append(Spacer(1, 4))

    # Bias status paragraph
    if abs(ma5_pct) > 8:
        bias_icon = "[严重]"
        if ma5_pct > 0:
            bias_body = (f"MA5乖离率 {ma5_pct:+.1f}%，现价{current_price:.2f}远离"
                         f"MA5({ma5_val:.2f})，短期严重超买。"
                         "均线虽呈多头排列但发散过大，追高风险极高。"
                         "建议耐心等待回踩MA5后再择机介入。")
        else:
            bias_body = (f"MA5乖离率 {ma5_pct:+.1f}%，现价{current_price:.2f}远离"
                         f"MA5({ma5_val:.2f})，短期严重超卖。"
                         "可关注反弹机会，但需确认企稳信号后再入场。")
    elif abs(ma5_pct) > 5:
        bias_icon = "[偏高]"
        bias_body = (f"MA5乖离率 {ma5_pct:+.1f}%，价格偏离均线较大，"
                     "存在回调修正的可能。建议控制仓位，不宜重仓追高。")
    else:
        bias_icon = "[正常]"
        bias_body = (f"MA5乖离率 {ma5_pct:+.1f}%，价格在均线附近运行，"
                     "短期走势相对健康。可根据评分建议正常操作。")

    flow.append(Paragraph(
        f"<font name='{bold_font}' color='{bias_color}'>{_esc(bias_icon)}</font> "
        f"{_esc(bias_body)}", st_body))
    flow.append(Spacer(1, 6))

    # Scenario + strategy text
    scenario_text = result.get("scenario_analysis", "")
    position_text = result.get("position_strategy", "")
    if scenario_text or position_text:
        combined = ""
        if scenario_text:
            combined += scenario_text.strip()
        if position_text:
            if combined:
                combined += " "
            combined += position_text.strip()
        # Trim to reasonable length
        if len(combined) > 500:
            combined = combined[:500] + "..."
        flow.append(Paragraph(
            f"<font name='{bold_font}' color='#1A3C6D'>[AI研判] </font>"
            f"{_esc(combined)}", st_body))
    flow.append(Spacer(1, 10))


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
        flow.append(Paragraph(f"<b>短线建议：</b>{_esc(short_hold)}；<b>中长线建议：</b>{_esc(medium_hold)}。", st_body))
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

    # （各Agent评分已在"多智能体加权评分体系"表格中展示，不再重复列出）

    # ── Bull vs Bear 辩论（仅结论） ──
    debate = result.get("debate_bull_bear") or {}
    if debate and (debate.get("bull_reason") or debate.get("bear_reason")):
        flow.append(Paragraph("Bull vs Bear 辩论", st_h))
        bull_role = debate.get("bull_role", debate.get("bull_agent_id", "看多"))
        bear_role = debate.get("bear_role", debate.get("bear_agent_id", "看空"))
        flow.append(Paragraph(
            f"<b>看多</b>（{_esc(bull_role)}）评分 {debate.get('bull_score', 0)}　｜　"
            f"<b>看空</b>（{_esc(bear_role)}）评分 {debate.get('bear_score', 0)}",
            st_body,
        ))
        judge_msg = str(debate.get("judge_msg", ""))
        if judge_msg:
            short_judge = (judge_msg[:80] + "…") if len(judge_msg) > 80 else judge_msg
            flow.append(Paragraph(f"<b>仲裁：</b>{_esc(short_judge)}", st_body))
        flow.append(Spacer(1, 6))

    # ── 财务健康度与成长性 ──
    _add_financial_health_table(flow, result, st_h, st_body, body_font)

    # ── 同业估值对比 ──
    _add_peer_comparison_table(flow, result, st_h, st_body, body_font)

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

    # ── 斐波那契 + 入场方案 ──
    _add_entry_plans_table(flow, result, st_h, st_body, body_font, bold_font)

    # ── 狙击点位 ──
    _add_sniper_points_table(flow, result, st_h, st_body, body_font, bold_font)

    # ── 空仓/持仓建议 ──
    _add_position_advice_table(flow, result, st_h, st_body, body_font, bold_font)

    # ── 报告总结卡片 ──
    _add_summary_card(flow, result, st_h, st_body, body_font, bold_font)

    flow.append(Paragraph(
        "风险提示：市场有风险，投资需谨慎。本报告仅供研究与参考，不构成投资建议。",
        st_body,
    ))

    doc.build(flow)
    return pdf_path


# ─────────────────────────────────────────────────────────────────
#  批量汇总 PDF
# ─────────────────────────────────────────────────────────────────

def _decision_label(score: float) -> tuple[str, str]:
    """返回 (决策中文, 颜色hex)。"""
    if score >= 70:
        return ("买入", "#C41E3A")
    if score >= 60:
        return ("弱买入", "#E74C3C")
    if score >= 50:
        return ("观望", "#0066CC")
    if score >= 40:
        return ("弱卖出", "#27AE60")
    return ("卖出", "#228B22")


def _bias_icon_text(pct: float) -> tuple[str, str]:
    """返回 (icon+pct_text, 颜色hex)。"""
    txt = f"{pct:+.1f}%"
    if abs(pct) > 8:
        return (f"[严重] {txt}", "#C41E3A")
    if abs(pct) > 5:
        return (f"[偏高] {txt}", "#E67E22")
    return (f"{txt}", "#27AE60")


def _extract_stock_data(result: dict) -> dict:
    """从 final_decision.json 提取批量汇总所需的全部字段。"""
    feats = result.get("analysis_features") or {}
    if not isinstance(feats, dict):
        feats = {}
    ki_day = feats.get("kline_indicators", {}).get("day", {})
    ki_week = feats.get("kline_indicators", {}).get("week", {})
    ma_day = ki_day.get("ma_system", {})
    ma_week = ki_week.get("ma_system", {})
    sp = result.get("sniper_points") or {}
    pa = result.get("position_advice") or {}

    ma5_data = ma_day.get("ma5", {})
    ma5_val = float(ma5_data.get("value", 0) or 0)
    ma5_pct = float(ma5_data.get("pct_above", 0) or 0)
    current_price = ma5_val * (1 + ma5_pct / 100) if ma5_val > 0 else 0

    model_totals = result.get("model_totals") or {}
    final_score = float(result.get("final_score", 0))

    return {
        "symbol": result.get("symbol", ""),
        "name": result.get("name", ""),
        "final_score": final_score,
        "model_totals": model_totals,
        "current_price": current_price,
        "ma5_pct": ma5_pct,
        "ma_day": ma_day,
        "ma_week": ma_week,
        "sniper_points": sp,
        "position_advice": pa,
        "scenarios": result.get("scenarios") or {},
    }


def build_batch_summary_pdf(
    results: list[dict[str, Any]],
    output_path: str | Path | None = None,
    title: str = "多智能体批量分析汇总报告",
) -> Path:
    """
    从多个 final_decision.json 结果生成批量汇总 PDF（5 张表格）。

    Parameters
    ----------
    results : list[dict]
        每个元素是一个 final_decision.json 的完整字典。
    output_path : str | Path | None
        输出文件路径。None 时自动生成到 output/ 目录。
    title : str
        报告标题。

    Returns
    -------
    Path  PDF 文件路径
    """
    body_font, bold_font = _register_fonts()

    # --- 提取数据 ---
    stocks = [_extract_stock_data(r) for r in results]
    stocks.sort(key=lambda s: s["final_score"], reverse=True)

    # --- 收集所有 provider 名称 ---
    all_providers: list[str] = []
    seen_p: set[str] = set()
    for s in stocks:
        for p in s["model_totals"]:
            if p not in seen_p:
                all_providers.append(p)
                seen_p.add(p)

    # --- 输出路径 ---
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"batch_summary_{ts}.pdf"
    output_path = Path(output_path)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )
    flow: list = []

    # --- 样式 ---
    st_title = ParagraphStyle(
        "batch_title", fontName=bold_font, fontSize=18, leading=24,
        textColor=colors.HexColor("#1A3C6D"), spaceAfter=6, alignment=1,
    )
    st_subtitle = ParagraphStyle(
        "batch_sub", fontName=body_font, fontSize=10, leading=14,
        textColor=colors.HexColor("#666666"), spaceAfter=10, alignment=1,
    )
    st_section = ParagraphStyle(
        "batch_sec", fontName=bold_font, fontSize=13, leading=17,
        textColor=colors.HexColor("#1A3C6D"), spaceBefore=12, spaceAfter=6,
    )
    st_body = ParagraphStyle(
        "batch_body", fontName=body_font, fontSize=9, leading=13,
        textColor=colors.HexColor("#333333"), wordWrap="CJK",
    )

    HDR_BG = _SUMMARY_HEADER_BG
    HDR_FG = _SUMMARY_HEADER_FG
    GRID_C = _SUMMARY_GRID

    def _hdr(text):
        return _summary_cell(text, bold_font, 9, True, HDR_FG)

    def _bc(text, sz=8, color="#333333", align=0, bold=False):
        return _summary_cell(text, bold_font if bold else body_font, sz, bold, color, align)

    def _std_style(n_rows):
        """标准表格样式 + 斑马条纹。"""
        s = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HDR_BG)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(HDR_FG)),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(GRID_C)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        for i in range(1, n_rows):
            bg = "#F8FBFF" if i % 2 == 1 else "#FFFFFF"
            s.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(bg)))
        return s

    # ================================================================
    #  标题
    # ================================================================
    flow.append(Paragraph(title, st_title))
    flow.append(Paragraph(
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"共 {len(stocks)} 只标的  |  "
        f"模型: {', '.join(p.upper() for p in all_providers)}",
        st_subtitle))

    # ================================================================
    #  表1: 综合评分排名
    # ================================================================
    flow.append(Paragraph("表1  综合评分排名", st_section))

    hdr1 = [_hdr("#"), _hdr("代码"), _hdr("名称")]
    for p in all_providers:
        hdr1.append(_hdr(p.upper()))
    hdr1 += [_hdr("均分"), _hdr("MA5乖离"), _hdr("建议")]
    rows1 = [hdr1]

    stats = {"买入": 0, "弱买入": 0, "观望": 0, "弱卖出": 0, "卖出": 0}
    score_min, score_max = 999, -999
    bias_warnings = []

    for idx, s in enumerate(stocks, 1):
        row = [_bc(str(idx), 8, align=1), _bc(s["symbol"], 8), _bc(s["name"], 8)]
        for p in all_providers:
            mt = s["model_totals"]
            if p in mt:
                v = mt[p]
                pv = float(v) if not isinstance(v, dict) else float(v.get("final_score", 0))
                row.append(_bc(f"{pv:.1f}", 8, align=1))
            else:
                row.append(_bc("—", 8, "#999999", 1))
        # Avg
        dec_label, dec_color = _decision_label(s["final_score"])
        row.append(_bc(f"{s['final_score']:.1f}", 9, dec_color, 1, True))
        # MA5 bias
        bias_txt, bias_c = _bias_icon_text(s["ma5_pct"])
        row.append(_bc(bias_txt, 8, bias_c, 1))
        # Decision
        row.append(_bc(dec_label, 9, dec_color, 1, True))
        rows1.append(row)

        stats[dec_label] = stats.get(dec_label, 0) + 1
        score_min = min(score_min, s["final_score"])
        score_max = max(score_max, s["final_score"])
        if abs(s["ma5_pct"]) > 5:
            bias_warnings.append(f"{s['symbol']} {s['name']} MA5乖离{s['ma5_pct']:+.1f}%")

    n_provider_cols = len(all_providers)
    cw1 = [8*mm, 18*mm, 22*mm] + [18*mm]*n_provider_cols + [18*mm, 22*mm, 18*mm]
    tbl1 = Table(rows1, colWidths=cw1)
    style1 = _std_style(len(rows1))
    tbl1.setStyle(TableStyle(style1))
    flow.append(tbl1)

    # Footer
    stat_parts = []
    for label in ["买入", "弱买入", "观望", "弱卖出", "卖出"]:
        cnt = stats.get(label, 0)
        if cnt > 0:
            stat_parts.append(f"{label}: {cnt}只")
    footer1 = f"{' | '.join(stat_parts)}    评分区间: {score_min:.1f} ~ {score_max:.1f}"
    flow.append(Paragraph(_esc(footer1), st_body))
    if bias_warnings:
        flow.append(Paragraph(
            f"<font color='#E67E22'>乖离预警: {_esc(' / '.join(bias_warnings))}</font>",
            st_body))
    flow.append(Spacer(1, 6))

    # ================================================================
    #  表2: 狙击点位
    # ================================================================
    flow.append(Paragraph("表2  狙击点位", st_section))
    hdr2 = [_hdr("#"), _hdr("代码"), _hdr("名称"), _hdr("现价"),
            _hdr("首选买入"), _hdr("次选买入"), _hdr("止损"),
            _hdr("止盈①"), _hdr("止盈②"), _hdr("仓位")]
    rows2 = [hdr2]
    for idx, s in enumerate(stocks, 1):
        sp = s["sniper_points"]
        cp = s["current_price"]

        def _pv(key, positive=True):
            v = sp.get(key)
            if v is None:
                return "—", "—"
            try:
                fv = float(v)
            except (ValueError, TypeError):
                return str(v), "—"
            diff = _pct_diff(cp, fv) if cp > 0 else "—"
            return f"{fv:.2f}", diff

        ib_p, ib_d = _pv("ideal_buy")
        sb_p, sb_d = _pv("secondary_buy")
        sl_p, sl_d = _pv("stop_loss")
        tp1_p, tp1_d = _pv("take_profit_1")
        tp2_p, tp2_d = _pv("take_profit_2")
        ratio = s["position_advice"].get("position_ratio", "—")

        rows2.append([
            _bc(str(idx), 8, align=1),
            _bc(s["symbol"], 8),
            _bc(s["name"], 8),
            _bc(f"{cp:.2f}" if cp > 0 else "—", 8, "#1A3C6D", 1, True),
            _bc(f"{ib_p} ({ib_d})", 7, "#2E7D32"),
            _bc(f"{sb_p} ({sb_d})", 7, "#2E7D32"),
            _bc(f"{sl_p} ({sl_d})", 7, "#C62828"),
            _bc(f"{tp1_p} ({tp1_d})", 7, "#B8860B"),
            _bc(f"{tp2_p} ({tp2_d})", 7, "#B8860B"),
            _bc(str(ratio), 8, align=1),
        ])

    cw2 = [8*mm, 16*mm, 20*mm, 16*mm, 22*mm, 22*mm, 22*mm, 22*mm, 22*mm, 14*mm]
    tbl2 = Table(rows2, colWidths=cw2)
    style2 = _std_style(len(rows2))
    # Color buy rows green, stop-loss red, TP yellow for header cells
    style2.append(("BACKGROUND", (4, 0), (5, 0), colors.HexColor("#2E7D32")))
    style2.append(("BACKGROUND", (6, 0), (6, 0), colors.HexColor("#C62828")))
    style2.append(("BACKGROUND", (7, 0), (8, 0), colors.HexColor("#B8860B")))
    tbl2.setStyle(TableStyle(style2))
    flow.append(tbl2)
    flow.append(Spacer(1, 6))

    # ================================================================
    #  表3: 多周期均线支撑压力
    # ================================================================
    flow.append(Paragraph("表3  多周期均线支撑/压力", st_section))
    hdr3 = [_hdr("#"), _hdr("代码"), _hdr("名称"), _hdr("现价"),
            _hdr("MA5"), _hdr("MA10"), _hdr("MA20"), _hdr("MA60"),
            _hdr("周MA5"), _hdr("周MA20"), _hdr("短期评估")]
    rows3 = [hdr3]
    for idx, s in enumerate(stocks, 1):
        cp = s["current_price"]
        ma_d = s["ma_day"]
        ma_w = s["ma_week"]

        def _mav(src, key):
            v = src.get(key, {}).get("value")
            return f"{float(v):.2f}" if v else "—"

        # Assessment: compare current price vs MA20
        ma20_v = ma_d.get("ma20", {}).get("value")
        ma5_pct_val = s["ma5_pct"]
        if abs(ma5_pct_val) > 8:
            assess, assess_c = "超买" if ma5_pct_val > 0 else "超卖", "#C41E3A"
        elif abs(ma5_pct_val) > 5:
            assess, assess_c = "偏高" if ma5_pct_val > 0 else "偏低", "#E67E22"
        else:
            assess, assess_c = "正常", "#27AE60"

        rows3.append([
            _bc(str(idx), 8, align=1),
            _bc(s["symbol"], 8),
            _bc(s["name"], 8),
            _bc(f"{cp:.2f}" if cp > 0 else "—", 8, "#1A3C6D", 1, True),
            _bc(_mav(ma_d, "ma5"), 8, align=1),
            _bc(_mav(ma_d, "ma10"), 8, align=1),
            _bc(_mav(ma_d, "ma20"), 8, align=1),
            _bc(_mav(ma_d, "ma60"), 8, align=1),
            _bc(_mav(ma_w, "ma5"), 8, align=1),
            _bc(_mav(ma_w, "ma20"), 8, align=1),
            _bc(assess, 8, assess_c, 1, True),
        ])

    cw3 = [8*mm, 16*mm, 20*mm, 16*mm] + [16*mm]*6 + [16*mm]
    tbl3 = Table(rows3, colWidths=cw3)
    tbl3.setStyle(TableStyle(_std_style(len(rows3))))
    flow.append(tbl3)
    flow.append(Spacer(1, 6))

    # ================================================================
    #  表4: 仓位策略
    # ================================================================
    flow.append(Paragraph("表4  仓位策略 (空仓 vs 持仓)", st_section))
    hdr4 = [_hdr("#"), _hdr("代码"), _hdr("名称"),
            _hdr("空仓建议"), _hdr("持仓建议")]
    rows4 = [hdr4]
    for idx, s in enumerate(stocks, 1):
        pa = s["position_advice"]
        no_p = pa.get("no_position", "—")
        has_p = pa.get("has_position", "—")
        rows4.append([
            _bc(str(idx), 8, align=1),
            _bc(s["symbol"], 8),
            _bc(s["name"], 8),
            _bc(str(no_p), 7, "#2E7D32"),
            _bc(str(has_p), 7, "#1565C0"),
        ])

    cw4 = [8*mm, 16*mm, 20*mm, 68*mm, 68*mm]
    tbl4 = Table(rows4, colWidths=cw4)
    style4 = _std_style(len(rows4))
    style4.append(("BACKGROUND", (3, 0), (3, 0), colors.HexColor("#2E7D32")))
    style4.append(("BACKGROUND", (4, 0), (4, 0), colors.HexColor("#1565C0")))
    tbl4.setStyle(TableStyle(style4))
    flow.append(tbl4)
    flow.append(Spacer(1, 6))

    # ================================================================
    #  表5: 乖离预警区
    # ================================================================
    alert_stocks = [s for s in stocks if abs(s["ma5_pct"]) > 5]
    if alert_stocks:
        flow.append(Paragraph("表5  乖离率预警区", st_section))
        hdr5 = [_hdr("预警"), _hdr("代码"), _hdr("名称"),
                _hdr("MA5乖离"), _hdr("现价 vs MA5"), _hdr("风险提示")]
        rows5 = [hdr5]
        for s in alert_stocks:
            ma5_v = s["ma_day"].get("ma5", {}).get("value", 0)
            pct = s["ma5_pct"]
            if abs(pct) > 8:
                level_txt, level_c = "严重", "#C41E3A"
                risk = "严重超买，不宜追高" if pct > 0 else "严重超卖，关注反弹"
            else:
                level_txt, level_c = "注意", "#E67E22"
                risk = "偏离较大，注意风险" if pct > 0 else "偏离较大，关注企稳"
            rows5.append([
                _bc(level_txt, 9, level_c, 1, True),
                _bc(s["symbol"], 8),
                _bc(s["name"], 8),
                _bc(f"{pct:+.1f}%", 9, level_c, 1, True),
                _bc(f"{s['current_price']:.2f} vs {float(ma5_v):.2f}" if ma5_v else "—",
                    8, align=1),
                _bc(risk, 8, level_c),
            ])

        cw5 = [14*mm, 18*mm, 22*mm, 20*mm, 36*mm, 70*mm]
        tbl5 = Table(rows5, colWidths=cw5)
        style5 = _std_style(len(rows5))
        # Highlight alert rows
        for i in range(1, len(rows5)):
            bg = "#FFEBEE" if abs(alert_stocks[i-1]["ma5_pct"]) > 8 else "#FFF3E0"
            style5.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(bg)))
        tbl5.setStyle(TableStyle(style5))
        flow.append(tbl5)
    else:
        flow.append(Paragraph("表5  乖离率预警区", st_section))
        flow.append(Paragraph("所有标的MA5乖离率均在正常范围内，无需预警。", st_body))

    flow.append(Spacer(1, 16))
    flow.append(Paragraph(
        "风险提示：市场有风险，投资需谨慎。本报告仅供研究与参考，不构成投资建议。",
        st_body))

    doc.build(flow)
    return output_path
