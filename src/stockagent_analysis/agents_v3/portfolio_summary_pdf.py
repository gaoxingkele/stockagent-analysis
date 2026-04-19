"""组合汇总 PDF - 一份文件集中呈现所有股票的 v3 裁决。

结构:
  封面 + 统计 → BUY 区 → HOLD 区 → SELL 区

每只股票的"名片":
  Row 1: 代码 | 名称 | 综合 | 量化 | 辩论 | 风控 | 等级
  Row 2: 回踩入场 / 目标1 / 目标2 / 止损 / RR
  Row 3: 买入理由(Judge key_reasons 合并)
  Row 4: 风险/止损条件
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.graphics.shapes import Drawing, Circle, Line, String, Rect, Polygon

from ..report_pdf import _register_fonts, _esc, _cell, _safe_filename
from .report_pdf_v3 import _register_font_family, _LEVEL_CN, _LEVEL_COLOR, _score_color, _trim


# FOMC Dot Plot: 4 专家颜色编码
_EXPERT_COLORS = {
    "structure_expert": "#E74C3C",    # K 走势 - 红
    "wave_expert": "#27AE60",         # 波浪 - 绿
    "intraday_t_expert": "#3498DB",   # 短 T - 蓝
    "martingale_expert": "#F39C12",   # 马丁 - 橙
}
_EXPERT_SHORT = {
    "structure_expert": "K",
    "wave_expert": "浪",
    "intraday_t_expert": "T",
    "martingale_expert": "马",
}


def _make_dot_plot(experts: list[dict], avg: float, median: float,
                   width: float = 380, height: float = 28) -> Drawing:
    """FOMC 点阵图: 横轴 0-100, 4 专家彩点 + 均值(灰▽) + 中位数(蓝▲)。"""
    d = Drawing(width, height)
    margin = 12
    track_w = width - 2 * margin

    # 背景条
    d.add(Rect(margin, height / 2 - 1, track_w, 2,
               fillColor=colors.HexColor("#E5E5E5"), strokeColor=None))

    # 刻度: 0/25/50/75/100
    for v in [0, 25, 50, 75, 100]:
        x = margin + track_w * v / 100
        d.add(Line(x, height / 2 - 3, x, height / 2 + 3,
                   strokeColor=colors.HexColor("#BBBBBB"), strokeWidth=0.5))
        d.add(String(x, 1, str(v), fontSize=6,
                     textAnchor="middle",
                     fillColor=colors.HexColor("#999999")))

    # 50 分中线(深灰)
    x50 = margin + track_w * 0.5
    d.add(Line(x50, height / 2 - 6, x50, height / 2 + 6,
               strokeColor=colors.HexColor("#888888"), strokeWidth=0.7))

    # 专家彩点(放背景条上方, 大圆 + 白色边框易识别重叠)
    y_dot = height / 2 + 4
    for e in experts:
        try:
            s = float(e.get("score", 50))
            role = e.get("role", "")
            col = _EXPERT_COLORS.get(role, "#666666")
            x = margin + track_w * s / 100
            d.add(Circle(x, y_dot, 3.8,
                         fillColor=colors.HexColor(col),
                         strokeColor=colors.white, strokeWidth=1))
        except Exception:
            continue

    # 均值: 灰色小三角(朝下, 在顶部)
    x_avg = margin + track_w * avg / 100
    d.add(Polygon(points=[x_avg - 3, height - 2, x_avg + 3, height - 2, x_avg, height - 6],
                  fillColor=colors.HexColor("#808080"), strokeColor=None))

    # 中位数: 深蓝三角(朝上, 在底部)
    x_med = margin + track_w * median / 100
    d.add(Polygon(points=[x_med - 3, 8, x_med + 3, 8, x_med, 12],
                  fillColor=colors.HexColor("#0D3B66"), strokeColor=None))

    return d


def _dot_plot_legend(bold_font: str) -> str:
    """图例文字(放在 Dot Plot 下方)。"""
    return (
        f"<font color='#E74C3C'>●</font>K走势  "
        f"<font color='#27AE60'>●</font>波浪  "
        f"<font color='#3498DB'>●</font>短T  "
        f"<font color='#F39C12'>●</font>马丁  "
        f"<font color='#808080'>▽</font>均值  "
        f"<font color='#0D3B66'>▲</font>中位数"
    )


_CURRENT_BODY = "STSong-Light"
_CURRENT_BOLD = "STSong-Light"


def _c(text, size: float = 9, bold: bool = False, color: str = "#222222", align: int = 0):
    font = _CURRENT_BOLD if bold else _CURRENT_BODY
    return _cell(text, font=font, size=size, bold=bold, color=color, align=align)


def _collect_all_results(runs_dir: Path, since: datetime) -> list[dict[str, Any]]:
    """从 runs_v3 收集指定时间后的所有 final_decision_v3.json。"""
    rows = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            ts = datetime.strptime(d.name[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if ts < since:
            continue
        f = d / "final_decision_v3.json"
        if f.exists():
            try:
                data = json.load(f.open(encoding="utf-8"))
                data["_run_dir_name"] = d.name
                rows.append(data)
            except Exception:
                continue
    return rows


def build_portfolio_summary_pdf(
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str = "v3 组合汇总决策报告",
) -> Path:
    """生成组合汇总 PDF。

    rows 按本函数内部分区+排序(不要求预先排序)。
    """
    global _CURRENT_BODY, _CURRENT_BOLD
    body_font, bold_font = _register_fonts()
    _register_font_family(body_font, bold_font)
    _CURRENT_BODY = body_font
    _CURRENT_BOLD = bold_font

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title=title,
    )

    styles = getSampleStyleSheet()
    st_title = ParagraphStyle(
        "psTitle", parent=styles["Title"],
        fontName=bold_font, fontSize=18, leading=22,
        textColor=colors.HexColor("#0D3B66"),
    )
    st_h1 = ParagraphStyle(
        "psH1", parent=styles["Heading1"],
        fontName=bold_font, fontSize=14, leading=18,
        textColor=colors.HexColor("#0D3B66"), spaceBefore=10, spaceAfter=6,
    )
    st_h2 = ParagraphStyle(
        "psH2", parent=styles["Heading2"],
        fontName=bold_font, fontSize=11, leading=14,
        textColor=colors.HexColor("#1E3A5F"), spaceBefore=4, spaceAfter=2,
    )
    st_body = ParagraphStyle(
        "psBody", parent=styles["BodyText"],
        fontName=body_font, fontSize=9.5, leading=13,
        textColor=colors.black,
    )

    flow: list = []

    # ── 封面 ──
    flow.append(Paragraph(title, st_title))
    flow.append(Paragraph(
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  标的数量: {len(rows)}",
        ParagraphStyle("cover", fontName=body_font, fontSize=10, textColor=colors.HexColor("#666")),
    ))

    # FOMC 点阵图图例说明
    legend_para = ParagraphStyle("legend2", fontName=body_font, fontSize=9,
                                  leading=13, textColor=colors.HexColor("#333"))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(
        "<b>FOMC 点阵图说明:</b>  "
        "<font color='#E74C3C'>●</font>K走势结构  "
        "<font color='#27AE60'>●</font>波浪理论  "
        "<font color='#3498DB'>●</font>短线做T  "
        "<font color='#F39C12'>●</font>马丁网格  "
        "<font color='#808080'>▽</font>均值  "
        "<font color='#0D3B66'>▲</font>中位数", legend_para))
    flow.append(Paragraph(
        "<b>分歧级别:</b>  "
        "<font color='#27AE60'>✓ 共识良好</font> = 4 位专家分差 &lt; 15 分  |  "
        "<font color='#E67E22'>◐ 同向分歧</font> = 有专家偏离中位但与辩论方向一致  |  "
        "<font color='#C0392B'>⚠ 高分歧决策</font> = 辩论层判 BUY 但专家异议偏空(或反之), 建议降仓/等确认",
        legend_para))
    flow.append(Spacer(1, 8))

    # ── 统计概览 ──
    _section_overview(flow, rows, st_h1, st_body)

    # ── 按决策分区 ──
    buys = sorted([r for r in rows if r.get("final_decision") == "buy"],
                  key=lambda r: r["final_score"], reverse=True)
    holds = sorted([r for r in rows if r.get("final_decision") == "hold"],
                   key=lambda r: r["final_score"], reverse=True)
    sells = sorted([r for r in rows if r.get("final_decision") == "sell"],
                   key=lambda r: r["final_score"])

    if buys:
        flow.append(PageBreak())
        flow.append(Paragraph(f"一、BUY 买入建议 ({len(buys)} 只) · 按综合评分降序",
                              st_h1))
        _render_stock_group(flow, buys, st_h2, st_body, "buy")

    if holds:
        flow.append(PageBreak())
        flow.append(Paragraph(f"二、HOLD 观望 ({len(holds)} 只) · 按综合评分降序",
                              st_h1))
        _render_stock_group(flow, holds, st_h2, st_body, "hold")

    if sells:
        flow.append(PageBreak())
        flow.append(Paragraph(f"三、SELL 卖出/减仓建议 ({len(sells)} 只) · 按综合评分升序",
                              st_h1))
        _render_stock_group(flow, sells, st_h2, st_body, "sell")

    doc.build(flow)
    return out_path


def _section_overview(flow, rows, st_h1, st_body):
    flow.append(Paragraph("组合概览", st_h1))
    finals = sorted(r["final_score"] for r in rows)
    from collections import Counter
    decs = Counter(r.get("final_decision") for r in rows)
    lvls = Counter(r.get("decision_level") for r in rows)

    n = len(rows)
    stat_rows = [
        [_c("总数", bold=True), _c(str(n))],
        [_c("评分范围", bold=True), _c(f"{finals[0]:.1f} ~ {finals[-1]:.1f}  (span={finals[-1]-finals[0]:.1f})")],
        [_c("评分中位", bold=True), _c(f"{finals[n//2]:.1f}")],
        [_c("BUY 数量", bold=True, color="#C41E3A"),
         _c(f"{decs.get('buy',0)} 只  ({decs.get('buy',0)*100//n}%)", color="#C41E3A")],
        [_c("HOLD 数量", bold=True, color="#0066CC"),
         _c(f"{decs.get('hold',0)} 只  ({decs.get('hold',0)*100//n}%)", color="#0066CC")],
        [_c("SELL 数量", bold=True, color="#27AE60"),
         _c(f"{decs.get('sell',0)} 只  ({decs.get('sell',0)*100//n}%)", color="#27AE60")],
    ]
    stat_tbl = Table(stat_rows, colWidths=[30 * mm, 70 * mm])
    stat_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F8FC")),
    ]))
    flow.append(stat_tbl)
    flow.append(Spacer(1, 6))

    # 等级细分
    level_order = ["strong_buy", "weak_buy", "hold", "watch_sell", "weak_sell", "strong_sell"]
    lvl_rows = [[_c("细分等级", bold=True), _c("数量", bold=True, align=1), _c("占比", bold=True, align=1)]]
    for k in level_order:
        cnt = lvls.get(k, 0)
        cn = _LEVEL_CN.get(k, k)
        col = _LEVEL_COLOR.get(k, "#333")
        lvl_rows.append([
            _c(f"{cn}", color=col, bold=True),
            _c(f"{cnt}", align=1, color=col),
            _c(f"{cnt*100/n:.1f}%", align=1, color=col),
        ])
    lvl_tbl = Table(lvl_rows, colWidths=[45 * mm, 25 * mm, 25 * mm])
    lvl_tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
    ]))
    flow.append(lvl_tbl)


def _pick_entry_plan(entry_plans: list, prefer: str = "回踩") -> dict:
    """优先选回踩/确认方案(更稳); 没有则取第一个。"""
    if not isinstance(entry_plans, list):
        return {}
    for p in entry_plans:
        if isinstance(p, dict) and prefer in str(p.get("strategy", "")):
            return p
    for p in entry_plans:
        if isinstance(p, dict):
            return p
    return {}


def _format_reasons(reasons: list, max_items: int = 3, max_chars: int = 220) -> str:
    if not reasons:
        return "-"
    out = []
    for i, r in enumerate(reasons[:max_items], 1):
        s = str(r).strip()
        if len(s) > max_chars:
            s = s[:max_chars] + "…"
        out.append(f"{i}. {s}")
    return "\n".join(out)


def _render_stock_group(flow, rows: list[dict], st_h2, st_body, group: str):
    """渲染一组股票, 每只用一个紧凑 4 行卡片(复合表格)。"""
    for rank, r in enumerate(rows, 1):
        _render_one_card(flow, r, rank, st_h2, st_body, group)


def _render_one_card(flow, r: dict, rank: int, st_h2, st_body, group: str):
    symbol = r.get("symbol", "")
    name = r.get("name", "")
    sc = r.get("score_components") or {}
    ip = r.get("investment_plan") or {}
    tp = r.get("trading_plan") or {}
    rp = r.get("risk_policy") or {}

    final_score = float(r.get("final_score", 0))
    level = r.get("decision_level", "")
    level_cn = _LEVEL_CN.get(level, level)
    level_color = _LEVEL_COLOR.get(level, "#333")
    expert_avg = sc.get("expert_avg", 0)
    judge_adj = sc.get("judge_adj", 0)
    risk_mapped = sc.get("risk_mapped", 0)
    bonus = sc.get("consensus_bonus", 0)

    # Row 1: 代码/名称/综合/量化/辩论/风控/等级
    row1_header = [
        _c("#" + str(rank), size=8.5, color="#888", align=1),
        _c(f"{symbol}  {name}", bold=True, size=10.5, color="#111"),
        _c("综合", size=8, color="#666", align=1),
        _c("量化", size=8, color="#666", align=1),
        _c("辩论", size=8, color="#666", align=1),
        _c("风控", size=8, color="#666", align=1),
        _c("奖励", size=8, color="#666", align=1),
        _c("等级", size=8, color="#666", align=1),
    ]
    row1_vals = [
        _c("", size=7),   # 占位
        _c("", size=7),
        _c(f"{final_score:.1f}", bold=True, size=12, color=level_color, align=1),
        _c(f"{expert_avg:.1f}", size=10, color=_score_color(expert_avg), align=1),
        _c(f"{judge_adj:.1f}", size=10, color=_score_color(judge_adj), align=1),
        _c(f"{risk_mapped:.1f}", size=10, color=_score_color(risk_mapped), align=1),
        _c(f"{bonus:+.1f}" if bonus else "0", size=9, color="#888", align=1),
        _c(level_cn, bold=True, size=10, color=level_color, align=1),
    ]
    # 入场方案
    ep = _pick_entry_plan(tp.get("entry_plans") or [], prefer="回踩")

    def _v(x):
        return str(x) if x is not None else "-"

    row2_header = [
        _c("建议策略", size=8, color="#666"),
        _c(f"{ep.get('strategy','-')} 入场", size=8.5, color="#555", bold=True),
        _c("目标1", size=8, color="#666", align=1),
        _c("目标2", size=8, color="#666", align=1),
        _c("止损", size=8, color="#666", align=1),
        _c("盈亏比", size=8, color="#666", align=1),
        _c("首仓%", size=8, color="#666", align=1),
        _c("风险评级", size=8, color="#666", align=1),
    ]
    row2_vals = [
        _c("", size=7),
        _c(f"{_v(ep.get('entry'))}", bold=True, size=10.5, color="#222"),
        _c(f"{_v(ep.get('target_1'))}", size=10, color="#C41E3A", align=1),
        _c(f"{_v(ep.get('target_2'))}", size=10, color="#C41E3A", align=1),
        _c(f"{_v(ep.get('stop'))}", size=10, color="#27AE60", align=1),
        _c(f"{_v(ep.get('rr'))}", size=10, align=1),
        _c(f"{float(rp.get('initial_position_ratio',0)) * 100:.0f}%" if rp.get('initial_position_ratio') else "-",
           size=10, align=1),
        _c(f"{rp.get('final_risk_rating','-')}",
           size=10, align=1,
           color={"低": "#27AE60", "中": "#E67E22", "高": "#C0392B"}.get(rp.get('final_risk_rating', ''), "#333")),
    ]

    # Dot Plot 行: FOMC 点阵图 + 图例
    expert_details = sc.get("expert_details") or []
    expert_median = sc.get("expert_median", expert_avg)
    expert_std = sc.get("expert_std", 0)
    dissent = sc.get("dissent") or []

    from reportlab.platypus import Paragraph as _P
    from reportlab.lib.styles import ParagraphStyle as _PS
    legend_style = _PS("legend", fontName=_CURRENT_BODY, fontSize=7.5,
                       textColor=colors.HexColor("#555"), leading=10)
    dot_plot = _make_dot_plot(expert_details, expert_avg, expert_median,
                              width=380, height=26) if expert_details else _c("-")

    # 分析"方向冲突"的异议: 辩论层 BUY 但专家异议偏空 / 辩论层 SELL 但专家异议偏多
    j_dir_upper = (ip.get("direction") or "HOLD").upper()
    judge_adj_val = sc.get("judge_adj", 0)

    conflict_dissent = []
    aligned_dissent = []
    for d in dissent:
        ddir = d.get("direction", "")
        if j_dir_upper == "BUY" and ddir == "偏空":
            conflict_dissent.append(d)
        elif j_dir_upper == "SELL" and ddir == "偏多":
            conflict_dissent.append(d)
        else:
            aligned_dissent.append(d)

    # 异议标注
    if conflict_dissent:
        # 高分歧决策警告(方向冲突)
        parts = [f"⚠ 高分歧决策  辩论层判 {j_dir_upper}({judge_adj_val:.1f})"]
        conflict_str = " / ".join(
            f"{d['role_cn']}唱{d['direction'].replace('偏','')} {d['score']:.0f}(偏离中位{abs(d['deviation']):.0f})"
            for d in conflict_dissent
        )
        parts.append(f"但 {conflict_str}")
        parts.append("→ 建议降仓/等确认")
        dissent_text = "  ·  ".join(parts)
        dissent_color = "#C0392B"    # 红
        dissent_bold = True
    elif aligned_dissent:
        # 同向异议(不警告, 只提示)
        dissent_text = "◐ 同向分歧: " + " | ".join(
            f"{d['role_cn']}({d['score']:.0f},{d['direction']}{abs(d['deviation']):.0f})"
            for d in aligned_dissent[:3]
        ) + f"    σ={expert_std:.1f}"
        dissent_color = "#E67E22"    # 橙
        dissent_bold = False
    else:
        dissent_text = f"✓ 专家共识良好  μ={expert_avg:.1f} / 中位={expert_median:.1f} / σ={expert_std:.1f}"
        dissent_color = "#27AE60"    # 绿
        dissent_bold = False

    dissent_style = _PS(
        "dissent", fontName=_CURRENT_BOLD if dissent_bold else _CURRENT_BODY,
        fontSize=8.5 if dissent_bold else 8, leading=12,
        textColor=colors.HexColor(dissent_color),
    )

    # 第 3 行: 买入/持有理由
    reason_label_buy = "买入理由" if group == "buy" else "核心判断" if group == "hold" else "Judge 点评"
    if group == "sell":
        # SELL: 重点展示 Bear 胜点 + winning_points
        reasons_text = _format_reasons(ip.get("winning_points", []), max_items=3, max_chars=220)
    else:
        reasons_text = _format_reasons(ip.get("key_reasons", []), max_items=3, max_chars=220)

    # 第 4 行: 风险/止损纪律
    sl = rp.get("stop_loss_discipline") or {}
    tp_rule = rp.get("take_profit_rule") or {}
    risk_parts = []
    if sl.get("description"):
        risk_parts.append(f"止损: {_trim(sl.get('description',''), 150)}")
    if tp_rule.get("description"):
        risk_parts.append(f"止盈: {_trim(tp_rule.get('description',''), 120)}")
    pm_sum = rp.get("pm_summary", "")
    if pm_sum:
        risk_parts.append(f"风控: {_trim(pm_sum, 200)}")
    risk_text = "\n".join(risk_parts) if risk_parts else "-"

    # 组装 6 行表格(8 列), 新增 Dot Plot 行 + 异议/共识行
    data = [
        row1_header,                                                          # Row 0
        row1_vals,                                                            # Row 1
        row2_header,                                                          # Row 2
        row2_vals,                                                            # Row 3
        [_c("专家点阵图", bold=True, size=8.5, color="#0D3B66"),               # Row 4
         dot_plot, _c(""), _c(""), _c(""), _c(""), _c(""), _c("")],
        [_c("", size=8), _P(dissent_text, dissent_style),                     # Row 5
         _c(""), _c(""), _c(""), _c(""), _c(""), _c("")],
        [_c(reason_label_buy, bold=True, size=8.5, color="#0D3B66"),           # Row 6
         _c(reasons_text, size=9, color="#222"), _c(""), _c(""), _c(""), _c(""), _c(""), _c("")],
        [_c("风险纪律", bold=True, size=8.5, color="#0D3B66"),                 # Row 7
         _c(risk_text, size=9, color="#555"), _c(""), _c(""), _c(""), _c(""), _c(""), _c("")],
    ]
    col_widths = [14 * mm, 38 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm, 27 * mm]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F8FC")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F5F8FC")),
        ("SPAN", (0, 0), (0, 1)),
        ("SPAN", (1, 0), (1, 1)),
        ("SPAN", (1, 4), (-1, 4)),   # Dot plot 跨整行
        ("SPAN", (1, 5), (-1, 5)),   # 异议/共识 跨整行
        ("SPAN", (1, 6), (-1, 6)),   # 买入理由
        ("SPAN", (1, 7), (-1, 7)),   # 风险纪律
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D0D0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("VALIGN", (0, 4), (-1, 4), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBEFORE", (0, 0), (0, -1), 2, colors.HexColor(level_color)),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 4))


# ────────────────────────────────────────────────
# CLI 入口
# ────────────────────────────────────────────────

def main_generate(since_str: str | None = None, out_path: Path | None = None) -> Path:
    """命令行入口: 从 output/runs_v3 收集数据生成汇总 PDF。"""
    if since_str:
        since = datetime.strptime(since_str, "%Y-%m-%d %H:%M")
    else:
        since = datetime(2026, 4, 19, 10, 30)   # 本批 94 只开始时间
    if out_path is None:
        out_path = Path(f"output/portfolio_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    runs_dir = Path("output/runs_v3")
    rows = _collect_all_results(runs_dir, since)
    if not rows:
        raise RuntimeError(f"未找到 {since} 之后的 v3 run")
    print(f"收集到 {len(rows)} 只股票")
    pdf = build_portfolio_summary_pdf(rows, out_path)
    print(f"PDF 生成: {pdf}  ({pdf.stat().st_size // 1024} KB)")
    return pdf


if __name__ == "__main__":
    import sys
    since = sys.argv[1] if len(sys.argv) > 1 else None
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    main_generate(since, out)
