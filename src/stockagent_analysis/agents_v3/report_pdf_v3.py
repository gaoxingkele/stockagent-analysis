"""v3 机构级 PDF 报告 - 包含 6 Phase 全链路的可读性渲染。

复用 report_pdf.py 的辅助函数(字体注册/_cell/_cells/_esc)。
产出结构:
  1. 封面
  2. 执行摘要卡片
  3. Phase 0 量化事实(技术/结构/资金 3 节)
  4. Phase 1 专家研判表(4 角色)
  5. Phase 2 投资决策 - Bull/Bear 辩论 + Judge
  6. Phase 3 首席交易员方案
  7. Phase 4 风控 Portfolio Manager + 三方辩论
  8. 附录: 融合公式说明
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
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak,
)

from ..report_pdf import _register_fonts, _esc, _cell, _cells, _safe_filename


def _register_font_family(body_font: str, bold_font: str) -> None:
    """注册 reportlab 的 FontFamily, 让 <b>...</b> 标签能正确映射到粗体字族。

    若未注册 family, reportlab 会调 ps2tt(font) 查找 family,
    对 STSong-Light/SimSun 这类自定义字体会报错。
    """
    try:
        pdfmetrics.registerFontFamily(
            body_font,
            normal=body_font, bold=bold_font,
            italic=body_font, boldItalic=bold_font,
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────
# 颜色/等级辅助
# ─────────────────────────────────────────────────────────────────

_LEVEL_CN = {
    "strong_buy": "强烈买入",
    "weak_buy": "弱买入",
    "hold": "观望",
    "watch_sell": "观望(偏卖)",
    "weak_sell": "弱卖出",
    "strong_sell": "强烈卖出",
}
_LEVEL_COLOR = {
    "strong_buy": "#C41E3A",
    "weak_buy": "#E74C3C",
    "hold": "#0066CC",
    "watch_sell": "#805AD5",
    "weak_sell": "#27AE60",
    "strong_sell": "#228B22",
}
_DIR_CN = {"BUY": "买入", "HOLD": "观望", "SELL": "卖出", "": "-"}


def _rating_color(rating: str) -> str:
    return {"低": "#27AE60", "中": "#E67E22", "高": "#C0392B"}.get(rating, "#333333")


def _score_color(score: float) -> str:
    if score >= 72:
        return "#C41E3A"
    if score >= 62:
        return "#0066CC"
    if score >= 52:
        return "#805AD5"
    if score >= 42:
        return "#27AE60"
    return "#228B22"


def _trim(s: str, n: int = 280) -> str:
    s = (s or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    if len(s) <= n:
        return s
    return s[:n] + "…"


# ─────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────

_CURRENT_BODY_FONT: str = "STSong-Light"
_CURRENT_BOLD_FONT: str = "STSong-Light"


def _c(text, size: float = 9, bold: bool = False, color: str = "#222222", align: int = 0):
    """_cell 的 v3 包装: 自动用全局已注册字体, 避免 STSong-Light 回退。"""
    font = _CURRENT_BOLD_FONT if bold else _CURRENT_BODY_FONT
    return _cell(text, font=font, size=size, bold=bold, color=color, align=align)


def _cs(row, size: float = 9, bold: bool = False):
    font = _CURRENT_BOLD_FONT if bold else _CURRENT_BODY_FONT
    return [_cell(str(c), font=font, size=size, bold=bold) for c in row]


def build_investor_pdf_v3(run_dir: Path, result: dict[str, Any]) -> Path:
    """从 v3 final_decision_v3.json 生成机构级 PDF。"""
    global _CURRENT_BODY_FONT, _CURRENT_BOLD_FONT
    body_font, bold_font = _register_fonts()
    _register_font_family(body_font, bold_font)
    _CURRENT_BODY_FONT = body_font
    _CURRENT_BOLD_FONT = bold_font

    symbol = str(result.get("symbol", "")).strip()
    name = str(result.get("name", "")).strip()
    pdf_name = _safe_filename(f"{symbol}_{name}_v3投研报告.pdf")
    pdf_path = Path(run_dir) / pdf_name

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="v3 多角色投研报告",
    )

    styles = getSampleStyleSheet()
    st_title = ParagraphStyle(
        "v3Title", parent=styles["Title"],
        fontName=bold_font, fontSize=18, leading=22,
        textColor=colors.HexColor("#0D3B66"),
    )
    st_h1 = ParagraphStyle(
        "v3H1", parent=styles["Heading1"],
        fontName=bold_font, fontSize=13, leading=18,
        textColor=colors.HexColor("#0D3B66"), spaceBefore=10, spaceAfter=4,
    )
    st_h2 = ParagraphStyle(
        "v3H2", parent=styles["Heading2"],
        fontName=bold_font, fontSize=11, leading=15,
        textColor=colors.HexColor("#1E3A5F"), spaceBefore=6, spaceAfter=2,
    )
    st_body = ParagraphStyle(
        "v3Body", parent=styles["BodyText"],
        fontName=body_font, fontSize=10, leading=15,
        textColor=colors.black,
    )
    st_quote = ParagraphStyle(
        "v3Quote", parent=st_body,
        fontSize=9, leading=13, leftIndent=10,
        textColor=colors.HexColor("#555555"),
        borderPadding=4,
    )

    flow: list = []

    # ── 1. 封面 / 标题 ──
    _section_cover(flow, result, st_title, st_h2, st_body, bold_font)

    # ── 2. 执行摘要卡片 ──
    _section_executive_summary(flow, result, st_h1, st_body, body_font, bold_font)

    # ── 3. Phase 0 量化事实 ──
    flow.append(PageBreak())
    _section_phase0_facts(flow, result, run_dir, st_h1, st_h2, st_body, st_quote, body_font, bold_font)

    # ── 4. Phase 1 专家研判 ──
    flow.append(PageBreak())
    _section_phase1_experts(flow, result, st_h1, st_h2, st_body, body_font, bold_font)

    # ── 5. Phase 2 投资决策 ──
    flow.append(PageBreak())
    _section_phase2_debate(flow, result, st_h1, st_h2, st_body, st_quote, body_font, bold_font)

    # ── 6. Phase 3 首席交易员 ──
    flow.append(PageBreak())
    _section_phase3_trader(flow, result, st_h1, st_h2, st_body, body_font, bold_font)

    # ── 7. Phase 4 风控 ──
    flow.append(PageBreak())
    _section_phase4_risk(flow, result, st_h1, st_h2, st_body, st_quote, body_font, bold_font)

    # ── 8. 附录 ──
    flow.append(PageBreak())
    _section_appendix(flow, result, st_h1, st_h2, st_body, body_font, bold_font)

    doc.build(flow)
    return pdf_path


# ─────────────────────────────────────────────────────────────────
# Section 1: 封面
# ─────────────────────────────────────────────────────────────────

def _section_cover(flow, result, st_title, st_h2, st_body, bold_font):
    symbol = result.get("symbol", "")
    name = result.get("name", "")
    generated = result.get("generated_at", datetime.now().isoformat())
    final_score = float(result.get("final_score", 0))
    final_dec = str(result.get("final_decision", "hold")).upper()
    level = result.get("decision_level", "hold")
    level_cn = _LEVEL_CN.get(level, level)
    level_color = _LEVEL_COLOR.get(level, "#333333")
    dur = result.get("duration_sec", 0)

    flow.append(Paragraph("v3 多智能体投研报告", st_title))
    flow.append(Paragraph("LLM 角色化架构 · Phase 0-5 流水线",
                          ParagraphStyle("sub", fontName=bold_font, fontSize=10,
                                         textColor=colors.HexColor("#666666"))))
    flow.append(Spacer(1, 8))

    # 核心信息表格
    row1 = [_c(f"{symbol}  {name}", bold=True, size=14, color="#111111"),
            _c(f"生成时间: {generated[:19]}", size=10, color="#555555", align=2)]
    row2 = [_c(f"综合评分  {final_score:.2f}", bold=True, size=16, color=level_color),
            _c(f"{_DIR_CN.get(final_dec, final_dec)}  · {level_cn}",
                  bold=True, size=14, color=level_color, align=2)]
    row3 = [_c(f"处理耗时: {dur}s | 辩论 {result.get('config',{}).get('debate_rounds','?')} 轮 + 风控 {result.get('config',{}).get('risk_rounds','?')} 轮", size=9, color="#888888"),
            _c("", size=9)]

    tbl = Table([row1, row2, row3], colWidths=[100 * mm, 70 * mm])
    tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F5F8FC")),
        ("LINEBELOW", (0, 1), (-1, 1), 1, colors.HexColor(level_color)),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    flow.append(tbl)


# ─────────────────────────────────────────────────────────────────
# Section 2: 执行摘要
# ─────────────────────────────────────────────────────────────────

def _section_executive_summary(flow, result, st_h1, st_body, body_font, bold_font):
    flow.append(Paragraph("执行摘要", st_h1))

    sc = result.get("score_components") or {}
    ip = result.get("investment_plan") or {}
    tp = result.get("trading_plan") or {}
    rp = result.get("risk_policy") or {}

    # 评分拆解
    header = _cs(["分量", "评分", "权重", "加权贡献", "说明"], bold=True, size=9.5)
    rows = [header]
    for key, w, desc in [
        ("expert_avg", 0.50, "4 位专家平均分"),
        ("judge_adj",  0.32, "Judge 仲裁分(强信号放大后)"),
        ("risk_mapped", 0.18, "PM 风险映射分"),
    ]:
        v = float(sc.get(key, 0))
        rows.append(_cs([key, f"{v:.2f}", f"{w:.2f}", f"{v*w:.2f}", desc], size=9))
    bonus = float(sc.get("consensus_bonus", 0))
    rows.append(_cs(["一致性奖励", f"{bonus:+.2f}", "-", f"{bonus:+.2f}",
                        sc.get("bonus_reason", "")], size=9))
    rows.append(_cs(["最终评分", f"{sc.get('final_score', 0):.2f}",
                        "", "", ""], bold=True, size=10))
    tbl = Table(rows, colWidths=[28 * mm, 20 * mm, 16 * mm, 22 * mm, 80 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#FFF6E5")),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 6))

    # 三层决策对比
    flow.append(Paragraph("三层决策对比", st_h1))
    j_dir = ip.get("direction", "-")
    t_dec = tp.get("final_decision", "-")
    pm_align = rp.get("alignment_with", "-")
    pm_rating = rp.get("final_risk_rating", "-")

    rows2 = [_cs(["层级", "输出", "方向/立场", "核心"], bold=True, size=9.5)]
    rows2.append(_cs(["Phase 2 研究主管",
                         f"score={ip.get('overall_score','?')} conf={ip.get('confidence','?')}",
                         _DIR_CN.get(j_dir, j_dir),
                         _trim(ip.get("judge_comment", ""), 200)], size=9))
    rows2.append(_cs(["Phase 3 首席交易员",
                         f"pos={tp.get('initial_position_ratio', '?')}  {tp.get('primary_strategy','')}",
                         _DIR_CN.get(t_dec, t_dec),
                         _trim(tp.get("reasoning", ""), 200)], size=9))
    rows2.append(_cs(["Phase 4 风控主管",
                         f"maxpos={rp.get('max_position_ratio','?')}  rating={pm_rating}",
                         pm_align,
                         _trim(rp.get("pm_summary", ""), 200)], size=9))
    tbl2 = Table(rows2, colWidths=[32 * mm, 40 * mm, 20 * mm, 74 * mm])
    tbl2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(tbl2)


# ─────────────────────────────────────────────────────────────────
# Section 3: Phase 0 量化事实
# ─────────────────────────────────────────────────────────────────

def _section_phase0_facts(flow, result, run_dir, st_h1, st_h2, st_body, st_quote, body_font, bold_font):
    flow.append(Paragraph("Phase 0 · 量化事实层", st_h1))
    flow.append(Paragraph(
        "由纯 Python 计算(不含 LLM), 产出 6 份客观事实报告, 作为下游所有 LLM 角色的共享输入。",
        st_body))
    flow.append(Spacer(1, 4))

    # 从 phase0_bundle.md 读取(会比较长, 做摘要)
    bundle_md = Path(run_dir) / "data" / "phase0_bundle.md"
    if bundle_md.exists():
        text = bundle_md.read_text(encoding="utf-8")
        # 切分为 6 份 section
        sections = _split_phase0_sections(text)
        # 选择 3 个关键节展示: 技术面/K线结构/资金筹码
        for key in ["技术面报告", "K线结构报告", "资金/筹码报告"]:
            content = sections.get(key, "")
            if content:
                flow.append(Paragraph(key, st_h2))
                for line in content.split("\n")[:30]:   # 限制长度
                    line = line.strip()
                    if not line:
                        flow.append(Spacer(1, 2))
                        continue
                    if line.startswith("##"):
                        continue   # 标题已显示
                    flow.append(Paragraph(_esc(line), st_body))
                flow.append(Spacer(1, 6))
    else:
        flow.append(Paragraph("(Phase 0 bundle 文件未找到)", st_body))


def _split_phase0_sections(md: str) -> dict[str, str]:
    """把 phase0_bundle.md 切成 6 份。"""
    sections: dict[str, str] = {}
    current_key = None
    buf: list[str] = []
    for line in md.split("\n"):
        if line.startswith("## "):
            title = line[3:].strip()
            if current_key:
                sections[current_key] = "\n".join(buf).strip()
            current_key = title
            buf = []
        elif current_key:
            buf.append(line)
    if current_key:
        sections[current_key] = "\n".join(buf).strip()
    return sections


# ─────────────────────────────────────────────────────────────────
# Section 4: Phase 1 专家研判
# ─────────────────────────────────────────────────────────────────

def _section_phase1_experts(flow, result, st_h1, st_h2, st_body, body_font, bold_font):
    flow.append(Paragraph("Phase 1 · 专业视角分析师", st_h1))
    flow.append(Paragraph(
        "4 个 LLM 专家角色并行分析 Phase 0 报告, 各自从专业视角输出研判。",
        st_body))
    flow.append(Spacer(1, 4))

    experts = result.get("experts") or {}
    rows = [_cs(["角色", "模型", "评分", "研判结论", "风险提示"], bold=True, size=9.5)]
    for role_id in ["structure_expert", "wave_expert", "intraday_t_expert", "martingale_expert"]:
        e = experts.get(role_id) or {}
        if not e:
            continue
        score = float(e.get("score", 50))
        rows.append([
            _c(str(e.get("role_cn", role_id)), bold=True, size=9),
            _c(str(e.get("provider", "-")), size=8, color="#666666"),
            _c(f"{score:.1f}", bold=True, size=10, color=_score_color(score), align=1),
            _c(_trim(e.get("analysis", ""), 220), size=8.5),
            _c(_trim(e.get("risk", ""), 150), size=8, color="#888"),
        ])
    tbl = Table(rows, colWidths=[26 * mm, 18 * mm, 14 * mm, 76 * mm, 36 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 6))

    # 专家关键数据 (key_data)
    flow.append(Paragraph("专家关键结构化输出", st_h2))
    for role_id, title in [("structure_expert", "K 走势结构"), ("wave_expert", "波浪理论"),
                            ("intraday_t_expert", "短线做 T"), ("martingale_expert", "马丁/网格")]:
        e = experts.get(role_id) or {}
        kd = e.get("key_data") or {}
        if not kd:
            continue
        flow.append(Paragraph(f"<b>{title}</b>",
                              ParagraphStyle("ExpKey", fontName=bold_font, fontSize=9.5,
                                             textColor=colors.HexColor("#1E3A5F"), leading=13)))
        for k, v in kd.items():
            v_str = str(v)
            if len(v_str) > 150:
                v_str = v_str[:150] + "…"
            flow.append(Paragraph(f"• <b>{_esc(k)}</b>: {_esc(v_str)}", st_body))
        flow.append(Spacer(1, 3))


# ─────────────────────────────────────────────────────────────────
# Section 5: Phase 2 多空辩论
# ─────────────────────────────────────────────────────────────────

def _section_phase2_debate(flow, result, st_h1, st_h2, st_body, st_quote, body_font, bold_font):
    flow.append(Paragraph("Phase 2 · 多空辩论 + 研究主管仲裁", st_h1))

    ip = result.get("investment_plan") or {}
    bull_rounds = result.get("bull_rounds") or []
    bear_rounds = result.get("bear_rounds") or []

    # Judge 仲裁表
    flow.append(Paragraph("研究主管 InvestmentPlan", st_h2))
    rows = [_cs(["字段", "值"], bold=True, size=9.5)]
    for label, key in [
        ("方向",       "direction"),
        ("置信度",     "confidence"),
        ("胜方",       "winner"),
        ("综合评分",   "overall_score"),
        ("时间维度",   "time_horizon"),
        ("支撑位",     "key_support"),
        ("阻力位",     "key_resistance"),
        ("上行目标",   "target_price_up"),
        ("下行目标",   "target_price_down"),
    ]:
        v = ip.get(key)
        rows.append(_cs([label, str(v) if v is not None else "-"], size=9))

    # 关键理由(数组)
    reasons = ip.get("key_reasons") or []
    reason_text = "\n".join(f"• {r}" for r in reasons) if reasons else "-"
    rows.append([_c("关键理由", bold=True, size=9),
                 _c(_trim(reason_text, 450), size=8.5)])
    rows.append([_c("仲裁点评", bold=True, size=9),
                 _c(_trim(ip.get("judge_comment", ""), 400), size=8.5)])

    tbl = Table(rows, colWidths=[28 * mm, 142 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F8FC")),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 6))

    # 辩论摘要 - 只显示最后一轮
    flow.append(Paragraph("多空辩论摘要(最后一轮)", st_h2))
    if bull_rounds:
        flow.append(Paragraph(f"<b>多方(Bull) 最终陈词:</b>", st_body))
        flow.append(Paragraph(_esc(_trim(bull_rounds[-1], 700)), st_quote))
        flow.append(Spacer(1, 4))
    if bear_rounds:
        flow.append(Paragraph(f"<b>空方(Bear) 最终陈词:</b>", st_body))
        flow.append(Paragraph(_esc(_trim(bear_rounds[-1], 700)), st_quote))


# ─────────────────────────────────────────────────────────────────
# Section 6: Phase 3 首席交易员
# ─────────────────────────────────────────────────────────────────

def _section_phase3_trader(flow, result, st_h1, st_h2, st_body, body_font, bold_font):
    flow.append(Paragraph("Phase 3 · 首席交易员方案", st_h1))
    tp = result.get("trading_plan") or {}

    # 基础信息
    decision = tp.get("final_decision", "HOLD")
    pos_ratio = tp.get("initial_position_ratio", 0)
    horizon = tp.get("time_horizon", "-")
    strategy = tp.get("primary_strategy", "-")

    head = Table([
        [_c(f"决策: {_DIR_CN.get(decision, decision)}", bold=True, size=13,
               color=_LEVEL_COLOR.get("weak_buy" if decision == "BUY" else "weak_sell" if decision == "SELL" else "hold")),
         _c(f"首仓: {pos_ratio}  |  主策略: {strategy}  |  周期: {horizon}",
               size=10, color="#444")]
    ], colWidths=[55 * mm, 115 * mm])
    head.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#888888")),
    ]))
    flow.append(head)
    flow.append(Spacer(1, 4))

    # 入场方案表
    flow.append(Paragraph("入场策略(三套)", st_h2))
    rows = [_cs(["策略", "入场价", "目标1", "目标2", "止损价", "R:R", "触发条件"], bold=True, size=9.5)]
    for ep in (tp.get("entry_plans") or []):
        if not isinstance(ep, dict):
            continue
        rows.append(_cs([
            str(ep.get("strategy", "-")),
            str(ep.get("entry", "-")),
            str(ep.get("target_1", "-")),
            str(ep.get("target_2", "-")),
            str(ep.get("stop", "-")),
            str(ep.get("rr", "-")),
            _trim(str(ep.get("trigger", "-")), 60),
        ], size=9))
    tbl = Table(rows, colWidths=[22 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 14 * mm, 62 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (5, -1), "CENTER"),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 6))

    # 逻辑与条件
    for title, key in [("交易逻辑", "reasoning"),
                        ("持有条件", "hold_conditions"),
                        ("离场条件", "exit_conditions"),
                        ("风险警示", "risk_alert")]:
        v = tp.get(key, "")
        if v:
            flow.append(Paragraph(f"<b>{title}</b>: {_esc(_trim(v, 350))}", st_body))
            flow.append(Spacer(1, 2))

    # FINAL PROPOSAL 行
    final_line = tp.get("final_line", "")
    if final_line:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(f"<b><font color='#C41E3A'>{_esc(final_line)}</font></b>",
                              ParagraphStyle("final", fontName=bold_font, fontSize=11, alignment=1)))


# ─────────────────────────────────────────────────────────────────
# Section 7: Phase 4 风控
# ─────────────────────────────────────────────────────────────────

def _section_phase4_risk(flow, result, st_h1, st_h2, st_body, st_quote, body_font, bold_font):
    flow.append(Paragraph("Phase 4 · 风控主管政策 + 三方辩论", st_h1))
    rp = result.get("risk_policy") or {}

    rating = rp.get("final_risk_rating", "-")
    rating_color = _rating_color(rating)
    pos = rp.get("max_position_ratio", 0)
    init_pos = rp.get("initial_position_ratio", 0)
    align = rp.get("alignment_with", "neutral")

    # 核心政策表
    header = Table([
        [_c(f"风险评级: {rating}", bold=True, size=13, color=rating_color),
         _c(f"最大仓位: {pos}  |  首仓: {init_pos}  |  认同: {align}",
               size=10, color="#444")]
    ], colWidths=[45 * mm, 125 * mm])
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(rating_color)),
    ]))
    flow.append(header)
    flow.append(Spacer(1, 4))

    # 纪律与规则
    sl = rp.get("stop_loss_discipline") or {}
    tp_rule = rp.get("take_profit_rule") or {}

    rows = [_cs(["项", "内容"], bold=True, size=9.5)]
    if sl:
        rows.append([_c("止损纪律", bold=True, size=9),
                     _c(f"{sl.get('type','-')} @ {sl.get('value','-')} — {_trim(sl.get('description',''), 200)}", size=9)])
    if tp_rule:
        targets = tp_rule.get("targets") or []
        rows.append([_c("止盈规则", bold=True, size=9),
                     _c(f"{tp_rule.get('type','-')} 目标: {targets} — {_trim(tp_rule.get('description',''), 200)}", size=9)])
    for label, key in [("加仓条件", "add_position_condition"),
                        ("减仓条件", "reduce_position_condition"),
                        ("黑天鹅应对", "black_swan_response"),
                        ("修正说明", "override_comment"),
                        ("PM 总结", "pm_summary")]:
        v = rp.get(key, "")
        if v:
            rows.append([_c(label, bold=True, size=9),
                         _c(_trim(v, 320), size=9)])
    tbl = Table(rows, colWidths=[26 * mm, 144 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F8FC")),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 6))

    # 三方辩论摘要 - 只展示最后一轮
    flow.append(Paragraph("风控三方辩论摘要(最后一轮)", st_h2))
    for label, key in [("激进派", "risk_aggressive_rounds"),
                        ("保守派", "risk_conservative_rounds"),
                        ("中性派", "risk_neutral_rounds")]:
        rounds = result.get(key) or []
        if rounds:
            flow.append(Paragraph(f"<b>{label}:</b>", st_body))
            flow.append(Paragraph(_esc(_trim(rounds[-1], 450)), st_quote))
            flow.append(Spacer(1, 3))


# ─────────────────────────────────────────────────────────────────
# Section 8: 附录
# ─────────────────────────────────────────────────────────────────

def _section_appendix(flow, result, st_h1, st_h2, st_body, body_font, bold_font):
    flow.append(Paragraph("附录 · 融合公式与架构说明", st_h1))
    cfg = result.get("config") or {}

    flow.append(Paragraph("评分融合公式", st_h2))
    formula = (
        "<b>final = 0.50·expert_avg + 0.32·judge_adj + 0.18·risk_mapped + consensus_bonus</b><br/>"
        "• expert_avg = 4 位 Phase 1 专家评分算术平均(0-100)<br/>"
        "• judge_adj = Judge.overall_score × 1.08 (BUY+conf≥0.65) 或 ×0.92 (SELL+conf≥0.65), 否则原值<br/>"
        "• risk_mapped = 20 + max_position_ratio × 60 + rating_bonus(低:+8/中:0/高:-10)<br/>"
        "• consensus_bonus = Judge+Trader 共识 BUY: +5, 共识 SELL: -5, 冲突: -4, HOLD: 0"
    )
    flow.append(Paragraph(formula, st_body))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("决策等级阈值", st_h2))
    levels_text = (
        "• ≥80 strong_buy 强烈买入 &nbsp;&nbsp;"
        "• 72-80 weak_buy 弱买入<br/>"
        "• 62-72 hold 观望 &nbsp;&nbsp;"
        "• 52-62 watch_sell 观望(偏卖)<br/>"
        "• 42-52 weak_sell 弱卖出 &nbsp;&nbsp;"
        "• <42 strong_sell 强烈卖出"
    )
    flow.append(Paragraph(levels_text, st_body))
    flow.append(Spacer(1, 6))

    flow.append(Paragraph("流水线配置", st_h2))
    cfg_rows = [_cs(["配置项", "值"], bold=True, size=9.5)]
    for label, key in [
        ("Bull Provider", "bull_provider"),
        ("Bear Provider", "bear_provider"),
        ("Judge Provider", "judge_provider"),
        ("Trader Provider", "trader_provider"),
        ("PM Provider", "pm_provider"),
        ("多空辩论轮数", "debate_rounds"),
        ("风控辩论轮数", "risk_rounds"),
    ]:
        cfg_rows.append(_cs([label, str(cfg.get(key, "-"))], size=9))
    tbl = Table(cfg_rows, colWidths=[45 * mm, 125 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0FE")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 8))

    flow.append(Paragraph(
        "生成工具: stockagent-analysis v3 · LLM 角色化架构 · 分支 feat/llm-role-refactor",
        ParagraphStyle("foot", fontName=body_font, fontSize=8,
                        textColor=colors.HexColor("#888"), alignment=1),
    ))
