"""半导体板块 V12+V11 综合评测 PDF 报告生成."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether,
)

ROOT = Path(__file__).resolve().parent

# 中文字体
pdfmetrics.registerFont(TTFont("msyh", "C:/Windows/Fonts/msyh.ttc"))
pdfmetrics.registerFont(TTFont("msyhbd", "C:/Windows/Fonts/msyhbd.ttc"))

# 配色 (Linear 风格)
ACCENT = colors.HexColor("#7B61FF")
SUCCESS = colors.HexColor("#3FB950")
WARN = colors.HexColor("#D29922")
DANGER = colors.HexColor("#F85149")
DIM = colors.HexColor("#5A5F6B")
MUTED = colors.HexColor("#8A8F9A")
BG_CARD = colors.HexColor("#F7F8FA")
LINE = colors.HexColor("#E5E7EB")

# 样式
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="msyhbd",
                     fontSize=22, leading=28, spaceAfter=8, textColor=ACCENT, alignment=TA_LEFT)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="msyhbd",
                     fontSize=15, leading=22, spaceBefore=14, spaceAfter=6,
                     textColor=colors.HexColor("#1F2937"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="msyhbd",
                     fontSize=12, leading=18, spaceBefore=8, spaceAfter=4,
                     textColor=colors.HexColor("#374151"))
P = ParagraphStyle("P", parent=styles["BodyText"], fontName="msyh",
                    fontSize=10, leading=15, spaceAfter=4, textColor=colors.HexColor("#1F2937"))
PSmall = ParagraphStyle("PSmall", parent=P, fontSize=8.5, leading=12, textColor=DIM)
PSmallMuted = ParagraphStyle("PSmallMuted", parent=PSmall, textColor=MUTED)


def load_data():
    v11 = pd.read_csv(ROOT / "output/v12_inference/v11_semi_top50_20260508.csv",
                       dtype={"ts_code": str})
    v12 = pd.read_csv(ROOT / "output/v7c_full_inference/v7c_inference_20260508.csv",
                       dtype={"ts_code": str})
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")
    m = v11.merge(
        v12[["ts_code","industry","buy_score","sell_score","r10_pred","r20_pred",
             "sell_20_v6_prob","quadrant","v7c_recommend"]],
        on="ts_code", how="left",
    )
    m = m.merge(basic[["ts_code", "name"]], on="ts_code", how="left")
    m["comp_score"] = (
        (m["r20_pred"].clip(-5, 20) + 5) / 25 * 40
        + m["bull_prob"].fillna(0.3) * 30
        + (1 - m["sell_score"]/100) * 30
    )
    m = m.sort_values("comp_score", ascending=False).reset_index(drop=True)
    m["rank"] = m.index + 1
    return m


def make_summary_box():
    """概览卡片."""
    cells = [
        ["报告日期", "2026-05-08 截面"],
        ["板块", "半导体 (申万二级)"],
        ["全板块股数", "187 只"],
        ["本次评测样本", "51 只 (Top 50 按 r20_pred + 300604)"],
        ["LLM 解析成功率", "50/51 (98%)"],
        ["bull≥0.5 共识看多", "16 只 (32%)"],
        ["V7c 主推", "0 只 (整体过热, 无标的通过 5 铁律)"],
        ["V12 评分版本", "V7c LGBM (r10_v4 + r20_v4 + sell_v6)"],
        ["LLM 评估版本", "V11 (claude-sonnet-4-6 视觉, 月+周+日 3 框架)"],
    ]
    t = Table(cells, colWidths=[42*mm, 130*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("BACKGROUND", (0,0), (0,-1), BG_CARD),
        ("TEXTCOLOR", (0,0), (0,-1), DIM),
        ("FONTNAME", (0,0), (0,-1), "msyhbd"),
        ("LINEBELOW", (0,0), (-1,-2), 0.4, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def make_top30_table(m: pd.DataFrame):
    """Top 30 综合推荐表."""
    headers = ["#", "代码", "中文名", "buy", "sell", "r20%", "bull", "bear", "trend", "象限", "综合"]
    rows = [headers]
    for _, r in m.head(30).iterrows():
        bull = f"{r['bull_prob']:.2f}" if pd.notna(r['bull_prob']) else "—"
        bear = f"{r['bear_prob']:.2f}" if pd.notna(r['bear_prob']) else "—"
        trend = (r.get("trend_strength") or "")[:8]
        name = str(r.get("name") or "—")[:8]
        rows.append([
            str(int(r["rank"])),
            r["ts_code"],
            name,
            f"{r['buy_score']:.0f}",
            f"{r['sell_score']:.0f}",
            f"{r['r20_pred']:+.2f}",
            bull, bear, trend,
            str(r.get("quadrant","") or ""),
            f"{r['comp_score']:.1f}",
        ])
    col_widths = [8*mm, 20*mm, 22*mm, 12*mm, 12*mm, 15*mm, 12*mm, 12*mm, 17*mm, 16*mm, 13*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8.5),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG_CARD]),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ])
    # bull≥0.5 行高亮; v7c理想多绿色; 矛盾段橙色 (列号+1 因加了中文名列)
    for i, (_, r) in enumerate(m.head(30).iterrows(), 1):
        bp = r.get("bull_prob")
        if pd.notna(bp) and bp >= 0.5:
            style.add("FONTNAME", (6,i), (6,i), "msyhbd")
            style.add("TEXTCOLOR", (6,i), (6,i), SUCCESS)
        if r.get("quadrant") == "理想多":
            style.add("TEXTCOLOR", (9,i), (9,i), SUCCESS)
            style.add("FONTNAME", (9,i), (9,i), "msyhbd")
        elif r.get("quadrant") == "矛盾段":
            style.add("TEXTCOLOR", (9,i), (9,i), WARN)
        # r20 红涨绿跌 (A 股惯例)
        try:
            v = float(r["r20_pred"])
            if v > 0:
                style.add("TEXTCOLOR", (5,i), (5,i), DANGER)
            elif v < 0:
                style.add("TEXTCOLOR", (5,i), (5,i), SUCCESS)
        except Exception:
            pass
        # 中文名加粗
        style.add("FONTNAME", (2,i), (2,i), "msyhbd")
    t.setStyle(style)
    return t


def make_strong_bull_section(m: pd.DataFrame):
    """LLM 强多头 16 只详细."""
    strong = m[(m["status"] == "ok") & (m["bull_prob"] >= 0.5)].sort_values(
        "bull_prob", ascending=False
    ).reset_index(drop=True)
    rows = [["代码", "中文名", "bull/trend", "r20", "sell", "形态 + Elliott"]]
    for _, r in strong.iterrows():
        bull = f"{r['bull_prob']:.2f}/{r['trend_strength']}"
        pat = (r.get("key_pattern") or "").strip()
        ell = (r.get("elliott_phase") or "").strip()
        name = str(r.get("name") or "—")[:7]
        combined = Paragraph(f"<b>{pat}</b><br/><font color='#5A5F6B' size='8'>{ell}</font>", PSmall)
        rows.append([
            r["ts_code"], name, bull, f"{r['r20_pred']:+.2f}%",
            f"{r['sell_score']:.0f}", combined,
        ])
    col_widths = [20*mm, 20*mm, 22*mm, 14*mm, 12*mm, 84*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8.5),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), SUCCESS),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("ALIGN", (0,1), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG_CARD]),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def make_position_table():
    """实战仓位分级."""
    rows = [
        ["级别", "标的 (代码 + 中文名 + 综合分 + 关键信号)", "建议仓位"],
        ["★★★\n核心组合", Paragraph(
            "<b>603078.SH 江化微</b> (61.7) 半导体材料·主升 3 浪延伸 + 三买加速<br/>"
            "<b>688173.SH 希荻微</b> (58.6) 电源管理芯片·V 反转 + 类一买<br/>"
            "<b>688230.SH 芯导科技</b> (58.8) 功率半导体·V 反转 + 二买<br/>"
            "<b>688620.SH 安凯微</b> (65.9) 模拟芯片·一类买点 + r20=+10.5%", PSmall),
         "≤ 2% / 股"],
        ["★★\n高潜力", Paragraph(
            "<b>688130 晶华微</b> (10.41%) / <b>688061 灿瑞科技</b> (8.70%) / "
            "<b>603375 盛景微</b> (8.61%) / <b>688216 气派科技</b> (9.50%, <b>封测</b>) / "
            "<b>688512 慧智微-U</b> (8.89%) / <b>688381 帝奥微</b> (8.90%) / "
            "<b>688653 康希通信</b> (7.67%) — bull≥0.55, r20 高", PSmall),
         "≤ 1% / 股"],
        ["★\n第三梯队", Paragraph(
            "<b>920139.BJ 华岭股份</b> (r20=+18.42% 全场最高但 LLM 仅 0.35) / "
            "<b>300241 瑞丰光电</b> / <b>002449 国星光电</b> / "
            "<b>688538 和辉光电-U</b> / <b>688469 芯联集成-U</b>", PSmall),
         "≤ 0.5% / 股"],
        ["⚠\n警惕", Paragraph(
            "<font color='#F85149'><b>300604 长川科技 (#49/51, 综合 30.8)</b></font> — "
            "封测设备·bull=0.25 / bear=-12% 顶分型 + MACD 顶背离<br/>"
            "<b>002449 国星光电</b> 月线下降通道 + 周线空头排列 (量化看好但 LLM 否定)", PSmall),
         "不建议参与"],
    ]
    t = Table(rows, colWidths=[24*mm, 124*mm, 24*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (2,0), (2,-1), "CENTER"),
        ("FONTNAME", (0,1), (0,-1), "msyhbd"),
        ("TEXTCOLOR", (0,1), (0,1), SUCCESS),
        ("TEXTCOLOR", (0,2), (0,2), ACCENT),
        ("TEXTCOLOR", (0,3), (0,3), MUTED),
        ("TEXTCOLOR", (0,4), (0,4), DANGER),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("BACKGROUND", (0,1), (-1,-1), colors.white),
        ("LINEBELOW", (0,0), (-1,-1), 0.4, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def make_sector_observation():
    """板块观察."""
    rows = [
        ["维度", "状态"],
        ["LLM 整体多头比例", "32% (bull≥0.5), 远高于矛盾段板 23%"],
        ["量化主推", "0/187 半导体股进 V7c 主推 — 板块整体过热"],
        ["主流形态", "主升 3 浪 / 三类买点 / 收敛三角形末端突破 — 高频出现"],
        ["风险信号", "16 只 bull=0.65 中多数 trend=moderate/weak — 视觉确认度未达极强"],
        ["集中度", "688 创业/科创板占 37/51 = 73% — 高估值高弹性"],
        ["定性结论", "板块主升浪中段, 但量化已识别短期过热"],
    ]
    t = Table(rows, colWidths=[42*mm, 130*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#374151")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("BACKGROUND", (0,1), (0,-1), BG_CARD),
        ("FONTNAME", (0,1), (0,-1), "msyhbd"),
        ("TEXTCOLOR", (0,1), (0,-1), DIM),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def make_focus_300604():
    """300604 特别关注卡."""
    rows = [
        [Paragraph("<b>300604 长川科技 (半导体设备 / 封装测试设备)</b>", H3),
         Paragraph("<font color='#F85149'><b>综合分 #49/51 = 30.8 (倒数第三)</b></font>", PSmall)],
        ["V12 量化",
         Paragraph("buy=81.6 / sell=61.1 / r20_pred=+2.27% / 中性区<br/>"
                   "5 铁律命中 1/4 (sell>30, pyr_velocity 不吸筹, "
                   "<b>f1_neg1=-0.7249</b> 大跌日主力异常流出)", PSmall)],
        ["V11 LLM 视觉",
         Paragraph("bull=0.25 / base=0.45 / <b>bear=0.30</b><br/>"
                   "Bull +8.0% / Base -3.0% / <b>Bear -12.0%</b><br/>"
                   "形态: 月线完美 HH/HL 上升通道, 周线突破布林上轨后回踩, "
                   "<b>日线形成顶分型+中枢震荡</b><br/>"
                   "Elliott: <b>主升 3 浪末端或 5 浪顶部, MACD 顶背离</b>, "
                   "等待修正浪 a-b-c", PSmall)],
        ["综合诊断",
         Paragraph("<b>三引擎一致看空</b>: V7c 量化 + V11 LLM 视觉 + 行业 beta 全部偏空<br/>"
                   "建议: <font color='#F85149'><b>当前未持有不建仓; "
                   "持有者考虑减持/止盈</b></font>", PSmall)],
    ]
    t = Table(rows, colWidths=[36*mm, 136*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#FFF7E6")),
        ("BACKGROUND", (0,1), (0,-1), BG_CARD),
        ("FONTNAME", (0,1), (0,-1), "msyhbd"),
        ("TEXTCOLOR", (0,1), (0,-1), DIM),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE),
        ("BOX", (0,0), (-1,-1), 1.0, WARN),
    ]))
    return t


def make_alternatives_to_300604():
    """与 300604 同细分赛道 (半导体设备/封测/材料) 的优质替代标的."""
    headers = ["代码", "中文名", "细分赛道", "综合分", "buy/sell", "r20", "bull/trend", "关键信号"]
    data = [
        # (code, name, sub, comp, buy, sell, r20, bull, trend, key)
        ("603078.SH", "江化微", "半导体材料 (湿电子化学品)",
         61.7, 94, 28, "+7.85%", 0.65, "strong",
         "周线突破中枢回踩确认, 日线三类买点后加速上涨, 主升 3 浪延伸"),
        ("688432.SH", "有研硅", "半导体材料 (大硅片)",
         54.2, 95, 51, "+7.55%", 0.65, "strong",
         "月线突破长期下降趋势线, 主升 3 浪延伸已创新高, 动能强劲"),
        ("688216.SH", "气派科技", "封装测试 (与长川同细分)",
         56.1, 99, 45, "+9.50%", 0.55, "moderate",
         "周线中枢突破后回踩, 日线二类买点构筑中, 主升 3 浪修正完成"),
        ("688419.SH", "耐科装备", "封测设备 (塑封设备, 直接对标)",
         55.6, 95, 22, "+6.78%", 0.45, "moderate",
         "月线突破长期中枢回踩确认, 周线二类买点, 日线中枢上沿震荡"),
        ("688352.SH", "颀中科技", "封装测试 (晶圆级封装)",
         55.6, 90, 9, "+6.15%", 0.35, "moderate",
         "周线中枢形成, 日线底分型后反弹遇 MA60 压制 (sell=9 极干净)"),
        ("688729.SH", "屹唐股份", "晶圆制造设备 (去胶机)",
         55.4, 92, 14, "+6.92%", 0.35, "moderate",
         "周线中枢震荡, 日线底分型后反弹测试 MA20, 疑似二类买点"),
        ("688605.SH", "先锋精科", "半导体设备零部件",
         53.9, 91, 18, "+6.69%", 0.35, "weak",
         "月线长期横盘, 周线 W 底后回落, 日线布林中轨附近底分型"),
    ]
    rows = [headers]
    for code, name, sub, comp, buy, sell, r20, bull, trend, key in data:
        rows.append([
            code, name, sub, f"{comp:.1f}",
            f"{buy}/{sell}", r20, f"{bull:.2f}/{trend}",
            Paragraph(f"<font size='8'>{key}</font>", PSmall),
        ])
    col_widths = [19*mm, 19*mm, 38*mm, 13*mm, 18*mm, 14*mm, 22*mm, 29*mm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,0), 9),
        ("FONTSIZE", (0,1), (-1,-1), 8.5),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("FONTNAME", (1,1), (1,-1), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), SUCCESS),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("ALIGN", (7,1), (7,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG_CARD]),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE),
        ("BOX", (0,0), (-1,-1), 0.8, SUCCESS),
        # r20 红色 (都为正)
        ("TEXTCOLOR", (5,1), (5,-1), DANGER),
        ("FONTNAME", (5,1), (5,-1), "msyhbd"),
    ])
    # bull≥0.5 字体加粗
    for i in range(1, len(rows)):
        bull_str = rows[i][6]
        try:
            bv = float(bull_str.split("/")[0])
            if bv >= 0.5:
                style.add("FONTNAME", (6,i), (6,i), "msyhbd")
                style.add("TEXTCOLOR", (6,i), (6,i), SUCCESS)
        except Exception:
            pass
    t.setStyle(style)
    return t


def build():
    out = ROOT / "output" / "v12_inference" / "半导体板块_V12+V11_综合评测_20260508.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title="半导体板块 V12+V11 综合评测", author="V12 评分系统",
    )

    m = load_data()
    story = []

    # 封面/标题
    story.append(Paragraph("半导体板块 V12+V11 综合评测", H1))
    story.append(Paragraph(
        "V7c LightGBM 量化 × V11 LLM 视觉技术分析 双引擎融合 "
        "&middot; 51 只样本 &middot; 2026-05-08 截面",
        PSmallMuted))
    story.append(Spacer(1, 6))

    # 概览
    story.append(Paragraph("📊 报告概览", H2))
    story.append(make_summary_box())

    # Top 30
    story.append(Paragraph("🏆 综合推荐 Top 30", H2))
    story.append(Paragraph(
        "排序方法: 综合分 = (r20_pred 归一化) × 40% + (LLM bull_prob) × 30% + "
        "(1 - sell_score/100) × 30%。绿色 bull 字体 = LLM 看多 (≥0.5); "
        "理想多象限绿色; 矛盾段橙色; r20 红涨绿跌 (A 股惯例)。",
        PSmallMuted))
    story.append(Spacer(1, 4))
    story.append(make_top30_table(m))

    story.append(PageBreak())

    # 强多头 16 只
    story.append(Paragraph("⭐ LLM 强多头信号 (bull ≥ 0.5) 详细", H2))
    story.append(Paragraph(
        "16 只通过 V11 LLM 视觉技术分析 (claude-sonnet-4-6) 获得 bull_prob ≥ 0.5 "
        "的标的, 表示 LLM 在 5 大域 (Dow / 支撑阻力 / MA / 成交量 / 缠论+Elliott) "
        "综合判断下倾向多头场景。",
        PSmallMuted))
    story.append(Spacer(1, 4))
    story.append(make_strong_bull_section(m))

    story.append(PageBreak())

    # 仓位
    story.append(Paragraph("🎯 实战仓位分级", H2))
    story.append(make_position_table())

    story.append(Spacer(1, 10))

    # 300604 特别关注
    story.append(Paragraph("🔍 特别关注: 300604 长川科技", H2))
    story.append(make_focus_300604())

    story.append(Spacer(1, 10))

    # 同细分赛道优质替代标的
    story.append(Paragraph("✅ 同细分赛道优质替代标的 (与长川科技同属半导体设备/封测/材料)", H2))
    story.append(Paragraph(
        "若投资者关注半导体设备/封测产业链, 但 300604 长川科技当前形态见顶, "
        "可考虑以下 <b>7 只同细分赛道但评分更优</b>的替代标的 (全部综合分高于长川 30.8): "
        "<font color='#F85149'>r20 红色 = 看涨幅度</font>, "
        "<font color='#3FB950'>bull 绿色 = LLM 强多头共识 (≥0.5)</font>。",
        PSmallMuted))
    story.append(Spacer(1, 4))
    story.append(make_alternatives_to_300604())

    story.append(Spacer(1, 10))

    # 板块观察
    story.append(Paragraph("📈 板块层面观察", H2))
    story.append(make_sector_observation())

    story.append(Spacer(1, 14))

    # 免责
    story.append(Paragraph("⚠️ 风险提示与方法局限", H2))
    disclaim = (
        "1. <b>评分模型局限</b>: V7c 量化基于 LightGBM 监督学习, 训练数据截止 2026-01, "
        "灾难月 (如 202508) 出现 -4.15pp 系统性失效记录, 灾难月识别能力待补强。<br/>"
        "2. <b>LLM 视觉局限</b>: V11 仅 200 样本 PoC 验证 +3.76pp (矛盾段), "
        "1000 样本显著性未确认; 本次半导体板块直接应用属扩展使用, 仅供参考。<br/>"
        "3. <b>板块过热警示</b>: 半导体板块 0/187 通过 V7c 5 铁律全过滤, "
        "整体 sell_score 偏高, 当前已处主升浪中后段。<br/>"
        "4. <b>仓位约束</b>: V12 单股 ≤2-5%, 总仓位分散 20+ 标的, "
        "单一行业 (如半导体) 总仓位建议 ≤30%。<br/>"
        "5. <b>非投资建议</b>: 本报告为量化模型 + LLM 形态识别的双引擎评分输出, "
        "不构成投资建议, 实操需结合资金面 / 政策面 / 公告面综合判断。"
    )
    story.append(Paragraph(disclaim, PSmall))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "—— 报告由 V12 评分系统自动生成 &middot; "
        "数据源: Tushare Pro &middot; "
        "评分模块: stockagent_analysis.v12_scoring + v11_vision",
        ParagraphStyle("foot", parent=PSmallMuted, alignment=TA_CENTER)))

    doc.build(story)
    print(f"OK: {out}")
    print(f"size: {out.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    build()
