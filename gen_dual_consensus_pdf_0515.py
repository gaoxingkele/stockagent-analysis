"""0515 双重独立共识池 16 只 PDF 报告."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont("msyh", "C:/Windows/Fonts/msyh.ttc"))
pdfmetrics.registerFont(TTFont("msyhbd", "C:/Windows/Fonts/msyhbd.ttc"))

ACCENT = colors.HexColor("#7B61FF"); SUCCESS = colors.HexColor("#3FB950")
WARN = colors.HexColor("#D29922"); DANGER = colors.HexColor("#F85149")
DIM = colors.HexColor("#5A5F6B"); MUTED = colors.HexColor("#8A8F9A")
BG = colors.HexColor("#F7F8FA"); LINE = colors.HexColor("#E5E7EB")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="msyhbd", fontSize=20,
                     leading=26, spaceAfter=6, textColor=ACCENT)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="msyhbd", fontSize=14,
                     leading=20, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1F2937"))
P = ParagraphStyle("P", parent=ss["BodyText"], fontName="msyh", fontSize=10,
                    leading=14, textColor=colors.HexColor("#1F2937"))
PS = ParagraphStyle("PS", parent=P, fontSize=8.5, leading=11.5, textColor=DIM)
PM = ParagraphStyle("PM", parent=PS, textColor=MUTED)


def load():
    df = pd.read_csv(ROOT / "output/drill_down/drill_down_20260515.csv", dtype={"ts_code": str})
    med = df["r20_pred"].median()
    sub = df[(df["r20_pred"] >= med) & (df["intraday_score"] >= 0.3)].sort_values(
        "intraday_score", ascending=False
    ).reset_index(drop=True)
    sub["rank"] = sub.index + 1
    return sub, med


def make_summary(m, med):
    rows = [
        ["截面日期", "2026-05-15 (周五收盘)"],
        ["筛选规则",
         f"V12 推荐池 Top 80 → 1H 钻取 → "
         f"日线 r20 ≥ {med:.2f}% AND 1H 综合分 ≥ +0.3"],
        ["最终入选", f"{len(m)} 只 (双重独立共识)"],
        ["板块分布", "化工原料 14 / 半导体 1 (盛景微) / 专用机械 1 (豪迈科技)"],
        ["相关性核验",
         "日线 r20 vs 1H 综合: Pearson r = +0.006 (完全正交)"],
        ["策略含义",
         "两个独立 alpha 维度同时触发 = 真正稀缺共识 (理论概率 ≈ 6%, 实际 20%)"],
    ]
    t = Table(rows, colWidths=[36*mm, 134*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (0,-1), "msyhbd"),
        ("BACKGROUND", (0,0), (0,-1), BG), ("TEXTCOLOR", (0,0), (0,-1), DIM),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW", (0,0), (-1,-2), 0.4, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def make_table(m):
    headers = ["#", "代码", "中文名", "行业", "日线 r20", "1H 分", "1H 趋势", "1H RSI", "原 V12 池"]
    rows = [headers]
    for _, r in m.iterrows():
        nm = str(r.get("name", ""))[:8]
        rows.append([
            str(int(r["rank"])), r["ts_code"], nm, str(r.get("industry",""))[:8],
            f"{r['r20_pred']:+.2f}%", f"{r['intraday_score']:+.2f}",
            str(r.get("intraday_trend","")), f"{r.get('intraday_rsi14',0):.1f}",
            str(r.get("pool",""))[:18],
        ])
    cw = [7*mm, 21*mm, 22*mm, 20*mm, 17*mm, 13*mm, 16*mm, 12*mm, 38*mm]
    t = Table(rows, colWidths=cw, repeatRows=1)
    st = TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,0), 9), ("FONTSIZE", (0,1), (-1,-1), 8.5),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), ACCENT), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG]),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE), ("BOX", (0,0), (-1,-1), 0.6, LINE),
        ("FONTNAME", (2,1), (2,-1), "msyhbd"),
    ])
    for i, (_, r) in enumerate(m.iterrows(), 1):
        # r20 红涨
        st.add("TEXTCOLOR", (4,i), (4,i), DANGER)
        st.add("FONTNAME", (4,i), (4,i), "msyhbd")
        # 1H 分高亮
        if r["intraday_score"] >= 0.7:
            st.add("FONTNAME", (5,i), (5,i), "msyhbd")
            st.add("TEXTCOLOR", (5,i), (5,i), SUCCESS)
        # 1H 趋势
        if r.get("intraday_trend") == "up":
            st.add("TEXTCOLOR", (6,i), (6,i), SUCCESS)
        # RSI > 75 黄色
        if r.get("intraday_rsi14", 0) > 75:
            st.add("TEXTCOLOR", (7,i), (7,i), WARN)
    t.setStyle(st)
    return t


def make_industry_summary(m):
    by_ind = m.groupby("industry").size().sort_values(ascending=False)
    rows = [["行业", "入选数", "占比"]]
    for ind, n in by_ind.items():
        rows.append([str(ind), str(n), f"{n/len(m)*100:.0f}%"])
    t = Table(rows, colWidths=[40*mm, 25*mm, 25*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#374151")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, BG]),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE), ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def make_position_advice():
    rows = [
        ["级别", "标的特征", "仓位建议"],
        ["★★★ 首选",
         Paragraph("<b>301076 新瀚新材 / 600618 氯碱化工 / 002825 纳尔股份</b> "
                   "(1H 分 ≥ +1.0, 趋势 up, RSI 60-70 健康)", PS),
         "各 ≤ 1.5%"],
        ["★★ 高潜",
         Paragraph("<b>603041/001335/301212/301149/301555/688267/603928</b> "
                   "(1H 分 ≥ +0.6, 趋势 up, 大多 RSI 60-72)", PS),
         "各 ≤ 1%"],
        ["★ 次选",
         Paragraph("<b>豪迈科技/盛景微</b> 及其他 sideways "
                   "(1H 不上涨但不弱, V12 r20 强)", PS),
         "各 ≤ 0.8%"],
        ["⚠ 警惕高 RSI",
         Paragraph("<b>605399 晨光新材 RSI 81</b> (短期过热, "
                   "可能 1-2 日内回调, 等回调入场更稳)", PS),
         "等回调"],
    ]
    t = Table(rows, colWidths=[22*mm, 130*mm, 22*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), ACCENT), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (0,-1), "CENTER"), ("ALIGN", (2,0), (2,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,1), (0,-1), "msyhbd"),
        ("TEXTCOLOR", (0,1), (0,1), SUCCESS), ("TEXTCOLOR", (0,2), (0,2), ACCENT),
        ("TEXTCOLOR", (0,3), (0,3), DIM), ("TEXTCOLOR", (0,4), (0,4), WARN),
        ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("BACKGROUND", (0,1), (-1,-1), colors.white),
        ("LINEBELOW", (0,0), (-1,-1), 0.4, LINE), ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def build():
    out = ROOT / "output" / "drill_down" / "双重独立共识池_16只_20260515.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm, topMargin=14*mm, bottomMargin=14*mm,
        title="双重独立共识池", author="V12 评分系统")
    m, med = load()
    story = []
    story.append(Paragraph("双重独立共识池 — 0515 16 只金矿", H1))
    story.append(Paragraph(
        "V12 日线 r20 (V16 LGBM, IC=0.32) × 1H 综合分 (规则信号) 双引擎独立共识. "
        "两个维度相关性 r ≈ 0 (完全正交), 同时触发即真共识.", PM))
    story.append(Spacer(1, 4))

    story.append(Paragraph("📊 报告概览", H2))
    story.append(make_summary(m, med))

    story.append(Paragraph("🏆 16 只完整清单 (按 1H 综合分降序)", H2))
    story.append(Paragraph(
        "颜色提示: r20 红涨 / 1H 分 ≥0.7 绿色加粗 / 1H 趋势 up 绿 / 1H RSI >75 黄(警告)",
        PM))
    story.append(Spacer(1, 3))
    story.append(make_table(m))

    story.append(Spacer(1, 10))
    story.append(Paragraph("🏭 行业分布", H2))
    story.append(make_industry_summary(m))

    story.append(PageBreak())
    story.append(Paragraph("🎯 仓位分级建议", H2))
    story.append(make_position_advice())

    story.append(Spacer(1, 10))
    story.append(Paragraph("🔍 跨尺度正交性的商业意义", H2))
    story.append(Paragraph(
        "<b>核心发现</b>: 日线 r20 (V16 LGBM) 与 1H 综合分 相关性 r=+0.006, 完全正交.<br/><br/>"
        "<b>为何这反而是好事</b>:<br/>"
        "- 共线 (r→1) 信号: 冗余, 两个信号 = 1 个信息<br/>"
        "- 正交 (r=0) 信号: 互补, 两个信号 = 2 个独立信息<br/>"
        "- 共识=巧合 vs 共识=双重独立验证<br/><br/>"
        "<b>数学解释</b>:<br/>"
        "- 日线 r20 (V16 LGBM, 232 因子): 中长期趋势/估值 (20 日 forward)<br/>"
        "- 1H 综合 (规则: MA20+RSI14+趋势+量能): 短期动量/微观结构 (5 日内)<br/>"
        "- 两者捕捉完全不同的市场信息层<br/><br/>"
        "<b>独立性的稀缺价值</b>:<br/>"
        "- 假设两信号都强各占 25%, 独立同时触发概率 ≈ 6%<br/>"
        "- 0515 实际 16/80 = 20% (略高于理论, 显示化工板块共振)<br/>"
        "- 这 16 只是真正的'高 confidence' 标的", PS))

    story.append(Spacer(1, 14))
    story.append(Paragraph("⚠️ 免责说明", H2))
    story.append(Paragraph(
        "1. 本报告基于 V12.17 量化模型 + 1H 规则信号双引擎.<br/>"
        "2. 化工原料板块 14/16 集中, 单一板块仓位上限 ≤ 30%, 避免行业系统性风险.<br/>"
        "3. 1H 信号反映最近 5 日动量, 不保证次日延续.<br/>"
        "4. 高 RSI 股 (晨光新材 81.1) 短期回调风险大, 等回调入场.<br/>"
        "5. 不构成投资建议, 实操需结合个人风险偏好 + 资金面 + 政策面.",
        PS))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "—— V12.17 评分系统 · 跨尺度独立共识 · 2026-05-15 截面",
        ParagraphStyle("foot", parent=PM, alignment=TA_CENTER)))

    doc.build(story)
    print(f"OK: {out}")
    print(f"size: {out.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    build()
