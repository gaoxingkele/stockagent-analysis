"""用户清单 25 只 0514 V12 评测 PDF."""
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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

ROOT = Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont("msyh", "C:/Windows/Fonts/msyh.ttc"))
pdfmetrics.registerFont(TTFont("msyhbd", "C:/Windows/Fonts/msyhbd.ttc"))

ACCENT = colors.HexColor("#7B61FF")
SUCCESS = colors.HexColor("#3FB950")
WARN = colors.HexColor("#D29922")
DANGER = colors.HexColor("#F85149")
DIM = colors.HexColor("#5A5F6B")
MUTED = colors.HexColor("#8A8F9A")
BG_CARD = colors.HexColor("#F7F8FA")
LINE = colors.HexColor("#E5E7EB")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="msyhbd", fontSize=20,
                     leading=26, spaceAfter=6, textColor=ACCENT)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="msyhbd", fontSize=14,
                     leading=20, spaceBefore=12, spaceAfter=5,
                     textColor=colors.HexColor("#1F2937"))
P = ParagraphStyle("P", parent=styles["BodyText"], fontName="msyh", fontSize=10,
                    leading=14, spaceAfter=3, textColor=colors.HexColor("#1F2937"))
PSmall = ParagraphStyle("PSmall", parent=P, fontSize=8.5, leading=11.5, textColor=DIM)
PMuted = ParagraphStyle("PMuted", parent=PSmall, textColor=MUTED)


def load_data():
    user = pd.read_csv(ROOT / "output/v12_inference/user_list_0514.csv", dtype={"ts_code": str})
    user = user.sort_values("r20_pred", ascending=False).reset_index(drop=True)
    user["rank"] = user.index + 1
    return user


def make_summary(m):
    n_ideal = (m["quadrant"] == "理想多").sum()
    n_contra = (m["quadrant"] == "矛盾段").sum()
    n_zombie = m["is_zombie"].fillna(False).astype(bool).sum()
    n_v7c = m["v7c_recommend"].fillna(False).astype(bool).sum()
    rows = [
        ["报告日期", "2026-05-14 截面 (V12.5 含 zombie filter)"],
        ["评测样本", "25 只 A 股 (123266 博士转债跳过)"],
        ["全市场 V7c 主推", "165 / 5128 (3.2%)"],
        ["清单 V7c 主推", f"{n_v7c} / 25 (含 6 铁律)"],
        ["清单理想多", f"{n_ideal} / 25 (sell ≤ 30, 量化看多)"],
        ["清单矛盾段", f"{n_contra} / 25"],
        ["清单僵尸区", f"{n_zombie} / 25"],
        ["最高 r20_pred", f"{m['r20_pred'].iloc[0]:+.2f}% ({m['name'].iloc[0]})"],
        ["最低 r20_pred", f"{m['r20_pred'].iloc[-1]:+.2f}% ({m['name'].iloc[-1]})"],
    ]
    t = Table(rows, colWidths=[40*mm, 130*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (0,-1), "msyhbd"),
        ("BACKGROUND", (0,0), (0,-1), BG_CARD),
        ("TEXTCOLOR", (0,0), (0,-1), DIM),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LINEBELOW", (0,0), (-1,-2), 0.4, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def make_full_table(m):
    headers = ["#", "代码", "中文名", "行业", "buy", "sell", "r20%", "象限", "横盘%", "僵尸"]
    rows = [headers]
    for _, r in m.iterrows():
        nm = str(r.get("name", ""))[:8]
        zp = f"{r['zombie_days_pct']*100:.0f}%" if pd.notna(r.get("zombie_days_pct")) else "—"
        iz = "🚨" if r.get("is_zombie") else "✓"
        rows.append([
            str(int(r["rank"])),
            r["ts_code"],
            nm,
            str(r.get("industry", ""))[:8],
            f"{r['buy_score']:.0f}",
            f"{r['sell_score']:.0f}",
            f"{r['r20_pred']:+.2f}",
            str(r.get("quadrant", "")),
            zp, iz,
        ])
    col_widths = [7*mm, 21*mm, 22*mm, 22*mm, 11*mm, 11*mm, 16*mm, 16*mm, 14*mm, 12*mm]
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
        ("FONTNAME", (2,1), (2,-1), "msyhbd"),
    ])
    for i, (_, r) in enumerate(m.iterrows(), 1):
        # 象限高亮
        if r.get("quadrant") == "理想多":
            style.add("TEXTCOLOR", (7,i), (7,i), SUCCESS)
            style.add("FONTNAME", (7,i), (7,i), "msyhbd")
        elif r.get("quadrant") == "矛盾段":
            style.add("TEXTCOLOR", (7,i), (7,i), WARN)
        # r20 红涨绿跌
        v = r.get("r20_pred")
        if pd.notna(v):
            if v > 0:
                style.add("TEXTCOLOR", (6,i), (6,i), DANGER)
            else:
                style.add("TEXTCOLOR", (6,i), (6,i), SUCCESS)
        # sell≥70 标红
        sv = r.get("sell_score", 0)
        if sv >= 70:
            style.add("TEXTCOLOR", (5,i), (5,i), DANGER)
            style.add("FONTNAME", (5,i), (5,i), "msyhbd")
        elif sv <= 30:
            style.add("TEXTCOLOR", (5,i), (5,i), SUCCESS)
        # 僵尸高亮
        if r.get("is_zombie"):
            style.add("TEXTCOLOR", (9,i), (9,i), DANGER)
            style.add("FONTNAME", (9,i), (9,i), "msyhbd")
        # buy>85 顶部警告
        bv = r.get("buy_score", 0)
        if bv > 85:
            style.add("TEXTCOLOR", (4,i), (4,i), WARN)
    t.setStyle(style)
    return t


def make_tier_table():
    rows = [
        ["级别", "标的 (代码 + 中文名 + 关键信号)", "仓位"],
        ["⭐⭐⭐\n首选", Paragraph(
            "<b>300525.SH 博思软件</b> (sell=15.1 全场最干净, 信创)<br/>"
            "<b>601866.SH 中远海发</b> (sell=0.1 全场最低, +5.80%, MA60 上行)<br/>"
            "<b>300917.SZ 特发服务</b> (sell=25.3, 6 日持续优胜)<br/>"
            "<b>002571.SZ 德力股份</b> (sell=19.1, 家居小盘)", PSmall),
         "≤ 2% / 股"],
        ["⭐⭐\n高潜力", Paragraph(
            "<b>300545.SZ 联得装备</b> (r20=+7.71% 第 1, buy=89.5)<br/>"
            "<b>301187.SZ 欧圣电气</b> (buy=90.8 最高, r20=+7.54%)<br/>"
            "<b>300530.SZ 领湃科技</b> (锂电设备, r20=+6.74%)", PSmall),
         "≤ 1% / 股"],
        ["⭐\n中性区潜力", Paragraph(
            "<b>688720.SH 艾森股份</b> (半导体材料, r20=+5.59%, sell=64.5)<br/>"
            "<b>300450.SZ 先导智能</b> (锂电设备龙头, r20=+5.58%)<br/>"
            "<b>301186.SZ 超达装备</b> / <b>002709.SZ 天赐材料</b>", PSmall),
         "≤ 0.5% / 股 / 观察"],
        ["❌\n避免", Paragraph(
            "<font color='#F85149'><b>300604.SZ 长川科技</b> r20=-1.99% + sell=80, 半导体设备见顶</font><br/>"
            "<font color='#F85149'><b>688409.SH 富创精密</b> sell=84 极端避雷</font><br/>"
            "<b>300895.SZ 铜牛信息</b> / <b>688037.SH 芯源微</b> sell=75 / <b>688012.SH 中微公司</b><br/>"
            "<b>300567 精测电子</b> / <b>300679 电连技术</b> / <b>000021 深科技</b> / <b>002922 伊戈尔</b> 信号弱", PSmall),
         "不参与"],
    ]
    t = Table(rows, colWidths=[22*mm, 124*mm, 24*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("ALIGN", (0,0), (0,-1), "CENTER"),
        ("ALIGN", (2,0), (2,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
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


def make_observations():
    rows = [
        ["主题", "观察"],
        ["半导体设备链集体回调",
         Paragraph("长川-1.99% / 芯源微+0.10% / 富创精密+0.31% / 中微公司+0.78% - "
                   "国产半导体设备龙头清单当前均偏弱, 与 0508-0511 大涨后的回调一致.<br/>"
                   "<b>艾森股份</b> (半导体材料, +5.59%) 是该板块唯一仍有读数的强势股 (但 sell=64.5 偏高)", PSmall)],
        ["持续优胜组",
         Paragraph("<b>300545 / 301187 / 300525 / 300530 / 300917</b> 连续 6 个交易日 (0508-0514) "
                   "都在 V12 评分前列. <b>多日一致信号是质量保证</b>, 可优先考虑.", PSmall)],
        ["中远海发 (601866) 特殊状态",
         Paragraph("buy=75.6 / sell=0.1 / r20=+5.80% / 100% 横盘, 但 <b>MA60 斜率为正</b>, "
                   "不算 zombie. '横盘 + MA60 上行' 是文章规则的智能性体现 - "
                   "经典 '上升中休整' 形态, 突破后可能加速.", PSmall)],
        ["V7c 主推 0/25",
         Paragraph("用户清单 <b>0 只通过 6 铁律</b>. 前 6 名虽是理想多, 但 buy 多数 >85 (V7c 上限被卡), "
                   "或 pyr_velocity 未在吸筹段, 或 f1/f2 不静默. 全市场 V7c 主推 165 只 (3.2%) "
                   "都未在用户清单中.", PSmall)],
        ["zombie 全部 0",
         Paragraph("25 只 <b>无一进入僵尸区</b> (尽管多只 100% 横盘, 但 MA60 斜率均非走平/下). "
                   "说明用户清单整体处于上升趋势的健康整理或加速段.", PSmall)],
    ]
    t = Table(rows, colWidths=[36*mm, 134*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "msyh"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("FONTNAME", (0,0), (-1,0), "msyhbd"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#374151")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,1), (0,-1), BG_CARD),
        ("FONTNAME", (0,1), (0,-1), "msyhbd"),
        ("TEXTCOLOR", (0,1), (0,-1), DIM),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW", (0,0), (-1,-1), 0.3, LINE),
        ("BOX", (0,0), (-1,-1), 0.6, LINE),
    ]))
    return t


def build():
    out = ROOT / "output" / "v12_inference" / "用户清单25只_V12评测_20260514.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm, topMargin=14*mm, bottomMargin=14*mm,
        title="用户清单 V12 评测", author="V12 评分系统",
    )
    m = load_data()
    story = []
    story.append(Paragraph("用户清单 25 只 V12 评测", H1))
    story.append(Paragraph(
        "V7c 6 铁律 (含 zombie filter) · 2026-05-14 截面 · 含 buy/sell/r20/象限/横盘/僵尸",
        PMuted))
    story.append(Spacer(1, 4))

    story.append(Paragraph("📊 报告概览", H2))
    story.append(make_summary(m))

    story.append(Paragraph("🏆 25 只完整评分表 (按 r20_pred 降序)", H2))
    story.append(Paragraph(
        "颜色提示: r20 红涨绿跌 / sell≤30 绿色 / sell≥70 红色 / buy>85 黄色(V7c 上限警告)",
        PMuted))
    story.append(Spacer(1, 3))
    story.append(make_full_table(m))

    story.append(PageBreak())
    story.append(Paragraph("🎯 实战仓位分级", H2))
    story.append(make_tier_table())

    story.append(Spacer(1, 10))
    story.append(Paragraph("🔍 关键观察", H2))
    story.append(make_observations())

    story.append(Spacer(1, 14))
    story.append(Paragraph("⚠️ 免责说明", H2))
    story.append(Paragraph(
        "1. V12.5 = V7c 6 铁律 (LGBM 4 模型 + 双向评分 + zombie 过滤). "
        "OOS 净 α vs hs300 +4.83pp/月, 胜率 70%.<br/>"
        "2. r20_pred 是 20 个交易日预期, 不保证当日兑现; 0511 验证显示 95% 方向准确率, "
        "5% 短期反向 (V12 唯一弱点).<br/>"
        "3. 全市场 V7c 主推 165 / 5128 = 3.2%, 用户清单 0 只通过 — "
        "若坚持配置, 优先 ⭐⭐⭐ 首选组合, 单股仓位 ≤2%.<br/>"
        "4. 灾难月 (如 202508) V7c 系统性 -4pp, zombie 过滤仅改善 +0.08pp — "
        "组合层面需配置防御股做对冲.<br/>"
        "5. <b>不构成投资建议</b>, 实操需结合个人风险偏好、资金面、政策面.",
        PSmall))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "—— V12.5 评分系统自动生成 · Tushare Pro 数据 · zombie filter 来自量化访谈共识",
        ParagraphStyle("foot", parent=PMuted, alignment=TA_CENTER)))

    doc.build(story)
    print(f"OK: {out}")
    print(f"size: {out.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    build()
