# -*- coding: utf-8 -*-
"""V12.31 因子审计 → 一页 PDF (承重墙/可替代/冗余 + SHAP top)。"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/daily_pick/V12因子审计报告.pdf"

shap_rows = [
    ["1", "cyb_rel_strength 创业板相对强度", "2.80", "cross_sectional_rel"],
    ["2", "industry_id 行业", "1.41", "market_context"],
    ["3", "mkt_ret_20d 大盘20日动量", "0.92", "market_context"],
    ["4", "mkt_ret_60d", "0.73", "market_context"],
    ["5", "mkt_ret_5d", "0.26", "market_context"],
]
cls_rows = [
    ["🧱 承重墙", "market_context", "ΔIC CI[-0.154,-0.024]", "地基: 大盘择时/行业, 剔除重伤"],
    ["🧱 承重墙", "moneyflow", "CI[-0.026,-0.002]", "资金流, 不可替代"],
    ["🧱 边缘承重", "fundamental_chip", "CI[-0.025,+0.0003]", "几乎显著"],
    ["🔧 可替代", "cross_sectional_rel", "CI[-0.027,+0.008]", "SHAP第一但块可替代(轻钢龙骨)"],
    ["🔧 可替代", "pyramid/volatility/candle", "CI 含0", "共线, 剔除有补位"],
    ["🔧 可替代", "oscillator/breakout/volume", "CI 含0", "贡献但可被平替"],
    ["🗑 冗余倾向", "trend_ma", "Δ+0.0027(剔了略升)", "边际噪声"],
    ["🗑 冗余倾向", "valuation_size", "Δ+0.0005(剔了略升)", "论文同款: 估值=冗余"],
]


def table(ax, rows, headers, title, colw):
    ax.axis("off"); ax.set_title(title, fontsize=13, family="Microsoft YaHei", pad=10, loc="left")
    t = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="left", colWidths=colw)
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_text_props(family="Microsoft YaHei")
        if r == 0:
            cell.set_facecolor("#1772e6"); cell.set_text_props(color="white", family="Microsoft YaHei")


def main():
    with PdfPages(OUT) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.06, 0.95, "V12.31 因子审计报告", fontsize=18, family="Microsoft YaHei", weight="bold")
        fig.text(0.06, 0.915, "TreeSHAP 归因 × 消融诊断 (对标 QuantML A股 XGBoost 论文) · r20 选股模型 · 19月 embargo-WF",
                 fontsize=10, family="Microsoft YaHei", color="#555")
        ax1 = fig.add_axes([0.06, 0.62, 0.88, 0.22])
        table(ax1, shap_rows, ["#", "因子", "mean|SHAP|", "所属块"],
              "一、TreeSHAP 全局驱动 (pred_r20 = base 2.4171 + Σ SHAP)", [0.05, 0.45, 0.15, 0.35])
        ax2 = fig.add_axes([0.06, 0.10, 0.88, 0.42])
        table(ax2, cls_rows, ["类别", "块", "消融 ΔIC 95%CI", "解读"],
              "二、承重墙 / 可替代 / 冗余 三分类 (baseline IC=+0.1091, 无净负块)", [0.13, 0.24, 0.25, 0.38])
        fig.text(0.06, 0.065, "结论: r20 靠 行为/价量(大盘择时·资金流·相对强度), 不靠估值/基本面; 无可剪块(V12.31-clean不动).",
                 fontsize=9.5, family="Microsoft YaHei", color="#1772e6")
        fig.text(0.06, 0.04, "caveat: 仅r20选股层归因(非全book); 效应落21月噪声带; 我们embargo-WF口径(比论文无embargo严谨).",
                 fontsize=8.5, family="Microsoft YaHei", color="#999")
        pdf.savefig(fig); plt.close(fig)
    print(f"PDF: {OUT}")


if __name__ == "__main__":
    main()
