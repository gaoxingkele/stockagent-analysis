# -*- coding: utf-8 -*-
"""好基金优选池 V12.31 评分 → PDF (多页表格)。"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
DATE = "20260618"
CSV = ROOT / f"output/daily_pick/fund_pool_score_{DATE}.csv"
OUT = ROOT / f"output/daily_pick/好基金优选池_V12评分_{DATE}.pdf"
ROWS_PER_PAGE = 26


def main():
    df = pd.read_parquet(CSV) if CSV.suffix == ".parquet" else pd.read_csv(CSV)
    df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
    df = df.sort_values("ratio", ascending=False).reset_index(drop=True)
    df.insert(0, "#", range(1, len(df) + 1))

    def row(p):
        return [p["#"], p["ts_code"], str(p.get("name", ""))[:8], str(p.get("industry", ""))[:6],
                int(p.get("n_funds", 0)), f"{p.get('buy_r20_score', 0):.0f}",
                f"{p['pump_score']:.3f}", f"{p['pump_down_score']:.3f}",
                f"{p['ratio']:.1f}x", f"{p.get('past_r5', 0)*100:+.0f}%",
                "★" if p.get("v7c_recommend") else ""]
    headers = ["#", "代码", "名称", "行业", "持有基金", "r20", "pump↑", "pump↓", "ratio", "past5", "主推"]

    nrec = int(df["v7c_recommend"].sum())
    with PdfPages(OUT) as pdf:
        # 封面/摘要页
        fig = plt.figure(figsize=(11.7, 8.3)); fig.clf()
        txt = (f"好基金优选池 · V12.31 评分排序\n\n"
               f"日期: {DATE}    池: top68 高收益基金 2026Q1 重仓中被 ≥2 只持有的股 ({len(df)} 只)\n"
               f"排序: ratio = pump↑ / (pump↓ + 0.01) 降序\n\n"
               f"摘要:\n"
               f"  · 池子全为 AI/算力/半导体/光通信/元器件 主题 (好基金抱团主战场)\n"
               f"  · 入 V12.31 V7c 主推 ★: {nrec}/{len(df)} (这批是机构追高强势龙头, 与V12.31抄底回调DNA相反)\n"
               f"  · 机构确信度看'持有基金'列: 中际旭创64/新易盛63/东山精密38/源杰科技37/天孚通信34 为核心抱团\n"
               f"  · ratio 高=强势确认(非买入信号, 同步不领先); '高抱团+低ratio回调中'(天孚/新易盛)=回调观察点\n\n"
               f"注: ratio 是强度温度计非预测信号; 数据 Tushare; 生产线 V12.31 只读未动。")
        fig.text(0.07, 0.95, txt, va="top", ha="left", fontsize=13, family="Microsoft YaHei")
        pdf.savefig(fig); plt.close(fig)

        # 表格页
        for start in range(0, len(df), ROWS_PER_PAGE):
            chunk = df.iloc[start:start + ROWS_PER_PAGE]
            fig, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis("off")
            ax.set_title(f"好基金优选池 V12.31 评分 {DATE}  (第 {start//ROWS_PER_PAGE+1} 页)",
                         fontsize=13, family="Microsoft YaHei", pad=12)
            cells = [row(p) for _, p in chunk.iterrows()]
            t = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
            t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.45)
            for (r_, c_), cell in t.get_celld().items():
                cell.set_text_props(family="Microsoft YaHei")
                if r_ == 0:
                    cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", family="Microsoft YaHei")
                elif cells[r_-1][-1] == "★":
                    cell.set_facecolor("#fdebd0")
            pdf.savefig(fig); plt.close(fig)
    print(f"PDF: {OUT}")
    return str(OUT)


if __name__ == "__main__":
    print(main())
