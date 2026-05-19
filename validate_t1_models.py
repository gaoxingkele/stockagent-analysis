"""Sprint 4.6 验证 - 三 T+1 模型 + v2 r20 对比.

对比维度:
A. 每模型在自己 label 上的 IC (基础质量)
B. EOD (15:00) bar 上的 IC (实际触发时段)
C. 47 日 EOD Top Decile 实际 r1_next_open 收益 (统一用 r1 算实战)
D. 不同时段 (10:30 / 11:30 / 13:30 / 14:30) Top Decile 表现差异

输出: output/cross_scale/t1_validation_report.md
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
PROD = ROOT / "output" / "production"
OUT = ROOT / "output" / "cross_scale"
OUT.mkdir(parents=True, exist_ok=True)

OOS_START = "20260301"

MODELS = [
    ("r20_1h_v2",      "r20_1h"),
    ("r1_next_open_v3", "r1_next_open"),
    ("r4_next_morn_v3", "r4_next_morn"),
    ("r8_next_day_v3",  "r8_next_day"),
]


def load_model(name):
    d = PROD / name
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta["feature_cols"]


def main():
    t0 = time.time()
    print("\n=== Sprint 4.6 T+1 模型对比 ===\n")
    df = pd.read_parquet(F3)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"v3 因子 {len(df):,} × {len(df.columns)}", flush=True)

    oos = df[df["trade_date"] >= OOS_START].copy()
    print(f"OOS ≥ {OOS_START}: {len(oos):,}", flush=True)

    # 每个模型推理
    rows_ic = []
    for name, label in MODELS:
        d = PROD / name
        if not (d / "classifier.txt").exists():
            print(f"  [{name}] 缺模型, 跳过"); continue
        b, feat_cols = load_model(name)
        # clip
        oos_c = oos.copy()
        for c in feat_cols:
            if c not in oos_c.columns: oos_c[c] = 0.0
            oos_c[c] = oos_c[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        X = oos_c[feat_cols].astype("float32")
        oos[f"pred_{name}"] = b.predict(X)
        del oos_c
        print(f"  [{name}] 推理完成", flush=True)

    # === A. 自标签 IC ===
    print("\n[A] 自标签 IC (全量 1H bar)")
    for name, label in MODELS:
        col = f"pred_{name}"
        if col not in oos.columns: continue
        sub = oos.dropna(subset=[col, label])
        sub = sub[sub[label].abs() <= 20]
        ic = stats.pearsonr(sub[col], sub[label])[0]
        rank = stats.spearmanr(sub[col], sub[label])[0]
        rows_ic.append({"model": name, "label": label, "mode": "all_bars",
                          "IC": ic, "RankIC": rank, "n": len(sub)})
        print(f"  {name:25s} {label:18s} IC={ic:+.4f}  RankIC={rank:+.4f}  n={len(sub):,}")

    # === B. EOD bar IC ===
    print("\n[B] EOD (15:00) bar IC")
    eod = oos[oos["trade_time"].dt.hour == 15].copy()
    for name, label in MODELS:
        col = f"pred_{name}"
        if col not in eod.columns: continue
        sub = eod.dropna(subset=[col, label])
        sub = sub[sub[label].abs() <= 20]
        ic = stats.pearsonr(sub[col], sub[label])[0]
        rank = stats.spearmanr(sub[col], sub[label])[0]
        rows_ic.append({"model": name, "label": label, "mode": "EOD_only",
                          "IC": ic, "RankIC": rank, "n": len(sub)})
        print(f"  {name:25s} {label:18s} IC={ic:+.4f}  RankIC={rank:+.4f}  n={len(sub):,}")

    # === C. EOD Top Decile 用 r1_next_open 统一实战 ===
    print("\n[C] EOD Top Decile r1_next_open 实际收益 (T+1 真实可执行 α)")
    rows_top = []
    for name, label in MODELS:
        col = f"pred_{name}"
        if col not in eod.columns: continue
        by_date = []
        for d_, g in eod.groupby("trade_date"):
            g = g.dropna(subset=[col, "r1_next_open"])
            g = g[g["r1_next_open"].abs() <= 20]
            if len(g) < 200: continue
            n10 = max(1, len(g) // 10)
            top = g.nlargest(n10, col)
            ret_top = top["r1_next_open"].mean()
            ret_mkt = g["r1_next_open"].mean()
            by_date.append({"date": d_, "n": len(g), "n_top": n10,
                              "ret_top": ret_top, "ret_mkt": ret_mkt,
                              "alpha": ret_top - ret_mkt})
        if by_date:
            bd = pd.DataFrame(by_date)
            row = {
                "model": name, "n_days": len(bd),
                "ret_top_mean": bd["ret_top"].mean(),
                "ret_mkt_mean": bd["ret_mkt"].mean(),
                "alpha_mean": bd["alpha"].mean(),
                "alpha_median": bd["alpha"].median(),
                "alpha_per_month": bd["alpha"].mean() * 20,  # ~20 交易日/月
            }
            rows_top.append(row)
            print(f"  {name:25s} 日数 {len(bd)}  Top={bd['ret_top'].mean():+.3f}%  "
                  f"市场={bd['ret_mkt'].mean():+.3f}%  α={bd['alpha'].mean():+.3f}pp  "
                  f"月化={row['alpha_per_month']:+.2f}pp")

    # === D. 不同时段 Top Decile 表现 (用最优模型) ===
    best_model = max(rows_top, key=lambda r: r["alpha_mean"])["model"] if rows_top else None
    print(f"\n[D] 不同时段 Top Decile (用 {best_model})")
    if best_model:
        col = f"pred_{best_model}"
        for hour in [10, 11, 13, 14, 15]:
            bar = oos[oos["trade_time"].dt.hour == hour].copy()
            by_date = []
            for d_, g in bar.groupby("trade_date"):
                g = g.dropna(subset=[col, "r1_next_open"])
                g = g[g["r1_next_open"].abs() <= 20]
                if len(g) < 200: continue
                n10 = max(1, len(g) // 10)
                top = g.nlargest(n10, col)
                ret_top = top["r1_next_open"].mean()
                ret_mkt = g["r1_next_open"].mean()
                by_date.append({"alpha": ret_top - ret_mkt})
            if by_date:
                bd = pd.DataFrame(by_date)
                print(f"  {hour:02d}:30 触发, n_days={len(bd)}, α 均值={bd['alpha'].mean():+.3f}pp  "
                      f"月化={bd['alpha'].mean()*20:+.2f}pp/月")

    # === 报告 ===
    md = [f"# Sprint 4.6 T+1 模型对比报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至今\n\n",
            "## A & B. IC 对比\n\n",
            "| Model | Label | Mode | IC | RankIC | n |\n|---|---|---|---|---|---|\n"]
    for r in rows_ic:
        md.append(f"| {r['model']} | {r['label']} | {r['mode']} | {r['IC']:+.4f} | {r['RankIC']:+.4f} | {r['n']:,} |\n")

    md.append("\n## C. EOD Top Decile r1_next_open 实战 α\n\n")
    md.append("| Model | n_days | Top% | Mkt% | α (pp) | α 月化 (pp/月) |\n|---|---|---|---|---|---|\n")
    for r in rows_top:
        md.append(f"| {r['model']} | {r['n_days']} | {r['ret_top_mean']:+.3f} | {r['ret_mkt_mean']:+.3f} | "
                   f"{r['alpha_mean']:+.3f} | {r['alpha_per_month']:+.2f} |\n")
    md.append("\n## 结论\n\n")
    if rows_top:
        best = max(rows_top, key=lambda r: r["alpha_mean"])
        md.append(f"- 最优模型: **{best['model']}** (月化 α = {best['alpha_per_month']:+.2f}pp)\n")
        md.append(f"- vs v2 r20: 看哪个表现更好\n")
    p = OUT / "t1_validation_report.md"
    p.write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {p}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
