"""深度版评分一致性验证 (用户 0519 提问).

对比:
1. 1H R20 (r20_1h_v2) vs 日线 R5 (r5_v17_all) - 理论同尺度 5 日 forward
2. 1H R20 (r20_1h_v2) vs 日线 R20 (r20_v16_all) - 1H 看 5日 vs 日线看 20日
3. 日线 R5 vs 日线 R20 - 同特征不同 label

输出: 评分相关性 + Top decile 重叠 + label 一致性 baseline
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PROD = ROOT / "output" / "production"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
OUT = ROOT / "output" / "cross_scale"
OUT.mkdir(parents=True, exist_ok=True)

OOS_START = "20260301"
OOS_END = "20260415"  # labels 极限 (r20 需 +20 个交易日)


def load_model(name):
    d = PROD / name
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta["feature_cols"]


def main():
    t0 = time.time()
    print("\n=== 深度版评分一致性验证 ===\n")

    # === 1. 加载日线 factor (train_v15_refresh.load_window) ===
    print("[1] 加载日线 factor + labels (OOS 期)...", flush=True)
    from train_v15_refresh import load_window
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    print(f"   daily: {len(daily):,} × {len(daily.columns)} (industry 列存在={'industry' in daily.columns})", flush=True)

    # 加 industry_id
    if "industry" in daily.columns:
        daily["industry_id"] = pd.Categorical(daily["industry"].fillna("unknown")).codes
    else:
        daily["industry_id"] = 0

    # === 2. 日线 r5/r20 模型推理 ===
    print("\n[2] 日线 r5_v17_all + r20_v16_all 推理...", flush=True)
    daily_preds = daily[["ts_code", "trade_date"]].copy()
    for name in ["r5_v17_all", "r20_v16_all"]:
        b, feat_cols = load_model(name)
        # 缺失列填 0
        miss = [c for c in feat_cols if c not in daily.columns]
        if miss:
            print(f"   [{name}] 缺 {len(miss)} 个特征, 填 0: {miss[:5]}...", flush=True)
            for c in miss: daily[c] = 0.0
        X = daily[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily_preds[f"pred_{name}"] = b.predict(X)
        print(f"   [{name}] pred 均值 {daily_preds[f'pred_{name}'].mean():+.3f} std {daily_preds[f'pred_{name}'].std():.3f}", flush=True)

    # === 3. 1H R20 (v2) 推理 - EOD bar only ===
    print("\n[3] 加载 1H factors_v3 + r20_1h_v2 EOD 推理...", flush=True)
    df1h = pd.read_parquet(F3, columns=None)
    df1h["trade_time"] = pd.to_datetime(df1h["trade_time"])
    df1h["trade_date"] = df1h["trade_date"].astype(str)
    eod = df1h[(df1h["trade_time"].dt.hour == 15) &
                 (df1h["trade_date"] >= OOS_START) &
                 (df1h["trade_date"] <= OOS_END)].copy()
    print(f"   EOD bar: {len(eod):,}", flush=True)
    del df1h

    b, feat_cols = load_model("r20_1h_v2")
    for c in feat_cols:
        if c not in eod.columns: eod[c] = 0.0
        eod[c] = eod[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    X = eod[feat_cols].astype("float32")
    eod["pred_1h_r20"] = b.predict(X)
    eod_preds = eod[["ts_code", "trade_date", "pred_1h_r20", "r20_1h"]].drop_duplicates(["ts_code","trade_date"])

    # === 4. 合并所有预测 + label ===
    print("\n[4] 合并预测 + 日线 label...", flush=True)
    # 日线 r5 / r20 真实 label (从 daily 的 r10/r20 列)
    label_cols = [c for c in ["r5", "r10", "r20"] if c in daily.columns]
    if not label_cols:
        # 尝试 r5_close 等
        label_cols = [c for c in ["r5_close", "r10_close", "r20_close", "r10", "r20"] if c in daily.columns]
    print(f"   日线 label 列: {label_cols}", flush=True)
    daily_labs = daily[["ts_code", "trade_date"] + label_cols].copy()

    merged = daily_preds.merge(eod_preds, on=["ts_code", "trade_date"], how="inner")
    merged = merged.merge(daily_labs, on=["ts_code", "trade_date"], how="left")
    print(f"   合并后: {len(merged):,} 行", flush=True)

    # === 5. 评分相关性分析 ===
    print("\n=== A. 评分两两相关 ===\n")
    pairs_score = [
        ("pred_1h_r20", "pred_r5_v17_all", "1H R20 vs 日线 R5"),
        ("pred_1h_r20", "pred_r20_v16_all", "1H R20 vs 日线 R20"),
        ("pred_r5_v17_all", "pred_r20_v16_all", "日线 R5 vs 日线 R20"),
    ]
    score_rows = []
    for a, b_, name in pairs_score:
        sub = merged.dropna(subset=[a, b_])
        if len(sub) < 100: continue
        pear = stats.pearsonr(sub[a], sub[b_])[0]
        sper = stats.spearmanr(sub[a], sub[b_])[0]
        # Top decile 重叠
        n10 = max(1, len(sub) // 10)
        top_a = set(sub.nlargest(n10, a).index)
        top_b = set(sub.nlargest(n10, b_).index)
        overlap = len(top_a & top_b) / n10
        score_rows.append({"pair": name, "pearson": pear, "spearman": sper,
                              "n": len(sub), "top10_overlap": overlap})
        print(f"  {name:30s} Pearson={pear:+.4f}  Spearman={sper:+.4f}  "
              f"Top10%重叠={overlap*100:.1f}%  n={len(sub):,}")

    # === 6. label 一致性 baseline ===
    print("\n=== B. Label 一致性 baseline (真实 r5 vs r20) ===\n")
    if "r5_close" in merged.columns and "r20_close" in merged.columns:
        sub = merged.dropna(subset=["r5_close", "r20_close"])
        sub = sub[(sub["r5_close"].abs() < 50) & (sub["r20_close"].abs() < 50)]
        pear = stats.pearsonr(sub["r5_close"], sub["r20_close"])[0]
        sper = stats.spearmanr(sub["r5_close"], sub["r20_close"])[0]
        print(f"  真实 r5 vs r20:  Pearson={pear:+.4f}  Spearman={sper:+.4f}  n={len(sub):,}")
        baseline_r5_r20 = (pear, sper)
    else:
        baseline_r5_r20 = None
        print("  缺日线 label, 跳过")

    # 1H r20_1h vs 日线 r20
    if "r20_1h" in merged.columns and "r20" in merged.columns:
        sub = merged.dropna(subset=["r20_1h", "r20"])
        sub = sub[(sub["r20_1h"].abs() < 50) & (sub["r20"].abs() < 50)]
        if len(sub) > 100:
            pear = stats.pearsonr(sub["r20_1h"], sub["r20"])[0]
            sper = stats.spearmanr(sub["r20_1h"], sub["r20"])[0]
            print(f"  1H r20_1h vs 日线 r20:  Pearson={pear:+.4f}  Spearman={sper:+.4f}  n={len(sub):,}")

    # === 7. 按日子统计 (取均值, 看稳定性) ===
    print("\n=== C. 按日子统计 (50 个 OOS 日) ===\n")
    by_day = []
    for d_, g in merged.groupby("trade_date"):
        if len(g) < 200: continue
        g = g.dropna(subset=["pred_1h_r20", "pred_r5_v17_all", "pred_r20_v16_all"])
        if len(g) < 200: continue
        r1 = stats.spearmanr(g["pred_1h_r20"], g["pred_r5_v17_all"])[0]
        r2 = stats.spearmanr(g["pred_1h_r20"], g["pred_r20_v16_all"])[0]
        r3 = stats.spearmanr(g["pred_r5_v17_all"], g["pred_r20_v16_all"])[0]
        by_day.append({"date": d_, "n": len(g),
                          "1h_r20_vs_d_r5": r1, "1h_r20_vs_d_r20": r2,
                          "d_r5_vs_d_r20": r3})
    bd_df = pd.DataFrame(by_day)
    if len(bd_df):
        print(f"  评测日数: {len(bd_df)}")
        for col in ["1h_r20_vs_d_r5", "1h_r20_vs_d_r20", "d_r5_vs_d_r20"]:
            print(f"  {col}: 均值 {bd_df[col].mean():+.4f}, 中位 {bd_df[col].median():+.4f}, "
                  f"std {bd_df[col].std():.4f}, "
                  f"min {bd_df[col].min():+.4f}, max {bd_df[col].max():+.4f}")
        bd_df.to_csv(OUT / "score_consistency_by_day.csv", index=False)

    # === 8. 报告 ===
    md = [f"# 深度版评分一致性报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至 {OOS_END}\n\n",
            "## A. 评分两两相关\n\n",
            "| 对比 | Pearson | Spearman | Top10% 重叠 | n |\n|---|---|---|---|---|\n"]
    for r in score_rows:
        md.append(f"| {r['pair']} | {r['pearson']:+.4f} | {r['spearman']:+.4f} | "
                   f"{r['top10_overlap']*100:.1f}% | {r['n']:,} |\n")

    if baseline_r5_r20:
        md.append(f"\n## B. 真实 Label baseline\n\n")
        md.append(f"- 真实 r5 vs r20: Pearson={baseline_r5_r20[0]:+.4f}, Spearman={baseline_r5_r20[1]:+.4f}\n")

    if len(bd_df):
        md.append(f"\n## C. 按日子统计 (n={len(bd_df)} 日)\n\n")
        md.append("| Pair | 均值 | 中位 | std | min | max |\n|---|---|---|---|---|---|\n")
        for col in ["1h_r20_vs_d_r5", "1h_r20_vs_d_r20", "d_r5_vs_d_r20"]:
            md.append(f"| {col} | {bd_df[col].mean():+.4f} | {bd_df[col].median():+.4f} | "
                       f"{bd_df[col].std():.4f} | {bd_df[col].min():+.4f} | {bd_df[col].max():+.4f} |\n")

    md.append("\n## 解读\n\n")
    if score_rows:
        r1 = score_rows[0]["spearman"] if score_rows else 0
        r3 = score_rows[2]["spearman"] if len(score_rows) >= 3 else 0
        md.append(f"- **1H R20 vs 日线 R5** (同 5 日 forward 但不同尺度): Spearman={r1:+.3f}\n")
        md.append(f"- **日线 R5 vs 日线 R20** (同特征不同 label): Spearman={r3:+.3f}\n")
        if abs(r1) < 0.3:
            md.append(f"- 1H R20 vs 日线 R5 低相关 (<0.3) → **跨尺度独立 alpha 源**, 共识可作金矿\n")
        if abs(r3) > 0.5:
            md.append(f"- 日线 R5 vs 日线 R20 高相关 (>0.5) → 日线模型本质共享同一组动量特征\n")

    p = OUT / "score_consistency_report.md"
    p.write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {p}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
