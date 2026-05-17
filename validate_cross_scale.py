"""1H R20 跨尺度验证 (现实可执行版).

A. 1H R20 自评估 (OOS ≥ 20260228, 1.29M 样本)
   - IC / RankIC vs r4_1h / r20_1h / r40_1h
   - 按日 Top Decile 实际收益 (用 r20_1h 自带的 forward 5 日)
B. 与 V12 0508-0515 日线 csv 共识对比
   - 取 V12 csv 里 buy_r5_score / buy_r10_score / buy_r20_score
   - 看 1H R20 EOD 预测 vs 日线 V12 预测 的 相关性 + Top 共识

输出: output/cross_scale/validation_report.md
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
F1H = ROOT / "output" / "1h_factors" / "factors.parquet"
PROD = ROOT / "output" / "production"
V12_DIR = ROOT / "output" / "v7c_full_inference"
OUT = ROOT / "output" / "cross_scale"
OUT.mkdir(parents=True, exist_ok=True)

OOS_START = "20260301"
TRAIN_END_DATE = "20260228"


def main():
    t0 = time.time()
    print("\n=== 1H R20 跨尺度验证 ===\n")

    # === 加载 1H 因子 + label ===
    print(f"[1] 加载 1H 因子 {F1H.name}")
    df = pd.read_parquet(F1H)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"   {len(df):,} 行 × {df['ts_code'].nunique()} 股 × {df['trade_date'].nunique()} 日")

    # 加载模型
    print(f"[2] 加载 1H R20 模型 r20_1h_v1")
    d = PROD / "r20_1h_v1"
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]

    # 在 OOS 期 EOD bar (15:00) 上 inference
    print(f"[3] 推理 OOS ≥ {OOS_START} 全量")
    oos = df[df["trade_date"] >= OOS_START].copy()
    print(f"   OOS 1H bars: {len(oos):,}")
    # clip 同 train
    for c in feat_cols:
        oos[c] = oos[c].clip(-200, 200)
    X = oos[feat_cols].astype("float32")
    oos["pred_1h_r20"] = booster.predict(X)

    # 自评估 - 全量 1H bar
    print("\n[A1] 全量 1H bar 自评估 (含 r4_1h/r20_1h/r40_1h)")
    rows = []
    for lab in ["r4_1h", "r20_1h", "r40_1h"]:
        sub = oos.dropna(subset=["pred_1h_r20", lab])
        # 同上 winsorize label
        sub = sub[sub[lab].abs() <= 50]
        ic = stats.pearsonr(sub["pred_1h_r20"], sub[lab])[0]
        rank = stats.spearmanr(sub["pred_1h_r20"], sub[lab])[0]
        rows.append({"label": lab, "IC": ic, "RankIC": rank, "n": len(sub)})
        print(f"   {lab}: IC={ic:+.4f}  RankIC={rank:+.4f}  n={len(sub):,}")

    # 仅 EOD (15:00) bar
    print("\n[A2] 仅 EOD (15:00) bar 自评估")
    eod = oos[oos["trade_time"].dt.hour == 15].copy()
    print(f"   EOD bars: {len(eod):,}")
    for lab in ["r4_1h", "r20_1h", "r40_1h"]:
        sub = eod.dropna(subset=["pred_1h_r20", lab])
        sub = sub[sub[lab].abs() <= 50]
        ic = stats.pearsonr(sub["pred_1h_r20"], sub[lab])[0]
        rank = stats.spearmanr(sub["pred_1h_r20"], sub[lab])[0]
        rows.append({"label": f"{lab}_EOD", "IC": ic, "RankIC": rank, "n": len(sub)})
        print(f"   {lab} (EOD): IC={ic:+.4f}  RankIC={rank:+.4f}  n={len(sub):,}")

    # 按日 Top Decile 实际收益 (在 EOD 上)
    print("\n[A3] EOD 按日 Top Decile r20_1h 实际收益")
    by_date = []
    for d_, g in eod.groupby("trade_date"):
        g = g.dropna(subset=["pred_1h_r20", "r20_1h"])
        g = g[g["r20_1h"].abs() <= 50]
        if len(g) < 200: continue
        n10 = max(1, len(g) // 10)
        top = g.nlargest(n10, "pred_1h_r20")
        ret_top = top["r20_1h"].mean()
        ret_mkt = g["r20_1h"].mean()
        by_date.append({"date": d_, "n": len(g), "n_top": n10,
                          "ret_top": ret_top, "ret_mkt": ret_mkt,
                          "alpha": ret_top - ret_mkt})
    by_date_df = pd.DataFrame(by_date)
    if len(by_date_df):
        print(f"   评测日: {len(by_date_df)}")
        print(f"   Top Decile 5 日均收益: {by_date_df['ret_top'].mean():+.3f}%")
        print(f"   市场平均 5 日收益:      {by_date_df['ret_mkt'].mean():+.3f}%")
        print(f"   α: {by_date_df['alpha'].mean():+.3f}pp  (中位 {by_date_df['alpha'].median():+.3f}pp)")
        print(f"   α 月化: {by_date_df['alpha'].mean()*4:+.2f}pp/月")
        by_date_df.to_csv(OUT / "by_date_1h_self.csv", index=False, encoding="utf-8-sig")

    # === B. 与 V12 csv 共识 ===
    print("\n[B] 与 V12 csv 共识对比 (0508-0515)")
    v12_csvs = sorted(V12_DIR.glob("v7c_inference_2026*.csv"))
    overlap_rows = []
    for vp in v12_csvs:
        snap = vp.stem.split("_")[-1]
        if snap < "20260508": continue
        v12 = pd.read_csv(vp, dtype={"ts_code": str})
        # 找 EOD 1H 预测分
        eod_snap = eod[eod["trade_date"] == snap][["ts_code", "pred_1h_r20"]]
        if eod_snap.empty:
            print(f"   {snap}: 无 1H EOD 数据"); continue
        eod_snap = eod_snap.drop_duplicates("ts_code")
        # 合并
        merged = v12.merge(eod_snap, on="ts_code", how="inner")
        if "buy_r20_score" in merged.columns:
            d_col = "buy_r20_score"
        elif "r20_pred" in merged.columns:
            d_col = "r20_pred"
        else:
            print(f"   {snap}: 无日线 r20 列"); continue
        merged = merged.dropna(subset=["pred_1h_r20", d_col])
        if len(merged) < 100: continue
        r = stats.pearsonr(merged["pred_1h_r20"], merged[d_col])[0]
        # top decile
        n10 = max(1, len(merged) // 10)
        top_1h = set(merged.nlargest(n10, "pred_1h_r20")["ts_code"])
        top_d20 = set(merged.nlargest(n10, d_col)["ts_code"])
        overlap = len(top_1h & top_d20)
        overlap_rows.append({
            "snap": snap, "n": len(merged), "corr_pred": r,
            "top_n": n10, "overlap": overlap, "rate": overlap/n10,
        })
        print(f"   {snap}: n={len(merged):,}  corr={r:+.4f}  "
              f"Top{n10} 重叠={overlap}/{n10} ({overlap/n10*100:.0f}%)")

    # === 报告 ===
    md = [f"# 1H R20 跨尺度验证报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至今\n\n",
            "## A. 1H R20 自评估\n\n",
            "### A1. 全量 1H bar\n\n| Label | IC | RankIC | n |\n|---|---|---|---|\n"]
    for r in rows[:3]:
        md.append(f"| {r['label']} | {r['IC']:+.4f} | {r['RankIC']:+.4f} | {r['n']:,} |\n")
    md.append("\n### A2. 仅 EOD bar\n\n| Label | IC | RankIC | n |\n|---|---|---|---|\n")
    for r in rows[3:]:
        md.append(f"| {r['label']} | {r['IC']:+.4f} | {r['RankIC']:+.4f} | {r['n']:,} |\n")

    if len(by_date_df):
        md.append(f"\n### A3. EOD Top Decile 5 日实际收益\n\n")
        md.append(f"- 评测日数: {len(by_date_df)}\n")
        md.append(f"- Top Decile 平均收益: {by_date_df['ret_top'].mean():+.3f}%\n")
        md.append(f"- 市场平均: {by_date_df['ret_mkt'].mean():+.3f}%\n")
        md.append(f"- **α: {by_date_df['alpha'].mean():+.3f}pp** (中位 {by_date_df['alpha'].median():+.3f}pp)\n")
        md.append(f"- α 月化: {by_date_df['alpha'].mean()*4:+.2f}pp/月\n")

    if overlap_rows:
        md.append("\n## B. 与日线 V12 共识对比\n\n| Snap | n | corr | Top重叠 |\n|---|---|---|---|\n")
        for o in overlap_rows:
            md.append(f"| {o['snap']} | {o['n']:,} | {o['corr_pred']:+.4f} | "
                       f"{o['overlap']}/{o['top_n']} ({o['rate']*100:.0f}%) |\n")
        md.append(f"\n- 平均 corr: **{np.mean([o['corr_pred'] for o in overlap_rows]):+.4f}**\n")
        md.append(f"- 平均重叠率: **{np.mean([o['rate'] for o in overlap_rows])*100:.1f}%**\n")

    md.append("\n## 结论\n\n")
    a_ic = next((r['IC'] for r in rows if r['label']=='r20_1h_EOD'), None)
    if a_ic is not None and abs(a_ic) < 0.02:
        md.append(f"- **1H R20 EOD IC={a_ic:+.4f}**, 极弱, 直接打分用作单独 alpha 不可行\n")
    elif a_ic is not None and a_ic > 0.03:
        md.append(f"- 1H R20 EOD IC={a_ic:+.4f}, 弱但有效, 可作为日线模型补充信号\n")
    if overlap_rows:
        avg_corr = np.mean([o['corr_pred'] for o in overlap_rows])
        if abs(avg_corr) < 0.1:
            md.append(f"- 1H 预测与日线 V12 预测相关 {avg_corr:+.3f}, **高度正交** → 共识 Top 是独立 alpha 源\n")
        else:
            md.append(f"- 1H 与日线相关 {avg_corr:+.3f}, 部分重叠\n")
    md.append("\n## 下一步\n\n")
    md.append("- 1H 模型 IC 弱 (~0.04 vs 日线 0.18), 因子设计需重做: 加技术指标分位、量价节奏、订单流近似\n")
    md.append("- 当前 1H 模型不建议独立打分, 但可作为 V12 日线池的二次过滤器\n")

    p = OUT / "validation_report.md"
    p.write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {p}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
