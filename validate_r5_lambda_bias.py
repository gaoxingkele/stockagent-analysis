"""验证 r5_v17_long_lambda_bias_nost vs r5_v17_long_nost 在 R5 反向过滤中的效果.

V12 双轨架构核心 alpha 源:
  P1 R5 反向 d5_bot30: 长 OOS Sharpe 2.67, 月化 +2.43%

如果新 lambdarank+bias 模型让 R5 反向 Sharpe 提升, 双轨架构 alpha 同步提升.

输出: output/cross_scale/r5_lambda_bias_compare.md
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

OUT = ROOT / "output" / "cross_scale"
PROD = ROOT / "output" / "production"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"

# 3 路对比
R5_MODELS = {
    "OLD": "r5_v17_long",                       # 旧, 含 ST
    "NOST": "r5_v17_long_nost",                 # 排 ST, regression
    "LAMBDA_BIAS": "r5_v17_long_lambda_bias_nost",  # 排 ST, lambdarank + bias
}
R20_MODEL = "r20_v16_long_nost"   # anchor 用统一 r20 模型


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"]


def main():
    t0 = time.time()
    print(f"\n=== R5 lambdarank+bias 对比 (3 路 × d5_bot30) ===\n", flush=True)

    # 加载日线 + ST 已源头排除 (load_window)
    print("[1] 加载日线数据 + merge long_return ...", flush=True)
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)

    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")
        new_cols = [c for c in lf.columns if c not in ("ts_code", "trade_date")]
        for c in new_cols:
            if c.endswith("_decile"):
                daily[c] = pd.to_numeric(daily[c], errors="coerce").astype("float32")

    # 推理 r20_v16_long_nost (anchor)
    print("[2] r20 anchor 推理 ...", flush=True)
    b20, fc20 = load_model(R20_MODEL)
    industry_map = json.loads((PROD / R20_MODEL / "feature_meta.json").read_text(encoding="utf-8")
                                ).get("industry_map", {})
    if "industry" in daily.columns:
        daily["industry_id"] = daily["industry"].fillna("unknown").map(industry_map).fillna(-1).astype(int)
    miss = [c for c in fc20 if c not in daily.columns]
    for c in miss: daily[c] = 0.0
    X20 = daily[fc20].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    daily["pred_r20_anchor"] = b20.predict(X20)

    # 推理 3 个 R5 模型
    for tag, name in R5_MODELS.items():
        if not (PROD / name / "classifier.txt").exists():
            print(f"!! {name} 不存在, 跳过 {tag}", flush=True); continue
        print(f"[3] R5 模型 {tag}={name} ...", flush=True)
        b, fc = load_model(name)
        # 重算 industry_id (可能 r5 用的 map 跟 r20 不同)
        im = json.loads((PROD / name / "feature_meta.json").read_text(encoding="utf-8")
                          ).get("industry_map", {})
        if im and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(im).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily[f"pred_r5_{tag}"] = b.predict(X)

    # 拉 1H EOD 取 dist 过滤
    print("[4] 加载 1H factors_v3 EOD dist ...", flush=True)
    df1h = pd.read_parquet(F3, columns=["ts_code", "trade_date", "trade_time",
                                            "dist_to_upper_limit_pct", "dist_to_lower_limit_pct"])
    df1h["trade_time"] = pd.to_datetime(df1h["trade_time"])
    df1h["trade_date"] = df1h["trade_date"].astype(str)
    eod = df1h[(df1h["trade_time"].dt.hour == 15) &
                 (df1h["trade_date"] >= OOS_START) &
                 (df1h["trade_date"] <= OOS_END)][
        ["ts_code", "trade_date", "dist_to_upper_limit_pct", "dist_to_lower_limit_pct"]
    ].drop_duplicates(["ts_code", "trade_date"])
    del df1h

    m = daily.merge(eod, on=["ts_code", "trade_date"], how="inner")
    m["bad"] = (m["dist_to_upper_limit_pct"].fillna(100) < 2.0) | \
                (m["dist_to_lower_limit_pct"].fillna(100) < 1.0)
    m["r20_cap"] = m["r20"].clip(-30, 30)
    print(f"  merged: {len(m):,}, 日数 {m['trade_date'].nunique()}", flush=True)

    # 实验: r20 top 20% anchor 池内, 按 r5_<TAG> 拆 d5_bot30
    summary = []
    for tag in R5_MODELS:
        pred_col = f"pred_r5_{tag}"
        if pred_col not in m.columns: continue
        rows = []
        for d_, g in m.groupby("trade_date"):
            g = g.dropna(subset=["pred_r20_anchor", pred_col, "r20_cap"])
            gf = g[~g["bad"]]
            if len(gf) < 500: continue
            n_top = max(1, int(len(gf) * 0.20))
            anchor = gf.nlargest(n_top, "pred_r20_anchor").copy()
            anchor["d5_rank"] = anchor[pred_col].rank(pct=True, method="first")
            sub = anchor[anchor["d5_rank"] < 0.30]
            if len(sub):
                rows.append({"date": d_, "n": len(sub), "r20": sub["r20_cap"].mean()})
        df_r = pd.DataFrame(rows)
        if df_r.empty: continue
        sharpe = df_r["r20"].mean() / (df_r["r20"].std() + 1e-9) * np.sqrt(50)
        summary.append({
            "tag": tag, "model": R5_MODELS[tag],
            "n_days": len(df_r), "pool_avg": df_r["n"].mean(),
            "r20_mean_pct": df_r["r20"].mean(),
            "r20_std": df_r["r20"].std(),
            "sharpe": sharpe,
        })

    res = pd.DataFrame(summary)
    res.to_csv(OUT / "r5_lambda_bias_compare.csv", index=False)

    # 报告
    md = [f"# R5 LambdaRank+Bias vs Regression 对比 (P1 R5 反向 d5_bot30)\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START}-{OOS_END}\n",
            f"Anchor: {R20_MODEL} top 20% 池\n",
            f"切分: d5_bot30 (R5 评分最低 30%, 短期超跌)\n\n",
            "| 模型 | 日数 | 池均股数 | r20 均 % | std | Sharpe |\n|---|---|---|---|---|---|\n"]
    for _, r in res.iterrows():
        md.append(f"| **{r['tag']}** ({r['model']}) | {int(r['n_days'])} | "
                   f"{r['pool_avg']:.1f} | {r['r20_mean_pct']:+.3f} | "
                   f"{r['r20_std']:.2f} | **{r['sharpe']:.2f}** |\n")

    Path(OUT / "r5_lambda_bias_compare.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'r5_lambda_bias_compare.md'}")

    for _, r in res.iterrows():
        print(f"  {r['tag']:12s}: n={int(r['n_days']):3d} 池={r['pool_avg']:5.1f} "
               f"r20={r['r20_mean_pct']:+.3f}% std={r['r20_std']:5.2f} Sharpe={r['sharpe']:5.2f}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
