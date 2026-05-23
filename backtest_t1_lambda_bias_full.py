"""Phase 2 最终对比回测: 4 模型联测 (含 Phase 2 + 长 OOS).

模型:
  OLD          : r1_next_open_v3_long          (regression, 含 ST 偏见)
  NOST         : r1_next_open_v3_long_nost     (regression, 排 ST)
  LAMBDA       : r1_next_open_v3_long_lambda_nost (lambdarank, Phase 1)
  LAMBDA_BIAS  : r1_next_open_v3_long_lambda_bias_nost (lambdarank + 偏置, Phase 2) ⭐

输出: output/backtest_t1_lambda_bias/report.md
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
PROD = ROOT / "output" / "production"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT = ROOT / "output" / "backtest_t1_lambda_bias"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)

OOS_START = "20251001"
OOS_END = "20260331"
MODELS = {
    "OLD": "r1_next_open_v3_long",
    "NOST": "r1_next_open_v3_long_nost",
    "LAMBDA": "r1_next_open_v3_long_lambda_nost",
    "LAMBDA_BIAS": "r1_next_open_v3_long_lambda_bias_nost",
}
COST_BPS = 35.0 / 10000
R1_CAP = 3.0
DIST_THRESHOLD = 3.0
TOP_NS = [5, 10, 20, 50]
CATEGORICAL_DECILE = [
    "long_return_252d_decile", "long_return_504d_decile",
    "industry_return_504d_decile", "rs_in_decile",
]


def load_model(name):
    d = PROD / name
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta["feature_cols"]


def main():
    t0 = time.time()
    print(f"\n=== 4 模型对比回测 (OLD/NOST/LAMBDA/LAMBDA_BIAS) ===\n", flush=True)

    df = pd.read_parquet(F3)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)
    oos = df[(df["trade_date"] >= OOS_START) & (df["trade_date"] <= OOS_END) &
              (df["trade_time"].dt.hour == 15)].copy()
    del df
    oos = oos.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    print(f"  OOS EOD bar: {len(oos):,}, 日数 {oos['trade_date'].nunique()}", flush=True)

    # merge long_return features (for LAMBDA_BIAS 推理)
    if LONG_FEAT_P.exists():
        print(f"  merge long_return features ...", flush=True)
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        oos = oos.merge(lf, on=["ts_code", "trade_date"], how="left")
        # decile 转 int (与训练一致)
        for cd in CATEGORICAL_DECILE:
            if cd in oos.columns:
                oos[cd] = oos[cd].astype("Int8").astype("Int16").fillna(-1).astype("int16")

    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    oos = oos.merge(basic, on="ts_code", how="left")
    oos["is_st"] = oos["name"].fillna("").str.contains("ST", regex=False)

    # 4 模型推理
    for tag, name in MODELS.items():
        if not (PROD / name / "classifier.txt").exists():
            print(f"!! {name} 不存在, 跳过 {tag}", flush=True); continue
        b, fc = load_model(name)
        # 为缺失列填 0 (新模型可能用到长期收益因子,旧模型不会)
        for c in fc:
            if c not in oos.columns:
                oos[c] = 0.0
            if c not in CATEGORICAL_DECILE:
                oos[c] = oos[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        oos[f"pred_{tag}"] = b.predict(oos[fc].astype("float32"))
        print(f"  {tag}={name}: pred μ={oos[f'pred_{tag}'].mean():+.3f} σ={oos[f'pred_{tag}'].std():.3f}",
               flush=True)

    oos["is_near_upper"] = oos["dist_to_upper_limit_pct"].fillna(100) < DIST_THRESHOLD
    oos["is_at_lower"] = oos["dist_to_lower_limit_pct"].fillna(100) < 1.0

    valid = oos.dropna(subset=["r1_next_open"])
    valid = valid[valid["r1_next_open"].abs() <= 20].copy()
    valid["r1_capped"] = valid["r1_next_open"].clip(-R1_CAP, R1_CAP)

    # 8 配置 = 4 模型 × 2 ST
    results = []
    for tag in MODELS:
        pred_col = f"pred_{tag}"
        if pred_col not in valid.columns: continue
        for exclude_st in [False, True]:
            bad = valid["is_near_upper"] | valid["is_at_lower"]
            if exclude_st: bad = bad | valid["is_st"]
            for top_n in TOP_NS:
                curve = []
                for d_, g in valid.groupby("trade_date"):
                    gf = g[~bad.loc[g.index]]
                    if len(gf) < top_n: continue
                    top = gf.nlargest(top_n, pred_col)
                    net = top["r1_capped"].mean() - COST_BPS * 100
                    mkt = g["r1_capped"].mean()
                    curve.append({"date": d_, "ret_net_pct": net, "ret_mkt_pct": mkt,
                                    "alpha_pct": net - mkt})
                if not curve: continue
                cv = pd.DataFrame(curve).sort_values("date")
                cv["nav_net"] = (1 + cv["ret_net_pct"] / 100).cumprod()
                cv["nav_mkt"] = (1 + cv["ret_mkt_pct"] / 100).cumprod()
                cv["dd"] = cv["nav_net"] / cv["nav_net"].cummax() - 1
                st_tag = "ExclST" if exclude_st else "InclST"
                results.append({
                    "model": tag, "st_filter": st_tag, "top_n": top_n,
                    "n_days": len(cv),
                    "monthly_net_pct": cv["ret_net_pct"].mean() * 20,
                    "total_alpha_pct": (cv["nav_net"].iloc[-1] - cv["nav_mkt"].iloc[-1]) * 100,
                    "sharpe": cv["ret_net_pct"].mean() / (cv["ret_net_pct"].std() + 1e-9) * np.sqrt(252),
                    "mdd_pct": cv["dd"].min() * 100,
                    "win_rate_alpha": (cv["alpha_pct"] > 0).mean(),
                })
                cv.to_csv(OUT / "curves" / f"{tag}_{st_tag}_Top{top_n}.csv", index=False)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "results.csv", index=False)

    # 报告
    md = [f"# Phase 2 最终对比: 4 模型联测\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START}-{OOS_END} (~142 日)\n\n",
            "## 实盘配置 (排 ST, Top 5/10/20/50)\n\n",
            "| 模型 | Top5 | Top10 | Top20 | Top50 |\n",
            "|---|---|---|---|---|\n"]
    excl = res_df[res_df["st_filter"] == "ExclST"]
    for tag in MODELS:
        row = [f"| {tag}"]
        for top_n in TOP_NS:
            sub = excl[(excl["model"] == tag) & (excl["top_n"] == top_n)]
            if sub.empty: row.append("-"); continue
            r = sub.iloc[0]
            row.append(f"{r['monthly_net_pct']:+.2f}% / Sh{r['sharpe']:.1f}")
        md.append(" | ".join(row) + " |\n")

    md.append(f"\n## Phase 进展对比 (排 ST Top 10)\n\n")
    md.append("| Phase | 月化净 | Sharpe | MDD % | αwin |\n|---|---|---|---|---|\n")
    for tag in MODELS:
        sub = excl[(excl["model"] == tag) & (excl["top_n"] == 10)]
        if sub.empty: continue
        r = sub.iloc[0]
        md.append(f"| {tag} | {r['monthly_net_pct']:+.2f}% | {r['sharpe']:.2f} | "
                   f"{r['mdd_pct']:.1f} | {r['win_rate_alpha']*100:.0f}% |\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")

    print(f"\n--- 排 ST Top 10 ---")
    for tag in MODELS:
        sub = excl[(excl["model"] == tag) & (excl["top_n"] == 10)]
        if sub.empty: continue
        r = sub.iloc[0]
        print(f"  {tag:12s}: 月化={r['monthly_net_pct']:+.2f}% Sharpe={r['sharpe']:+.2f} "
               f"αwin={r['win_rate_alpha']*100:.0f}%")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
