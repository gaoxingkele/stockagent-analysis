"""Phase 1 关键验证: r1_lambdarank vs r1_nost (regression) 长 OOS 回测.

3 路对比:
  - OLD (regression, 含 ST 偏见)
  - NEW_nost (regression, 排 ST)
  - LAMBDA (lambdarank, 排 ST + Top N 直接优化)  ⭐ Phase 1 主测

输出: output/backtest_t1_lambda/report.md
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
PROD = ROOT / "output" / "production"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT = ROOT / "output" / "backtest_t1_lambda"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)

OOS_START = "20251001"
OOS_END = "20260331"
MODELS = {
    "OLD": "r1_next_open_v3_long",
    "NOST": "r1_next_open_v3_long_nost",
    "LAMBDA": "r1_next_open_v3_long_lambda_nost",
}
COST_BPS = 35.0 / 10000
R1_CAP = 3.0
DIST_THRESHOLD = 3.0
TOP_NS = [5, 10, 20, 50]


def load_model(name):
    d = PROD / name
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta["feature_cols"]


def main():
    t0 = time.time()
    print(f"\n=== r1 lambdarank vs regression (3 路对比) ===\n", flush=True)

    df = pd.read_parquet(F3)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)
    oos = df[(df["trade_date"] >= OOS_START) & (df["trade_date"] <= OOS_END) &
              (df["trade_time"].dt.hour == 15)].copy()
    del df
    oos = oos.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    print(f"  OOS EOD bar: {len(oos):,}, 日数 {oos['trade_date'].nunique()}", flush=True)

    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    oos = oos.merge(basic, on="ts_code", how="left")
    oos["is_st"] = oos["name"].fillna("").str.contains("ST", regex=False)

    # 三模型推理
    for tag, name in MODELS.items():
        if not (PROD / name / "classifier.txt").exists():
            print(f"!! {name} 不存在, 跳过 {tag}", flush=True); continue
        b, fc = load_model(name)
        for c in fc:
            if c not in oos.columns: oos[c] = 0.0
            oos[c] = oos[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        oos[f"pred_{tag}"] = b.predict(oos[fc].astype("float32"))
        print(f"  {tag}={name}: pred μ={oos[f'pred_{tag}'].mean():+.3f} σ={oos[f'pred_{tag}'].std():.3f}",
               flush=True)

    oos["is_near_upper"] = oos["dist_to_upper_limit_pct"].fillna(100) < DIST_THRESHOLD
    oos["is_at_lower"] = oos["dist_to_lower_limit_pct"].fillna(100) < 1.0

    valid = oos.dropna(subset=["r1_next_open"])
    valid = valid[valid["r1_next_open"].abs() <= 20].copy()
    valid["r1_capped"] = valid["r1_next_open"].clip(-R1_CAP, R1_CAP)

    # 6 配置 = 3 模型 × 2 (含/排 ST)
    results = []
    for tag in MODELS:
        pred_col = f"pred_{tag}"
        if pred_col not in valid.columns: continue
        for exclude_st in [False, True]:
            bad = valid["is_near_upper"] | valid["is_at_lower"]
            if exclude_st:
                bad = bad | valid["is_st"]
            for top_n in TOP_NS:
                curve = []
                for d_, g in valid.groupby("trade_date"):
                    gf = g[~bad.loc[g.index]]
                    if len(gf) < top_n: continue
                    top = gf.nlargest(top_n, pred_col)
                    gross = top["r1_capped"].mean()
                    net = gross - COST_BPS * 100
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
                    "total_net_pct": (cv["nav_net"].iloc[-1] - 1) * 100,
                    "total_mkt_pct": (cv["nav_mkt"].iloc[-1] - 1) * 100,
                    "monthly_net_pct": cv["ret_net_pct"].mean() * 20,
                    "sharpe": cv["ret_net_pct"].mean() / (cv["ret_net_pct"].std() + 1e-9) * np.sqrt(252),
                    "mdd_pct": cv["dd"].min() * 100,
                    "win_rate_alpha": (cv["alpha_pct"] > 0).mean(),
                })
                cv.to_csv(OUT / "curves" / f"{tag}_{st_tag}_Top{top_n}.csv", index=False)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "results.csv", index=False)

    # 报告
    md = [f"# r1 LambdaRank vs Regression 长 OOS 对比\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START}-{OOS_END} (~142 日)\n",
            f"约束: cap±{R1_CAP}%, dist<{DIST_THRESHOLD}%, cost 0.35%\n\n"]

    md.append("## 实盘配置 (排 ST)\n\n")
    md.append("| 模型 | TopN | 日数 | 月化净 % | α 累计 % | Sharpe | MDD % | αwin |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    excl = res_df[res_df["st_filter"] == "ExclST"]
    for _, r in excl.iterrows():
        md.append(f"| {r['model']} | {int(r['top_n'])} | {int(r['n_days'])} | "
                   f"{r['monthly_net_pct']:+.2f} | "
                   f"{r['total_net_pct'] - r['total_mkt_pct']:+.1f} | "
                   f"{r['sharpe']:.2f} | {r['mdd_pct']:.1f} | "
                   f"{r['win_rate_alpha']*100:.0f}% |\n")

    md.append(f"\n## 关键对比 (排 ST, 三模型对照)\n\n")
    md.append("| TopN | OLD | NOST | LAMBDA | LAMBDA vs OLD | LAMBDA vs NOST |\n")
    md.append("|---|---|---|---|---|---|\n")
    for top_n in TOP_NS:
        rows = {r["model"]: r for _, r in excl[excl["top_n"] == top_n].iterrows()}
        if "LAMBDA" not in rows: continue
        l, o, n = rows.get("LAMBDA"), rows.get("OLD"), rows.get("NOST")
        if l is None: continue
        delta_o = (l["monthly_net_pct"] - o["monthly_net_pct"]) if o is not None else None
        delta_n = (l["monthly_net_pct"] - n["monthly_net_pct"]) if n is not None else None
        do_s = f"{delta_o:+.2f}pp" if delta_o is not None else "-"
        dn_s = f"{delta_n:+.2f}pp" if delta_n is not None else "-"
        md.append(f"| {top_n} | "
                   f"{o['monthly_net_pct']:+.2f}%/Sh{o['sharpe']:.1f} | "
                   f"{n['monthly_net_pct']:+.2f}%/Sh{n['sharpe']:.1f} | "
                   f"**{l['monthly_net_pct']:+.2f}%**/Sh{l['sharpe']:.1f} | "
                   f"{do_s} | {dn_s} |\n")

    md.append("\n## 含 ST 配置 (对照)\n\n")
    md.append("| 模型 | TopN | 月化净 % | Sharpe |\n|---|---|---|---|\n")
    incl = res_df[res_df["st_filter"] == "InclST"]
    for _, r in incl.iterrows():
        md.append(f"| {r['model']} | {int(r['top_n'])} | {r['monthly_net_pct']:+.2f} | "
                   f"{r['sharpe']:.2f} |\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")

    # 终端打印 LAMBDA 排 ST
    print(f"\n--- LAMBDA (lambdarank), 排 ST (Phase 1 主测) ---")
    for _, r in excl[excl["model"] == "LAMBDA"].iterrows():
        print(f"  Top{int(r['top_n']):3d}: 月化={r['monthly_net_pct']:+.2f}% "
               f"Sharpe={r['sharpe']:+.2f} MDD={r['mdd_pct']:.1f}% "
               f"αwin={r['win_rate_alpha']*100:.0f}%")
    print(f"\n--- NOST (regression), 排 ST (基准) ---")
    for _, r in excl[excl["model"] == "NOST"].iterrows():
        print(f"  Top{int(r['top_n']):3d}: 月化={r['monthly_net_pct']:+.2f}% "
               f"Sharpe={r['sharpe']:+.2f} MDD={r['mdd_pct']:.1f}% "
               f"αwin={r['win_rate_alpha']*100:.0f}%")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
