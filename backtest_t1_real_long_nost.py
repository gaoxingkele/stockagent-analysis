"""验证: r1_next_open_v3_long_nost (训练时已排除 ST) 在长 OOS 上的实战表现.

关键问题: 排除 ST 后训练的新 r1 模型, 是不是真的有 alpha (vs 旧模型 ST 偏见)?

vs backtest_t1_real_long.py:
  - 模型: r1_next_open_v3_long_nost (新, 训练已排 ST)
  - 推理时也排除 ST (源头一致)
  - 其他完全一样: cap ±3%, dist<3%, cost 0.35%
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
OUT = ROOT / "output" / "backtest_t1_long_nost"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)

OOS_START = "20251001"
OOS_END = "20260331"
MODEL_NAME = "r1_next_open_v3_long_nost"
COMPARE_MODEL = "r1_next_open_v3_long"  # 旧模型 (含 ST)
COST_BPS = 35.0 / 10000
R1_CAP = 3.0
DIST_THRESHOLD = 3.0
TOP_NS = [5, 10, 20, 50, 100]
HOUR = 15  # 只测 EOD


def load_model(name):
    d = PROD / name
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta["feature_cols"]


def main():
    t0 = time.time()
    print(f"\n=== r1_long_nost 实战验证 (vs 旧版) ===\n", flush=True)
    print(f"模型: {MODEL_NAME} (训练已排 ST)", flush=True)
    print(f"对比: {COMPARE_MODEL} (旧, 训练含 ST)", flush=True)

    df = pd.read_parquet(F3)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)
    oos = df[(df["trade_date"] >= OOS_START) & (df["trade_date"] <= OOS_END) &
              (df["trade_time"].dt.hour == HOUR)].copy()
    del df
    print(f"  OOS EOD bar: {len(oos):,}, 日数 {oos['trade_date'].nunique()}", flush=True)

    # ST 标记 (用于双重过滤)
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    oos = oos.merge(basic, on="ts_code", how="left")
    oos["is_st"] = oos["name"].fillna("").str.contains("ST", regex=False)
    print(f"  ST bar: {oos['is_st'].sum():,} ({oos['is_st'].mean()*100:.1f}%)", flush=True)

    # 推理两个模型
    for name in [MODEL_NAME, COMPARE_MODEL]:
        b, fc = load_model(name)
        for c in fc:
            if c not in oos.columns: oos[c] = 0.0
            oos[c] = oos[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        oos[f"pred_{name}"] = b.predict(oos[fc].astype("float32"))
        print(f"  {name} pred 均值 {oos[f'pred_{name}'].mean():+.3f}", flush=True)

    # 过滤准备
    oos["is_near_upper"] = oos["dist_to_upper_limit_pct"].fillna(100) < DIST_THRESHOLD
    oos["is_at_lower"] = oos["dist_to_lower_limit_pct"].fillna(100) < 1.0

    valid = oos.dropna(subset=["r1_next_open"])
    valid = valid[valid["r1_next_open"].abs() <= 20].copy()
    valid["r1_capped"] = valid["r1_next_open"].clip(-R1_CAP, R1_CAP)

    # 4 组合: (model × ST 过滤)
    results = []
    for model in [MODEL_NAME, COMPARE_MODEL]:
        for exclude_st in [False, True]:
            pred_col = f"pred_{model}"
            for top_n in TOP_NS:
                bad = valid["is_near_upper"] | valid["is_at_lower"]
                if exclude_st:
                    bad = bad | valid["is_st"]
                curve = []
                for d_, g in valid.groupby("trade_date"):
                    gf = g[~bad.loc[g.index]]
                    if len(gf) < top_n: continue
                    top = gf.nlargest(top_n, pred_col)
                    gross = top["r1_capped"].mean()
                    net = gross - COST_BPS * 100
                    mkt = g["r1_capped"].mean()
                    curve.append({"date": d_, "ret_net_pct": net, "ret_mkt_pct": mkt,
                                    "alpha_pct": net - mkt, "n_pool": len(top)})
                if not curve: continue
                cv = pd.DataFrame(curve).sort_values("date")
                cv["nav_net"] = (1 + cv["ret_net_pct"] / 100).cumprod()
                cv["nav_mkt"] = (1 + cv["ret_mkt_pct"] / 100).cumprod()
                cv["dd"] = cv["nav_net"] / cv["nav_net"].cummax() - 1

                model_tag = "NEW_nost" if model == MODEL_NAME else "OLD"
                st_tag = "ExclST" if exclude_st else "InclST"
                tag = f"{model_tag}_{st_tag}_Top{top_n}"
                results.append({
                    "model": model_tag, "st_filter": st_tag, "top_n": top_n,
                    "n_days": len(cv),
                    "total_net_pct": (cv["nav_net"].iloc[-1] - 1) * 100,
                    "total_mkt_pct": (cv["nav_mkt"].iloc[-1] - 1) * 100,
                    "monthly_net_pct": cv["ret_net_pct"].mean() * 20,
                    "sharpe": cv["ret_net_pct"].mean() / (cv["ret_net_pct"].std() + 1e-9) * np.sqrt(252),
                    "mdd_pct": cv["dd"].min() * 100,
                    "win_rate_alpha": (cv["alpha_pct"] > 0).mean(),
                })
                cv.to_csv(OUT / "curves" / f"{tag}.csv", index=False)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "results.csv", index=False)

    # 报告
    md = [f"# r1_long_nost vs r1_long 实战对比 (142 日 OOS)\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"NEW_nost: {MODEL_NAME} (训练排 ST)\n",
            f"OLD:      {COMPARE_MODEL} (训练含 ST)\n",
            f"约束: cap ±{R1_CAP}%, dist<{DIST_THRESHOLD}%, cost 0.35%\n\n",
            "## 4 组合 × 5 TopN = 20 配置\n\n",
            "| 模型 | ST | TopN | 日数 | 月化净 % | α 累计 % | Sharpe | MDD % | αwin |\n",
            "|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in res_df.iterrows():
        md.append(f"| {r['model']} | {r['st_filter']} | {int(r['top_n'])} | "
                   f"{int(r['n_days'])} | {r['monthly_net_pct']:+.2f} | "
                   f"{r['total_net_pct'] - r['total_mkt_pct']:+.1f} | "
                   f"{r['sharpe']:.2f} | {r['mdd_pct']:.1f} | "
                   f"{r['win_rate_alpha']*100:.0f}% |\n")

    # 关键对比 (NEW ExclST vs OLD ExclST)
    md.append(f"\n## 关键对比: 排 ST 实盘配置下\n\n")
    md.append("| TopN | OLD 排 ST | NEW_nost 排 ST | 差异 |\n|---|---|---|---|\n")
    for top_n in TOP_NS:
        old_e = res_df[(res_df["model"] == "OLD") & (res_df["st_filter"] == "ExclST") & (res_df["top_n"] == top_n)]
        new_e = res_df[(res_df["model"] == "NEW_nost") & (res_df["st_filter"] == "ExclST") & (res_df["top_n"] == top_n)]
        if old_e.empty or new_e.empty: continue
        o, n = old_e.iloc[0], new_e.iloc[0]
        md.append(f"| {top_n} | 月化 {o['monthly_net_pct']:+.2f}% Sharpe {o['sharpe']:.2f} | "
                   f"月化 {n['monthly_net_pct']:+.2f}% Sharpe {n['sharpe']:.2f} | "
                   f"Δ月化 {n['monthly_net_pct']-o['monthly_net_pct']:+.2f}pp |\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")

    # 终端打印 NEW 排 ST 关键行
    print(f"\n--- NEW_nost 模型, 排除 ST (实盘配置) ---")
    nost_excl = res_df[(res_df["model"] == "NEW_nost") & (res_df["st_filter"] == "ExclST")]
    for _, r in nost_excl.iterrows():
        print(f"  Top{int(r['top_n']):3d}: 月化={r['monthly_net_pct']:+6.2f}% "
               f"Sharpe={r['sharpe']:6.2f} MDD={r['mdd_pct']:5.1f}% "
               f"αwin={r['win_rate_alpha']*100:3.0f}%")

    print(f"\n--- OLD 模型, 排除 ST (历史 ST 偏见证伪基准) ---")
    old_excl = res_df[(res_df["model"] == "OLD") & (res_df["st_filter"] == "ExclST")]
    for _, r in old_excl.iterrows():
        print(f"  Top{int(r['top_n']):3d}: 月化={r['monthly_net_pct']:+6.2f}% "
               f"Sharpe={r['sharpe']:6.2f} MDD={r['mdd_pct']:5.1f}% "
               f"αwin={r['win_rate_alpha']*100:3.0f}%")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
