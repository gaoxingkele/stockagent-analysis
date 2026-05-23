"""增量验证: 长 OOS 模型在 4 月+5 月新数据上的稳定性.

模型训练截断 2025-09-30, 长 OOS 验证到 2026-03-31.
新数据增量: 2026-04-01 至 2026-05-15 (~30 个交易日).

目的:
  1. R5 反向 (P1) 在 4-5 月新数据上是否依然 Sharpe ~2.7?
  2. r1 模型 ST 偏见 在 4-5 月是否仍是主导?
  3. V12 双轨架构在 4-5 月推荐池的回测净值

注: 5 月数据只到 5/15 (8 个交易日), 完整 r5 label 仅前几天, r1 label 几乎全有.

输出: output/cross_scale/validate_may2026.md
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROD = ROOT / "output" / "production"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT = ROOT / "output" / "cross_scale"

# 增量窗口: 长 OOS 之后的数据
APR_START = "20260401"
APR_END = "20260430"
MAY_START = "20260501"
MAY_END = "20260515"


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"]


def eval_r1_top_n(df: pd.DataFrame, top_n: int, exclude_st: bool,
                    label_col: str = "r1_next_open") -> dict:
    """Top N 简单回测 (cap±3, 成本 0.35%, dist<3 过滤)."""
    df = df.dropna(subset=["pred_r1", label_col]).copy()
    df = df[df[label_col].abs() <= 20]
    df["lab_cap"] = df[label_col].clip(-3, 3)
    df["bad"] = (df["dist_to_upper_limit_pct"].fillna(100) < 3.0) | \
                  (df["dist_to_lower_limit_pct"].fillna(100) < 1.0)
    if exclude_st:
        df["bad"] = df["bad"] | df["is_st"]
    rows = []
    for d_, g in df.groupby("trade_date"):
        gf = g[~g["bad"]]
        if len(gf) < top_n: continue
        top = gf.nlargest(top_n, "pred_r1")
        net = top["lab_cap"].mean() - 0.35
        mkt = g["lab_cap"].mean()
        rows.append({"d": d_, "net": net, "mkt": mkt})
    if not rows: return {}
    cv = pd.DataFrame(rows)
    return {
        "n_days": len(cv),
        "monthly_net_pct": cv["net"].mean() * 20,
        "monthly_mkt_pct": cv["mkt"].mean() * 20,
        "sharpe": cv["net"].mean() / (cv["net"].std() + 1e-9) * np.sqrt(252),
        "alpha_win_rate": (cv["net"] > cv["mkt"]).mean(),
    }


def main():
    t0 = time.time()
    print(f"\n=== 增量验证 (2026-04-01 至 2026-05-15) ===\n", flush=True)

    # 1. 1H EOD + r1 推理 (4-5 月)
    print(f"[1] 加载 factors_v3 4-5 月 EOD bar ...", flush=True)
    df1h = pd.read_parquet(F3)
    df1h["trade_time"] = pd.to_datetime(df1h["trade_time"])
    df1h["trade_date"] = df1h["trade_date"].astype(str)
    eod = df1h[(df1h["trade_time"].dt.hour == 15) &
                 (df1h["trade_date"] >= APR_START) &
                 (df1h["trade_date"] <= MAY_END)].copy()
    del df1h
    print(f"   EOD bar: {len(eod):,}, 日数 {eod['trade_date'].nunique()}", flush=True)

    print(f"\n[2] r1_next_open_v3_long 推理 ...", flush=True)
    b1, fc1 = load_model("r1_next_open_v3_long")
    for c in fc1:
        if c not in eod.columns: eod[c] = 0.0
        eod[c] = eod[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    eod["pred_r1"] = b1.predict(eod[fc1].astype("float32"))

    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    eod = eod.merge(basic, on="ts_code", how="left")
    eod["is_st"] = eod["name"].fillna("").str.contains("ST", regex=False)

    print(f"\n[3] r1 信号稳定性对比 (3 区间 × 含/排 ST × Top 10/20) ...", flush=True)

    long_oos_range = ("20251001", "20260331")
    rows = []
    for window_name, (s_, e_) in [("Apr (4 月)", (APR_START, APR_END)),
                                    ("May (5 月前半)", (MAY_START, MAY_END))]:
        sub = eod[(eod["trade_date"] >= s_) & (eod["trade_date"] <= e_)]
        for top_n in [10, 20]:
            for exc_st in [False, True]:
                m = eval_r1_top_n(sub, top_n, exc_st)
                if m:
                    rows.append({
                        "window": window_name, "top_n": top_n,
                        "exclude_st": exc_st, **m,
                    })

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT / "validate_may2026_r1.csv", index=False)

    # 4. R5 反向 (P1) 稳定性 - 在 4 月数据上回测
    print(f"\n[4] R5 反向 (P1) 4 月稳定性 (r5 label 5 日, 需 5 日前数据) ...", flush=True)
    # 直接用 r5_v17_long 推理 4 月 + 早 5 月数据, 看 d5_bot30 r20 收益
    # 这里简化: 用 r5_next 替代 (r5 label 5 日)
    from train_v15_refresh import load_window
    daily = load_window("20260301", MAY_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    meta_r5 = json.loads((PROD / "r5_v17_long" / "feature_meta.json").read_text(encoding="utf-8"))
    ind_map = meta_r5.get("industry_map", {})
    if "industry" in daily.columns:
        daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    else:
        daily["industry_id"] = -1
    b5, fc5 = load_model("r5_v17_long")
    miss = [c for c in fc5 if c not in daily.columns]
    for c in miss: daily[c] = 0.0
    X5 = daily[fc5].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    daily["pred_r5"] = b5.predict(X5)
    b20, fc20 = load_model("r20_v16_long")
    miss = [c for c in fc20 if c not in daily.columns]
    for c in miss: daily[c] = 0.0
    X20 = daily[fc20].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    daily["pred_r20"] = b20.predict(X20)
    daily = daily.merge(basic, on="ts_code", how="left")
    daily["is_st"] = daily["name"].fillna("").str.contains("ST", regex=False)

    # merge dist
    eod_d = eod[["ts_code","trade_date","dist_to_upper_limit_pct","dist_to_lower_limit_pct"]].drop_duplicates(["ts_code","trade_date"])
    d2 = daily.merge(eod_d, on=["ts_code","trade_date"], how="left")
    d2["bad"] = (d2["dist_to_upper_limit_pct"].fillna(100) < 2.0) | \
                  (d2["dist_to_lower_limit_pct"].fillna(100) < 1.0)
    if "r20" in d2.columns:
        d2["r20_cap"] = d2["r20"].clip(-30, 30)

    # 4 月有完整 r20 label (4/1 + 20 = 4/30 之前需 4/1 起 20 日 label, 但 r20 = 之后 20 日, 所以 4/1 r20 = 4/21 close)
    # 实际上能算的最迟 trade_date = 5/15 - 20 = 4/25 左右
    r5_rows = []
    for d_, g in d2.groupby("trade_date"):
        if d_ < APR_START: continue
        g = g.dropna(subset=["pred_r20", "pred_r5", "r20_cap"])
        gf = g[~g["bad"]]
        if len(gf) < 500: continue
        for label_st, mask_st in [("含 ST", False), ("排 ST", True)]:
            ggf = gf if not mask_st else gf[~gf["is_st"]]
            if len(ggf) < 500: continue
            n_top = max(1, int(len(ggf) * 0.20))
            anchor = ggf.nlargest(n_top, "pred_r20").copy()
            anchor["d5_rank"] = anchor["pred_r5"].rank(pct=True, method="first")
            sub = anchor[anchor["d5_rank"] < 0.30]
            if len(sub):
                r5_rows.append({"date": d_, "st_filter": label_st, "n": len(sub),
                                 "r20": sub["r20_cap"].mean()})
    r5df = pd.DataFrame(r5_rows)
    if not r5df.empty:
        r5df.to_csv(OUT / "validate_may2026_r5.csv", index=False)

    # 报告
    rep = [f"# 增量验证: 长 OOS 模型在 4-5 月新数据上的稳定性\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"长 OOS (训练后): {long_oos_range[0]}-{long_oos_range[1]} (142 日, 已完成)\n",
            f"增量数据: {APR_START}-{MAY_END} (4 月全 + 5 月前半)\n\n",
            "## 1. r1 模型: ST 偏见是否仍主导?\n\n",
            "| 区间 | TopN | 含 ST? | 日数 | 月化净 % | 月化市场 % | Sharpe | α 胜率 |\n",
            "|---|---|---|---|---|---|---|---|\n"]
    for _, r in rdf.iterrows():
        st_s = "含" if not r["exclude_st"] else "排"
        rep.append(f"| {r['window']} | {int(r['top_n'])} | {st_s} | "
                    f"{int(r['n_days'])} | {r['monthly_net_pct']:+.2f} | "
                    f"{r['monthly_mkt_pct']:+.2f} | {r['sharpe']:.2f} | "
                    f"{r['alpha_win_rate']*100:.0f}% |\n")

    rep.append(f"\n### 长 OOS 142 日基准 (已知):\n")
    rep.append(f"- Top10 含 ST: 月化 +20.96% Sharpe 18.48 α胜率 90%\n")
    rep.append(f"- Top10 排 ST: 月化 -4.60% Sharpe -4.97 α胜率 35%\n")

    rep.append(f"\n## 2. R5 反向过滤 (P1): 4-5 月稳定性\n\n")
    if not r5df.empty:
        rep.append("| ST 过滤 | 日数 | 池均 | r20 均 % | std | Sharpe |\n|---|---|---|---|---|---|\n")
        for k, gg in r5df.groupby("st_filter"):
            sharpe = gg["r20"].mean() / (gg["r20"].std() + 1e-9) * np.sqrt(50)
            rep.append(f"| {k} | {len(gg)} | {gg['n'].mean():.1f} | "
                        f"{gg['r20'].mean():+.3f} | {gg['r20'].std():.2f} | {sharpe:.2f} |\n")
        rep.append(f"\n### 长 OOS 142 日基准 (已知):\n")
        rep.append(f"- d5_bot30 含 ST: r20 +2.402% Sharpe 2.70\n")
        rep.append(f"- d5_bot30 排 ST: r20 +2.431% Sharpe 2.67\n")
    else:
        rep.append(f"(无足够 r20 label 数据, 4-5 月 r20 需 20 日 forward)\n")

    rep.append(f"\n## 3. 行动建议\n\n")
    rep.append(f"- 若 4-5 月 R5 反向 Sharpe 仍 > 2.0 → P1 集成保留, P3 双轨架构上线\n")
    rep.append(f"- 若 4-5 月 r1 排 ST 仍负 → P2 daily_r1_recommend 仅作辅助参考\n")
    rep.append(f"- 若两者都衰减严重 → 重训模型 (训练区间扩展到 2026-04)\n")

    out = OUT / "validate_may2026.md"
    out.write_text("".join(rep), encoding="utf-8")
    print(f"\n输出: {out}", flush=True)
    for _, r in rdf.iterrows():
        st_s = "含" if not r["exclude_st"] else "排"
        print(f"  {r['window']} Top{int(r['top_n'])} {st_s}ST: "
               f"月化={r['monthly_net_pct']:+.2f}% Sharpe={r['sharpe']:+.2f} "
               f"αwin={r['alpha_win_rate']*100:.0f}%")
    if not r5df.empty:
        for k, gg in r5df.groupby("st_filter"):
            sharpe = gg["r20"].mean() / (gg["r20"].std() + 1e-9) * np.sqrt(50)
            print(f"  R5_bot30 {k}: 天数 {len(gg)}, r20 均 {gg['r20'].mean():+.3f}%, "
                   f"Sharpe {sharpe:.2f}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
