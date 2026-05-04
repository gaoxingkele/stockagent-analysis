#!/usr/bin/env python3
"""全量 OOS 验证 v2 双向评分: buy_score / sell_score 分位胜率.

OOS 区间: 2025-05-01 → 2026-01-26 (~92 万样本)
4 模型: r10_all, r20_all, sell_10, sell_20

输出:
  output/production/oos_validation/
    decile_buy.csv      buy_score 十档 → 实际 r10/r20/胜率/max_dd
    decile_sell.csv     sell_score 十档 → 实际 max_dd/事件率
    quadrant.csv        buy×sell 4 象限组合分析
    summary.txt         核心结论
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
PROD_DIR = ROOT / "output" / "production"
OUT_DIR = PROD_DIR / "oos_validation"
OUT_DIR.mkdir(exist_ok=True)

TEST_START = "20250501"
TEST_END   = "20260126"

# OOS 锚定百分位 (与 lgbm_predictor.predict_dual 同步)
R10_ANCHOR = (0.72, 0.94, 1.40)
R20_ANCHOR = (-2.34, 2.50, 6.36)
SELL10_ANCHOR = (0.05, 0.20, 0.64)
SELL20_ANCHOR = (0.01, 0.07, 0.67)


def _map_anchored(v, p5, p50, p95):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50) * 50, out)
    return out


def load_oos():
    print(f"[{int(time.time())}s] 加载 OOS test 数据 {TEST_START} → {TEST_END}...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)]
        if not df.empty:
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    print(f"  base: {len(full):,} rows")

    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10","max_dd_10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")

    l20 = pd.read_parquet(LABELS_20, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")

    for path in [
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
        ROOT / "output" / "cogalpha_features" / "features.parquet",
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")

    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
                              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        full = full.merge(rg, on="trade_date", how="left")
    return full


def predict_model(df: pd.DataFrame, model_name: str) -> np.ndarray:
    d = PROD_DIR / model_name
    booster = lgb.Booster(model_file=str(d / "classifier.txt"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})

    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    X = df[feat_cols]
    pred = booster.predict(X)
    return pred


def decile_stats(df: pd.DataFrame, score_col: str, label_cols: list[str],
                  q: int = 10) -> pd.DataFrame:
    """按 score 十分位, 算每档 label 均值 + 胜率."""
    df = df.dropna(subset=[score_col]).copy()
    try:
        df["bucket"] = pd.qcut(df[score_col], q=q, labels=False, duplicates="drop")
    except ValueError:
        df["bucket"] = pd.cut(df[score_col], bins=q, labels=False)
    rows = []
    for b in sorted(df["bucket"].dropna().unique()):
        g = df[df["bucket"] == b]
        row = {"bucket": int(b), "n": len(g),
               "score_mean": g[score_col].mean()}
        for lc in label_cols:
            v = g[lc].dropna()
            if len(v) == 0:
                row[f"{lc}_mean"] = np.nan
                row[f"{lc}_winrate"] = np.nan
                continue
            row[f"{lc}_mean"] = v.mean()
            row[f"{lc}_winrate"] = (v > 0).mean() * 100
        if "max_dd_10" in g.columns:
            row["dd10_lt_5pct"] = (g["max_dd_10"] <= -5).mean() * 100
        if "max_dd_20" in g.columns:
            row["dd20_lt_8pct"] = (g["max_dd_20"] <= -8).mean() * 100
            row["dd20_lt_15pct"] = (g["max_dd_20"] <= -15).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    df = load_oos()
    print(f"[{int(time.time()-t0)}s] OOS shape: {df.shape}")

    # 4 模型预测
    print(f"[{int(time.time()-t0)}s] 预测 r10_all...")
    df["r10_pred"] = predict_model(df, "r10_all")
    print(f"[{int(time.time()-t0)}s] 预测 r20_all...")
    df["r20_pred"] = predict_model(df, "r20_all")
    print(f"[{int(time.time()-t0)}s] 预测 sell_10...")
    df["sell_10_prob"] = predict_model(df, "sell_10")
    print(f"[{int(time.time()-t0)}s] 预测 sell_20...")
    df["sell_20_prob"] = predict_model(df, "sell_20")

    # 评分
    s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df["r20_pred"].values, *R20_ANCHOR)
    df["buy_score"] = 0.5 * s10 + 0.5 * s20

    s10_sell = _map_anchored(df["sell_10_prob"].values, *SELL10_ANCHOR)
    s20_sell = _map_anchored(df["sell_20_prob"].values, *SELL20_ANCHOR)
    df["sell_score"] = 0.5 * s10_sell + 0.5 * s20_sell

    print(f"[{int(time.time()-t0)}s] 评分分布:")
    print(f"  buy:  p5={df['buy_score'].quantile(0.05):.1f}  p50={df['buy_score'].median():.1f}  p95={df['buy_score'].quantile(0.95):.1f}")
    print(f"  sell: p5={df['sell_score'].quantile(0.05):.1f}  p50={df['sell_score'].median():.1f}  p95={df['sell_score'].quantile(0.95):.1f}")

    # buy_score 十档
    print(f"\n[{int(time.time()-t0)}s] === buy_score 十档分析 ===")
    buy_df = decile_stats(df.copy(), "buy_score",
                            ["r10","r20","max_gain_20","max_dd_20"])
    print(buy_df.round(3).to_string())
    buy_df.to_csv(OUT_DIR / "decile_buy.csv", index=False, encoding="utf-8-sig")

    # sell_score 十档
    print(f"\n[{int(time.time()-t0)}s] === sell_score 十档分析 ===")
    sell_df = decile_stats(df.copy(), "sell_score",
                             ["r10","r20","max_dd_10","max_dd_20"])
    print(sell_df.round(3).to_string())
    sell_df.to_csv(OUT_DIR / "decile_sell.csv", index=False, encoding="utf-8-sig")

    # 4 象限分析
    print(f"\n[{int(time.time()-t0)}s] === buy × sell 4 象限 ===")
    df["buy_high"] = (df["buy_score"] >= 70).astype(int)
    df["buy_low"]  = (df["buy_score"] <= 30).astype(int)
    df["sell_high"]= (df["sell_score"] >= 70).astype(int)
    df["sell_low"] = (df["sell_score"] <= 30).astype(int)
    quadrants = []
    for label, mask in [
        ("BUY_high+SELL_low (理想多)", (df["buy_score"] >= 70) & (df["sell_score"] <= 30)),
        ("BUY_high+SELL_high (矛盾)",  (df["buy_score"] >= 70) & (df["sell_score"] >= 70)),
        ("BUY_low+SELL_high (理想空)", (df["buy_score"] <= 30) & (df["sell_score"] >= 70)),
        ("BUY_low+SELL_low (沉寂)",   (df["buy_score"] <= 30) & (df["sell_score"] <= 30)),
        ("中性区 (其他)",              ~((df["buy_score"] >= 70) | (df["buy_score"] <= 30) |
                                          (df["sell_score"] >= 70) | (df["sell_score"] <= 30))),
    ]:
        g = df[mask]
        if len(g) == 0: continue
        r10v = g["r10"].dropna()
        r20v = g["r20"].dropna()
        dd20 = g["max_dd_20"].dropna()
        quadrants.append({
            "quadrant": label, "n": len(g),
            "r10_mean": r10v.mean() if len(r10v) else np.nan,
            "r10_winrate": (r10v > 0).mean()*100 if len(r10v) else np.nan,
            "r20_mean": r20v.mean() if len(r20v) else np.nan,
            "r20_winrate": (r20v > 0).mean()*100 if len(r20v) else np.nan,
            "dd20_lt_8pct": (dd20 <= -8).mean()*100 if len(dd20) else np.nan,
            "dd20_lt_15pct": (dd20 <= -15).mean()*100 if len(dd20) else np.nan,
        })
    quad_df = pd.DataFrame(quadrants)
    print(quad_df.round(2).to_string())
    quad_df.to_csv(OUT_DIR / "quadrant.csv", index=False, encoding="utf-8-sig")

    # 极端段单独看
    print(f"\n[{int(time.time()-t0)}s] === 极端档 (top/bot 5%) ===")
    tops = df.nlargest(int(len(df)*0.05), "buy_score")
    bots = df.nsmallest(int(len(df)*0.05), "buy_score")
    print(f"  top 5% buy_score (n={len(tops):,}):  r20 mean={tops['r20'].mean():.3f}%  win={(tops['r20']>0).mean()*100:.1f}%")
    print(f"  bot 5% buy_score (n={len(bots):,}):  r20 mean={bots['r20'].mean():.3f}%  win={(bots['r20']>0).mean()*100:.1f}%")

    sell_tops = df.nlargest(int(len(df)*0.05), "sell_score")
    sell_bots = df.nsmallest(int(len(df)*0.05), "sell_score")
    print(f"  top 5% sell_score (n={len(sell_tops):,}): dd20<=-8% rate={((sell_tops['max_dd_20'] <= -8).mean()*100):.1f}%  dd20<=-15%={((sell_tops['max_dd_20']<=-15).mean()*100):.1f}%")
    print(f"  bot 5% sell_score: dd20<=-8% rate={((sell_bots['max_dd_20']<=-8).mean()*100):.1f}%")

    # summary
    summary_lines = [
        f"OOS 验证总耗时: {time.time()-t0:.0f}s",
        f"OOS 样本: {len(df):,} (从 {TEST_START} 到 {TEST_END})",
        "",
        "=== buy_score 分位胜率 ===",
        buy_df.round(3).to_string(),
        "",
        "=== sell_score 分位胜率 ===",
        sell_df.round(3).to_string(),
        "",
        "=== 4 象限 ===",
        quad_df.round(2).to_string(),
    ]
    Path(OUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n[{int(time.time()-t0)}s] 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
