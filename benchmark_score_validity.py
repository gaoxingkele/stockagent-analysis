#!/usr/bin/env python3
"""评分有效性内在测试 — 评分排序 vs 实际 max_gain_20 排序.

实验 A: 股票内部 IC
  在每个 t0, 全市场股票按 uptrend_prob 评分.
  看评分排序 vs max_gain_20 排序的 Spearman IC.
  按评分 5 分位看 max_gain_20 实际分布.

实验 B: 评分 vs ETF 指数同样评分
  在每个 t0, 给 5 个指数 (沪深300/中证500/创业板/北证50/中证1000) 算 uptrend_prob.
  比较: 我们 top 10 股票的评分 vs 各指数的评分.
  看: 评分高的资产是不是 max_gain_20 也高.

实验 C (简化): 评分 vs 基金技术评分
  基金 NAV 算 5/20 日斜率 + 简化"动量评分".
  比较跨资产排序 vs 实际 max_gain.
"""
from __future__ import annotations
import json, os, struct, time, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS = ROOT / "output" / "labels" / "max_gain_labels.parquet"
TRADES = ROOT / "output" / "portfolio_bt" / "trades.csv"
OUT_DIR = ROOT / "output" / "benchmark"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

INDICES = {
    "hs300": ("sh", "000300", "沪深300"),
    "cyb":   ("sz", "399006", "创业板指"),
    "zz500": ("sh", "000905", "中证500"),
    "zz1000":("sh", "000852", "中证1000"),
    "bj50":  ("bj", "899050", "北证50"),
}


def read_index_ohlc(market, code):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    rows = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append((d.strftime("%Y%m%d"), f[1]/100.0, f[2]/100.0,
                     f[3]/100.0, f[4]/100.0, float(f[6])))
    return pd.DataFrame(rows, columns=["trade_date","open","high","low","close","volume"])


def index_max_gain_20(idx_df, t0, lookahead=20):
    """同 t0+1 open 起 20 天期间最高涨幅 (%)."""
    if idx_df is None: return None
    dates = idx_df["trade_date"].tolist()
    if t0 not in dates: return None
    i = dates.index(t0)
    if i + lookahead >= len(dates): return None
    p0 = idx_df.iloc[i+1]["open"]
    if p0 <= 0: return None
    future = idx_df.iloc[i+1: i+1+lookahead]
    return (future["high"].max() / p0 - 1) * 100


def compute_index_features(idx_df: pd.DataFrame) -> pd.DataFrame:
    """给指数算个简化的"技术指标评分"特征 (用于跨资产可比)."""
    df = idx_df.copy()
    df["ma5"]  = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ret_5d"]  = df["close"].pct_change(5) * 100
    df["ret_20d"] = df["close"].pct_change(20) * 100
    df["ret_60d"] = df["close"].pct_change(60) * 100
    df["ma5_slope"] = df["ma5"].pct_change(5) * 100
    df["close_vs_ma20"] = (df["close"] / df["ma20"] - 1) * 100
    # RSI 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    # 成交量比率
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["vol_ma5"] / df["vol_ma20"]
    return df


def index_simple_score(row) -> float:
    """简化的指数技术评分 (0-100), 用同股票相似的逻辑."""
    if pd.isna(row.get("ret_20d")):
        return 50
    base = 50
    # 短期趋势: ret_5d
    r5 = row.get("ret_5d", 0)
    base += min(15, max(-15, r5 * 1.5))
    # MA5 斜率
    slope = row.get("ma5_slope", 0)
    base += min(8, max(-8, slope * 2))
    # close vs ma20 (站稳趋势)
    cv = row.get("close_vs_ma20", 0)
    base += min(8, max(-8, cv * 0.8))
    # 量能放大
    vr = row.get("vol_ratio", 1)
    if vr and vr > 1.2: base += 5
    elif vr and vr < 0.8: base -= 5
    # RSI
    rsi = row.get("rsi14", 50)
    if rsi:
        if rsi > 70: base -= 5  # 超买
        elif rsi < 30: base += 5  # 超卖
    return max(0, min(100, base))


def main():
    t0_start = time.time()

    # ── 实验 A: 股票内部 IC ────────────────────────────────────
    print("=" * 90)
    print("实验 A: 股票内部 IC (评分排序 vs max_gain_20 排序)")
    print("=" * 90)

    # 取所有 OOS 数据 + 标签
    print("加载 OOS 数据...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= "20250501") & (df["trade_date"] <= "20260126")]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_gain_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left").dropna(subset=["max_gain_20"])

    # merge 各种 features
    for path in [
        ROOT / "output" / "regimes" / "daily_regime.parquet",
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if path.name == "daily_regime.parquet":
            d = d.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                    "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14",
                                    "vol_ratio":"mkt_vol_ratio"})
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")

    # 批量预测 uptrend_prob
    print("批量预测 uptrend_prob...")
    booster = lgb.Booster(model_file="output/lgbm_uptrend/classifier.txt")
    meta = json.loads(Path("output/lgbm_uptrend/feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    indmap = meta["industry_map"]
    full["industry_id"] = full["industry"].fillna("unknown").map(
        lambda x: indmap.get(str(x), -1))
    for c in feat_cols:
        if c not in full.columns: full[c] = np.nan
    full["uptrend_prob"] = booster.predict(full[feat_cols].astype(float))
    print(f"  打分完成: {len(full)} 行")

    # 取换仓日子集
    if not TRADES.exists():
        print("没找到 trades.csv, 用所有 OOS 日期")
        rb_dates = sorted(full["trade_date"].unique())[::5]
    else:
        trades = pd.read_csv(TRADES)
        rb_dates = sorted(trades["rb_date"].astype(str).unique())
    print(f"分析 {len(rb_dates)} 个买点的内部 IC")

    # 每日内 Spearman IC
    daily_ics = []
    for d in rb_dates:
        sub = full[full["trade_date"] == d]
        if len(sub) < 100: continue
        ic = stats.spearmanr(sub["uptrend_prob"], sub["max_gain_20"])[0]
        if not np.isnan(ic):
            daily_ics.append({"date": d, "ic": ic, "n": len(sub)})
    ic_df = pd.DataFrame(daily_ics)
    print(f"\n每日 IC 分布 ({len(ic_df)} 个买点):")
    print(f"  平均 IC: {ic_df['ic'].mean():+.4f}")
    print(f"  中位 IC: {ic_df['ic'].median():+.4f}")
    print(f"  IC > 0 比例: {(ic_df['ic'] > 0).mean()*100:.1f}%")
    print(f"  IC > 0.10 比例: {(ic_df['ic'] > 0.10).mean()*100:.1f}%")
    print(f"  最高 IC: {ic_df['ic'].max():+.4f} ({ic_df.loc[ic_df['ic'].idxmax(), 'date']})")
    print(f"  最低 IC: {ic_df['ic'].min():+.4f} ({ic_df.loc[ic_df['ic'].idxmin(), 'date']})")

    # 5 分位 max_gain_20 分布
    print(f"\n=== 评分 5 分位 max_gain_20 实际分布 ===")
    full["score_q"] = full.groupby("trade_date")["uptrend_prob"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    sub_full = full[full["trade_date"].isin(rb_dates)]
    summary = sub_full.groupby("score_q").agg(
        prob_avg=("uptrend_prob", "mean"),
        max_gain_avg=("max_gain_20", "mean"),
        max_gain_med=("max_gain_20", "median"),
        gain_15=("max_gain_20", lambda x: (x>=15).mean()*100),
        gain_30=("max_gain_20", lambda x: (x>=30).mean()*100),
        n=("max_gain_20", "count"),
    ).round(3)
    print(summary.to_string())

    # ── 实验 B: 跨资产 (股票 vs ETF) ─────────────────────────────
    print("\n" + "=" * 90)
    print("实验 B: 同一买点, 我们 top 股票评分 vs ETF 评分")
    print("=" * 90)

    print("加载指数 OHLC + 算技术评分...")
    idx_data = {}
    for name, (m, c, cn) in INDICES.items():
        d = read_index_ohlc(m, c)
        if d is not None:
            d = compute_index_features(d.sort_values("trade_date").reset_index(drop=True))
            idx_data[name] = (cn, d)

    print(f"\n{'买点':<11} {'我们top10均分':>10} {'我们top10 max_gain':>15} "
          f"{'沪深300分':>10} {'沪深300 max_gain':>14} | 评分高的资产是否 max_gain 更高")
    print("-" * 105)

    cmp_rows = []
    for d in rb_dates:
        # 我们 top 10 平均评分
        sub = full[full["trade_date"] == d]
        if len(sub) < 50: continue
        top10 = sub.nlargest(10, "uptrend_prob")
        our_score = top10["uptrend_prob"].mean() * 100
        our_mg = top10["max_gain_20"].mean()

        # 各指数评分 + max_gain_20
        scores = {"date": d, "our_score": our_score, "our_max_gain": our_mg}
        for name, (cn, idx_df) in idx_data.items():
            row = idx_df[idx_df["trade_date"] == d]
            if len(row) == 0: continue
            sc = index_simple_score(row.iloc[0])
            mg = index_max_gain_20(idx_df, d)
            scores[f"{name}_score"] = sc
            scores[f"{name}_max_gain"] = mg
        cmp_rows.append(scores)

    cmp_df = pd.DataFrame(cmp_rows)

    # 总体对比 (vs 沪深300)
    sub = cmp_df.dropna(subset=["our_score","hs300_score","our_max_gain","hs300_max_gain"])
    print(f"\n=== 对比汇总 (n={len(sub)}) ===")
    score_diff_high_when_gain_high = (
        ((sub["our_score"] > sub["hs300_score"]) & (sub["our_max_gain"] > sub["hs300_max_gain"])) |
        ((sub["our_score"] < sub["hs300_score"]) & (sub["our_max_gain"] < sub["hs300_max_gain"]))
    ).sum()
    print(f"评分高低关系 vs 实际 max_gain 高低关系一致比例: "
          f"{score_diff_high_when_gain_high}/{len(sub)} = "
          f"{score_diff_high_when_gain_high/len(sub)*100:.1f}%")

    print("\n=== 各指数: 我们 top10 评分 vs 该指数评分 (排序一致性) ===")
    for name in INDICES:
        col_s = f"{name}_score"
        col_g = f"{name}_max_gain"
        if col_s not in cmp_df.columns: continue
        sub2 = cmp_df.dropna(subset=["our_score", col_s, "our_max_gain", col_g])
        if len(sub2) == 0: continue
        # 评分谁高
        we_higher_score = (sub2["our_score"] > sub2[col_s]).sum()
        # 实际 max_gain 谁高
        we_higher_gain = (sub2["our_max_gain"] > sub2[col_g]).sum()
        # 评分排序与实际涨幅排序一致的次数
        consistent = (
            ((sub2["our_score"] > sub2[col_s]) & (sub2["our_max_gain"] > sub2[col_g])) |
            ((sub2["our_score"] < sub2[col_s]) & (sub2["our_max_gain"] < sub2[col_g]))
        ).sum()
        cn = INDICES[name][2]
        print(f"  vs {cn:<10}: 评分我们>它 {we_higher_score:>2}/{len(sub2)}, "
              f"max_gain 我们>它 {we_higher_gain:>2}/{len(sub2)}, "
              f"排序一致 {consistent:>2}/{len(sub2)} ({consistent/len(sub2)*100:.0f}%)")

    cmp_df.to_csv(OUT_DIR / "score_validity.csv", index=False, encoding="utf-8-sig")

    # 关键问题: 我们的评分 vs 实际 max_gain 的相关性
    print(f"\n=== 关键测试: 我们评分越高 → max_gain 越高? ===")
    sub3 = cmp_df.dropna(subset=["our_score", "our_max_gain"])
    score_max_ic = stats.spearmanr(sub3["our_score"], sub3["our_max_gain"])[0]
    print(f"我们 top10 评分 vs top10 max_gain (跨日): IC = {score_max_ic:+.4f}")
    print("  (如果系统对, 评分高的日子, 选出来的股票 max_gain 应该也高)")

    print(f"\n总耗时 {time.time()-t0_start:.1f}s")


if __name__ == "__main__":
    main()
