"""Phase 2 长期收益偏置因子计算.

针对 r1 lambdarank 模型残余 -2% 月化的"明星股短期回调 vs 普通股 r1 信号" 问题,
加 11 个新因子让模型显式知道每只股的长期表现层级.

输出: output/long_return_features/features.parquet
       output/long_return_features/features_ext_{mmdd}.parquet (增量, 与 v12 pipeline 对齐)

因子设计 (3 层):

A. 个股层 (4 个)
   - long_return_252d         : close[t-1] / close[t-253] - 1     (1 年 forward leak 安全)
   - long_return_504d         : close[t-1] / close[t-505] - 1     (2 年)
   - long_return_252d_decile  : 当日全市场 pct rank × 10, 0-9, categorical
   - long_return_504d_decile  : 当日全市场 pct rank × 10, 0-9, categorical

B. 板块层 (4 个)
   - industry_return_252d         : 该日行业内股票 long_return_252d 均值
   - industry_return_504d         : 同上 504d
   - industry_return_504d_decile  : 行业层 decile (0-9)
   - concept_return_504d_mean     : 个股关联概念 504d 收益均值 (主要概念前 3)

C. 相对强度 (3 个)
   - relative_strength_252d  : long_return_252d - industry_return_252d
   - relative_strength_504d  : long_return_504d - industry_return_504d
   - rs_in_decile            : relative_strength_504d 在当日全市场的 decile (0-9)

Forward leak 防范:
   - close[t-1] / close[t-505], 用 t-1 作为窗口结束 (t 当天的 close 是 EOD 收盘后才确定)
   - decile 用当日横截面 pct rank, 不用未来分布
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
CONCEPT_DETAIL_P = ROOT / "output" / "tushare_cache" / "concept_detail.parquet"
CONCEPT_SUMMARY_P = ROOT / "output" / "tushare_cache" / "concept_member_summary.parquet"
OUT_DIR = ROOT / "output" / "long_return_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_daily_close() -> pd.DataFrame:
    """加载 daily cache 全历史 (ts_code, trade_date, close)."""
    files = sorted(DAILY_DIR.glob("*.parquet"))
    print(f"  加载 {len(files)} 个 daily 文件...", flush=True)
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
        parts.append(d)
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"  total {len(big):,} 行 / {big['ts_code'].nunique()} 股", flush=True)
    return big


def compute_individual_long_return(daily: pd.DataFrame) -> pd.DataFrame:
    """个股 long_return_252d/504d (shift 1 防 forward leak)."""
    print(f"\n[1/3] 个股 long_return (252d/504d) ...", flush=True)
    df = daily.copy()
    # close[t-1] / close[t-N-1] - 1, 用 t-1 作为窗口末日
    df["close_prev"] = df.groupby("ts_code")["close"].shift(1)
    df["close_252d_ago"] = df.groupby("ts_code")["close"].shift(253)
    df["close_504d_ago"] = df.groupby("ts_code")["close"].shift(505)
    df["long_return_252d"] = df["close_prev"] / df["close_252d_ago"] - 1
    df["long_return_504d"] = df["close_prev"] / df["close_504d_ago"] - 1

    # decile (当日横截面 pct rank, NaN 不参与排名)
    print(f"  decile 计算 (按日横截面 pct rank)...", flush=True)
    for col in ["long_return_252d", "long_return_504d"]:
        ranks = df.groupby("trade_date")[col].rank(pct=True, method="first")
        bins_float = (ranks * 10).clip(0, 9)
        df[f"{col}_decile"] = np.floor(bins_float.fillna(-1).values).astype("int8")
        df[f"{col}_decile"] = df[f"{col}_decile"].where(df[f"{col}_decile"] >= 0)

    return df[["ts_code", "trade_date",
                  "long_return_252d", "long_return_504d",
                  "long_return_252d_decile", "long_return_504d_decile"]]


def compute_industry_layer(ind_df: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    """板块层均值 + decile."""
    print(f"\n[2/3] 行业层 long_return 均值 + decile ...", flush=True)
    # merge industry
    df = ind_df.merge(basic[["ts_code", "industry"]], on="ts_code", how="left")
    df["industry"] = df["industry"].fillna("unknown")

    # 行业内当日均值
    agg = df.groupby(["trade_date", "industry"]).agg(
        industry_return_252d=("long_return_252d", "mean"),
        industry_return_504d=("long_return_504d", "mean"),
    ).reset_index()

    # 行业 decile (基于 industry_return_504d 在当日所有行业里的 pct rank)
    print(f"  行业 decile (按日横截面)...", flush=True)
    ranks = agg.groupby("trade_date")["industry_return_504d"].rank(pct=True, method="first")
    bins_float = (ranks * 10).clip(0, 9)
    agg["industry_return_504d_decile"] = np.floor(bins_float.fillna(-1).values).astype("int8")
    agg["industry_return_504d_decile"] = agg["industry_return_504d_decile"].where(
        agg["industry_return_504d_decile"] >= 0)

    # 回 merge 到个股
    df = df.merge(agg, on=["trade_date", "industry"], how="left")
    df["relative_strength_252d"] = df["long_return_252d"] - df["industry_return_252d"]
    df["relative_strength_504d"] = df["long_return_504d"] - df["industry_return_504d"]

    # rs_in_decile: relative_strength_504d 在当日全市场的 decile
    print(f"  rs_in_decile (全市场)...", flush=True)
    rs_ranks = df.groupby("trade_date")["relative_strength_504d"].rank(pct=True, method="first")
    bins_float = (rs_ranks * 10).clip(0, 9)
    df["rs_in_decile"] = np.floor(bins_float.fillna(-1).values).astype("int8")
    df["rs_in_decile"] = df["rs_in_decile"].where(df["rs_in_decile"] >= 0)

    return df[["ts_code", "trade_date",
                  "industry_return_252d", "industry_return_504d",
                  "industry_return_504d_decile",
                  "relative_strength_252d", "relative_strength_504d",
                  "rs_in_decile"]]


def compute_concept_layer(ind_df: pd.DataFrame,
                            concept_summary_p: Path,
                            concept_detail_p: Path) -> pd.DataFrame:
    """概念层 (用主要概念 top3 关联股票的 504d 均值).

    实现:
      1. concept_summary: 每股 → 主要概念 (top 3 最稀有的)
      2. 当日: 每个概念的 504d 均值 (该概念关联的所有股票 long_return_504d 均值)
      3. 个股 concept_return_504d_mean = 其主要 3 个概念的均值的均值

    如果概念数据不存在, 返回空 df (后续 merge 会忽略).
    """
    print(f"\n[3/3] 概念层 long_return ...", flush=True)
    if not concept_summary_p.exists() or not concept_detail_p.exists():
        print(f"  概念数据不存在, 跳过 (后续 fetch_concept_data.py 后重跑)", flush=True)
        return pd.DataFrame()

    summary = pd.read_parquet(concept_summary_p)  # ts_code, primary_concept_1/2/3, n_concepts
    detail = pd.read_parquet(concept_detail_p)    # ts_code, concept_name, concept_code
    print(f"  summary {len(summary)} 股, detail {len(detail)} 行 / "
           f"{detail['concept_name'].nunique()} 概念", flush=True)

    # 每股 long_return_504d → 每个概念 504d 均值 (当日)
    # detail (ts_code, concept_name) 是 N:N
    # ind_df (ts_code, trade_date, long_return_504d)
    # concept_daily = ind_df ⋈ detail → group by (trade_date, concept_name) → mean long_return_504d
    ind_slim = ind_df[["ts_code", "trade_date", "long_return_504d"]].dropna(
        subset=["long_return_504d"])
    print(f"  merge concept_detail x ind_df ...", flush=True)
    joined = detail[["ts_code", "concept_name"]].merge(
        ind_slim, on="ts_code", how="inner")
    print(f"  joined: {len(joined):,} 行 (含 {joined['concept_name'].nunique()} 概念)", flush=True)

    concept_daily = joined.groupby(["trade_date", "concept_name"])["long_return_504d"].mean(
        ).reset_index().rename(columns={"long_return_504d": "concept_504d"})

    # 每股: 主要 3 个概念的 concept_504d 均值
    # melt summary
    sum_melt = summary.melt(id_vars=["ts_code"],
                              value_vars=["primary_concept_1", "primary_concept_2", "primary_concept_3"],
                              var_name="rank", value_name="concept_name").dropna(subset=["concept_name"])

    # merge daily
    print(f"  每股 → 每日主要概念均值 ...", flush=True)
    # 需要 (ts_code) × (trade_date) 的 cross product, 然后 merge concept_504d
    # 因为概念是稳定的 (不随日期变), 只需对每个 trade_date 算 sum_melt × concept_daily
    days = sorted(ind_df["trade_date"].unique())
    parts = []
    for d in days:
        cd = concept_daily[concept_daily["trade_date"] == d][["concept_name", "concept_504d"]]
        if cd.empty: continue
        m = sum_melt.merge(cd, on="concept_name", how="inner")
        if m.empty: continue
        per_stock = m.groupby("ts_code")["concept_504d"].mean().rename(
            "concept_return_504d_mean").reset_index()
        per_stock["trade_date"] = d
        parts.append(per_stock)

    if not parts:
        print(f"  concept 计算无结果", flush=True)
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)[
        ["ts_code", "trade_date", "concept_return_504d_mean"]]


def main():
    t0 = time.time()
    print(f"\n=== 长期收益偏置因子计算 ===\n", flush=True)

    daily = load_daily_close()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name", "industry"]].drop_duplicates("ts_code")

    # 个股层
    ind = compute_individual_long_return(daily)
    del daily

    # 板块层
    ind_layer = compute_industry_layer(ind, basic)

    # 合并 A + B
    feats = ind.merge(ind_layer, on=["ts_code", "trade_date"], how="left")

    # 概念层
    cpt = compute_concept_layer(ind, CONCEPT_SUMMARY_P, CONCEPT_DETAIL_P)
    if not cpt.empty:
        feats = feats.merge(cpt, on=["ts_code", "trade_date"], how="left")
        print(f"  + 概念层: {feats['concept_return_504d_mean'].notna().sum():,} 非空", flush=True)
    else:
        feats["concept_return_504d_mean"] = np.nan

    # cast decile to nullable int (categorical for LightGBM)
    for col in ["long_return_252d_decile", "long_return_504d_decile",
                  "industry_return_504d_decile", "rs_in_decile"]:
        feats[col] = pd.array(feats[col], dtype="Int8")

    # 输出
    out_p = OUT_DIR / "features.parquet"
    feats.to_parquet(out_p, index=False)
    print(f"\n输出: {out_p}", flush=True)
    print(f"  {len(feats):,} 行 / {len(feats.columns)} 列", flush=True)
    print(f"  cols: {list(feats.columns)}", flush=True)
    print(f"  覆盖率 (非 NaN):")
    for c in feats.columns:
        if c in ("ts_code", "trade_date"): continue
        n = feats[c].notna().sum()
        print(f"    {c:40s}: {n:,} ({n/len(feats)*100:.1f}%)", flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
