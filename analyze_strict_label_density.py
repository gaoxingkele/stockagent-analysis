"""精确计算用户提出的严格 label 定义下, A 股真实样本数量.

日线:
  r5_strict: 未来 5 日最高涨幅 ≥ 10% AND 期间最大回撤 ≤ 5%
  r20_strict: 未来 20 日最高涨幅 ≥ 20% AND 期间最大回撤 ≤ 5%

1H:
  r5_1h_strict: 未来 5 个 1H bar 最高涨幅 ≥ 5% AND 回撤 ≤ 3%
  r20_1h_strict: 未来 20 个 1H bar 最高涨幅 ≥ 10% AND 回撤 ≤ 3%

涨幅: max(close[t+1..t+N]) / open[t+1] - 1
回撤: min(close[t+1..t+N]) / open[t+1] - 1  (越正越好, 阈值 -drawdown 是下界)
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent


def compute_daily_strict_labels():
    """日线层 r5/r20 严格 label."""
    print("\n=== 日线 ===", flush=True)
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    print(f"daily 文件: {len(files)} 个", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 排除 ST (源头一致)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name"]].drop_duplicates("ts_code")
    st_set = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    before = len(big)
    big = big[~big["ts_code"].isin(st_set)].reset_index(drop=True)
    print(f"ST 排除: {before - len(big):,} 行, 剩余 {len(big):,} bar", flush=True)

    # next_open[t+1]
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)

    for N, pct_thresh, dd_thresh in [(5, 0.10, 0.05), (20, 0.20, 0.05)]:
        print(f"\n--- r{N}_strict: 未来 {N} 日涨 ≥{pct_thresh*100:.0f}% & 回撤 ≤{dd_thresh*100:.0f}% ---",
               flush=True)
        # rolling max(high) / min(low) over next N bars
        # 用 high (最高价) 和 low (最低价) 而不是 close, 更严格
        big["high_max_next"] = (big.groupby("ts_code")["high"]
                                 .shift(-1).rolling(N, min_periods=N).max()
                                 .reset_index(level=0, drop=True))
        big["low_min_next"] = (big.groupby("ts_code")["low"]
                                .shift(-1).rolling(N, min_periods=N).min()
                                .reset_index(level=0, drop=True))
        # 注意 rolling 是反向: t 时刻看 t+1...t+N 的 max/min
        # 简化: 用 future shift 配 rolling
        big[f"max_high_{N}"] = (big.groupby("ts_code")["high"].apply(
            lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
        big[f"min_low_{N}"] = (big.groupby("ts_code")["low"].apply(
            lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))

        valid = big.dropna(subset=["next_open", f"max_high_{N}", f"min_low_{N}"]).copy()
        valid["upside"] = valid[f"max_high_{N}"] / valid["next_open"] - 1
        valid["downside"] = valid[f"min_low_{N}"] / valid["next_open"] - 1

        # label 条件
        valid[f"strict_{N}"] = (valid["upside"] >= pct_thresh) & (valid["downside"] >= -dd_thresh)
        n_total = len(valid)
        n_pos = int(valid[f"strict_{N}"].sum())
        # 也算松一点: 仅涨幅 (不限回撤)
        n_pos_upside_only = int((valid["upside"] >= pct_thresh).sum())
        # 仅 回撤 ≤
        n_pos_dd_only = int((valid["downside"] >= -dd_thresh).sum())

        print(f"  有效 bar (含未来 N 日): {n_total:,}", flush=True)
        print(f"  仅涨幅 ≥{pct_thresh*100:.0f}%: {n_pos_upside_only:,} "
               f"({n_pos_upside_only/n_total*100:.2f}%)", flush=True)
        print(f"  仅回撤 ≤{dd_thresh*100:.0f}%: {n_pos_dd_only:,} "
               f"({n_pos_dd_only/n_total*100:.2f}%)", flush=True)
        print(f"  [STRICT] 严格 label (两者都满足): {n_pos:,} "
               f"({n_pos/n_total*100:.3f}%)", flush=True)
        # 跨股分布: 多少股至少有一次满足
        stock_has_pos = valid[valid[f"strict_{N}"]]["ts_code"].nunique()
        total_stocks = valid["ts_code"].nunique()
        print(f"  至少 1 次启动子的股票: {stock_has_pos:,}/{total_stocks:,} "
               f"({stock_has_pos/total_stocks*100:.1f}%)", flush=True)
        # 各股平均触发次数
        avg_per_stock = valid.groupby("ts_code")[f"strict_{N}"].sum().mean()
        print(f"  平均每股触发: {avg_per_stock:.1f} 次", flush=True)
    return big


def compute_1h_strict_labels():
    """1H 层 r5/r20 严格 label."""
    print("\n\n=== 1H ===", flush=True)
    F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
    if not F3.exists():
        print(f"!! {F3} 不存在, 跳过 1H", flush=True); return
    cols = ["ts_code", "trade_date", "trade_time", "open", "high", "low", "close"]
    big = pd.read_parquet(F3, columns=cols)
    big["trade_time"] = pd.to_datetime(big["trade_time"])
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_time"]).reset_index(drop=True)

    # ST 排除
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name"]].drop_duplicates("ts_code")
    st_set = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    before = len(big)
    big = big[~big["ts_code"].isin(st_set)].reset_index(drop=True)
    print(f"ST 排除: {before - len(big):,} 行, 剩余 {len(big):,} 1H bar", flush=True)

    # 去重 (同 ts_code 同 trade_time 应只一条)
    big = big.drop_duplicates(subset=["ts_code", "trade_time"]).reset_index(drop=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)

    for N, pct_thresh, dd_thresh in [(5, 0.05, 0.03), (20, 0.10, 0.03)]:
        print(f"\n--- r{N}_1h_strict: 未来 {N} 个 1H bar 涨 ≥{pct_thresh*100:.0f}% & "
               f"回撤 ≤{dd_thresh*100:.0f}% ---", flush=True)
        big[f"max_high_{N}"] = (big.groupby("ts_code")["high"].apply(
            lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
        big[f"min_low_{N}"] = (big.groupby("ts_code")["low"].apply(
            lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))

        valid = big.dropna(subset=["next_open", f"max_high_{N}", f"min_low_{N}"]).copy()
        valid["upside"] = valid[f"max_high_{N}"] / valid["next_open"] - 1
        valid["downside"] = valid[f"min_low_{N}"] / valid["next_open"] - 1
        valid[f"strict_{N}"] = (valid["upside"] >= pct_thresh) & (valid["downside"] >= -dd_thresh)

        n_total = len(valid)
        n_pos = int(valid[f"strict_{N}"].sum())
        n_pos_up = int((valid["upside"] >= pct_thresh).sum())
        n_pos_dd = int((valid["downside"] >= -dd_thresh).sum())

        print(f"  有效 bar: {n_total:,}", flush=True)
        print(f"  仅涨幅 ≥{pct_thresh*100:.0f}%: {n_pos_up:,} "
               f"({n_pos_up/n_total*100:.2f}%)", flush=True)
        print(f"  仅回撤 ≤{dd_thresh*100:.0f}%: {n_pos_dd:,} "
               f"({n_pos_dd/n_total*100:.2f}%)", flush=True)
        print(f"  [STRICT] 严格 label: {n_pos:,} ({n_pos/n_total*100:.3f}%)", flush=True)
        stock_has_pos = valid[valid[f"strict_{N}"]]["ts_code"].nunique()
        total_stocks = valid["ts_code"].nunique()
        print(f"  至少 1 次启动子的股票: {stock_has_pos:,}/{total_stocks:,} "
               f"({stock_has_pos/total_stocks*100:.1f}%)", flush=True)
        avg_per_stock = valid.groupby("ts_code")[f"strict_{N}"].sum().mean()
        print(f"  平均每股触发: {avg_per_stock:.1f} 次", flush=True)


def main():
    t0 = time.time()
    print(f"=== 严格 label 样本密度分析 ===\n", flush=True)
    compute_daily_strict_labels()
    compute_1h_strict_labels()
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
