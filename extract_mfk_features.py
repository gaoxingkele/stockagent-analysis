#!/usr/bin/env python3
"""资金流 K 线风格 17 因子 (mfk_* = moneyflow K-line).

设计: 把 main_net 当成"价格", 围绕它构造 MA / 金叉 / MACD / RSI / 同步性 / 速度 体系.

6 层 17 因子:
  Layer A 趋势均线 (3): mfk_main_ma5, _ma20, _ma_diff
  Layer B 金叉死叉 (3): _cross_state, _days_in_cross, _cross_strength
  Layer C MACD (3):    _macd, _macd_hist, _macd_zero_dist
  Layer D 多空对比 (3): _smart_dumb_spread, _pyramid_top_heavy, _inst_acc_velocity
  Layer E 量价同步 (2): _main_price_sync_20d, _main_lead_5d
  Layer F 速度强度 (3): _main_velocity_5d, _main_consec_above_ma20, _main_acc_ratio_60d

输入: output/moneyflow/cache/{ts_code}.parquet + TDX OHLC
输出: output/mfk_features/features.parquet
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "output" / "moneyflow" / "cache"
OUT_DIR = ROOT / "output" / "mfk_features"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")
START = "20230101"
END   = "20260420"

THOUSAND_TO_YI = 1.0 / 1e5


def read_tdx(market, code):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    try: data = p.read_bytes()
    except: return None
    n = len(data) // 32
    if n == 0: return None
    rows = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append((d.strftime("%Y%m%d"), f[4]/100.0))  # date, close
    return rows


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def rsi14(s: pd.Series) -> pd.Series:
    """简化 RSI14: 上涨日均 / (上涨日均 + 下跌日均) * 100"""
    delta = s.diff()
    up = delta.where(delta > 0, 0).rolling(14, min_periods=7).mean()
    dn = (-delta.where(delta < 0, 0)).rolling(14, min_periods=7).mean()
    return 100 * up / (up + dn + 1e-9)


def compute_mfk(mf: pd.DataFrame, ohlc) -> pd.DataFrame:
    """单股 17 因子."""
    if mf.empty or not ohlc: return pd.DataFrame()
    mf = mf.sort_values("trade_date").reset_index(drop=True)
    px = pd.DataFrame(ohlc, columns=["trade_date","close"])
    df = mf.merge(px, on="trade_date", how="left")
    if df.empty: return pd.DataFrame()

    # 主力 / 散户 / 分层 (亿元)
    main = ((df["buy_lg_amount"] + df["buy_elg_amount"]
             - df["sell_lg_amount"] - df["sell_elg_amount"]) * THOUSAND_TO_YI)
    sm   = ((df["buy_sm_amount"] - df["sell_sm_amount"]) * THOUSAND_TO_YI)
    inst_buy = (df["buy_lg_amount"] + df["buy_elg_amount"]) * THOUSAND_TO_YI
    inst_total = (df["buy_lg_amount"] + df["buy_elg_amount"]
                   + df["sell_lg_amount"] + df["sell_elg_amount"]) * THOUSAND_TO_YI
    retail_total = (df["buy_sm_amount"] + df["sell_sm_amount"]
                     + df["buy_md_amount"] + df["sell_md_amount"]) * THOUSAND_TO_YI

    out = pd.DataFrame({"ts_code": mf["ts_code"], "trade_date": df["trade_date"]})

    # ── Layer A: 趋势均线 ──
    ma5  = main.rolling(5,  min_periods=3).mean()
    ma20 = main.rolling(20, min_periods=10).mean()
    out["mfk_main_ma5"]      = ma5
    out["mfk_main_ma20"]     = ma20
    out["mfk_main_ma_diff"]  = ma5 - ma20

    # ── Layer B: 金叉死叉 ──
    cross_state = np.sign(ma5 - ma20)
    out["mfk_main_cross_state"] = cross_state
    # days_in_cross: 当前态持续天数
    days = []
    cur = cnt = 0
    for v in cross_state:
        if pd.isna(v):
            days.append(np.nan); continue
        v = int(v)
        if v == cur: cnt += 1
        else: cur = v; cnt = 1
        days.append(cnt)
    out["mfk_main_days_in_cross"] = days
    out["mfk_main_cross_strength"] = (ma5 - ma20) / (ma20.abs() + 1e-3)

    # ── Layer C: MACD ──
    ema12 = main.ewm(span=12, adjust=False).mean()
    ema26 = main.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["mfk_main_macd"]      = macd
    out["mfk_main_macd_hist"] = macd - signal
    macd_std60 = macd.rolling(60, min_periods=20).std()
    out["mfk_main_macd_zero_dist"] = macd / (macd_std60 + 1e-6)

    # ── Layer D: 多空对比 ──
    out["mfk_smart_dumb_spread"] = rsi14(main) - rsi14(sm)
    inst_5d = inst_total.rolling(5, min_periods=3).sum()
    retail_5d = retail_total.rolling(5, min_periods=3).sum()
    out["mfk_pyramid_top_heavy"] = inst_5d / (retail_5d + 1e-3)
    inst_buy_5d = inst_buy.rolling(5, min_periods=3).sum()
    inst_buy_20d = inst_buy.rolling(20, min_periods=10).sum()
    out["mfk_inst_acc_velocity"] = (inst_buy_5d * 4) / (inst_buy_20d + 1e-3)

    # ── Layer E: 量价同步 ──
    close_ma5 = df["close"].rolling(5, min_periods=3).mean()
    out["mfk_main_price_sync_20d"] = ma5.rolling(20, min_periods=10).corr(close_ma5)
    # 资金 5日前 vs 价格变化 滞后相关
    main_lag5 = main.shift(5)
    close_pct = df["close"].pct_change()
    out["mfk_main_lead_5d"] = main_lag5.rolling(20, min_periods=10).corr(close_pct)

    # ── Layer F: 速度强度 ──
    out["mfk_main_velocity_5d"] = (ma5 - ma5.shift(5))
    # 连续在 ma20 上方的天数
    above_ma20 = (main > ma20).astype(int)
    consec = []; cnt = 0
    for v in above_ma20:
        if pd.isna(v): consec.append(np.nan); continue
        cnt = cnt + 1 if v else 0
        consec.append(cnt)
    out["mfk_main_consec_above_ma20"] = consec
    # 60d 分位
    def _rank60(s):
        return s.rolling(60, min_periods=20).apply(
            lambda x: (x[-1] >= x).mean() if not np.isnan(x[-1]) else np.nan, raw=True)
    out["mfk_main_acc_ratio_60d"] = _rank60(main.rolling(5, min_periods=3).sum())

    # 截区间 + ts_code 修正
    out = out[(out["trade_date"] >= START) & (out["trade_date"] <= END)].copy()
    out["ts_code"] = mf["ts_code"].iloc[0]
    return out


def main():
    t0 = time.time()
    cache_files = sorted(CACHE_DIR.glob("*.parquet"))
    print(f"moneyflow cache: {len(cache_files)} 文件")

    all_rows = []
    n_ok = n_skip = 0
    for i, p in enumerate(cache_files):
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(cache_files)}] {int(time.time()-t0)}s, "
                  f"ok={n_ok}, skip={n_skip}, rows={len(all_rows):,}")
        ts = p.stem
        c, m = code_market(ts)
        ohlc = read_tdx(m, c)
        if not ohlc: n_skip += 1; continue
        try:
            mf = pd.read_parquet(p)
            mf["trade_date"] = mf["trade_date"].astype(str)
        except: n_skip += 1; continue
        if mf.empty: n_skip += 1; continue
        feat = compute_mfk(mf, ohlc)
        if feat.empty: n_skip += 1; continue
        all_rows.append(feat)
        n_ok += 1

    if not all_rows:
        print("没有数据!"); return
    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(OUT_DIR / "features.parquet", index=False)
    print(f"\nfeatures: {len(full):,} 行 × {full['ts_code'].nunique()} 股, ok={n_ok}, skip={n_skip}")

    print("\n=== 17 因子分布 ===")
    feat_cols = [c for c in full.columns if c.startswith("mfk_")]
    print(full[feat_cols].describe(percentiles=[0.05,0.5,0.95]).round(4).to_string())
    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
