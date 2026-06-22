# -*- coding: utf-8 -*-
"""OA-VOL — 波动率因子族 build + Phase-1 正交筛查.

候选 (全 backward, 形成日 t 只用 ≤t 信息; 前向 r20 才是 label):
  ivol_60       特异波动率 = 个股日收益对市场收益(等权全市场均值)回归残差的 60日 std (低风险异象)
  total_vol_60  已实现总波动 = 日收益 60日 std (低波动异象: 预期 IC<0)
  beta_60       市场 beta (60日 OLS 斜率)
  max5_20       MAX/彩票效应 = 过去20日最大单日 pct_chg (预期 IC<0, 彩票偏好被高估)
  volofvol_60   波动的波动 = 20日滚动 vol 在过去60日的 std
  rskew_60      已实现偏度 (60日日收益; 预期负偏更危险)
  rkurt_60      已实现峰度 (60日)
  overnight_vol_60  隔夜收益(open/pre_close-1) 60日 std
  intraday_vol_60   日内收益(close/open-1) 60日 std
  on_in_ratio   隔夜/日内 波动比

数据: output/tushare_cache/daily/*.parquet (1075 交易日); 市场代理 = 当日全市场等权收益均值。
ST 源头排除 (SIGN-R06, stock_basic). 共享日线面板缓存 research/cache/oa_daily_panel.parquet
(VOL/VP 族复用; checkpoint SIGN-R08). 因子只输出在 54 个月度调仓日 (与控制面板对齐, 小而快)。

筛查: 复用 research/oa_screen.py (扣 FULL_CTRL 残差 IC + 消融 + 分regime + A/B/C/D, NW-t gate)。
纪律: R03/R04/R05/R06/R11/R12++/R15/R16 (见 oa_screen + guardrails)。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import oa_screen  # noqa: E402

DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
PANEL_CACHE = ROOT / "research" / "cache" / "oa_daily_panel.parquet"
CTRL_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
OUT_P = ROOT / "research" / "features" / "oa_vol_factors.parquet"

KEY = ["ts_code", "trade_date"]
W = 60          # 主窗口
W_MAX = 20      # MAX 效应窗口
MIN_OBS = 40    # 窗口内最少有效观测


def load_st_set() -> set:
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def build_panel() -> pd.DataFrame:
    """全历史日线面板 (ts_code, trade_date, ret, overnight, intraday, amount, vol); ST排除; 缓存复用。"""
    if PANEL_CACHE.exists():
        print(f"[panel] checkpoint 命中 {PANEL_CACHE.name}", flush=True)
        return pd.read_parquet(PANEL_CACHE)
    st = load_st_set()
    print(f"[st] ST 源头排除 {len(st)} 只", flush=True)
    files = sorted(DAILY_DIR.glob("*.parquet"))
    print(f"[panel] 读取 {len(files)} 个日线文件 ...", flush=True)
    cols = ["ts_code", "trade_date", "open", "pre_close", "close", "pct_chg", "amount", "vol"]
    parts = []
    t0 = time.time()
    for i, f in enumerate(files):
        d = pd.read_parquet(f, columns=cols)
        parts.append(d)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big[~big["ts_code"].isin(st)]
    big = big[(big["close"] > 0) & (big["pre_close"] > 0)]
    big = big.drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True)
    big["ret"] = big["close"] / big["pre_close"] - 1.0
    big["overnight"] = big["open"] / big["pre_close"] - 1.0
    big["intraday"] = big["close"] / big["open"].replace(0, np.nan) - 1.0
    big = big[["ts_code", "trade_date", "ret", "overnight", "intraday", "amount", "vol"]]
    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    big.to_parquet(PANEL_CACHE, index=False)
    print(f"[panel] 落盘 {PANEL_CACHE.name} ({len(big):,} 行 / {big['ts_code'].nunique()} 股, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return big


def compute_vol_factors(panel: pd.DataFrame, dates: set) -> pd.DataFrame:
    """滚动波动率因子; 只在 rebalance dates 输出。全 backward。"""
    panel = panel.sort_values(KEY).reset_index(drop=True)
    # 市场代理 = 当日全市场等权收益均值
    mkt = panel.groupby("trade_date")["ret"].mean().rename("mkt")
    panel = panel.merge(mkt, on="trade_date", how="left")

    g = panel.groupby("ts_code", sort=False)
    print("[vol] 滚动简单因子 (total_vol/max5/skew/kurt/overnight/intraday) ...", flush=True)
    panel["total_vol_60"] = g["ret"].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).std())
    panel["max5_20"] = g["ret"].transform(lambda s: s.rolling(W_MAX, min_periods=10).max())
    panel["rskew_60"] = g["ret"].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).skew())
    panel["rkurt_60"] = g["ret"].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).kurt())
    panel["overnight_vol_60"] = g["overnight"].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).std())
    panel["intraday_vol_60"] = g["intraday"].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).std())
    panel["on_in_ratio"] = panel["overnight_vol_60"] / (panel["intraday_vol_60"] + 1e-12)
    # vol-of-vol: 20日滚动vol的60日std
    vol20 = g["ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    panel["_vol20"] = vol20
    panel["volofvol_60"] = panel.groupby("ts_code")["_vol20"].transform(
        lambda s: s.rolling(W, min_periods=MIN_OBS).std())

    print("[vol] 向量化滚动 beta + ivol (cov/var 公式, 无 Python 内循环) ...", flush=True)
    t0 = time.time()
    panel["_rm"] = panel["ret"] * panel["mkt"]
    g2 = panel.groupby("ts_code", sort=False)
    rb = lambda col: g2[col].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).mean())
    m_r, m_m = rb("ret"), rb("mkt")
    m_rm = rb("_rm")
    v_m = g2["mkt"].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).var())
    v_r = g2["ret"].transform(lambda s: s.rolling(W, min_periods=MIN_OBS).var())
    cov = m_rm - m_r * m_m                      # 总体协方差近似 (用于 beta/ivol 排序, 量纲无关)
    beta = cov / v_m.replace(0, np.nan)
    ivol_var = (v_r - cov * cov / v_m.replace(0, np.nan)).clip(lower=0)
    panel["beta_60"] = beta
    panel["ivol_60"] = np.sqrt(ivol_var)
    print(f"[vol] beta/ivol 完成 ({time.time()-t0:.0f}s)", flush=True)

    out = panel[panel["trade_date"].isin(dates)].copy()
    factor_cols = ["ivol_60", "total_vol_60", "beta_60", "max5_20", "volofvol_60",
                   "rskew_60", "rkurt_60", "overnight_vol_60", "intraday_vol_60", "on_in_ratio"]
    out = out[KEY + factor_cols]
    return out, factor_cols


def main() -> int:
    t0 = time.time()
    print("\n=== OA-VOL 波动率因子 build + Phase-1 筛查 ===\n", flush=True)
    panel = build_panel()
    ctrl_dates = set(pd.read_parquet(CTRL_P, columns=["trade_date"])["trade_date"].astype(str).unique())
    print(f"[dates] 控制面板 {len(ctrl_dates)} 个月度调仓日", flush=True)

    out, factor_cols = compute_vol_factors(panel, ctrl_dates)
    oa_screen.assert_backward(factor_cols)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)} ({len(out):,} 行 × {len(factor_cols)} 因子)", flush=True)

    print("\n[screen] Phase-1 正交 IC 筛查 ...", flush=True)
    verdict = oa_screen.screen_factors(out, factor_cols, "OA-VOL", primary="ivol_60")
    print(f"\n[决策] 族裁决={verdict['verdict']}; 存活={verdict['survivors']}", flush=True)
    for f, c in verdict["classifications"].items():
        print(f"  {f:18s} {c['status']:10s} orth_ic={c['orth_ic']:+.4f} t={c['orth_t']} raw={c['raw_ic']:+.4f}", flush=True)
    print(f"\n[done] 耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
