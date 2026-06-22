# -*- coding: utf-8 -*-
"""OA-COM — 大宗商品/原材料供应链族 fetch + build + Phase-1 正交筛查.

供应链/投入成本渠道 (codex 排第2真正交 prior): 个股收益对大宗商品价格的暴露随行业/供应链
位置而异 → 横截面变化来自 beta (商品价格本身市场级, cross-section 恒定, 单独 IC=0)。候选:
  com_beta_basket_60   个股对等权商品篮子收益的 60日 beta (供应链暴露)
  com_beta_oil_60      对原油 SC 的 beta (能源/化工链)
  com_beta_metal_60    对工业金属(铜铝) 的 beta
  com_transmit_20      com_beta_basket_60 × 篮子20日动量 (暴露×近期商品趋势 = 传导信号)
  com_transmit_oil_20  油 beta × 油20日动量
  com_betavol_60       个股对篮子的 |残差| 代理 (商品特异冲击暴露)

商品 (主力连续, settle/pre_settle 收益): RB螺纹/HC热卷/CU铜/AL铝/AU金/SC原油/I铁矿/J焦炭/TA-PTA/M豆粕。
数据: pro.fut_daily 一次 range 调全历史 → output/tushare_cache/fut_daily/*.parquet (checkpoint)。
个股面板复用 oa_daily_panel.parquet。因子只输出 54 月度调仓日。全 backward。
筛查复用 oa_screen (NW-t gate)。纪律 R03/R04/R05/R06/R08/R11/R12++/R15/R16。
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import oa_screen          # noqa: E402
import oa_vol_factors as ov  # noqa: E402

FUT_DIR = ROOT / "output" / "tushare_cache" / "fut_daily"
CTRL_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
OUT_P = ROOT / "research" / "features" / "oa_com_factors.parquet"
KEY = ["ts_code", "trade_date"]

COMMODITIES = {
    "RB.SHF": "螺纹钢", "HC.SHF": "热卷", "CU.SHF": "铜", "AL.SHF": "铝",
    "AU.SHF": "黄金", "SC.INE": "原油", "I.DCE": "铁矿", "J.DCE": "焦炭",
    "TA.CZC": "PTA", "M.DCE": "豆粕",
}
OIL = ["SC.INE"]
METAL = ["CU.SHF", "AL.SHF"]
DATA_START, DATA_END = "20211001", "20260610"
W = 60
EPS = 1e-12


def _pro():
    if not os.environ.get("TUSHARE_TOKEN"):
        for line in open(ROOT / ".env", encoding="utf-8"):
            if line.startswith("TUSHARE_TOKEN="):
                os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
    import tushare as ts
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    return ts.pro_api()


def fetch_commodities() -> pd.DataFrame:
    """每商品 range 调一次 fut_daily, settle 收益; checkpoint 每商品一 parquet。"""
    FUT_DIR.mkdir(parents=True, exist_ok=True)
    pro = None
    frames = []
    for code in COMMODITIES:
        fp = FUT_DIR / f"{code.replace('.', '_')}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
        else:
            if pro is None:
                pro = _pro()
            for attempt in range(3):
                try:
                    df = pro.fut_daily(ts_code=code, start_date=DATA_START, end_date=DATA_END)
                    break
                except Exception as e:
                    print(f"  {code} 第{attempt+1}次失败: {repr(e)[:100]}", flush=True)
                    time.sleep(20)
            else:
                print(f"  {code} 放弃", flush=True)
                continue
            df = df[["ts_code", "trade_date", "pre_settle", "settle"]].copy()
            df.to_parquet(fp, index=False)
            time.sleep(0.6)
        df["trade_date"] = df["trade_date"].astype(str)
        df["cret"] = df["settle"] / df["pre_settle"].replace(0, np.nan) - 1.0
        frames.append(df[["ts_code", "trade_date", "cret"]])
        print(f"  {code} ({COMMODITIES[code]}): {len(df)} 日", flush=True)
    allc = pd.concat(frames, ignore_index=True)
    return allc


def build_commodity_series(allc: pd.DataFrame) -> pd.DataFrame:
    """按 trade_date 透视 → 等权篮子 / 油 / 金属 日收益。"""
    wide = allc.pivot_table(index="trade_date", columns="ts_code", values="cret")
    wide = wide.sort_index()
    basket = wide.mean(axis=1, skipna=True)
    oil = wide[[c for c in OIL if c in wide.columns]].mean(axis=1, skipna=True)
    metal = wide[[c for c in METAL if c in wide.columns]].mean(axis=1, skipna=True)
    com = pd.DataFrame({"trade_date": wide.index, "basket_ret": basket.values,
                        "oil_ret": oil.values, "metal_ret": metal.values})
    # 篮子 20日动量 (backward)
    com["basket_mom_20"] = com["basket_ret"].rolling(20, min_periods=10).sum()
    com["oil_mom_20"] = com["oil_ret"].rolling(20, min_periods=10).sum()
    return com.reset_index(drop=True)


def _roll_beta(panel, retcol, comcol, name):
    """个股 ret 对商品收益的向量化滚动 beta = cov/var。"""
    panel["_x"] = panel[retcol] * panel[comcol]
    g = panel.groupby("ts_code", sort=False)
    rb = lambda c: g[c].transform(lambda s: s.rolling(W, min_periods=40).mean())
    m_r, m_c, m_x = rb(retcol), rb(comcol), rb("_x")
    v_c = g[comcol].transform(lambda s: s.rolling(W, min_periods=40).var())
    cov = m_x - m_r * m_c
    panel[name] = cov / v_c.replace(0, np.nan)
    panel.drop(columns="_x", inplace=True)
    return panel


def compute_com_factors(panel, com, dates):
    panel = panel.sort_values(KEY).reset_index(drop=True)
    panel = panel.merge(com, on="trade_date", how="left")
    panel = _roll_beta(panel, "ret", "basket_ret", "com_beta_basket_60")
    panel = _roll_beta(panel, "ret", "oil_ret", "com_beta_oil_60")
    panel = _roll_beta(panel, "ret", "metal_ret", "com_beta_metal_60")
    panel["com_transmit_20"] = panel["com_beta_basket_60"] * panel["basket_mom_20"]
    panel["com_transmit_oil_20"] = panel["com_beta_oil_60"] * panel["oil_mom_20"]
    # 商品特异冲击暴露代理: |个股残差对篮子| 的滚动 std ~ 用 (ret - beta*basket) 的滚动std
    panel["_resid"] = panel["ret"] - panel["com_beta_basket_60"] * panel["basket_ret"]
    g = panel.groupby("ts_code", sort=False)
    panel["com_betavol_60"] = g["_resid"].transform(lambda s: s.rolling(W, min_periods=40).std())

    factor_cols = ["com_beta_basket_60", "com_beta_oil_60", "com_beta_metal_60",
                   "com_transmit_20", "com_transmit_oil_20", "com_betavol_60"]
    out = panel[panel["trade_date"].isin(dates)][KEY + factor_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, factor_cols


def main() -> int:
    t0 = time.time()
    print("\n=== OA-COM 大宗商品/供应链族 fetch+build+筛查 ===\n", flush=True)
    print("[fetch] 商品期货 fut_daily ...", flush=True)
    allc = fetch_commodities()
    com = build_commodity_series(allc)
    print(f"[com] 商品序列 {len(com)} 日 ({com['trade_date'].min()}~{com['trade_date'].max()})", flush=True)

    panel = ov.build_panel()
    ctrl_dates = set(pd.read_parquet(CTRL_P, columns=["trade_date"])["trade_date"].astype(str).unique())
    out, factor_cols = compute_com_factors(panel, com, ctrl_dates)
    oa_screen.assert_backward(factor_cols)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)} ({len(out):,} 行 × {len(factor_cols)} 因子)", flush=True)

    print("\n[screen] Phase-1 正交 IC 筛查 ...", flush=True)
    verdict = oa_screen.screen_factors(out, factor_cols, "OA-COM", primary="com_transmit_20")
    print(f"\n[决策] 族裁决={verdict['verdict']}; 存活={verdict['survivors']}", flush=True)
    for f, c in verdict["classifications"].items():
        print(f"  {f:20s} {c['status']:10s} orth_ic={c['orth_ic']:+.4f} t={c['orth_t']} raw={c['raw_ic']:+.4f}", flush=True)
    print(f"\n[done] 耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
