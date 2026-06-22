# -*- coding: utf-8 -*-
"""OA-VP — 未列入量价族 build + Phase-1 正交筛查.

聚焦与"总动量/反转(已是控制 mom_5/20/60)"正交的子空间, 避开 21 已否假设的 reskin:
  amihud_20/60     Amihud 非流动性 = mean(|ret|/amount) (非流动性溢价; 与 size 相关→残差见真章)
  illiq_trend      amihud_20/amihud_60 (非流动性动量)
  overnight_ret_20 隔夜收益累计 (Lou-Polk-Skouras: 隔夜动量持续) — 与总动量分解正交
  intraday_ret_20  日内收益累计 (日内倾向反转)
  on_minus_in_20   隔夜−日内 收益差 (最干净的隔夜vs日内分解信号)
  volret_corr_60   量(成交量变化)价(收益)相关 (量价配合度)
  kyle_lambda_60   价格冲击 = 滚动 |ret| ~ vol 斜率代理

数据/面板: 复用 oa_vol_factors.build_panel() 的共享日线面板 (ST排除, checkpoint)。
amount 单位千元; 因子只输出在 54 月度调仓日。筛查复用 oa_screen (NW-t gate, 扣全控制残差)。
全 backward (形成日 t 只用 ≤t); 纪律 R03/R04/R05/R06/R11/R12++/R15/R16。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import oa_screen          # noqa: E402
import oa_vol_factors as ov  # noqa: E402  (复用 build_panel)

CTRL_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
OUT_P = ROOT / "research" / "features" / "oa_vp_factors.parquet"
KEY = ["ts_code", "trade_date"]
EPS = 1e-9


def compute_vp_factors(panel: pd.DataFrame, dates: set):
    panel = panel.sort_values(KEY).reset_index(drop=True)
    panel["absret"] = panel["ret"].abs()
    panel["illiq"] = panel["absret"] / (panel["amount"] + EPS)   # 单位千元
    g = panel.groupby("ts_code", sort=False)

    print("[vp] amihud / overnight / intraday 累计 ...", flush=True)
    panel["amihud_20"] = g["illiq"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    panel["amihud_60"] = g["illiq"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    panel["illiq_trend"] = panel["amihud_20"] / (panel["amihud_60"] + EPS)
    panel["overnight_ret_20"] = g["overnight"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    panel["intraday_ret_20"] = g["intraday"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    panel["on_minus_in_20"] = panel["overnight_ret_20"] - panel["intraday_ret_20"]

    print("[vp] 量价相关 + kyle lambda (向量化滚动) ...", flush=True)
    panel["dvol"] = g["vol"].transform(lambda s: s.pct_change())
    panel["dvol"] = panel["dvol"].replace([np.inf, -np.inf], np.nan)
    panel["_rd"] = panel["ret"] * panel["dvol"]
    g2 = panel.groupby("ts_code", sort=False)
    rb = lambda col: g2[col].transform(lambda s: s.rolling(60, min_periods=40).mean())
    m_r, m_d, m_rd = rb("ret"), rb("dvol"), rb("_rd")
    v_r = g2["ret"].transform(lambda s: s.rolling(60, min_periods=40).var())
    v_d = g2["dvol"].transform(lambda s: s.rolling(60, min_periods=40).var())
    cov_rd = m_rd - m_r * m_d
    panel["volret_corr_60"] = cov_rd / (np.sqrt(v_r.clip(lower=0) * v_d.clip(lower=0)) + EPS)
    # kyle lambda 代理 = |ret| 对 vol 的滚动斜率
    panel["_av"] = panel["absret"] * panel["vol"]
    m_a = g2["absret"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    m_v = g2["vol"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    m_av = g2["_av"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    v_v = g2["vol"].transform(lambda s: s.rolling(60, min_periods=40).var())
    panel["kyle_lambda_60"] = (m_av - m_a * m_v) / (v_v.replace(0, np.nan))

    factor_cols = ["amihud_20", "amihud_60", "illiq_trend", "overnight_ret_20",
                   "intraday_ret_20", "on_minus_in_20", "volret_corr_60", "kyle_lambda_60"]
    out = panel[panel["trade_date"].isin(dates)][KEY + factor_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, factor_cols


def main() -> int:
    t0 = time.time()
    print("\n=== OA-VP 量价族 build + Phase-1 筛查 ===\n", flush=True)
    panel = ov.build_panel()
    ctrl_dates = set(pd.read_parquet(CTRL_P, columns=["trade_date"])["trade_date"].astype(str).unique())
    out, factor_cols = compute_vp_factors(panel, ctrl_dates)
    oa_screen.assert_backward(factor_cols)
    OUT_P.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)} ({len(out):,} 行 × {len(factor_cols)} 因子)", flush=True)

    print("\n[screen] Phase-1 正交 IC 筛查 ...", flush=True)
    verdict = oa_screen.screen_factors(out, factor_cols, "OA-VP", primary="overnight_ret_20")
    print(f"\n[决策] 族裁决={verdict['verdict']}; 存活={verdict['survivors']}", flush=True)
    for f, c in verdict["classifications"].items():
        print(f"  {f:18s} {c['status']:10s} orth_ic={c['orth_ic']:+.4f} t={c['orth_t']} raw={c['raw_ic']:+.4f}", flush=True)
    print(f"\n[done] 耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
