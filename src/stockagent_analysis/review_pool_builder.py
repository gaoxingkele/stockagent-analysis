"""池 0 看走眼 (反思学习池) 构建器.

三类典型 V12 误判:
  A. False Positive: V7c 主推, 实际 N 日跑输 hs300 >3pp
  B. Missed Rocket: 非 V7c 主推, 实际 N 日涨幅 > 15%
  C. LLM Reverse: V11 视觉 bull < 0.30, 实际 N 日涨幅 > 10%

输入:
  - output/v7c_full_inference/v7c_inference_{snap_date}.csv (V12 评分快照)
  - output/tushare_cache/daily/{date}.parquet (价格)
  - output/v12_inference/v11_*.csv (V11 视觉结果, 可选)

输出:
  - output/v12_review/wrongs_{snap_date}.parquet
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"
V7C_DIR = ROOT / "output" / "v7c_full_inference"
V11_DIR = ROOT / "output" / "v12_inference"  # v11_filter_results / v11_semi_top50 等
OUT_DIR = ROOT / "output" / "v12_review"

# 误判判据阈值
TYPE_A_THRESHOLD = -3.0  # V7c 推荐, 但 alpha < -3pp
TYPE_B_THRESHOLD = 15.0  # 非推荐, 但实际涨 > +15%
TYPE_C_THRESHOLD = 10.0  # V11 看空, 但实际涨 > +10%
TYPE_C_BULL_MAX = 0.30   # V11 bull_prob 上限


def _load_close(date: str) -> dict[str, float]:
    """ts_code -> close."""
    f = DAILY_CACHE / f"{date}.parquet"
    if not f.exists(): return {}
    df = pd.read_parquet(f)[["ts_code", "close"]]
    return dict(zip(df["ts_code"].astype(str), df["close"].astype(float)))


def _load_hs300_ret(start: str, end: str) -> float:
    """hs300 同期累计涨幅 (%)."""
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import tushare as ts
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    pro = ts.pro_api()
    df = pro.index_daily(ts_code="000300.SH", start_date=start, end_date=end)
    if df is None or df.empty: return 0.0
    df = df.sort_values("trade_date")
    p0 = float(df.iloc[0]["close"]); p1 = float(df.iloc[-1]["close"])
    return (p1 / p0 - 1) * 100


def _load_v11_bull(date: str) -> dict[str, float]:
    """ts_code -> bull_prob (V11 视觉结果, 可选)."""
    out = {}
    candidates = [
        V11_DIR / f"v11_semi_top50_{date}.csv",
        V11_DIR / f"v11_filter_results_{date}.csv",
        V11_DIR / f"v11_user_list_{date}.csv",
    ]
    for p in candidates:
        if not p.exists(): continue
        df = pd.read_csv(p, dtype={"ts_code": str})
        if "bull_prob" not in df.columns: continue
        for _, r in df.iterrows():
            ts = r["ts_code"]
            bp = r["bull_prob"]
            if pd.notna(bp):
                out[ts] = float(bp) if ts not in out else min(out[ts], float(bp))
    return out


def find_v12_wrongs(snap_date: str, end_date: str,
                     basic_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """找 snap_date 时点 V12 在 snap_date → end_date 期间看走眼的股.

    返回 DataFrame 含:
      ts_code, name, snap_date, end_date, holding_days,
      buy_score, sell_score, r20_pred, v7c_recommend, quadrant,
      real_ret_pct, hs300_ret_pct, alpha,
      wrong_type ('false_positive' / 'missed_rocket' / 'llm_reverse'),
      gap (实际 - 预期),
      v11_bull_prob (如有)
    """
    v7c_csv = V7C_DIR / f"v7c_inference_{snap_date}.csv"
    if not v7c_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(v7c_csv, dtype={"ts_code": str})

    p_snap = _load_close(snap_date)
    p_end = _load_close(end_date)
    if not p_snap or not p_end:
        return pd.DataFrame()

    # 价格 + 实际涨幅
    df["close_snap"] = df["ts_code"].map(p_snap)
    df["close_end"] = df["ts_code"].map(p_end)
    df = df.dropna(subset=["close_snap", "close_end"])
    df["real_ret_pct"] = (df["close_end"] / df["close_snap"] - 1) * 100

    # hs300 基准
    hs_ret = _load_hs300_ret(snap_date, end_date)
    df["hs300_ret_pct"] = hs_ret
    df["alpha"] = df["real_ret_pct"] - hs_ret

    # 交易日数
    daily_files = sorted([f.stem for f in DAILY_CACHE.glob("*.parquet")])
    holding_days = sum(1 for d in daily_files if snap_date < d <= end_date)
    df["holding_days"] = holding_days

    # V11 bull_prob (可选)
    v11_bull = _load_v11_bull(snap_date)
    df["v11_bull_prob"] = df["ts_code"].map(v11_bull)

    # 三类误判
    type_a_mask = (df["v7c_recommend"] == True) & (df["alpha"] < TYPE_A_THRESHOLD)
    type_b_mask = (df["v7c_recommend"] != True) & (df["real_ret_pct"] > TYPE_B_THRESHOLD)
    type_c_mask = (df["v11_bull_prob"].notna() &
                    (df["v11_bull_prob"] < TYPE_C_BULL_MAX) &
                    (df["real_ret_pct"] > TYPE_C_THRESHOLD))

    df["wrong_type"] = None
    df.loc[type_a_mask, "wrong_type"] = "false_positive"
    df.loc[type_b_mask, "wrong_type"] = "missed_rocket"
    df.loc[type_c_mask, "wrong_type"] = "llm_reverse"
    # gap = 实际 - 预测 (>0 表示低估)
    df["gap"] = df["real_ret_pct"] - df["r20_pred"]

    # 加 name
    if basic_df is None:
        basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
        if basic_p.exists():
            basic_df = pd.read_parquet(basic_p)[["ts_code", "name"]]
    if basic_df is not None:
        df = df.merge(basic_df, on="ts_code", how="left")

    df["snap_date"] = snap_date
    df["end_date"] = end_date
    wrongs = df[df["wrong_type"].notna()].copy()
    return wrongs.sort_values("gap", ascending=False).reset_index(drop=True)


def scan_history(snap_dates: list[str], end_date: str,
                  save: bool = True) -> pd.DataFrame:
    """批量扫描多个评测日."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_parts = []
    for sd in snap_dates:
        w = find_v12_wrongs(sd, end_date)
        if len(w) == 0: continue
        if save:
            w.to_parquet(OUT_DIR / f"wrongs_{sd}.parquet", index=False)
        all_parts.append(w)
        print(f"  {sd} → {end_date} ({len(w)} 看走眼)")
    if not all_parts:
        return pd.DataFrame()
    big = pd.concat(all_parts, ignore_index=True)
    if save:
        big.to_csv(OUT_DIR / "review_pool_all.csv", index=False, encoding="utf-8-sig")
    return big


def post_mortem_stats(wrongs: pd.DataFrame) -> dict:
    """池 0 模式总结: 哪些行业/regime/quadrant 错最多?"""
    out = {
        "n_total": len(wrongs),
        "by_type": wrongs.groupby("wrong_type").size().to_dict(),
        "top_industries": wrongs.groupby("industry").size().sort_values(ascending=False).head(10).to_dict(),
        "top_quadrants": wrongs.groupby("quadrant").size().to_dict(),
        "type_a_avg_alpha": wrongs[wrongs["wrong_type"]=="false_positive"]["alpha"].mean() if "false_positive" in wrongs["wrong_type"].values else None,
        "type_b_avg_real_ret": wrongs[wrongs["wrong_type"]=="missed_rocket"]["real_ret_pct"].mean() if "missed_rocket" in wrongs["wrong_type"].values else None,
        "type_b_top10_gap": wrongs[wrongs["wrong_type"]=="missed_rocket"].nlargest(10, "gap")[["ts_code","name","real_ret_pct","r20_pred","gap","industry"]].to_dict(orient="records"),
    }
    return out
