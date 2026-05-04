#!/usr/bin/env python3
"""V3 候选: 8 个 moneyflow 价量耦合因子 (突破 IC 0.07 天花板的尝试).

核心洞察: 现有 moneyflow 13 因子只算资金净额, 没和股价变动耦合.
CogAlpha 真正巧思就是价量耦合 (impact = price/amount).
Tushare Pro 4-tier 资金分层 + 价格联动 = 理论上能突破天花板.

8 个新因子:
  1. main_in_on_red_day_5d   : 大跌日(<-3%)主力净流入 5日累计 (亿) — 强吸筹+
  2. main_out_on_green_day_5d: 大涨日(>+3%)主力净流出 5日累计 (亿) — 派发-
  3. main_follow_breakout    : 突破日(20日新高)前后 5 日主力净流入累计
  4. dispersion_price_align  : (主力 vs 散户分歧度) × sign(5日涨幅)
  5. main_acc_velocity_60d   : main_net_5d 在 60d 历史的分位排名
  6. inst_buy_pressure_5d    : (buy_lg+buy_elg)/(sell_lg+sell_elg) 5 日均
  7. main_consec_silent_acc  : 主力连续流入 × max(0, 5-|累计涨幅|) — 静默吸筹
  8. quiet_distribution      : 主力连续流出 × max(0, 5-|累计跌幅|) — 静默派发

输入:
  output/moneyflow/cache/{ts_code}.parquet
  D:/tdx/vipdoc/{sh,sz,bj}/lday/{ts_code}.day

输出:
  output/moneyflow_v2/features.parquet (~3.6M 行)
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "output" / "moneyflow" / "cache"
OUT_DIR = ROOT / "output" / "moneyflow_v2"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")
START = "20230101"
END   = "20260126"

THOUSAND_TO_YI = 1.0 / 1e5  # tushare amount 千元 → 亿元


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
        rows.append((d.strftime("%Y%m%d"),
                     f[1]/100.0, f[2]/100.0, f[3]/100.0, f[4]/100.0))
    return rows


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def compute_v2_features(mf: pd.DataFrame, ohlc: list) -> pd.DataFrame:
    """单股计算 8 个 V2 因子.
    mf: moneyflow raw (按 trade_date 升序)
    ohlc: list of (date, open, high, low, close)
    """
    if mf.empty or not ohlc: return pd.DataFrame()
    mf = mf.sort_values("trade_date").reset_index(drop=True)

    # 价格 dataframe + 收益率
    px = pd.DataFrame(ohlc, columns=["trade_date","open","high","low","close"])
    px["ret_pct"] = px["close"].pct_change() * 100

    # 对齐 mf
    df = mf.merge(px[["trade_date","close","high","ret_pct"]],
                   on="trade_date", how="left")
    if df.empty: return pd.DataFrame()

    # 主力 + 散户净流入 (亿元)
    df["main_net"] = ((df["buy_lg_amount"] + df["buy_elg_amount"]
                       - df["sell_lg_amount"] - df["sell_elg_amount"])
                       * THOUSAND_TO_YI)
    df["sm_net"]   = ((df["buy_sm_amount"] - df["sell_sm_amount"])
                       * THOUSAND_TO_YI)

    # ── 因子 1: main_in_on_red_day_5d ──
    is_red = (df["ret_pct"] < -3).astype(float)
    df["f1_main_in_red"] = (df["main_net"] * is_red).rolling(5, min_periods=3).sum()

    # ── 因子 2: main_out_on_green_day_5d (取负值, 越大表示派发越严重) ──
    is_green = (df["ret_pct"] > 3).astype(float)
    df["f2_main_out_green"] = (-df["main_net"] * is_green).rolling(5, min_periods=3).sum()

    # ── 因子 3: main_follow_breakout ──
    # 简化: 当前 close 是否 ≥ 20日 close 最大值(突破), 是则取最近5日 main_net 累计
    rolling_max20 = df["close"].rolling(20, min_periods=10).max()
    is_breakout = (df["close"] >= rolling_max20).astype(float)
    df["f3_main_follow_breakout"] = (df["main_net"].rolling(5, min_periods=3).sum()
                                       * is_breakout)

    # ── 因子 4: dispersion_price_align ──
    main_net_5d = df["main_net"].rolling(5, min_periods=3).sum()
    sm_net_5d   = df["sm_net"].rolling(5, min_periods=3).sum()
    ret_5d      = df["ret_pct"].rolling(5, min_periods=3).sum()
    # +2 = 主力强 散户弱 上涨 (强吸筹后突破); -2 = 主力弱 散户强 下跌 (踩踏)
    df["f4_dispersion_price_align"] = (
        (np.sign(main_net_5d) - np.sign(sm_net_5d)) * np.sign(ret_5d)
    )

    # ── 因子 5: main_acc_velocity_60d ──
    # main_net_5d 在 60 天历史的分位 (0-1)
    def _rolling_rank(s, window=60):
        return s.rolling(window, min_periods=20).apply(
            lambda x: (x[-1] >= x).mean() if not np.isnan(x[-1]) else np.nan,
            raw=True
        )
    df["f5_main_acc_velocity_60d"] = _rolling_rank(main_net_5d, 60)

    # ── 因子 6: inst_buy_pressure_5d ──
    buy_inst = (df["buy_lg_amount"] + df["buy_elg_amount"]).rolling(5, min_periods=3).sum()
    sell_inst = (df["sell_lg_amount"] + df["sell_elg_amount"]).rolling(5, min_periods=3).sum()
    df["f6_inst_buy_pressure"] = (buy_inst / sell_inst.replace(0, np.nan)).clip(0, 5)

    # ── 因子 7: main_consec_silent_acc (静默吸筹) ──
    # 重新算连续流入天数
    main_pos = (df["main_net"] > 0).astype(int)
    consec_in = []
    cnt = 0
    for v in main_pos:
        cnt = cnt + 1 if v else 0
        consec_in.append(cnt)
    df["consec_in"] = consec_in
    # 静默 = 涨幅小 -> max(0, 5 - |ret_5d|), |ret| 越小因子越大
    silent_factor = np.maximum(0, 5 - ret_5d.abs())
    df["f7_silent_accumulation"] = df["consec_in"] * silent_factor

    # ── 因子 8: quiet_distribution (静默派发) ──
    main_neg = (df["main_net"] < 0).astype(int)
    consec_out = []
    cnt = 0
    for v in main_neg:
        cnt = cnt + 1 if v else 0
        consec_out.append(cnt)
    df["consec_out"] = consec_out
    df["f8_quiet_distribution"] = df["consec_out"] * silent_factor

    keep = ["ts_code","trade_date",
             "f1_main_in_red","f2_main_out_green",
             "f3_main_follow_breakout","f4_dispersion_price_align",
             "f5_main_acc_velocity_60d","f6_inst_buy_pressure",
             "f7_silent_accumulation","f8_quiet_distribution"]
    out = df[keep].copy()
    out["ts_code"] = mf["ts_code"].iloc[0]
    return out


def main():
    t0 = time.time()
    cache_files = sorted(CACHE_DIR.glob("*.parquet"))
    print(f"moneyflow cache 文件数: {len(cache_files)}")

    all_rows = []
    n_ok = n_skip = 0
    for i, p in enumerate(cache_files):
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(cache_files)}] {int(time.time()-t0)}s, "
                  f"ok={n_ok}, skip={n_skip}, rows={len(all_rows):,}")
        ts = p.stem
        c, m = code_market(ts)
        ohlc = read_tdx(m, c)
        if not ohlc:
            n_skip += 1; continue
        try:
            mf = pd.read_parquet(p)
            mf["trade_date"] = mf["trade_date"].astype(str)
        except Exception:
            n_skip += 1; continue
        if mf.empty:
            n_skip += 1; continue
        feat = compute_v2_features(mf, ohlc)
        if feat.empty:
            n_skip += 1; continue
        # 截取目标区间
        feat = feat[(feat["trade_date"] >= START) & (feat["trade_date"] <= END)]
        if feat.empty:
            n_skip += 1; continue
        all_rows.append(feat)
        n_ok += 1

    if not all_rows:
        print("没有数据!"); return
    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(OUT_DIR / "features.parquet", index=False)
    print(f"\nfeatures.parquet: {len(full):,} 行 × {full['ts_code'].nunique()} 股")

    print("\n=== 8 因子分布 ===")
    feat_cols = [c for c in full.columns if c.startswith("f")]
    print(full[feat_cols].describe(percentiles=[0.05, 0.5, 0.95]).round(4).to_string())

    print(f"\n总耗时 {time.time()-t0:.1f}s, ok={n_ok}, skip={n_skip}")


if __name__ == "__main__":
    main()
