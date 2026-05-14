#!/usr/bin/env python3
"""统一增量到 2026-05-14: amount + moneyflow_v1 + mfk + pyramid + v7_extras + regime + regime_extra.

数据源:
  - output/tushare_cache/daily/{YYYYMMDD}.parquet (565 日全市场 OHLC)
  - output/tushare_cache/moneyflow/{YYYYMMDD}.parquet (11 日 04-21 → 05-14 资金流)
  - output/moneyflow/cache/{ts_code}.parquet (单股 raw moneyflow 末日 04-20)

策略:
  1) 合并 moneyflow Tushare 增量到单股 cache (append, idempotent)
  2) 单股: 从 daily cache 取 OHLC, 从 cache 取 moneyflow, 算 5 个单股 feature 增量
  3) 市场级: 拉指数 daily, 算 regime + regime_extra 增量
  4) 输出 _ext_0514.parquet 到对应目录
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 复用已有算法
import extract_amount_features as _am_mod
import extract_mfk_features as _mfk_mod
import extract_pyramid_multiwindow as _pyr_mod
import extract_v7_extras as _v7_mod

# 把内部 END 常量推到 05-14, 否则 compute_xxx 内部会硬截到 04-20
_mfk_mod.END = "20260514"
_pyr_mod.END = "20260514"
_v7_mod.END  = "20260514"
_am_mod.END  = "20260514"

from extract_amount_features import compute_amount_features
from extract_mfk_features import compute_mfk
from extract_pyramid_multiwindow import compute_pyramid_v2
from extract_v7_extras import compute_v7
from src.stockagent_analysis.moneyflow.features import compute_features as compute_mf_v1

DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"
MF_BULK = ROOT / "output" / "tushare_cache" / "moneyflow"
MF_CACHE = ROOT / "output" / "moneyflow" / "cache"

OUT_AMOUNT = ROOT / "output" / "amount_features"
OUT_MF1    = ROOT / "output" / "moneyflow"
OUT_MFK    = ROOT / "output" / "mfk_features"
OUT_PYR    = ROOT / "output" / "pyramid_v2"
OUT_V7     = ROOT / "output" / "v7_extras"

NEW_START = "20260421"
NEW_END   = "20260514"
HIST_START = "20240101"  # 给 rolling 提供足够历史


# ─── Step 1: 合并 Tushare moneyflow bulk (trade_date 切) → 单股 cache append ───

def merge_moneyflow_bulk_to_cache():
    """把 04-21 → 05-14 的 trade_date-sliced moneyflow append 到单股 cache."""
    files = sorted(MF_BULK.glob("*.parquet"))
    if not files:
        print("  无 moneyflow bulk, 跳过 merge")
        return
    print(f"  读 {len(files)} 个 trade_date moneyflow ...")
    parts = [pd.read_parquet(f) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    print(f"  bulk 数据: {len(big):,} 行, {big['ts_code'].nunique()} 股")

    n_appended = 0
    for ts, g in big.groupby("ts_code"):
        cache_p = MF_CACHE / f"{ts}.parquet"
        if not cache_p.exists():
            continue  # 单股 cache 不存在则跳过(说明历史也没有)
        cur = pd.read_parquet(cache_p)
        cur["trade_date"] = cur["trade_date"].astype(str)
        last_date = cur["trade_date"].max()
        new = g[g["trade_date"] > last_date].copy()
        if new.empty:
            continue
        # 列对齐
        for col in cur.columns:
            if col not in new.columns:
                new[col] = pd.NA
        new = new[cur.columns]
        merged = pd.concat([cur, new], ignore_index=True).sort_values("trade_date").reset_index(drop=True)
        merged.to_parquet(cache_p, index=False)
        n_appended += len(new)
    print(f"  append: {n_appended:,} 行到单股 cache")


# ─── Step 2: 加载全市场 daily (内存字典 by ts_code) ───

def load_daily_by_code():
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    parts = [pd.read_parquet(f) for f in files if HIST_START <= f.stem <= NEW_END]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    # 补算 pre_close/change/pct_chg(已存在直接用)
    return {ts: g.reset_index(drop=True) for ts, g in big.groupby("ts_code")}


# ─── Step 3: 单股 5 feature 增量 ───

def daily_to_ohlc_list(daily_df):
    """compute_amount_features 期待 list of (date, o, h, l, c, vol)."""
    return list(zip(
        daily_df["trade_date"].tolist(),
        daily_df["open"].tolist(),
        daily_df["high"].tolist(),
        daily_df["low"].tolist(),
        daily_df["close"].tolist(),
        daily_df["vol"].tolist(),
    ))


def daily_to_close_list(daily_df):
    """mfk/v7 期待 list of (date, close)."""
    return list(zip(daily_df["trade_date"].tolist(), daily_df["close"].tolist()))


def update_per_stock(daily_by_code):
    amounts, mfs, mfks, pyrs, v7s = [], [], [], [], []

    ts_codes = sorted(daily_by_code.keys())
    print(f"  处理 {len(ts_codes)} 股 ...", flush=True)
    t = time.time()
    n_no_mf = 0
    for i, ts in enumerate(ts_codes, 1):
        daily = daily_by_code[ts]
        if len(daily) < 60:
            continue

        # amount: 用 OHLC list
        ohlc_list = daily_to_ohlc_list(daily)
        try:
            am = compute_amount_features(ohlc_list, NEW_START, NEW_END)
            if not am.empty:
                am["ts_code"] = ts
                amounts.append(am)
        except Exception:
            pass

        # 读 moneyflow 单股 cache
        mf_p = MF_CACHE / f"{ts}.parquet"
        if not mf_p.exists():
            n_no_mf += 1
        else:
            mf_raw = pd.read_parquet(mf_p)
            mf_raw["trade_date"] = mf_raw["trade_date"].astype(str)
            close_list = daily_to_close_list(daily)

            # moneyflow_v1
            try:
                mfv = compute_mf_v1(mf_raw)
                if not mfv.empty:
                    mfv = mfv[(mfv["trade_date"] >= NEW_START) & (mfv["trade_date"] <= NEW_END)]
                    if not mfv.empty:
                        mfs.append(mfv)
            except Exception:
                pass

            # mfk
            try:
                mfk = compute_mfk(mf_raw, close_list)
                if not mfk.empty:
                    mfk = mfk[(mfk["trade_date"] >= NEW_START) & (mfk["trade_date"] <= NEW_END)]
                    if not mfk.empty:
                        mfks.append(mfk)
            except Exception:
                pass

            # pyramid_v2
            try:
                py = compute_pyramid_v2(mf_raw)
                if not py.empty:
                    py = py[(py["trade_date"] >= NEW_START) & (py["trade_date"] <= NEW_END)]
                    if not py.empty:
                        pyrs.append(py)
            except Exception:
                pass

            # v7_extras
            try:
                v7 = compute_v7(mf_raw, close_list)
                if not v7.empty:
                    v7 = v7[(v7["trade_date"] >= NEW_START) & (v7["trade_date"] <= NEW_END)]
                    if not v7.empty:
                        v7s.append(v7)
            except Exception:
                pass

        if i % 500 == 0:
            print(f"    [{i}/{len(ts_codes)}] {time.time()-t:.0f}s", flush=True)

    if n_no_mf:
        print(f"  ! 无 moneyflow 单股 cache: {n_no_mf} 股")

    # 输出
    def write(parts, out_dir, name):
        if not parts:
            print(f"  {name}: 空")
            return 0
        df = pd.concat(parts, ignore_index=True)
        out_dir.mkdir(exist_ok=True)
        out_dir.joinpath(f"{name}_ext_0514.parquet").parent.mkdir(exist_ok=True)
        out = out_dir / f"{name}_ext_0514.parquet"
        df.to_parquet(out, index=False)
        print(f"  {name}: {len(df):,} 行 → {out}")
        return len(df)

    write(amounts, OUT_AMOUNT, "amount_features")
    write(mfs,     OUT_MF1,    "features")
    write(mfks,    OUT_MFK,    "features")
    write(pyrs,    OUT_PYR,    "features")
    write(v7s,     OUT_V7,     "features")


# ─── Step 4: 市场级 (regime + regime_extra) ───

def update_regimes_from_tushare():
    """拉三大指数 daily, 重算 regime + regime_extra (全量重写, 仅 ~580 行)."""
    import tushare as ts
    if not os.environ.get("TUSHARE_TOKEN"):
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
                break
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    pro = ts.pro_api()

    INDEX = {"hs300": "000300.SH", "cyb": "399006.SZ", "zz500": "000905.SH"}
    idx = {}
    for name, code in INDEX.items():
        df = None
        for attempt in range(3):
            try:
                df = pro.index_daily(ts_code=code, start_date="20230101", end_date=NEW_END)
                if df is not None and not df.empty and "trade_date" in df.columns:
                    break
            except Exception as e:
                print(f"  ! {name} attempt {attempt+1} 失败: {str(e)[:60]}")
            time.sleep(2)
        if df is None or df.empty or "trade_date" not in df.columns:
            print(f"  ! {name} {code} 拉取失败, 跳过")
            continue
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.sort_values("trade_date").reset_index(drop=True)
        idx[name] = df
        print(f"  {name} {code}: {len(df)} 行, 末日 {df['trade_date'].max()}")
        time.sleep(0.5)

    hs = idx["hs300"][["trade_date","open","high","low","close","vol"]].rename(columns={"vol":"volume"}).copy()
    hs["ret_5d"]  = hs["close"].pct_change(5)
    hs["ret_20d"] = hs["close"].pct_change(20)
    hs["ret_60d"] = hs["close"].pct_change(60)
    hs["ma60"]    = hs["close"].rolling(60).mean()
    hs["ma60_slope"] = hs["ma60"].pct_change(10)
    hs["std20"]   = hs["close"].pct_change().rolling(20).std()
    hs["bband_w"] = (4 * hs["std20"]) * 100
    delta = hs["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    hs["rsi14"] = 100 - 100 / (1 + rs)
    hs["vol_ma20"] = hs["volume"].rolling(20).mean()
    hs["vol_ratio"] = hs["volume"] / hs["vol_ma20"]
    hs["vol_z60"]  = (hs["volume"] - hs["volume"].rolling(60).mean()) / hs["volume"].rolling(60).std()

    cyb = idx["cyb"].set_index("trade_date")
    zz  = idx["zz500"].set_index("trade_date")
    hs["cyb_ret_20d"]   = hs["trade_date"].map(cyb["close"].pct_change(20))
    hs["zz500_ret_20d"] = hs["trade_date"].map(zz["close"].pct_change(20))

    # regime classify (复制 extract_regimes 的逻辑)
    REGIME_ID = {"bull_policy":1, "bull_fast":2, "bull_slow_diverge":3,
                 "bear":4, "sideways":5, "mixed":0}
    def classify(row):
        r5, r20, r60 = row["ret_5d"], row["ret_20d"], row["ret_60d"]
        rsi = row["rsi14"]; vol_r = row["vol_ratio"]; ma60_slope = row["ma60_slope"]
        bband = row["bband_w"]; cyb20 = row["cyb_ret_20d"]; zz20 = row["zz500_ret_20d"]
        if pd.isna(r20): return "mixed"
        if pd.notna(r5) and r5 >= 0.05 and pd.notna(vol_r) and vol_r >= 1.3: return "bull_policy"
        if pd.notna(r5) and r5 >= 0.03 and pd.notna(vol_r) and vol_r >= 1.5: return "bull_policy"
        if r20 >= 0.10 and ((pd.notna(rsi) and rsi > 65) or (pd.notna(ma60_slope) and ma60_slope > 0.005)): return "bull_fast"
        if pd.notna(r60) and r60 >= 0.20: return "bull_fast"
        hs300_mild = abs(r20) < 0.05
        small_up = ((pd.notna(zz20) and zz20 > 0.03) or (pd.notna(cyb20) and cyb20 > 0.03))
        if hs300_mild and small_up: return "bull_slow_diverge"
        if pd.notna(r60) and r60 <= -0.08: return "bear"
        if pd.notna(r20) and r20 <= -0.07 and pd.notna(ma60_slope) and ma60_slope < 0: return "bear"
        if abs(r20) < 0.03 and pd.notna(bband) and bband < 8: return "sideways"
        return "mixed"

    hs["regime"] = hs.apply(classify, axis=1)
    hs["regime_id"] = hs["regime"].map(REGIME_ID)
    out = hs[["trade_date","regime","regime_id","ret_5d","ret_20d","ret_60d",
              "rsi14","vol_ratio","vol_z60","cyb_ret_20d","zz500_ret_20d"]]
    out_p = ROOT / "output" / "regimes" / "daily_regime.parquet"
    out_p.parent.mkdir(exist_ok=True)
    out.to_parquet(out_p, index=False)
    print(f"  regimes: {len(out)} 行 → {out_p}")

    # regime_extra
    out2 = hs[["trade_date"]].copy()
    out2["regime_days_in"] = (hs["regime"] != hs["regime"].shift()).cumsum()
    out2["regime_days_in"] = out2.groupby(out2["regime_days_in"]).cumcount() + 1
    out2["regime_intensity"] = (hs["ret_20d"].abs() * 100).fillna(0)
    out2["hs300_ret60_z60"] = ((hs["ret_60d"] - hs["ret_60d"].rolling(60).mean()) /
                                hs["ret_60d"].rolling(60).std())
    out2["cyb_rel_strength"] = hs["cyb_ret_20d"] - hs["ret_20d"]
    out2["zz500_rel_strength"] = hs["zz500_ret_20d"] - hs["ret_20d"]
    out2_p = ROOT / "output" / "regime_extra" / "regime_extra.parquet"
    out2_p.parent.mkdir(exist_ok=True)
    out2.to_parquet(out2_p, index=False)
    print(f"  regime_extra: {len(out2)} 行 → {out2_p}")


# ─── Main ───

def main():
    t0 = time.time()
    print("=== Step 1: merge moneyflow bulk → 单股 cache ===", flush=True)
    merge_moneyflow_bulk_to_cache()

    print("\n=== Step 2: 加载全市场 daily ===", flush=True)
    daily_by_code = load_daily_by_code()
    print(f"  {len(daily_by_code)} 股, 耗时 {time.time()-t0:.0f}s", flush=True)

    print("\n=== Step 3: 单股 5 feature 增量 04-21 → 05-14 ===", flush=True)
    update_per_stock(daily_by_code)

    print("\n=== Step 4: 市场级 regime + regime_extra (全量重写) ===", flush=True)
    update_regimes_from_tushare()

    print(f"\n=== 完成, 总耗时 {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
