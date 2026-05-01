"""全因子大回测 (Stage 1: 本地纯计算 ~120 因子).

复用 output/backtest_factors_2026_04/ 已有数据:
  - raw_data/{ts_code}.json   (5149 股 1 年 daily 等时序)
  - group_results/group_NNN.jsonl  (100 万样本 + r5/r10/r20/r30/r40 + total_mv/pe/pb)
  - universe.json             (含 industry)

新增因子 (~120 个):
  Group A 技术指标 (25)   — MA/MACD/RSI/KDJ/BOLL/ATR/BIAS/ROC/CCI/WR/MFI/TRIX
  Group B 量价 (10)       — vol_ratio/amplitude/vol_price_match/OBV
  Group C TA-Lib K 线 (61) — talib.CDL* 全部
  Group D 趋势/通道 (5)
  Group J 事件因子 (8)    — 突破/缺口/涨停
  Group L Qlib 算子 (25)  — KBAR9 + 滚动算子精选
  Group M TA-Lib 动量 (17) — AROON/KAMA/SAR/CMO/BOP/PPO/AD/ADOSC/NATR/STOCHRSI

工程:
  - 5 worker 并行 (按 group_id)
  - per-group parquet checkpoint, 已 done 跳过
  - status.json 记录每 group 状态 (中断恢复)

用法:
  python factor_lab.py --phase 0     # 准备 (元数据 / 索引)
  python factor_lab.py --phase 1     # 算因子 (5 worker)
  python factor_lab.py --phase 2     # 聚合 IC 报告
  python factor_lab.py --phase all   # 全流程
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# 数据源切换 (env: FACTOR_LAB_DATA_SOURCE)
#   "1y"  → output/backtest_factors_2026_04 (241 天, 默认)
#   "3y"  → output/backtest_3y_2023_2026 (783 天)
DATA_SOURCE = os.environ.get("FACTOR_LAB_DATA_SOURCE", "1y").lower()
if DATA_SOURCE == "3y":
    SRC_DATA_DIR = ROOT / "output" / "backtest_3y_2023_2026"
    OUT_SUFFIX = "_3y"
else:
    SRC_DATA_DIR = ROOT / "output" / "backtest_factors_2026_04"
    OUT_SUFFIX = ""

RAW_DATA_DIR = SRC_DATA_DIR / "raw_data"
GR_DIR = SRC_DATA_DIR / "group_results"
UNIVERSE_FILE = SRC_DATA_DIR / "universe.json"

OUT_DIR = ROOT / "output" / f"factor_lab{OUT_SUFFIX}"
GROUPS_DIR = OUT_DIR / "factor_groups"
LOG_DIR = OUT_DIR / "logs"
STATUS_FILE = OUT_DIR / "status.json"
META_FILE = OUT_DIR / "factor_meta.json"
REPORT_FILE = OUT_DIR / "report.md"
REPORT_LAYER_FILE = OUT_DIR / "report_by_layer.md"

for d in (OUT_DIR, GROUPS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

NUM_WORKERS = 5
HOLD_PERIODS = [5, 10, 20, 30, 40]


# ──────────────────── 日志 ────────────────────
def setup_logger(name: str, file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(processName)s] %(message)s",
                             datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if file:
        fh = logging.FileHandler(file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ──────────────────── 因子计算 ────────────────────

def _safe_div(a, b):
    return np.where(np.abs(b) < 1e-12, np.nan, a / b)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_ha_ohlc(daily: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi K 线: 替换 OHLC, 保留 vol/amount/trade_date/pre_close/pct_chg."""
    df = daily.copy().reset_index(drop=True)
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    n = len(df)
    ha_close = (o + h + l + c) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (o[0] + c[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    out = df.copy()
    out["open"] = ha_open
    out["high"] = ha_high
    out["low"] = ha_low
    out["close"] = ha_close
    return out


def daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """日 → 周线 resample (按周末日期对齐). 用于周线因子."""
    df = daily.copy()
    df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("dt").set_index("dt")
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "vol": "sum", "amount": "sum",
    }
    if "pre_close" in df.columns:
        agg["pre_close"] = "first"
    if "pct_chg" in df.columns:
        # 周涨幅: (last_close / first_pre_close - 1) * 100
        pass
    w = df.resample("W-FRI").agg(agg).dropna(subset=["close"]).reset_index()
    if "pre_close" in w.columns:
        w["pre_close"] = w["close"].shift(1).fillna(w["pre_close"])
        w["pct_chg"] = (w["close"] / w["pre_close"] - 1) * 100
    else:
        w["pre_close"] = w["close"].shift(1).fillna(w["close"])
        w["pct_chg"] = (w["close"] / w["pre_close"] - 1) * 100
    w["trade_date"] = w["dt"].dt.strftime("%Y%m%d")
    w = w.drop(columns=["dt"]).reset_index(drop=True)
    return w


def compute_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """对单股 daily DataFrame (按时间升序) 算所有因子, 返回 DataFrame 同长度.

    daily 列: trade_date(str), open, high, low, close, pre_close, change, pct_chg, vol, amount
    """
    import talib

    df = daily.copy().reset_index(drop=True)
    n = len(df)
    if n < 30:
        return pd.DataFrame({"trade_date": df["trade_date"]})

    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    v = df["vol"].astype(float).values
    a = df["amount"].astype(float).values
    pc = df["pre_close"].astype(float).bfill().values
    pct = df["pct_chg"].astype(float).fillna(0).values
    closeS = pd.Series(c)
    volS = pd.Series(v)

    out = {"trade_date": df["trade_date"].values}

    # ───── Group A: 技术指标 ─────
    for w in (5, 10, 20, 60, 120):
        ma = closeS.rolling(w, min_periods=max(2, w // 2)).mean()
        out[f"ma_ratio_{w}"] = (c / ma - 1).values
    out["ma5_ma20"] = (closeS.rolling(5).mean() / closeS.rolling(20).mean() - 1).values
    out["ma20_ma60"] = (closeS.rolling(20).mean() / closeS.rolling(60).mean() - 1).values

    macd, macd_sig, macd_hist = talib.MACD(c, 12, 26, 9)
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    out["rsi_6"] = talib.RSI(c, 6)
    out["rsi_14"] = talib.RSI(c, 14)
    out["rsi_24"] = talib.RSI(c, 24)

    k, d = talib.STOCH(h, l, c, fastk_period=9, slowk_period=3, slowd_period=3)
    out["kdj_k"] = k
    out["kdj_d"] = d
    out["kdj_j"] = 3 * k - 2 * d

    boll_up, boll_mid, boll_dn = talib.BBANDS(c, 20, 2, 2)
    out["boll_pct"] = _safe_div(c - boll_dn, boll_up - boll_dn)
    out["boll_width"] = _safe_div(boll_up - boll_dn, boll_mid)

    out["atr_pct"] = _safe_div(talib.ATR(h, l, c, 14), c)
    for w in (5, 10, 20):
        ma = closeS.rolling(w, min_periods=max(2, w // 2)).mean()
        out[f"bias_{w}"] = (c / ma - 1).values

    out["roc_10"] = talib.ROC(c, 10)
    out["roc_20"] = talib.ROC(c, 20)
    out["cci_14"] = talib.CCI(h, l, c, 14)
    out["wr_14"] = talib.WILLR(h, l, c, 14)
    out["mfi_14"] = talib.MFI(h, l, c, v, 14)
    out["trix"] = talib.TRIX(c, 12)

    # ───── Group B: 量价 ─────
    for w in (5, 20):
        vma = volS.rolling(w, min_periods=max(2, w // 2)).mean()
        out[f"vol_ratio_{w}"] = (v / vma).values
    out["amplitude"] = _safe_div(h - l, pc)
    # turnover proxy: 当日 amount / 20日均 amount
    aS = pd.Series(a)
    out["amount_ratio_20"] = (aS / aS.rolling(20).mean()).values
    # 量价配合: 1=量增价涨, 2=缩量上涨, 3=量增价跌, 4=缩量下跌, 0=其他
    vol_up = volS.diff() > 0
    px_up = pct > 0
    vp = np.where(vol_up & px_up, 1,
         np.where(~vol_up & px_up, 2,
         np.where(vol_up & ~px_up, 3,
         np.where(~vol_up & ~px_up, 4, 0))))
    out["vol_price_match"] = vp
    out["obv"] = talib.OBV(c, v)
    out["obv_diff_20"] = pd.Series(out["obv"]).diff(20).values

    # ───── Group C: TA-Lib K 线形态 (61 种) ─────
    cdl_funcs = [name for name in dir(talib) if name.startswith("CDL")]
    for fn in cdl_funcs:
        try:
            out[fn.lower()] = getattr(talib, fn)(o, h, l, c)
        except Exception:
            pass

    # ───── Group D: 趋势 / 通道 ─────
    out["lr_slope_20"] = talib.LINEARREG_SLOPE(c, 20)
    out["lr_slope_60"] = talib.LINEARREG_SLOPE(c, 60)
    out["lr_angle_20"] = talib.LINEARREG_ANGLE(c, 20)
    # 通道位置: close 在 60 日 [min, max] 中位置
    rmax60 = closeS.rolling(60).max()
    rmin60 = closeS.rolling(60).min()
    out["channel_pos_60"] = ((c - rmin60.values) / (rmax60.values - rmin60.values + 1e-12))
    # R²
    try:
        # talib 没有 R², 自己算: 1 - SSE/SST
        # 使用 LINEARREG 的拟合值
        fit = talib.LINEARREG(c, 60)
        # roll var of close
        var60 = closeS.rolling(60).var().values
        sse = pd.Series((c - fit) ** 2).rolling(60).mean().values
        out["lr_r2_60"] = np.where(var60 > 1e-12, 1 - sse / (var60 + 1e-12), np.nan)
    except Exception:
        out["lr_r2_60"] = np.nan

    # ───── Group J: 事件因子 ─────
    rmax20 = closeS.rolling(20).max().shift(1)   # 不含今天
    rmax60 = closeS.rolling(60).max().shift(1)
    rmin20 = closeS.rolling(20).min().shift(1)
    out["break_high_20"] = (c > rmax20.values).astype(float)
    out["break_high_60"] = (c > rmax60.values).astype(float)
    out["break_low_20"] = (c < rmin20.values).astype(float)
    # 缺口: 跳空高开 / 低开
    out["gap_up"] = ((o - pc) / pc > 0.02).astype(float)
    out["gap_down"] = ((pc - o) / pc > 0.02).astype(float)
    # 涨停 dummy: pct_chg > 9.7 (创业板/科创板宽松, 这里只做粗筛)
    is_zt = (pct > 9.7).astype(float)
    out["limit_up"] = is_zt
    # 连板: 连续涨停天数
    consec = np.zeros(n)
    for i in range(n):
        if is_zt[i] == 1:
            consec[i] = consec[i - 1] + 1 if i > 0 else 1
    out["limit_consec"] = consec
    # 量比异常 (5 日量比, 已在上面 vol_ratio_5)
    # 巨量长阴 / 长阳: pct > 7 且 vol_ratio_5 > 2
    out["volume_spike_up"] = ((pct > 7) & (out["vol_ratio_5"] > 2)).astype(float)
    out["volume_spike_dn"] = ((pct < -7) & (out["vol_ratio_5"] > 2)).astype(float)

    # ───── Group L: Qlib KBAR + ROLL ─────
    # KBAR (9)
    out["k_kmid"] = (c - o) / (o + 1e-12)
    out["k_klen"] = (h - l) / (o + 1e-12)
    out["k_kmid2"] = (c - o) / (h - l + 1e-12)
    out["k_kup"] = (h - np.maximum(o, c)) / (o + 1e-12)
    out["k_kup2"] = (h - np.maximum(o, c)) / (h - l + 1e-12)
    out["k_klow"] = (np.minimum(o, c) - l) / (o + 1e-12)
    out["k_klow2"] = (np.minimum(o, c) - l) / (h - l + 1e-12)
    out["k_ksft"] = (2 * c - h - l) / (o + 1e-12)
    out["k_ksft2"] = (2 * c - h - l) / (h - l + 1e-12)

    # ROLL (主取 20 周期)
    W = 20
    out[f"qtlu_{W}"] = closeS.rolling(W).quantile(0.8).values / c - 1
    out[f"qtld_{W}"] = closeS.rolling(W).quantile(0.2).values / c - 1
    out[f"rank_{W}"] = closeS.rolling(W).rank(pct=True).values
    rmaxW = closeS.rolling(W).max()
    rminW = closeS.rolling(W).min()
    out[f"rsv_{W}"] = ((c - rminW.values) / (rmaxW.values - rminW.values + 1e-12))
    # IMAX/IMIN: 最高 / 最低出现位置 (距今天数, 0=今天)
    def rolling_idxmax(s, w):
        # idxmax 返回索引, 我们要"距今天数"
        arr = s.values
        out_arr = np.full_like(arr, np.nan, dtype=float)
        for i in range(w - 1, len(arr)):
            window = arr[i - w + 1:i + 1]
            out_arr[i] = (w - 1) - int(np.argmax(window))   # 距今: 0=今天
        return out_arr

    def rolling_idxmin(s, w):
        arr = s.values
        out_arr = np.full_like(arr, np.nan, dtype=float)
        for i in range(w - 1, len(arr)):
            window = arr[i - w + 1:i + 1]
            out_arr[i] = (w - 1) - int(np.argmin(window))
        return out_arr

    out[f"imax_{W}"] = rolling_idxmax(closeS, W)
    out[f"imin_{W}"] = rolling_idxmin(closeS, W)
    out[f"imxd_{W}"] = out[f"imax_{W}"] - out[f"imin_{W}"]

    # 量价相关
    rets = closeS.pct_change()
    out[f"corr_close_vol_{W}"] = closeS.rolling(W).corr(volS).values
    out[f"cord_ret_vol_{W}"] = rets.rolling(W).corr(volS.pct_change()).values

    # CNTP/CNTN (20 日内 涨/跌天数)
    up_day = (pct > 0).astype(float)
    dn_day = (pct < 0).astype(float)
    out[f"cntp_{W}"] = pd.Series(up_day).rolling(W).sum().values
    out[f"cntn_{W}"] = pd.Series(dn_day).rolling(W).sum().values
    out[f"cntd_{W}"] = out[f"cntp_{W}"] - out[f"cntn_{W}"]

    # SUMP/SUMN/SUMD (累计正涨幅 / 负跌幅)
    pos_pct = pd.Series(np.where(pct > 0, pct, 0))
    neg_pct = pd.Series(np.where(pct < 0, -pct, 0))
    out[f"sump_{W}"] = pos_pct.rolling(W).sum().values
    out[f"sumn_{W}"] = neg_pct.rolling(W).sum().values
    out[f"sumd_{W}"] = out[f"sump_{W}"] - out[f"sumn_{W}"]

    # 量类
    out[f"vma_{W}"] = volS.rolling(W).mean().values
    out[f"vstd_{W}"] = volS.rolling(W).std().values
    # 加权量 MA: 按 |pct| 加权
    w_abs_pct = pd.Series(np.abs(pct))
    out[f"wvma_{W}"] = (volS * w_abs_pct).rolling(W).sum().values / (w_abs_pct.rolling(W).sum().values + 1e-12)

    # ───── Group M: TA-Lib 动量 / 通道 / 统计 ─────
    aroon_dn, aroon_up = talib.AROON(h, l, 14)
    out["aroon_up"] = aroon_up
    out["aroon_dn"] = aroon_dn
    out["aroon_osc"] = talib.AROONOSC(h, l, 14)
    out["kama_20"] = talib.KAMA(c, 20)
    out["kama_dist"] = (c - out["kama_20"]) / (c + 1e-12)
    sar = talib.SAR(h, l)
    out["sar_signal"] = np.where(c > sar, 1.0, np.where(c < sar, -1.0, 0))
    out["cmo_14"] = talib.CMO(c, 14)
    out["bop"] = talib.BOP(o, h, l, c)
    out["ppo"] = talib.PPO(c, 12, 26)
    out["apo"] = talib.APO(c, 12, 26)
    out["ad"] = talib.AD(h, l, c, v)
    out["adosc"] = talib.ADOSC(h, l, c, v, 3, 10)
    out["natr_14"] = talib.NATR(h, l, c, 14)
    stoch_k, stoch_d = talib.STOCHRSI(c, 14, 5, 3)
    out["stochrsi_k"] = stoch_k
    out["stochrsi_d"] = stoch_d
    out["ht_trendmode"] = talib.HT_TRENDMODE(c).astype(float)

    rdf = pd.DataFrame(out)
    return rdf


# ──────────────────── Worker ────────────────────

def load_universe_index() -> dict:
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        uni = json.load(f)
    return {s["ts_code"]: {"industry": s.get("industry", ""), "name": s.get("name", "")}
            for s in uni["stocks"]}


def load_group_samples(group_id: int) -> pd.DataFrame:
    f = GR_DIR / f"group_{group_id:03d}.jsonl"
    if not f.exists():
        return pd.DataFrame()
    rows = []
    with open(f, encoding="utf-8") as fp:
        for line in fp:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_stock_daily(ts_code: str) -> pd.DataFrame | None:
    f = RAW_DATA_DIR / f"{ts_code}.json"
    if not f.exists():
        return None
    try:
        with open(f, encoding="utf-8") as fp:
            x = json.load(fp)
    except Exception:
        return None
    daily = x.get("ts", {}).get("daily") or []
    if not daily:
        return None
    df = pd.DataFrame(daily)
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


_WORKER_MODE = "raw"   # 'raw' | 'ha'  (设置在 phase1_run, 通过 globals 传给 worker)


def _suffix(mode: str) -> str:
    return "" if mode == "raw" else f"_{mode}"


def worker_run_group(args) -> dict:
    """args = (group_id, mode)"""
    if isinstance(args, tuple):
        group_id, mode = args
    else:
        group_id, mode = args, "raw"
    suf = _suffix(mode)
    log_file = LOG_DIR / f"worker_g{group_id:03d}{suf}.log"
    log = setup_logger(f"w_g{group_id}{suf}", log_file)

    out_file = GROUPS_DIR / f"group_{group_id:03d}{suf}.parquet"
    status_file = GROUPS_DIR / f"group_{group_id:03d}{suf}.status.json"

    if status_file.exists() and out_file.exists():
        try:
            with open(status_file, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("status") == "done":
                log.info("[skip] group %d 已完成", group_id)
                return s
        except Exception:
            pass

    samples = load_group_samples(group_id)
    if samples.empty:
        return {"group_id": group_id, "status": "empty", "samples": 0}

    log.info("group %d (mode=%s) 启动: %d 样本", group_id, mode, len(samples))
    t0 = time.time()
    industry_map = load_universe_index()

    all_factor_dfs = []
    samples_g = samples.groupby("ts_code")
    failed = []
    for i, (ts_code, sub) in enumerate(samples_g):
        try:
            daily = load_stock_daily(ts_code)
            if daily is None or len(daily) < 30:
                continue
            if mode == "ha":
                daily_for_factors = compute_ha_ohlc(daily)
            else:
                daily_for_factors = daily
            fac = compute_factors(daily_for_factors)
            # merge: sub 上加因子列 (按 trade_date)
            sub_dates = set(sub["trade_date"].astype(str).tolist())
            fac["trade_date"] = fac["trade_date"].astype(str)
            fac = fac[fac["trade_date"].isin(sub_dates)]
            merged = sub.merge(fac, on="trade_date", how="left")
            merged["industry"] = industry_map.get(ts_code, {}).get("industry", "")
            all_factor_dfs.append(merged)
        except Exception as e:
            failed.append({"ts_code": ts_code, "err": str(e)[:200]})
            log.error("[error] %s: %s", ts_code, str(e)[:120])
        if (i + 1) % 25 == 0:
            log.info("  group %d %d/%d", group_id, i + 1, len(samples_g))

    if not all_factor_dfs:
        return {"group_id": group_id, "status": "empty", "samples": 0}

    full = pd.concat(all_factor_dfs, ignore_index=True)
    full.to_parquet(out_file, index=False, compression="snappy")
    dur = time.time() - t0

    status = {
        "group_id": group_id,
        "status": "done",
        "samples": len(full),
        "n_stocks": full["ts_code"].nunique(),
        "n_factors": len([c for c in full.columns if c not in (
            "ts_code", "trade_date", "industry", "name") and not c.startswith(
            ("r", "dd")) and c not in ("total_mv", "pe", "pe_ttm", "pb")]),
        "failed": failed,
        "duration_sec": int(dur),
        "finished_at": datetime.now().isoformat(),
    }
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    log.info("group %d done: %d 样本, %d 列, %.1fs",
             group_id, len(full), len(full.columns), dur)
    return status


# ──────────────────── Phase 0: 元数据 ────────────────────

def phase0_meta():
    """枚举一只样本股的因子列, 写 factor_meta.json."""
    log = setup_logger("phase0", LOG_DIR / "phase0.log")
    sample = list(RAW_DATA_DIR.glob("*.json"))[0]
    log.info("用 %s 计算因子枚举...", sample.name)
    with open(sample, encoding="utf-8") as f:
        x = json.load(f)
    daily = pd.DataFrame(x["ts"]["daily"]).sort_values("trade_date").reset_index(drop=True)
    fac = compute_factors(daily)
    factor_cols = [c for c in fac.columns if c != "trade_date"]
    meta = {
        "n_factors": len(factor_cols),
        "factors": factor_cols,
        "by_group": {
            "A_tech": [c for c in factor_cols if c.startswith(("ma_ratio", "ma5_ma20", "ma20_ma60",
                                                               "macd", "rsi_", "kdj_", "boll_",
                                                               "atr_", "bias_", "roc_", "cci_",
                                                               "wr_", "mfi_", "trix"))],
            "B_volprice": [c for c in factor_cols if c.startswith(("vol_ratio", "amplitude",
                                                                   "amount_ratio", "vol_price",
                                                                   "obv"))],
            "C_candle": [c for c in factor_cols if c.startswith("cdl")],
            "D_trend": [c for c in factor_cols if c.startswith(("lr_", "channel_"))],
            "J_event": [c for c in factor_cols if c.startswith(("break_", "gap_", "limit_",
                                                                "volume_spike"))],
            "L_qlib": [c for c in factor_cols if c.startswith(("k_k", "qtlu", "qtld", "rank_",
                                                               "rsv_", "imax", "imin", "imxd",
                                                               "corr_", "cord_", "cntp", "cntn",
                                                               "cntd", "sump", "sumn", "sumd",
                                                               "vma_", "vstd_", "wvma_"))],
            "M_momentum": [c for c in factor_cols if c.startswith(("aroon", "kama", "sar_",
                                                                   "cmo_", "bop", "ppo", "apo",
                                                                   "ad", "adosc", "natr",
                                                                   "stochrsi", "ht_"))],
        },
        "generated_at": datetime.now().isoformat(),
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("factor_meta 写入 %d 因子, 分组: %s",
             meta["n_factors"], {k: len(v) for k, v in meta["by_group"].items()})
    return meta


# ──────────────────── Phase 1: 跑全部 group ────────────────────

def phase1_run(only_group: int | None = None, mode: str = "raw"):
    log = setup_logger(f"phase1_{mode}", LOG_DIR / f"phase1_{mode}.log")

    # 列出待跑 groups
    all_groups = sorted([int(f.stem.split("_")[1]) for f in GR_DIR.glob("group_*.jsonl")])
    log.info("总 group 数: %d, mode=%s", len(all_groups), mode)
    suf = _suffix(mode)

    if only_group is not None:
        log.info("单跑 group %d", only_group)
        s = worker_run_group((only_group, mode))
        update_status(s, mode)
        return

    pending = []
    for g in all_groups:
        sf = GROUPS_DIR / f"group_{g:03d}{suf}.status.json"
        of = GROUPS_DIR / f"group_{g:03d}{suf}.parquet"
        if sf.exists() and of.exists():
            try:
                with open(sf, encoding="utf-8") as fp:
                    s = json.load(fp)
                if s.get("status") == "done":
                    continue
            except Exception:
                pass
        pending.append(g)

    log.info("待跑 %d 个 group, %d worker 并行", len(pending), NUM_WORKERS)
    if not pending:
        log.info("全部完成")
        return

    args_list = [(g, mode) for g in pending]
    t0 = time.time()
    done = 0
    with mp.Pool(processes=NUM_WORKERS) as pool:
        for s in pool.imap_unordered(worker_run_group, args_list):
            done += 1
            update_status(s, mode)
            elapsed = time.time() - t0
            eta = elapsed / done * (len(pending) - done)
            log.info("g%d done | %d/%d | 已耗时 %.1fmin | ETA %.1fmin",
                     s["group_id"], done, len(pending), elapsed / 60, eta / 60)


def update_status(g_status: dict, mode: str = "raw"):
    suf = _suffix(mode)
    sf = OUT_DIR / f"status{suf}.json"
    prog = {}
    if sf.exists():
        try:
            prog = json.loads(sf.read_text(encoding="utf-8"))
        except Exception:
            pass
    prog[str(g_status["group_id"])] = {
        "status": g_status["status"],
        "samples": g_status.get("samples", 0),
        "n_stocks": g_status.get("n_stocks", 0),
        "duration_sec": g_status.get("duration_sec", 0),
        "finished_at": g_status.get("finished_at", ""),
    }
    sf.write_text(json.dumps(prog, ensure_ascii=False, indent=2), encoding="utf-8")


# ──────────────────── Phase 2: 聚合 IC 报告 ────────────────────

MV_BUCKETS = [(0, 50), (50, 100), (100, 300), (300, 1000), (1000, float("inf"))]
MV_LABELS = ["20-50亿", "50-100亿", "100-300亿", "300-1000亿", "1000亿+"]

PE_BUCKETS = [(-float("inf"), 0), (0, 15), (15, 30), (30, 50), (50, 100), (100, float("inf"))]
PE_LABELS = ["亏损", "0-15", "15-30", "30-50", "50-100", "100+"]


def bucket_mv(v):
    if pd.isna(v):
        return None
    bil = v / 10000   # total_mv 万元 -> 亿元
    for (lo, hi), lab in zip(MV_BUCKETS, MV_LABELS):
        if lo <= bil < hi:
            return lab
    return None


def bucket_pe(v):
    if pd.isna(v):
        return None
    for (lo, hi), lab in zip(PE_BUCKETS, PE_LABELS):
        if lo <= v < hi:
            return lab
    return None


def phase2_report():
    log = setup_logger("phase2", LOG_DIR / "phase2.log")

    log.info("加载所有 group parquet...")
    files = sorted(GROUPS_DIR.glob("group_*.parquet"))
    if not files:
        log.error("无 parquet 文件")
        return
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            log.error("读 %s 失败: %s", f, e)
    full = pd.concat(dfs, ignore_index=True)
    log.info("总样本: %d, 列数: %d", len(full), len(full.columns))

    # 因子列 = 所有非元数据 / 收益 / 风险列
    excluded = {"ts_code", "trade_date", "industry", "name", "total_mv", "pe", "pe_ttm", "pb"} | \
        {f"r{h}" for h in (5, 10, 20, 30, 40)} | {f"dd{h}" for h in (5, 10, 20, 30, 40)} | \
        {"market_score_adj", "adx", "winner_rate", "main_net", "holder_pct",
         "mf_divergence", "mf_strength", "mf_consecutive"}
    factor_cols = [c for c in full.columns if c not in excluded]
    log.info("因子数: %d", len(factor_cols))

    # 添加分桶
    full["mv_bucket"] = full["total_mv"].apply(bucket_mv)
    full["pe_bucket"] = full["pe"].apply(bucket_pe)

    # ───── 1) 全市场单因子 IC ─────
    log.info("计算全市场单因子 IC...")
    ic_rows = []
    for fc in factor_cols:
        x = full[fc]
        if not np.issubdtype(x.dtype, np.number):
            continue
        valid_mask = x.notna()
        if valid_mask.sum() < 1000:
            continue
        sigma = float(x.std())
        row = {"factor": fc, "n_valid": int(valid_mask.sum()), "sigma": sigma}
        for h in (5, 20, 40):
            r = full[f"r{h}"]
            mask = valid_mask & r.notna()
            if mask.sum() < 1000:
                row[f"ic_{h}"] = np.nan
                continue
            xv = x[mask].values
            yv = r[mask].values
            # 避免常数列
            if np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
                row[f"ic_{h}"] = np.nan
            else:
                row[f"ic_{h}"] = float(np.corrcoef(xv, yv)[0, 1])
        ic_rows.append(row)
    ic_df = pd.DataFrame(ic_rows).sort_values("ic_20", key=lambda s: s.abs(), ascending=False)

    # ───── 2) 分层 IC ─────
    def layer_ic(group_col: str, labels: list[str], min_n=500, min_bucket_n=5000):
        rows = []
        for bk in labels:
            sub = full[full[group_col] == bk]
            if len(sub) < min_bucket_n:
                continue
            for fc in factor_cols:
                x = sub[fc]
                if not np.issubdtype(x.dtype, np.number) or x.notna().sum() < min_n:
                    continue
                for h in (5, 20, 40):
                    r = sub[f"r{h}"]
                    mask = x.notna() & r.notna()
                    if mask.sum() < min_n:
                        continue
                    xv = x[mask].values
                    yv = r[mask].values
                    if np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
                        continue
                    ic = float(np.corrcoef(xv, yv)[0, 1])
                    if abs(ic) > 0.005:
                        rows.append({"factor": fc, "bucket": bk, "h": h,
                                     "ic": ic, "n": int(mask.sum())})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    log.info("计算 mv_bucket 分层 IC...")
    layer_df_mv = layer_ic("mv_bucket", MV_LABELS)
    log.info("计算 pe_bucket 分层 IC...")
    layer_df_pe = layer_ic("pe_bucket", PE_LABELS)

    # 行业分层 - 取样本数 ≥10000 的主要行业
    log.info("计算 industry 分层 IC...")
    ind_counts = full["industry"].value_counts()
    main_inds = ind_counts[ind_counts >= 10000].index.tolist()
    layer_df_ind = layer_ic("industry", main_inds, min_n=300, min_bucket_n=10000)

    # ───── 3) 写报告 ─────
    log.info("写报告...")
    write_report(full, ic_df, layer_df_mv, layer_df_pe, layer_df_ind, factor_cols)
    log.info("DONE")


def write_report(full: pd.DataFrame, ic_df: pd.DataFrame,
                  layer_df_mv: pd.DataFrame, layer_df_pe: pd.DataFrame,
                  layer_df_ind: pd.DataFrame, factor_cols: list[str]):
    n_total = len(full)
    n_stocks = full["ts_code"].nunique()
    industry_n = full["industry"].nunique()

    lines = []
    lines.append(f"# 全因子大回测报告\n\n")
    lines.append(f"> 生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"- 样本: **{n_total:,}**\n")
    lines.append(f"- 股票: **{n_stocks}**\n")
    lines.append(f"- 行业: **{industry_n}**\n")
    lines.append(f"- 因子: **{len(factor_cols)}**\n\n")

    lines.append(f"## 一、Top 30 单因子 IC (按 |IC(20d)| 排序)\n\n")
    lines.append("| 因子 | 有效样本 | σ | IC(5d) | IC(20d) | IC(40d) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for _, r in ic_df.head(30).iterrows():
        lines.append(f"| {r['factor']} | {r['n_valid']:,} | {r['sigma']:.3f} | "
                     f"{r.get('ic_5', float('nan')):+.4f} | "
                     f"{r.get('ic_20', float('nan')):+.4f} | "
                     f"{r.get('ic_40', float('nan')):+.4f} |\n")
    lines.append("\n")

    # 各组 top 5
    lines.append(f"## 二、各 Group 最强因子 (前5)\n\n")
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        for grp_name, grp_facs in meta.get("by_group", {}).items():
            sub_ic = ic_df[ic_df["factor"].isin(grp_facs)]
            if sub_ic.empty:
                continue
            lines.append(f"### {grp_name} ({len(grp_facs)} 因子)\n\n")
            lines.append("| 因子 | IC(5d) | IC(20d) | IC(40d) |\n")
            lines.append("|---|---|---|---|\n")
            for _, r in sub_ic.head(5).iterrows():
                lines.append(f"| {r['factor']} | "
                             f"{r.get('ic_5', float('nan')):+.4f} | "
                             f"{r.get('ic_20', float('nan')):+.4f} | "
                             f"{r.get('ic_40', float('nan')):+.4f} |\n")
            lines.append("\n")

    REPORT_FILE.write_text("".join(lines), encoding="utf-8")

    # 分层报告
    lines2 = [f"# 因子分层 IC 报告 (市值 + PE + 行业)\n\n",
              f"> 生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"]

    # ── 市值分层 ──
    if not layer_df_mv.empty:
        lines2.append("## 一、市值分层\n\n### 1.1 各市值段最强因子 (按 |IC(20d)| 前 20)\n\n")
        for bk in MV_LABELS:
            sub = layer_df_mv[(layer_df_mv["bucket"] == bk) & (layer_df_mv["h"] == 20)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["abs_ic"] = sub["ic"].abs()
            sub = sub.sort_values("abs_ic", ascending=False).head(20)
            lines2.append(f"#### {bk}\n\n| 因子 | IC(20d) | 样本 |\n|---|---|---|\n")
            for _, r in sub.iterrows():
                lines2.append(f"| {r['factor']} | {r['ic']:+.4f} | {r['n']:,} |\n")
            lines2.append("\n")

        lines2.append("### 1.2 跨市值方向反转的因子 (大盘 vs 小盘)\n\n")
        wide = layer_df_mv[layer_df_mv["h"] == 20].pivot_table(
            index="factor", columns="bucket", values="ic", aggfunc="first")
        if "20-50亿" in wide.columns and "1000亿+" in wide.columns:
            flip = []
            for fc in wide.index:
                small = wide.at[fc, "20-50亿"]
                big = wide.at[fc, "1000亿+"]
                if pd.isna(small) or pd.isna(big):
                    continue
                if small * big < 0 and abs(small - big) > 0.03:
                    flip.append({"factor": fc, "ic_small": small, "ic_big": big,
                                  "diff": big - small})
            flip.sort(key=lambda r: abs(r["diff"]), reverse=True)
            if flip:
                lines2.append("| 因子 | 小盘 IC(20d) | 大盘 IC(20d) | 差值 |\n|---|---|---|---|\n")
                for r in flip[:40]:
                    lines2.append(f"| {r['factor']} | {r['ic_small']:+.4f} | "
                                  f"{r['ic_big']:+.4f} | {r['diff']:+.4f} |\n")
                lines2.append("\n")

    # ── PE 分层 ──
    if not layer_df_pe.empty:
        lines2.append("## 二、PE 分层\n\n### 2.1 各 PE 段最强因子 (按 |IC(20d)| 前 15)\n\n")
        for bk in PE_LABELS:
            sub = layer_df_pe[(layer_df_pe["bucket"] == bk) & (layer_df_pe["h"] == 20)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["abs_ic"] = sub["ic"].abs()
            sub = sub.sort_values("abs_ic", ascending=False).head(15)
            lines2.append(f"#### PE {bk}\n\n| 因子 | IC(20d) | 样本 |\n|---|---|---|\n")
            for _, r in sub.iterrows():
                lines2.append(f"| {r['factor']} | {r['ic']:+.4f} | {r['n']:,} |\n")
            lines2.append("\n")

        lines2.append("### 2.2 跨 PE 方向反转的因子 (高 PE vs 低 PE)\n\n")
        wide_pe = layer_df_pe[layer_df_pe["h"] == 20].pivot_table(
            index="factor", columns="bucket", values="ic", aggfunc="first")
        if "0-15" in wide_pe.columns and "100+" in wide_pe.columns:
            flip = []
            for fc in wide_pe.index:
                lo = wide_pe.at[fc, "0-15"]
                hi = wide_pe.at[fc, "100+"]
                if pd.isna(lo) or pd.isna(hi):
                    continue
                if lo * hi < 0 and abs(lo - hi) > 0.03:
                    flip.append({"factor": fc, "ic_low": lo, "ic_high": hi,
                                  "diff": hi - lo})
            flip.sort(key=lambda r: abs(r["diff"]), reverse=True)
            if flip:
                lines2.append("| 因子 | 低PE(0-15) IC | 高PE(100+) IC | 差值 |\n|---|---|---|---|\n")
                for r in flip[:30]:
                    lines2.append(f"| {r['factor']} | {r['ic_low']:+.4f} | "
                                  f"{r['ic_high']:+.4f} | {r['diff']:+.4f} |\n")
                lines2.append("\n")

    # ── 行业分层 ──
    if not layer_df_ind.empty:
        lines2.append("## 三、行业分层 (主要行业, 样本 ≥10000)\n\n")
        lines2.append("### 3.1 各行业最强因子 (按 |IC(20d)| 前 10)\n\n")
        ind_buckets = layer_df_ind["bucket"].unique().tolist()
        ind_buckets = sorted(ind_buckets,
                             key=lambda b: -full[full["industry"] == b].shape[0])
        for bk in ind_buckets[:30]:
            sub = layer_df_ind[(layer_df_ind["bucket"] == bk) & (layer_df_ind["h"] == 20)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["abs_ic"] = sub["ic"].abs()
            sub = sub.sort_values("abs_ic", ascending=False).head(10)
            n_samples = full[full["industry"] == bk].shape[0]
            lines2.append(f"#### {bk} (样本 {n_samples:,})\n\n| 因子 | IC(20d) | 样本 |\n|---|---|---|\n")
            for _, r in sub.iterrows():
                lines2.append(f"| {r['factor']} | {r['ic']:+.4f} | {r['n']:,} |\n")
            lines2.append("\n")

    REPORT_LAYER_FILE.write_text("".join(lines2), encoding="utf-8")


# ──────────────────── Main ────────────────────

def phase2_report_mode(mode: str = "raw"):
    """支持不同 mode 的聚合报告."""
    global REPORT_FILE, REPORT_LAYER_FILE
    log = setup_logger(f"phase2_{mode}", LOG_DIR / f"phase2_{mode}.log")
    suf = _suffix(mode)

    log.info("加载所有 group{%s} parquet...", suf)
    files = sorted(GROUPS_DIR.glob(f"group_*{suf}.parquet"))
    if not files:
        log.error("无 parquet")
        return
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            log.error("读 %s 失败: %s", f, e)
    full = pd.concat(dfs, ignore_index=True)
    log.info("总样本 %d, 列 %d", len(full), len(full.columns))

    excluded = {"ts_code", "trade_date", "industry", "name", "total_mv", "pe", "pe_ttm", "pb"} | \
        {f"r{h}" for h in (5, 10, 20, 30, 40)} | {f"dd{h}" for h in (5, 10, 20, 30, 40)} | \
        {"market_score_adj", "adx", "winner_rate", "main_net", "holder_pct",
         "mf_divergence", "mf_strength", "mf_consecutive", "mv_bucket", "pe_bucket"}
    factor_cols = [c for c in full.columns if c not in excluded]

    full["mv_bucket"] = full["total_mv"].apply(bucket_mv)
    full["pe_bucket"] = full["pe"].apply(bucket_pe)

    ic_rows = []
    for fc in factor_cols:
        x = full[fc]
        if not np.issubdtype(x.dtype, np.number):
            continue
        valid_mask = x.notna()
        if valid_mask.sum() < 1000:
            continue
        sigma = float(x.std())
        row = {"factor": fc, "n_valid": int(valid_mask.sum()), "sigma": sigma}
        for h in (5, 20, 40):
            r = full[f"r{h}"]
            mask = valid_mask & r.notna()
            if mask.sum() < 1000:
                row[f"ic_{h}"] = np.nan
                continue
            xv = x[mask].values
            yv = r[mask].values
            if np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
                row[f"ic_{h}"] = np.nan
            else:
                row[f"ic_{h}"] = float(np.corrcoef(xv, yv)[0, 1])
        ic_rows.append(row)
    ic_df = pd.DataFrame(ic_rows).sort_values("ic_20", key=lambda s: s.abs(), ascending=False)

    # 临时切换报告文件路径
    rep = OUT_DIR / f"report{suf}.md"
    rep_layer = OUT_DIR / f"report_by_layer{suf}.md"
    old_rep, old_layer = REPORT_FILE, REPORT_LAYER_FILE
    REPORT_FILE = rep
    REPORT_LAYER_FILE = rep_layer

    # 分层
    def layer_ic(group_col, labels, min_n=500, min_bucket_n=5000):
        rows = []
        for bk in labels:
            sub = full[full[group_col] == bk]
            if len(sub) < min_bucket_n:
                continue
            for fc in factor_cols:
                x = sub[fc]
                if not np.issubdtype(x.dtype, np.number) or x.notna().sum() < min_n:
                    continue
                for h in (5, 20, 40):
                    r = sub[f"r{h}"]
                    mask = x.notna() & r.notna()
                    if mask.sum() < min_n:
                        continue
                    xv = x[mask].values
                    yv = r[mask].values
                    if np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
                        continue
                    ic = float(np.corrcoef(xv, yv)[0, 1])
                    if abs(ic) > 0.005:
                        rows.append({"factor": fc, "bucket": bk, "h": h,
                                     "ic": ic, "n": int(mask.sum())})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    log.info("分层 mv...")
    mv_df = layer_ic("mv_bucket", MV_LABELS)
    log.info("分层 pe...")
    pe_df = layer_ic("pe_bucket", PE_LABELS)
    log.info("分层 行业...")
    ind_counts = full["industry"].value_counts()
    main_inds = ind_counts[ind_counts >= 10000].index.tolist()
    ind_df = layer_ic("industry", main_inds, min_n=300, min_bucket_n=10000)

    write_report(full, ic_df, mv_df, pe_df, ind_df, factor_cols)
    REPORT_FILE = old_rep
    REPORT_LAYER_FILE = old_layer
    log.info("done. 写到 %s, %s", rep, rep_layer)


# ──────────────────── Phase 3: validity_matrix.json ────────────────────

VALIDITY_FILE = OUT_DIR / "validity_matrix.json"
VALIDITY_STATS_FILE = OUT_DIR / "validity_matrix_stats.md"

# 激活规则 (v2, 跨段可比):
#   active = (best_win - q3_win) >= EXCESS_THRESHOLD AND best_win >= ABS_MIN
#
# 用 Q3 (中位桶, 信号无关基线) 作为该段段内基线.
# v1 用 60% 绝对阈值导致大盘段全失效 (Q3 中位 47.8%), 小盘段过度激活 (Q3 中位 59.7%).
EXCESS_THRESHOLD = 0.05   # 比中位桶高 5pp 才算"真信号"
ABS_MIN = 0.50            # 绝对底线: best_win 至少 50% (避免负胜率假激活)

# 各维度最低样本要求
MIN_SAMPLES_MV = 5000
MIN_SAMPLES_PE = 5000
MIN_SAMPLES_IND = 10000
# Q 桶内样本最低 (避免单桶噪音)
MIN_BUCKET_SAMPLES = 200


def _q_bucket_stats(sub: pd.DataFrame, factor: str, n_buckets: int = 5):
    """对 sub 按 factor 分 5 分位, 返回每桶 D+20 胜率列表 + best_q + sign + spread + q_thresholds."""
    x = sub[factor]
    valid = sub[x.notna() & sub["r20"].notna()]
    if len(valid) < MIN_BUCKET_SAMPLES * n_buckets:
        return None
    try:
        ranks = valid[factor].rank(pct=True, method="first")
        bucket = pd.cut(ranks, bins=n_buckets,
                         labels=[f"Q{i+1}" for i in range(n_buckets)],
                         include_lowest=True)
        # q_thresholds: P20/P40/P60/P80 边界, 用于运行时判定股票 q 桶
        q_thresholds = np.percentile(valid[factor].values, [20, 40, 60, 80]).tolist()
    except Exception:
        return None
    wins = []
    avgs = []
    ns = []
    for q_label in [f"Q{i+1}" for i in range(n_buckets)]:
        b = valid[bucket == q_label]
        if len(b) < MIN_BUCKET_SAMPLES:
            wins.append(None)
            avgs.append(None)
            ns.append(int(len(b)))
            continue
        r = b["r20"].dropna()
        wins.append(float((r > 0).mean()))
        avgs.append(float(r.mean()))
        ns.append(int(len(b)))
    # best_q (1-indexed): 胜率最大的桶
    valid_wins = [(i + 1, w) for i, w in enumerate(wins) if w is not None]
    if not valid_wins:
        return None
    best_q, best_w = max(valid_wins, key=lambda r: r[1])
    worst_q, worst_w = min(valid_wins, key=lambda r: r[1])
    spread = best_w - worst_w
    # sign: +1 表示因子值越高收益越好, -1 反之
    # best_q == 5 → sign +1; best_q == 1 → sign -1; 中间桶 sign 0
    if best_q == n_buckets:
        sign = 1
    elif best_q == 1:
        sign = -1
    else:
        sign = 0
    # 新激活规则: best_win 比 Q3 (中位桶) 高 EXCESS_THRESHOLD 且绝对值 >= ABS_MIN
    q3_w = wins[2] if len(wins) >= 3 else None
    excess = (best_w - q3_w) if q3_w is not None else None
    is_active = (
        q3_w is not None
        and excess >= EXCESS_THRESHOLD
        and best_w >= ABS_MIN
    )
    return {
        "q_thresholds": [round(t, 6) for t in q_thresholds],
        "q_wins": [round(w, 4) if w is not None else None for w in wins],
        "q_avgs": [round(a, 4) if a is not None else None for a in avgs],
        "q_ns": ns,
        "best_q": best_q,
        "best_win": round(best_w, 4),
        "q3_win": round(q3_w, 4) if q3_w is not None else None,
        "excess": round(excess, 4) if excess is not None else None,
        "worst_q": worst_q,
        "worst_win": round(worst_w, 4),
        "spread": round(spread, 4),
        "sign": sign,
        "active": is_active,
    }


def phase3_validity_matrix():
    """从 raw parquet 算 validity_matrix.json — sparse_layered 数据底座."""
    log = setup_logger("phase3", LOG_DIR / "phase3.log")

    log.info("加载所有 group parquet...")
    files = sorted(GROUPS_DIR.glob("group_???.parquet"))
    if not files:
        log.error("无 parquet")
        return
    dfs = [pd.read_parquet(f) for f in files]
    full = pd.concat(dfs, ignore_index=True)
    log.info("总样本 %d, 列 %d", len(full), len(full.columns))

    # 上下文分桶
    full["mv_bucket"] = full["total_mv"].apply(bucket_mv)
    full["pe_bucket"] = full["pe"].apply(bucket_pe)

    # 因子列
    excluded = {"ts_code", "trade_date", "industry", "name",
                "total_mv", "pe", "pe_ttm", "pb",
                "mv_bucket", "pe_bucket"} | \
        {f"r{h}" for h in (5, 10, 20, 30, 40)} | \
        {f"dd{h}" for h in (5, 10, 20, 30, 40)} | \
        {"market_score_adj", "adx", "winner_rate", "main_net", "holder_pct",
         "mf_divergence", "mf_strength", "mf_consecutive"}
    factor_cols = [c for c in full.columns
                   if c not in excluded and pd.api.types.is_numeric_dtype(full[c])]
    log.info("因子数 %d", len(factor_cols))

    # 全市场基准胜率
    base_win = float((full["r20"] > 0).mean())
    log.info("基准 D+20 胜率 %.4f", base_win)

    # 主要行业 (Top 30, 样本 ≥ MIN_SAMPLES_IND)
    ind_counts = full["industry"].value_counts()
    main_inds = ind_counts[ind_counts >= MIN_SAMPLES_IND].index.tolist()[:30]
    log.info("主要行业 %d 个", len(main_inds))

    # ───── 算每个因子在每个 (dim, segment) 下的 5 分位胜率 ─────
    matrix = {}   # factor → {"global": {...}, "mv": {seg: {...}}, "pe": {...}, "industry": {...}}
    n_active_total = 0
    n_factor_active = 0
    counter = 0

    for fc in factor_cols:
        counter += 1
        if counter % 30 == 0:
            log.info("  progress %d/%d", counter, len(factor_cols))

        entry = {"global": None, "mv": {}, "pe": {}, "industry": {}}

        # 全市场
        g_stats = _q_bucket_stats(full, fc)
        if g_stats:
            entry["global"] = g_stats

        # mv 分层
        for mv_seg in MV_LABELS:
            sub = full[full["mv_bucket"] == mv_seg]
            if len(sub) < MIN_SAMPLES_MV:
                continue
            s = _q_bucket_stats(sub, fc)
            if s:
                entry["mv"][mv_seg] = s
                if s["active"]:
                    n_active_total += 1

        # pe 分层
        for pe_seg in PE_LABELS:
            sub = full[full["pe_bucket"] == pe_seg]
            if len(sub) < MIN_SAMPLES_PE:
                continue
            s = _q_bucket_stats(sub, fc)
            if s:
                entry["pe"][pe_seg] = s
                if s["active"]:
                    n_active_total += 1

        # industry 分层
        for ind in main_inds:
            sub = full[full["industry"] == ind]
            if len(sub) < MIN_SAMPLES_IND:
                continue
            s = _q_bucket_stats(sub, fc)
            if s:
                entry["industry"][ind] = s
                if s["active"]:
                    n_active_total += 1

        # 该因子是否在任何分层段下有 active
        any_active = (
            (entry["global"] and entry["global"]["active"]) or
            any(v["active"] for v in entry["mv"].values()) or
            any(v["active"] for v in entry["pe"].values()) or
            any(v["active"] for v in entry["industry"].values())
        )
        if any_active:
            n_factor_active += 1

        matrix[fc] = entry

    # ───── 写 JSON ─────
    output = {
        "meta": {
            "n_samples": int(len(full)),
            "n_stocks": int(full["ts_code"].nunique()),
            "n_factors": len(factor_cols),
            "n_industries": len(main_inds),
            "base_win_rate": round(base_win, 4),
            "activation_rule": "v2: (best_win - q3_win) >= excess_threshold AND best_win >= abs_min",
            "excess_threshold": EXCESS_THRESHOLD,
            "abs_min": ABS_MIN,
            "hold_period": "D+20",
            "min_bucket_samples": MIN_BUCKET_SAMPLES,
            "data_period": [str(full["trade_date"].min()), str(full["trade_date"].max())],
            "industries": main_inds,
            "mv_labels": MV_LABELS,
            "pe_labels": PE_LABELS,
            "generated_at": datetime.now().isoformat(),
        },
        "factors": matrix,
    }
    VALIDITY_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    log.info("validity_matrix 写入 %s", VALIDITY_FILE)
    log.info("总激活段数 %d, 至少有一段激活的因子 %d/%d",
             n_active_total, n_factor_active, len(factor_cols))

    # ───── 统计报告 ─────
    write_validity_stats(matrix, factor_cols, main_inds, base_win)


def write_validity_stats(matrix: dict, factor_cols: list[str],
                          main_inds: list[str], base_win: float):
    """统计报告: 激活率分布 + Top 激活因子."""
    lines = ["# Validity Matrix 统计\n\n"]
    lines.append(f"> 生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> **激活规则 v2**: `(best_win - q3_win) >= {EXCESS_THRESHOLD*100:.0f}pp` AND `best_win >= {ABS_MIN*100:.0f}%`\n")
    lines.append(f"> Q3 = 中位桶, 作为该段段内 \"信号无关\" 基线 (跨段可比)\n")
    lines.append(f"> 全市场基准 D+20 胜率: {base_win*100:.2f}%\n\n")

    # 1) 各维度激活段数统计
    lines.append("## 一、激活段分布 (按维度)\n\n")
    lines.append("| 维度 | 总段数 | 激活段 | 激活率 |\n|---|---|---|---|\n")

    for dim, label, segs in [
        ("global", "全市场", ["global"]),
        ("mv", "市值", MV_LABELS),
        ("pe", "PE", PE_LABELS),
        ("industry", "行业", main_inds),
    ]:
        total = 0
        active = 0
        for fc in factor_cols:
            entry = matrix.get(fc, {})
            if dim == "global":
                if entry.get("global"):
                    total += 1
                    if entry["global"]["active"]:
                        active += 1
            else:
                d = entry.get(dim, {})
                for seg in segs:
                    if seg in d:
                        total += 1
                        if d[seg]["active"]:
                            active += 1
        rate = active / total * 100 if total else 0
        lines.append(f"| {label} | {total} | {active} | {rate:.1f}% |\n")
    lines.append("\n")

    # 2) Top 20 最强 (best_win) 段
    lines.append("## 二、Top 20 最强激活段 (按 best_win)\n\n")
    lines.append("| 因子 | 维度 | 段 | best_q | best_win | sign | 样本 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    rows = []
    for fc, entry in matrix.items():
        for dim_name, dim_data in [("mv", entry.get("mv", {})),
                                    ("pe", entry.get("pe", {})),
                                    ("industry", entry.get("industry", {}))]:
            for seg, s in dim_data.items():
                if s["active"]:
                    n_samp = s["q_ns"][s["best_q"] - 1]
                    rows.append({
                        "factor": fc, "dim": dim_name, "seg": seg,
                        "best_q": s["best_q"], "best_win": s["best_win"],
                        "sign": s["sign"], "n": n_samp,
                    })
    rows.sort(key=lambda r: r["best_win"], reverse=True)
    for r in rows[:20]:
        lines.append(f"| {r['factor']} | {r['dim']} | {r['seg']} | "
                     f"Q{r['best_q']} | {r['best_win']*100:.1f}% | "
                     f"{r['sign']:+d} | {r['n']:,} |\n")
    lines.append("\n")

    # 3) 各维度激活段最多的 Top 10 因子
    lines.append("## 三、各因子激活段统计\n\n")
    factor_active_counts = []
    for fc, entry in matrix.items():
        n_mv = sum(1 for s in entry.get("mv", {}).values() if s["active"])
        n_pe = sum(1 for s in entry.get("pe", {}).values() if s["active"])
        n_ind = sum(1 for s in entry.get("industry", {}).values() if s["active"])
        n_tot = n_mv + n_pe + n_ind
        if n_tot:
            factor_active_counts.append({
                "factor": fc, "mv": n_mv, "pe": n_pe, "ind": n_ind, "total": n_tot
            })
    factor_active_counts.sort(key=lambda r: r["total"], reverse=True)
    lines.append("| 因子 | mv 段 | PE 段 | 行业段 | 总激活段 |\n|---|---|---|---|---|\n")
    for r in factor_active_counts[:25]:
        lines.append(f"| {r['factor']} | {r['mv']} | {r['pe']} | "
                     f"{r['ind']} | **{r['total']}** |\n")
    lines.append(f"\n激活率最高的 25 个因子覆盖了大部分有效信号。\n\n")

    # 4) 各 mv 段激活的因子数量
    lines.append("## 四、各市值段下的激活因子数\n\n")
    lines.append("| 市值段 | 激活因子数 |\n|---|---|\n")
    for mv in MV_LABELS:
        n = sum(1 for fc in factor_cols
                if matrix.get(fc, {}).get("mv", {}).get(mv, {}).get("active"))
        lines.append(f"| {mv} | {n} |\n")
    lines.append("\n## 五、各 PE 段下的激活因子数\n\n")
    lines.append("| PE 段 | 激活因子数 |\n|---|---|\n")
    for pe in PE_LABELS:
        n = sum(1 for fc in factor_cols
                if matrix.get(fc, {}).get("pe", {}).get(pe, {}).get("active"))
        lines.append(f"| PE {pe} | {n} |\n")

    VALIDITY_STATS_FILE.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["0", "1", "2", "3", "all"], default="all")
    parser.add_argument("--group", type=int)
    parser.add_argument("--mode", choices=["raw", "ha"], default="raw")
    args = parser.parse_args()

    log = setup_logger("main", LOG_DIR / "master.log")

    if args.phase in ("0", "all"):
        log.info("===== Phase 0: 因子元数据 =====")
        phase0_meta()

    if args.phase in ("1", "all"):
        log.info("===== Phase 1: 跑因子 (mode=%s) =====", args.mode)
        phase1_run(only_group=args.group, mode=args.mode)

    if args.phase in ("2", "all"):
        log.info("===== Phase 2: 聚合报告 (mode=%s) =====", args.mode)
        if args.mode == "raw":
            phase2_report()
        else:
            phase2_report_mode(args.mode)

    if args.phase in ("3", "all"):
        log.info("===== Phase 3: validity_matrix =====")
        phase3_validity_matrix()

    log.info("DONE")
