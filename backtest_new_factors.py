"""新因子单因子回测 (8 因子 × 4 周期)

Universe: 市值≥20亿, 排除 ST, 全 A 股
Period:   12 个月 (今天往回 365 天, 留 60 天给 D+40)
Hold:     D+5 / D+20 / D+30 / D+40 — 涨幅 / 胜率 / 最大回撤 + IC
Factors:
  1. mf_divergence       (Tushare moneyflow_dc, 个股 MA3)
  2. mf_strength         (Tushare moneyflow_dc, 个股 MA3 流入率)
  3. mf_consecutive      (Tushare moneyflow_dc, 主力连续净流入天数)
  4. market_score_adj    (Tushare moneyflow_mkt_dc, MA5 全市场)
  5. adx                 (Tushare stk_factor_pro)
  6. winner_rate         (Tushare cyq_perf)
  7. main_net            (Tushare moneyflow 旧接口绝对量)
  8. holder_pct          (Tushare stk_holdernumber 季度环比)

架构:
  Phase 1  build_universe   → output/backtest_factors_2026_04/universe.json
  Phase 2  run_groups       → 多进程 5 worker, 每组 100 只, 等差数列分配
  Phase 3  aggregate_report → output/backtest_factors_2026_04/report.md

CheckPoint:
  - per-stock 原始时序: raw_data/{ts_code}.json (存在则跳过下载)
  - per-group 状态:     group_results/group_NNN.status.json (done 则跳过)
  - per-group 样本:     group_results/group_NNN.jsonl (一行一条样本)
  - 任意阶段中断后, 重新执行同命令即可恢复

用法:
  python backtest_new_factors.py --phase 1                  # 只建股票池
  python backtest_new_factors.py --phase 2                  # 多进程跑所有组
  python backtest_new_factors.py --phase 2 --group 7        # 单跑第 7 组 (debug)
  python backtest_new_factors.py --phase 3                  # 聚合 + 出报告
  python backtest_new_factors.py --phase all                # 全流程
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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

# ─── 路径 ───
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 让子进程也能找到 .env (TUSHARE_TOKEN)
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.cloubic", override=False)

# ─── 输出目录 ───
OUTPUT_DIR = ROOT / "output" / "backtest_factors_2026_04"
RAW_DATA_DIR = OUTPUT_DIR / "raw_data"
GROUP_RESULTS_DIR = OUTPUT_DIR / "group_results"
LOG_DIR = OUTPUT_DIR / "logs"
UNIVERSE_FILE = OUTPUT_DIR / "universe.json"
MARKET_FILE = OUTPUT_DIR / "market_data.json"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
REPORT_FILE = OUTPUT_DIR / "report.md"

for d in (OUTPUT_DIR, RAW_DATA_DIR, GROUP_RESULTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── 配置 ───
GROUP_SIZE = 100
NUM_WORKERS = 5
PERIOD_DAYS = 365
SAFE_TAIL_DAYS = 60        # 保留 D+40 + 缓冲
HOLD_PERIODS = [5, 20, 30, 40]
MARKET_CAP_MIN = 20.0      # 亿元 (total_mv 单位是万元, 阈值 200000)
RATE_LIMIT_SLEEP = 0.4     # 每次 Tushare 调用后睡 (5 worker × 150 calls/min ≈ 12.5 calls/sec)
MAX_RETRY = 3

# ─── 日志 ───
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


# ─── Tushare client (lazy, per-process) ───
_PRO = None


def get_pro():
    global _PRO
    if _PRO is not None:
        return _PRO
    import tushare as ts
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 缺失")
    _PRO = ts.pro_api(token=token, timeout=30)
    return _PRO


def call_tushare(fn_name: str, **kwargs):
    """带重试的 Tushare 调用。"""
    pro = get_pro()
    fn = getattr(pro, fn_name)
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            df = fn(**kwargs)
            time.sleep(RATE_LIMIT_SLEEP)
            return df
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "rate" in msg or "limit" in msg or "频率" in msg:
                time.sleep(5 + attempt * 5)
            else:
                time.sleep(2)
    raise last_err


# ─── 日期工具 ───
def today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def days_ago_str(n: int) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y%m%d")


def safe_cutoff_str() -> str:
    return (datetime.now() - timedelta(days=SAFE_TAIL_DAYS)).strftime("%Y%m%d")


# ═══════════════════════════════════════════════════════════════
# Phase 1: build_universe
# ═══════════════════════════════════════════════════════════════

def phase1_build_universe(force: bool = False):
    """筛全市场 → 分组 → 保存 universe.json"""
    log = setup_logger("phase1", LOG_DIR / "phase1.log")
    if UNIVERSE_FILE.exists() and not force:
        with open(UNIVERSE_FILE, encoding="utf-8") as f:
            uni = json.load(f)
        log.info("universe 已存在: %d 只 %d 组, 跳过 (用 --force 重建)",
                 uni["total"], uni["num_groups"])
        return uni

    log.info("开始拉股票列表...")
    df = call_tushare("stock_basic",
                      exchange="", list_status="L",
                      fields="ts_code,symbol,name,area,industry,list_date")
    log.info("Tushare 返回 %d 只在市股", len(df))

    # 排除 ST
    df = df[~df["name"].str.contains(r"\*?ST|退", regex=True, na=False)]
    log.info("排除 ST/退市: 剩 %d 只", len(df))

    # 拉最新 daily_basic 取市值
    log.info("拉最新 daily_basic 取市值 (用最近交易日)...")
    cutoff = safe_cutoff_str()
    df_basic = None
    for back in range(0, 10):
        d_try = (datetime.now() - timedelta(days=SAFE_TAIL_DAYS + back)).strftime("%Y%m%d")
        df_basic = call_tushare("daily_basic", trade_date=d_try,
                                fields="ts_code,total_mv")
        if df_basic is not None and not df_basic.empty:
            log.info("市值快照日: %s, 行数 %d", d_try, len(df_basic))
            break
    if df_basic is None or df_basic.empty:
        raise RuntimeError("拉不到 daily_basic")

    # total_mv 单位: 万元; 20 亿 = 200000 万元
    df_basic = df_basic[df_basic["total_mv"] >= MARKET_CAP_MIN * 10000]
    log.info("市值 ≥ %.0f 亿: %d 只", MARKET_CAP_MIN, len(df_basic))

    big = set(df_basic["ts_code"].tolist())
    df = df[df["ts_code"].isin(big)]
    df = df.sort_values("ts_code").reset_index(drop=True)

    # 分组 (稳定: 按 ts_code 升序, 100/组)
    stocks = []
    for i, row in df.iterrows():
        gid = i // GROUP_SIZE + 1
        stocks.append({
            "ts_code": row["ts_code"],
            "symbol": row["symbol"],
            "name": row["name"],
            "industry": row.get("industry", ""),
            "group_id": gid,
        })
    num_groups = (len(stocks) + GROUP_SIZE - 1) // GROUP_SIZE

    # 市值快照日, 留作 sample window 锚点
    universe = {
        "built_at": datetime.now().isoformat(),
        "criteria": {
            "market_cap_min_billion": MARKET_CAP_MIN,
            "exclude_st": True,
            "no_pe_filter": True,
        },
        "snapshot_date": d_try,
        "total": len(stocks),
        "num_groups": num_groups,
        "group_size": GROUP_SIZE,
        "stocks": stocks,
    }
    UNIVERSE_FILE.write_text(json.dumps(universe, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    log.info("universe 写入 %s: %d 只, %d 组", UNIVERSE_FILE, len(stocks), num_groups)
    return universe


# ═══════════════════════════════════════════════════════════════
# 因子计算: 各 fetcher (per-stock 时序拉取)
# ═══════════════════════════════════════════════════════════════

def fetch_stock_timeseries(ts_code: str, start: str, end: str) -> dict:
    """拉单股全套时序数据, 返回字典。"""
    out = {}
    # 1. daily (OHLCV)
    df = call_tushare("daily", ts_code=ts_code, start_date=start, end_date=end)
    if df is not None and not df.empty:
        df = df.sort_values("trade_date")
        out["daily"] = df.to_dict(orient="records")
    else:
        out["daily"] = []
    # 2. moneyflow_dc (主力/散户分层)
    try:
        df = call_tushare("moneyflow_dc", ts_code=ts_code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            df = df.sort_values("trade_date")
            out["moneyflow_dc"] = df.to_dict(orient="records")
        else:
            out["moneyflow_dc"] = []
    except Exception as e:
        out["moneyflow_dc"] = []
        out["moneyflow_dc_err"] = str(e)[:120]
    # 3. moneyflow (旧接口, main_net)
    try:
        df = call_tushare("moneyflow", ts_code=ts_code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            df = df.sort_values("trade_date")
            out["moneyflow"] = df.to_dict(orient="records")
        else:
            out["moneyflow"] = []
    except Exception as e:
        out["moneyflow"] = []
    # 4. stk_factor_pro (ADX, 真实字段带复权后缀)
    try:
        df = call_tushare("stk_factor_pro", ts_code=ts_code, start_date=start, end_date=end,
                         fields="ts_code,trade_date,dmi_adx_qfq,dmi_pdi_qfq,dmi_mdi_qfq")
        if df is not None and not df.empty:
            df = df.sort_values("trade_date")
            # 重命名为不带后缀, 后续算因子代码不用改
            df = df.rename(columns={
                "dmi_adx_qfq": "dmi_adx",
                "dmi_pdi_qfq": "dmi_pdi",
                "dmi_mdi_qfq": "dmi_mdi",
            })
            out["stk_factor_pro"] = df.to_dict(orient="records")
        else:
            out["stk_factor_pro"] = []
    except Exception as e:
        out["stk_factor_pro"] = []
    # 5. cyq_perf (winner_rate)
    try:
        df = call_tushare("cyq_perf", ts_code=ts_code, start_date=start, end_date=end,
                         fields="ts_code,trade_date,winner_rate")
        if df is not None and not df.empty:
            df = df.sort_values("trade_date")
            out["cyq_perf"] = df.to_dict(orient="records")
        else:
            out["cyq_perf"] = []
    except Exception as e:
        out["cyq_perf"] = []
    # 6. stk_holdernumber (季度)
    try:
        df = call_tushare("stk_holdernumber", ts_code=ts_code,
                         start_date=days_ago_str(PERIOD_DAYS + 365),  # 多拉一年保证有 ≥2 期
                         end_date=end,
                         fields="ts_code,end_date,holder_num")
        if df is not None and not df.empty:
            df = df.sort_values("end_date")
            out["holders"] = df.to_dict(orient="records")
        else:
            out["holders"] = []
    except Exception as e:
        out["holders"] = []
    return out


def fetch_market_moneyflow(start: str, end: str) -> list[dict]:
    """全市场主力/散户资金面 (一次性, worker 共享)。"""
    df = call_tushare("moneyflow_mkt_dc", start_date=start, end_date=end)
    if df is None or df.empty:
        return []
    df = df.sort_values("trade_date")
    return df.to_dict(orient="records")


def ensure_market_data(start: str, end: str, log: logging.Logger) -> list[dict]:
    if MARKET_FILE.exists():
        try:
            with open(MARKET_FILE, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("rows") and d.get("start") == start and d.get("end") == end:
                log.info("market_moneyflow 缓存命中: %d 行", len(d["rows"]))
                return d["rows"]
        except Exception:
            pass
    log.info("拉 moneyflow_mkt_dc %s ~ %s", start, end)
    rows = fetch_market_moneyflow(start, end)
    MARKET_FILE.write_text(json.dumps({"start": start, "end": end, "rows": rows},
                                       ensure_ascii=False), encoding="utf-8")
    log.info("market_moneyflow 写入 %d 行", len(rows))
    return rows


# ═══════════════════════════════════════════════════════════════
# 因子计算
# ═══════════════════════════════════════════════════════════════

def _safe_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def compute_mf_factors_at(ts: dict, idx_in_mf_dc: int) -> dict:
    """idx_in_mf_dc: moneyflow_dc 列表中的位置 (按时间升序)。
    返回 {mf_divergence: float, mf_strength: float, mf_consecutive: int}。
    """
    rows = ts.get("moneyflow_dc") or []
    if idx_in_mf_dc < 2:   # 不够 3 日 MA
        return {}
    window = rows[max(0, idx_in_mf_dc - 2): idx_in_mf_dc + 1]   # 3 日窗口
    main_nets = []
    retail_nets = []
    main_rates = []
    for r in window:
        # moneyflow_dc 字段: buy_lg_amount/buy_elg_amount 已是净值(买-卖)
        # 主力 = 大单+特大单
        mn = (_safe_float(r.get("buy_lg_amount")) or 0) + (_safe_float(r.get("buy_elg_amount")) or 0)
        # 散户 = 小单
        rn = _safe_float(r.get("buy_sm_amount")) or 0
        main_nets.append(mn)
        retail_nets.append(rn)
        # rate 字段
        if r.get("buy_lg_amount_rate") is not None and r.get("buy_elg_amount_rate") is not None:
            main_rates.append((_safe_float(r["buy_lg_amount_rate"]) or 0) +
                              (_safe_float(r["buy_elg_amount_rate"]) or 0))

    main_ma3 = float(np.mean(main_nets))
    retail_ma3 = float(np.mean(retail_nets))
    main_rate_ma3 = float(np.mean(main_rates)) if main_rates else None

    # 1. mf_divergence: 把规则离散化为 score
    if main_ma3 < 0 and retail_ma3 > 0:
        divergence = -12.0   # distribution
    elif main_ma3 > 0 and retail_ma3 < 0:
        divergence = +8.0    # smart_accumulating
    elif main_ma3 > 0 and retail_ma3 > 0:
        divergence = +3.0    # consensus_buy
    elif main_ma3 < 0 and retail_ma3 < 0:
        divergence = -5.0    # consensus_sell
    else:
        divergence = 0.0

    # 2. mf_strength: rate 优先, 绝对量回退
    strength = 0.0
    if main_rate_ma3 is not None:
        r = main_rate_ma3
        if r >= 3.0: strength = 10.0
        elif r >= 1.0: strength = 5.0
        elif r <= -3.0: strength = -10.0
        elif r <= -1.0: strength = -5.0
    else:
        mn = main_ma3
        if mn >= 5000: strength = 10.0
        elif mn >= 1000: strength = 5.0
        elif mn <= -5000: strength = -10.0
        elif mn <= -1000: strength = -5.0

    # 3. mf_consecutive: 从 idx 往回数连续主力净流入天数
    consec = 0
    for j in range(idx_in_mf_dc, -1, -1):
        r = rows[j]
        mn = (_safe_float(r.get("buy_lg_amount")) or 0) + (_safe_float(r.get("buy_elg_amount")) or 0)
        if mn > 0:
            consec += 1
        else:
            break
    if consec >= 5: consec_score = 5.0
    elif consec >= 3: consec_score = 3.0
    else: consec_score = 0.0

    return {
        "mf_divergence": divergence,
        "mf_strength": strength,
        "mf_consecutive": consec_score,
        "_consec_days": consec,
        "_main_rate_ma3": main_rate_ma3,
        "_main_ma3": main_ma3,
    }


def compute_market_score_adj_at(mkt_rows: list[dict], idx: int) -> float | None:
    """全市场主力/散户 5 日 MA 信号 → ±5 分。"""
    if idx < 4:
        return None
    window = mkt_rows[max(0, idx - 4): idx + 1]   # 5 日
    mains = []
    retails = []
    for r in window:
        m = (_safe_float(r.get("buy_lg_amount")) or 0) + (_safe_float(r.get("buy_elg_amount")) or 0)
        rt = _safe_float(r.get("buy_sm_amount")) or 0
        mains.append(m)
        retails.append(rt)
    main_ma5 = np.mean(mains)
    retail_ma5 = np.mean(retails)
    if main_ma5 < 0 and retail_ma5 > 0:
        return -5.0    # distribution
    if main_ma5 > 0 and retail_ma5 < 0:
        return +4.0    # smart_accumulating
    if main_ma5 > 0 and retail_ma5 > 0:
        return +2.0
    if main_ma5 < 0 and retail_ma5 < 0:
        return -3.0
    return 0.0


def compute_adx_at(ts: dict, target_date: str) -> float | None:
    """ADX 趋势强度 → score。"""
    rows = ts.get("stk_factor_pro") or []
    for r in reversed(rows):
        if r.get("trade_date") <= target_date:
            adx = _safe_float(r.get("dmi_adx"))
            pdi = _safe_float(r.get("dmi_pdi"))
            mdi = _safe_float(r.get("dmi_mdi"))
            if adx is None or pdi is None or mdi is None:
                return None
            up = pdi > mdi
            if adx >= 30 and up: return 10.0
            if adx >= 25 and up: return 6.0
            if adx >= 25 and not up: return -8.0
            if adx < 20: return -2.0
            return 0.0
    return None


def compute_winner_rate_at(ts: dict, target_date: str) -> float | None:
    rows = ts.get("cyq_perf") or []
    for r in reversed(rows):
        if r.get("trade_date") <= target_date:
            wr = _safe_float(r.get("winner_rate"))
            if wr is None:
                return None
            if wr <= 20: return 12.0
            if wr <= 35: return 6.0
            if wr >= 85: return -10.0
            if wr >= 75: return -5.0
            return 0.0
    return None


def compute_main_net_at(ts: dict, target_date: str) -> float | None:
    """旧接口 main_net (10 日累计) → score。"""
    rows = ts.get("moneyflow") or []
    # 找 target_date 及之前的 10 行
    selected = [r for r in rows if r.get("trade_date") <= target_date]
    if len(selected) < 10:
        return None
    last10 = selected[-10:]
    total_main = 0.0
    for r in last10:
        # buy_lg_vol / buy_elg_vol / sell_lg_vol / sell_elg_vol  (单位手, 改用 amount)
        buy_lg = _safe_float(r.get("buy_lg_amount")) or 0
        sell_lg = _safe_float(r.get("sell_lg_amount")) or 0
        buy_elg = _safe_float(r.get("buy_elg_amount")) or 0
        sell_elg = _safe_float(r.get("sell_elg_amount")) or 0
        # 单位是万元
        total_main += (buy_lg - sell_lg) + (buy_elg - sell_elg)
    if total_main >= 5000: return 10.0
    if total_main >= 1000: return 5.0
    if total_main <= -5000: return -10.0
    if total_main <= -1000: return -5.0
    return 0.0


def compute_holder_pct_at(ts: dict, target_date: str) -> float | None:
    rows = ts.get("holders") or []
    valid = [r for r in rows if r.get("end_date") <= target_date]
    if len(valid) < 2:
        return None
    first = _safe_float(valid[-2].get("holder_num"))
    last = _safe_float(valid[-1].get("holder_num"))
    if not first or not last or first <= 0:
        return None
    pct = (last - first) / first * 100
    if abs(pct) > 500:
        return None   # 异常
    if pct <= -5: return 8.0
    if pct <= -2: return 4.0
    if pct >= 5: return -6.0
    if pct >= 2: return -3.0
    return 0.0


# ═══════════════════════════════════════════════════════════════
# 收益 / 回撤计算
# ═══════════════════════════════════════════════════════════════

def compute_returns_and_dd(daily: list[dict], idx_t: int) -> dict:
    """从 idx_t 这天买入, 计算各持有期的收益和最大回撤。"""
    if idx_t >= len(daily):
        return {}
    close_t = _safe_float(daily[idx_t].get("close"))
    if not close_t:
        return {}
    out = {}
    for h in HOLD_PERIODS:
        if idx_t + h >= len(daily):
            continue
        future = daily[idx_t + 1: idx_t + h + 1]
        closes = [_safe_float(r.get("close")) for r in future]
        closes = [c for c in closes if c]
        if len(closes) < h:
            continue
        end_close = closes[-1]
        ret_pct = (end_close - close_t) / close_t * 100
        # 最大回撤: 持有期内 (close[i] - max_so_far) / max_so_far
        peak = close_t
        max_dd = 0.0
        for c in closes:
            peak = max(peak, c)
            dd = (c - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
        out[f"r{h}"] = ret_pct
        out[f"dd{h}"] = max_dd   # 负数
    return out


# ═══════════════════════════════════════════════════════════════
# Per-group worker
# ═══════════════════════════════════════════════════════════════

def worker_run_group(group_id: int) -> dict:
    """跑一组 100 只股票, 返回状态摘要。"""
    log_file = LOG_DIR / f"worker_group_{group_id:03d}.log"
    log = setup_logger(f"worker_g{group_id}", log_file)

    status_file = GROUP_RESULTS_DIR / f"group_{group_id:03d}.status.json"
    samples_file = GROUP_RESULTS_DIR / f"group_{group_id:03d}.jsonl"

    # 已 done 跳过
    if status_file.exists():
        try:
            with open(status_file, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("status") == "done":
                log.info("[skip] group %d 已完成", group_id)
                return s
        except Exception:
            pass

    # 加载 universe
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        uni = json.load(f)
    stocks = [s for s in uni["stocks"] if s["group_id"] == group_id]
    if not stocks:
        return {"group_id": group_id, "status": "empty"}

    start = days_ago_str(PERIOD_DAYS)
    end = today_str()

    log.info("group %d 启动, %d 只, 期间 %s ~ %s", group_id, len(stocks), start, end)

    # 加载/拉 market 数据
    mkt_rows = ensure_market_data(start, end, log)
    mkt_date_idx = {r["trade_date"]: i for i, r in enumerate(mkt_rows)}

    t0 = time.time()
    samples_written = 0
    fail_stocks = []

    # 追加模式 (resume 时会重复, 用 set 去重比较麻烦, 简单起见: 每次重跑该组就清空)
    if samples_file.exists():
        samples_file.unlink()

    with open(samples_file, "w", encoding="utf-8") as fout:
        for i, st in enumerate(stocks):
            ts_code = st["ts_code"]
            try:
                ts = ensure_stock_data(ts_code, start, end, log)
                daily = ts.get("daily") or []
                if len(daily) < 50:
                    log.warning("[skip] %s 数据不足 (%d 天)", ts_code, len(daily))
                    continue

                mf_dc_date_idx = {r["trade_date"]: idx for idx, r in
                                  enumerate(ts.get("moneyflow_dc") or [])}

                for idx_t, row in enumerate(daily):
                    td = row.get("trade_date")
                    # 留够 D+40 数据
                    if idx_t + max(HOLD_PERIODS) >= len(daily):
                        continue

                    # 计算 8 因子
                    factors = {}

                    # 1-3. mf_*
                    if td in mf_dc_date_idx:
                        idx_mf = mf_dc_date_idx[td]
                        f_mf = compute_mf_factors_at(ts, idx_mf)
                        if f_mf:
                            factors["mf_divergence"] = f_mf.get("mf_divergence")
                            factors["mf_strength"] = f_mf.get("mf_strength")
                            factors["mf_consecutive"] = f_mf.get("mf_consecutive")

                    # 4. market_score_adj
                    if td in mkt_date_idx:
                        factors["market_score_adj"] = compute_market_score_adj_at(
                            mkt_rows, mkt_date_idx[td])

                    # 5-8
                    factors["adx"] = compute_adx_at(ts, td)
                    factors["winner_rate"] = compute_winner_rate_at(ts, td)
                    factors["main_net"] = compute_main_net_at(ts, td)
                    factors["holder_pct"] = compute_holder_pct_at(ts, td)

                    # 收益与回撤
                    ret = compute_returns_and_dd(daily, idx_t)
                    if not ret:
                        continue

                    sample = {
                        "ts_code": ts_code,
                        "trade_date": td,
                        **factors,
                        **ret,
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    samples_written += 1

                if (i + 1) % 10 == 0:
                    log.info("group %d 进度 %d/%d, samples=%d", group_id,
                             i + 1, len(stocks), samples_written)
            except Exception as e:
                log.error("[error] %s: %s\n%s", ts_code, e, traceback.format_exc())
                fail_stocks.append({"ts_code": ts_code, "err": str(e)[:200]})

    dur = time.time() - t0
    status = {
        "group_id": group_id,
        "status": "done",
        "stocks": len(stocks),
        "samples": samples_written,
        "duration_sec": int(dur),
        "failed_stocks": fail_stocks,
        "finished_at": datetime.now().isoformat(),
    }
    status_file.write_text(json.dumps(status, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    log.info("group %d 完成: samples=%d 失败=%d 耗时=%.1fs",
             group_id, samples_written, len(fail_stocks), dur)
    return status


def ensure_stock_data(ts_code: str, start: str, end: str, log: logging.Logger) -> dict:
    """单股原始数据缓存。存在则跳过下载。"""
    f = RAW_DATA_DIR / f"{ts_code}.json"
    if f.exists():
        try:
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            if d.get("start") == start and d.get("end") == end:
                return d.get("ts", {})
        except Exception:
            pass
    log.debug("下载 %s 时序", ts_code)
    ts = fetch_stock_timeseries(ts_code, start, end)
    f.write_text(json.dumps({"start": start, "end": end, "ts": ts},
                            ensure_ascii=False), encoding="utf-8")
    return ts


# ═══════════════════════════════════════════════════════════════
# Phase 2: 多进程跑全部组
# ═══════════════════════════════════════════════════════════════

def assign_groups_to_workers(num_groups: int, num_workers: int) -> dict:
    """等差数列分配: worker k 接 group (k, k+W, k+2W, ...)"""
    out = {k: [] for k in range(num_workers)}
    for g in range(1, num_groups + 1):
        worker_idx = (g - 1) % num_workers
        out[worker_idx].append(g)
    return out


def update_progress(group_status: dict):
    prog = {}
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                prog = json.load(f)
        except Exception:
            pass
    prog[str(group_status["group_id"])] = {
        "status": group_status["status"],
        "samples": group_status.get("samples", 0),
        "stocks": group_status.get("stocks", 0),
        "duration_sec": group_status.get("duration_sec", 0),
        "failed_stocks": len(group_status.get("failed_stocks", [])),
        "finished_at": group_status.get("finished_at", ""),
    }
    PROGRESS_FILE.write_text(json.dumps(prog, ensure_ascii=False, indent=2),
                              encoding="utf-8")


def phase2_run_groups(only_group: int | None = None):
    """启动多进程跑分组。"""
    log = setup_logger("phase2", LOG_DIR / "phase2.log")

    if not UNIVERSE_FILE.exists():
        raise RuntimeError("universe 不存在, 先跑 --phase 1")
    with open(UNIVERSE_FILE, encoding="utf-8") as f:
        uni = json.load(f)
    num_groups = uni["num_groups"]

    if only_group is not None:
        log.info("单跑 group %d", only_group)
        s = worker_run_group(only_group)
        update_progress(s)
        return

    # 计算尚未完成的组
    pending = []
    for g in range(1, num_groups + 1):
        sf = GROUP_RESULTS_DIR / f"group_{g:03d}.status.json"
        if sf.exists():
            try:
                with open(sf, encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("status") == "done":
                    continue
            except Exception:
                pass
        pending.append(g)

    log.info("总组数=%d 待跑=%d worker=%d", num_groups, len(pending), NUM_WORKERS)
    if not pending:
        log.info("所有组都已完成")
        return

    # 等差数列分配
    assignments = assign_groups_to_workers(num_groups, NUM_WORKERS)
    for w, gs in assignments.items():
        worker_pending = [g for g in gs if g in pending]
        log.info("worker %d 负责 %d 组: %s%s", w + 1, len(worker_pending),
                 worker_pending[:8], " ..." if len(worker_pending) > 8 else "")

    # 多进程: 用 imap 看到完成顺序
    t0 = time.time()
    done_count = 0
    total_samples = 0

    with mp.Pool(processes=NUM_WORKERS) as pool:
        for s in pool.imap_unordered(worker_run_group, pending):
            done_count += 1
            update_progress(s)
            total_samples += s.get("samples", 0)
            elapsed = time.time() - t0
            eta_sec = elapsed / done_count * (len(pending) - done_count)
            log.info(
                "✓ group %d done | %d/%d (%.1f%%) | samples 累计 %d | 已耗时 %.1fmin | ETA %.1fmin",
                s["group_id"], done_count, len(pending),
                done_count / len(pending) * 100, total_samples,
                elapsed / 60, eta_sec / 60,
            )

    log.info("Phase 2 全部完成, 累计 samples=%d", total_samples)


# ═══════════════════════════════════════════════════════════════
# Phase 3: 聚合 + 报告
# ═══════════════════════════════════════════════════════════════

FACTOR_BINS = {
    # 离散因子: 几个固定值
    "mf_divergence": [-12, -5, 0, 3, 8],
    "mf_consecutive": [0, 3, 5],
    "market_score_adj": [-5, -3, 0, 2, 4],
    # 连续因子: 离散区间
    "mf_strength": [-10, -5, 0, 5, 10],
    "adx": [-8, -2, 0, 6, 10],
    "winner_rate": [-10, -5, 0, 6, 12],
    "main_net": [-10, -5, 0, 5, 10],
    "holder_pct": [-6, -3, 0, 4, 8],
}


def phase3_aggregate_report():
    log = setup_logger("phase3", LOG_DIR / "phase3.log")
    if not UNIVERSE_FILE.exists():
        raise RuntimeError("universe 不存在")

    # 收集所有 samples
    log.info("加载 samples...")
    all_samples = []
    for f in sorted(GROUP_RESULTS_DIR.glob("group_*.jsonl")):
        with open(f, encoding="utf-8") as fp:
            for line in fp:
                try:
                    all_samples.append(json.loads(line))
                except Exception:
                    pass
    log.info("加载完成: %d 样本", len(all_samples))

    if not all_samples:
        raise RuntimeError("无样本")

    # 输出报告
    lines = []
    lines.append(f"# 8 因子单因子回测报告\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> 样本: {len(all_samples):,}, 期间 {days_ago_str(PERIOD_DAYS)} ~ {today_str()}\n")
    lines.append(f"> Universe: 市值≥{MARKET_CAP_MIN:.0f}亿 / 排除 ST / 全 A 股\n\n")

    # 概览: 每因子有效样本数 + IC
    lines.append("## 一、因子概览\n\n")
    lines.append("| 因子 | 有效样本 | σ | " +
                 " | ".join(f"IC({h}d)" for h in HOLD_PERIODS) + " |\n")
    lines.append("|---|---|---|" + "---|" * len(HOLD_PERIODS) + "\n")
    for fname in FACTOR_BINS.keys():
        vals = [s.get(fname) for s in all_samples if s.get(fname) is not None]
        if not vals:
            continue
        sigma = float(np.std(vals))
        row = f"| {fname} | {len(vals):,} | {sigma:.2f} |"
        for h in HOLD_PERIODS:
            x = []
            y = []
            for s in all_samples:
                fv = s.get(fname)
                rv = s.get(f"r{h}")
                if fv is None or rv is None: continue
                x.append(fv)
                y.append(rv)
            if len(x) > 30:
                ic = float(np.corrcoef(x, y)[0, 1])
                row += f" {ic:+.4f} |"
            else:
                row += " - |"
        lines.append(row + "\n")
    lines.append("\n")

    # 每因子分桶详情
    lines.append("## 二、各因子分桶详情\n\n")
    for fname, _bins in FACTOR_BINS.items():
        vals = sorted({s.get(fname) for s in all_samples if s.get(fname) is not None})
        if not vals:
            continue
        lines.append(f"### {fname}\n\n")

        # 各唯一值/区间作为桶
        # 离散因子直接按值分桶
        bucket_keys = sorted(set(s[fname] for s in all_samples if s.get(fname) is not None))
        total_n = sum(1 for s in all_samples if s.get(fname) is not None)

        header = "| 桶 | 占比 | 样本 |"
        for h in HOLD_PERIODS:
            header += f" D+{h} 均涨 | D+{h} 胜率 | D+{h} 最大回撤 |"
        sep = "|---|---|---|" + "---|---|---|" * len(HOLD_PERIODS)
        lines.append(header + "\n")
        lines.append(sep + "\n")

        for bk in bucket_keys:
            bucket = [s for s in all_samples if s.get(fname) == bk]
            n = len(bucket)
            row = f"| {bk:+.1f} | {n / total_n * 100:.1f}% | {n:,} |"
            for h in HOLD_PERIODS:
                rs = [s.get(f"r{h}") for s in bucket if s.get(f"r{h}") is not None]
                dds = [s.get(f"dd{h}") for s in bucket if s.get(f"dd{h}") is not None]
                if rs:
                    avg = float(np.mean(rs))
                    wr = float(np.mean([1 if r > 0 else 0 for r in rs])) * 100
                    avg_dd = float(np.mean(dds)) if dds else 0
                    row += f" {avg:+.2f}% | {wr:.1f}% | {avg_dd:.2f}% |"
                else:
                    row += " - | - | - |"
            lines.append(row + "\n")
        lines.append("\n")

    REPORT_FILE.write_text("".join(lines), encoding="utf-8")
    log.info("report 写入 %s", REPORT_FILE)
    print(f"\n报告: {REPORT_FILE}\n")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--group", type=int, help="单跑某组 (debug)")
    parser.add_argument("--force", action="store_true", help="强制重建 universe")
    args = parser.parse_args()

    main_log = setup_logger("main", LOG_DIR / "master.log")

    if args.phase in ("1", "all"):
        main_log.info("===== Phase 1: 建立股票池 =====")
        phase1_build_universe(force=args.force)

    if args.phase in ("2", "all"):
        main_log.info("===== Phase 2: 跑分组 =====")
        phase2_run_groups(only_group=args.group)

    if args.phase in ("3", "all"):
        main_log.info("===== Phase 3: 聚合报告 =====")
        phase3_aggregate_report()

    main_log.info("DONE")
