# -*- coding: utf-8 -*-
"""V7综合评分回测 — 在V6基础上叠加市场环境调节(±8分)。

思路: 预计算6大宽基指数每日的趋势状态和综合大盘评分,
然后在每个(股票,日期)记录上用 compute_market_adjustment 调节 composite_score。

用法:
    python backtest_composite_v7.py [--max N]
"""
import sys, os, struct, time, statistics, pickle
import argparse
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from backtest_composite import (
    scan_tdx_symbols, composite_score, compute_channel_scores,
    batch_stretch_scores, analyze, print_result,
    WEIGHTS, _NEUTRAL_THRESHOLD, _KEY_DIMS,
)
from backtest_agents import (
    load_tdx_daily, rolling_indicators,
    score_trend_momentum, score_capital_liquidity, score_divergence,
    score_chanlun, score_resonance,
    score_ichimoku, _f_amt_ratio,
)
from stockagent_analysis.market_context import (
    _read_tdx_day_file, classify_trend_state, TrendState,
    MAJOR_INDICES, _compute_market_score,
)


# ── 大盘/ETF历史状态预计算 ───────────────────────────────────────

def precompute_market_states(lookback_days: int = 600) -> dict:
    """预计算6大指数的滚动 TrendState,返回 {date_str: market_score}。

    思路: 从每根K线回溯60根计算当日状态,得到逐日的大盘评分序列。
    """
    print("[预计算] 读取6大指数历史K线...", flush=True)
    index_dfs = {}
    for code_with_mkt, name in MAJOR_INDICES.items():
        market = code_with_mkt[:2]
        code = code_with_mkt[2:]
        df = _read_tdx_day_file(market, code, days=lookback_days + 60)
        if df is not None and len(df) >= 100:
            index_dfs[code_with_mkt] = df
            print(f"  {name} ({code_with_mkt}): {len(df)} 行", flush=True)
        else:
            print(f"  [警告] {name} 数据不足, 跳过", flush=True)

    if not index_dfs:
        print("[错误] 无任何指数数据", flush=True)
        return {}

    # 用日期对齐所有指数
    print("[预计算] 滚动计算每日大盘评分...", flush=True)
    # 取主指数(沪深300)的日期作为基准
    base_key = "sh000300" if "sh000300" in index_dfs else list(index_dfs.keys())[0]
    base_df = index_dfs[base_key]
    date_col = base_df["date"].dt.strftime("%Y-%m-%d") if hasattr(base_df["date"], 'dt') else base_df["date"].astype(str)
    base_df = base_df.copy()
    base_df["date_str"] = date_col.values

    # 准备每个指数的 date->row 映射
    idx_maps = {}
    for code, df in index_dfs.items():
        df2 = df.copy()
        dstr = df2["date"].dt.strftime("%Y-%m-%d") if hasattr(df2["date"], 'dt') else df2["date"].astype(str)
        df2["date_str"] = dstr.values
        df2 = df2.set_index("date_str")
        idx_maps[code] = df2

    # 滚动: 对每个交易日,用最近60根K线算该日的 TrendState
    market_scores: dict[str, float] = {}
    market_phases: dict[str, str] = {}

    date_list = list(base_df["date_str"].tail(lookback_days).values)
    for di, dt_str in enumerate(date_list):
        if di < 60:
            continue
        states = []
        for code, df2 in idx_maps.items():
            if dt_str not in df2.index:
                continue
            loc = df2.index.get_loc(dt_str)
            # loc 可能是 int 或 slice,取 int
            if isinstance(loc, slice):
                loc = loc.stop - 1
            # 取 loc 及之前60根
            start = max(0, loc - 60)
            sub = df2.iloc[start:loc + 1].copy().reset_index(drop=True)
            if len(sub) < 60:
                continue
            ts = classify_trend_state(sub)
            ts.code = code
            ts.name = MAJOR_INDICES.get(code, code)
            states.append(ts)
        if not states:
            continue
        score, phase, phase_cn = _compute_market_score(states)
        market_scores[dt_str] = score
        market_phases[dt_str] = phase

    print(f"[预计算] 完成, 覆盖 {len(market_scores)} 个交易日", flush=True)
    return {"scores": market_scores, "phases": market_phases}


# ── V7评分: V6 composite + 市场调节 ────────────────────────────

def composite_score_v7(agent_scores: dict, market_score: float | None) -> float:
    """V7 = V6 composite + 简化版市场调节。

    回测里无概念板块和ETF数据(需要行业匹配,复杂),
    只用 market_score(大盘) 做调节:
      - 大盘强(>65): 对个股评分>50加成, <50无影响(避免"弱势股被牛市救")
      - 大盘弱(<35): 对个股评分<50加重看空, >50压制(避免"牛股被熊市埋没"过度)
      - 极端分(>85或<15)调节减半

    调节幅度: ±5分 (比实盘小,因为回测样本多,过度调节放大噪声)
    """
    base = composite_score(agent_scores)
    if market_score is None:
        return base

    adj = 0.0
    # 大盘环境(±5)
    if market_score >= 65:
        # 强势: 加强买入信号
        if base > 50:
            adj += min(5, (market_score - 65) * 0.15)
    elif market_score <= 35:
        # 弱势: 加强卖出信号
        if base < 50:
            adj -= min(5, (35 - market_score) * 0.15)
        elif base > 65:
            # 压制虚高(熊市不跟风追多)
            adj -= min(3, (35 - market_score) * 0.08)

    # 极端分调节减半
    if base > 85 or base < 15:
        adj *= 0.5

    return max(0.0, min(100.0, base + adj))


# ── 回测主循环(改造自 V6) ────────────────────────────────────────

CHECKPOINT_FILE = Path("output/v7_backtest_checkpoint.pkl")
CHECKPOINT_EVERY = 100  # 每N只股票保存一次


def _load_checkpoint() -> dict | None:
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[checkpoint] 读取失败: {e}", flush=True)
        return None


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    tmp.replace(CHECKPOINT_FILE)


def run_backtest_v7(symbols: list[str], market_data: dict,
                     resume: bool = False) -> tuple[list, list]:
    market_scores = market_data.get("scores", {})
    total = len(symbols)

    # 尝试加载 checkpoint
    records: list[tuple] = []
    v6_records: list[tuple] = []
    skipped = 0
    processed_set: set[str] = set()
    start_idx = 0

    if resume:
        cp = _load_checkpoint()
        if cp:
            records = cp.get("records", [])
            v6_records = cp.get("v6_records", [])
            skipped = cp.get("skipped", 0)
            processed_set = set(cp.get("processed", []))
            print(f"[checkpoint] 恢复: 已处理{len(processed_set)}只, "
                  f"V7样本={len(records)}, V6样本={len(v6_records)}", flush=True)

    for si, sym in enumerate(symbols):
        if sym in processed_set:
            continue
        if (si + 1) % 50 == 0:
            print(f"  进度: {si+1}/{total} (样本={len(records)}, 跳过={skipped})...", flush=True)
        if (si + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint({
                "records": records, "v6_records": v6_records,
                "skipped": skipped, "processed": list(processed_set),
            })
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 200:
                skipped += 1
                continue

            cr_scores = compute_channel_scores(df)
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty:
                skipped += 1
                continue

            close_arr = df["close"].values
            date_arr = df["date"].dt.strftime("%Y-%m-%d").values if hasattr(df["date"], 'dt') else df["date"].astype(str).values
            rows = [r.to_dict() for _, r in ind.iterrows()]

            for row in rows:
                idx = int(row["idx"])
                r5  = (close_arr[idx+5]  / close_arr[idx] - 1)*100 if idx+5  < len(close_arr) else np.nan
                r10 = (close_arr[idx+10] / close_arr[idx] - 1)*100 if idx+10 < len(close_arr) else np.nan
                r20 = (close_arr[idx+20] / close_arr[idx] - 1)*100 if idx+20 < len(close_arr) else np.nan

                _tm = score_trend_momentum(row)
                _ich = score_ichimoku(row)
                _res_rev = 100.0 - score_resonance(row)
                _mr = (_tm + _ich + _res_rev) / 3.0

                agent_scores = {
                    "capital_liquidity": score_capital_liquidity(row),
                    "divergence":        score_divergence(row),
                    "chanlun":           score_chanlun(row),
                    "f_amt_ratio":       max(10, min(90, _f_amt_ratio(row))),
                    "mean_reversion":    _mr,
                    "channel_reversal":  float(cr_scores[idx]) if idx < len(cr_scores) else 50.0,
                }

                # V6 baseline
                v6_score = composite_score(agent_scores)
                v6_records.append((v6_score, r5, r10, r20))

                # V7: 查找该日大盘评分
                dt_str = date_arr[idx] if idx < len(date_arr) else None
                mkt_score = market_scores.get(dt_str) if dt_str else None

                score_v7 = composite_score_v7(agent_scores, mkt_score)
                records.append((score_v7, r5, r10, r20))

        except Exception as e:
            print(f"  {sym} 失败: {e}", flush=True)
            skipped += 1
        finally:
            processed_set.add(sym)

    # 最后保存一次
    _save_checkpoint({
        "records": records, "v6_records": v6_records,
        "skipped": skipped, "processed": list(processed_set),
    })
    return records, v6_records


# ── 主流程 ───────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="限制股票数(调试)")
    ap.add_argument("--lookback", type=int, default=600, help="大盘预计算覆盖天数")
    ap.add_argument("--resume", action="store_true", help="从checkpoint恢复")
    ap.add_argument("--reset", action="store_true", help="删除checkpoint重新开始")
    args = ap.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"[checkpoint] 已删除 {CHECKPOINT_FILE}", flush=True)

    t0 = time.time()

    # 1. 预计算大盘状态
    market_data = precompute_market_states(lookback_days=args.lookback)
    if not market_data.get("scores"):
        print("[错误] 大盘数据不可用", flush=True)
        sys.exit(1)

    # 2. 扫描股票池 - 优先读 backtest_stock_pool.txt
    pool_file = Path("backtest_stock_pool.txt")
    if pool_file.exists():
        symbols = [line.strip() for line in pool_file.read_text().splitlines() if line.strip()]
        print(f"\n[股票池] 从 {pool_file} 加载 {len(symbols)} 只", flush=True)
    else:
        print("\n[扫描] TDX全量A股...", flush=True)
        symbols = scan_tdx_symbols()
        print(f"[扫描] {len(symbols)} 只股票", flush=True)
    if args.max > 0:
        symbols = symbols[:args.max]
        print(f"[限制] 截取前 {args.max} 只", flush=True)

    # 3. 回测 (同时跑V6 baseline 用于对比)
    print("\n[回测] 开始...", flush=True)
    t_bt = time.time()
    v7_records, v6_records = run_backtest_v7(symbols, market_data, resume=args.resume)
    print(f"[回测] 完成, {len(v7_records)} 个样本, 耗时 {time.time()-t_bt:.1f}s", flush=True)

    # 4. 展宽
    v7_stretched = batch_stretch_scores(v7_records)
    v6_stretched = batch_stretch_scores(v6_records)

    # 5. 分析
    print("\n" + "="*60)
    print("V6 Baseline (无市场调节)")
    print("="*60)
    v6_result = analyze(v6_stretched)
    print_result(v6_result)

    print("\n" + "="*60)
    print("V7 (叠加市场环境调节)")
    print("="*60)
    v7_result = analyze(v7_stretched)
    print_result(v7_result)

    # 6. 对比
    print("\n" + "="*60)
    print("V6 vs V7 对比")
    print("="*60)
    print(f"{'指标':<20} {'V6':>10} {'V7':>10} {'Δ':>10}")
    print("-"*55)
    for k in ("ic5", "ic10", "ic20"):
        v6 = v6_result.get(k, 0)
        v7 = v7_result.get(k, 0)
        label = {"ic5": "IC(5d)", "ic10": "IC(10d)", "ic20": "IC(20d)"}[k]
        print(f"{label:<20} {v6:>+10.4f} {v7:>+10.4f} {v7-v6:>+10.4f}")

    # 高分段胜率对比(65+)
    print()
    print(f"{'分数段':<20} {'V6样本':>8} {'V6胜率20d':>10} {'V7样本':>8} {'V7胜率20d':>10}")
    print("-"*65)
    v6_buckets = {b["range"]: b for b in v6_result["buckets"]}
    v7_buckets = {b["range"]: b for b in v7_result["buckets"]}
    for rng in ("65-70", "70-75", "75-80", "80-85", "85-90", "90-100"):
        v6b = v6_buckets.get(rng, {})
        v7b = v7_buckets.get(rng, {})
        v6wr = v6b.get("wr_20d")
        v7wr = v7b.get("wr_20d")
        print(f"{rng:<20} {v6b.get('count', 0):>8} {(v6wr if v6wr else 0):>10.1f}% "
              f"{v7b.get('count', 0):>8} {(v7wr if v7wr else 0):>10.1f}%")

    print(f"\n总耗时: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
