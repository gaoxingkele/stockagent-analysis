#!/usr/bin/env python3
"""18个月回测 (2024-11 → 2026-05)

Phase 1: 滚动月度 IC 验证 (SpearmanR vs r20)
Phase 2: 周频组合回测 + checkpoint, 对比4个版本:
  - base : 无 EB, 无 K-weight
  - eb   : 有 EB 收缩, 无 K-weight
  - k    : 无 EB, 有 K-weight
  - full : 有 EB + K-weight  (当前最新版本)

输出:
  output/backtest_18m/
    ic_monthly.parquet        Phase 1 滚动 IC
    ic_summary.csv            因子 IC 汇总
    portfolio_returns.parquet Phase 2 周频收益
    perf_summary.csv          4版本 Sharpe/DD/IC IR 对比
    checkpoint.json           断点续跑状态
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest_18m")

# ── 配置 ──────────────────────────────────────────────────────────────
BACKTEST_START = "20241101"
BACKTEST_END   = "20260501"
OUTPUT_DIR = Path("output/backtest_18m")
CKPT_FILE  = OUTPUT_DIR / "checkpoint.json"

FACTOR_DIR = Path("output/factor_lab_3y/factor_groups")
MATRIX_PATH = Path("output/factor_lab_3y/validity_matrix.json")

# 持仓周期 (用 r5 = 5日前向收益)
HOLD_DAYS  = 5
RET_COL    = "r5"

TOP_PCT    = 0.20
BOT_PCT    = 0.20
MIN_STOCKS = 50    # 每个组合最少持仓数

VERSIONS = {
    "base": dict(use_eb=False, use_k_weight=False),
    "eb":   dict(use_eb=True,  use_k_weight=False),
    "k":    dict(use_eb=False, use_k_weight=True),
    "full": dict(use_eb=True,  use_k_weight=True),
}

# 需要的列
META_COLS    = ["ts_code", "trade_date", "total_mv", "pe", "pe_ttm", "industry",
                "market_score_adj", "mf_divergence", "mf_strength", "mf_consecutive"]
RETURN_COLS  = ["r5", "r20", "r40"]
FACTOR_COLS_EXCLUDE = set(META_COLS + RETURN_COLS + ["r30", "dd5", "dd20", "dd30", "dd40"])


# ── 工具函数 ──────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CKPT_FILE.exists():
        return json.loads(CKPT_FILE.read_text(encoding="utf-8"))
    return {"phase1_done": False, "phase2_weeks_done": []}


def save_checkpoint(ckpt: dict):
    CKPT_FILE.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_factors() -> pd.DataFrame:
    """加载所有 factor_groups parquet, 过滤到回测区间, 只保留必要列."""
    log.info("加载 factor_groups (%d 个文件)...", len(list(FACTOR_DIR.glob("*.parquet"))))
    parts = []
    for p in sorted(FACTOR_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df = df[(df["trade_date"] >= BACKTEST_START) & (df["trade_date"] <= BACKTEST_END)]
        if not df.empty:
            parts.append(df)
    if not parts:
        raise RuntimeError("没有找到回测期间数据")
    full = pd.concat(parts, ignore_index=True)
    full["trade_date"] = full["trade_date"].astype(str)
    log.info("合并后: %d 行, %d 列, %d 支股票, 日期 %s → %s",
             len(full), len(full.columns),
             full["ts_code"].nunique(),
             full["trade_date"].min(), full["trade_date"].max())
    return full


def get_factor_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FACTOR_COLS_EXCLUDE and c not in ("r30","dd30")]


def get_weekly_dates(df: pd.DataFrame) -> list[str]:
    """取每周最后一个交易日 (用于周频换仓)."""
    dates = sorted(df["trade_date"].unique())
    dt_series = pd.to_datetime(dates, format="%Y%m%d")
    df_dates = pd.DataFrame({"date_str": dates, "dt": dt_series})
    df_dates["week"] = df_dates["dt"].dt.isocalendar().week.astype(int)
    df_dates["year"] = df_dates["dt"].dt.isocalendar().year.astype(int)
    weekly = df_dates.groupby(["year", "week"])["date_str"].last().tolist()
    # 确保最后一日有 r5 数据 (至少留5个交易日空间)
    last_usable = sorted(df["trade_date"].unique())[-6]
    weekly = [d for d in weekly if d <= last_usable]
    log.info("周频换仓日: %d 周, %s → %s", len(weekly), weekly[0], weekly[-1])
    return weekly


def mf_state_from_row(row) -> str | None:
    """从 parquet 行推导资金流状态."""
    div = row.get("mf_divergence", 0) or 0
    strength = row.get("mf_strength", 0) or 0
    consec = row.get("mf_consecutive", 0) or 0
    if div > 0.3 and consec >= 2:
        return "main_inflow_3d"
    if div > 0.1:
        return "main_inflow"
    if div < -0.3 and consec <= -2:
        return "main_outflow_3d"
    return None


def regime_from_score(market_score: float | None) -> dict | None:
    if market_score is None:
        return None
    if market_score >= 70:
        return {"trend": "fast_bull"}
    if market_score >= 55:
        return {"trend": "slow_bull"}
    if market_score <= 30:
        return {"trend": "bear"}
    return None


# ── Phase 1: 滚动 IC ──────────────────────────────────────────────────

def run_phase1(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    log.info("=== Phase 1: 滚动月度 IC ===")
    dates = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.copy()
    df["_dt"] = dates
    df["_ym"] = df["_dt"].dt.to_period("M")

    months = sorted(df["_ym"].unique())
    records = []

    for ym in months:
        chunk = df[df["_ym"] == ym]
        if len(chunk) < 200:
            continue
        r20 = pd.to_numeric(chunk["r20"], errors="coerce").values
        valid_ret = ~np.isnan(r20)
        for fc in factor_cols:
            fv = pd.to_numeric(chunk[fc], errors="coerce").values
            valid_f = ~np.isnan(fv)
            mask = valid_ret & valid_f
            if mask.sum() < 100:
                continue
            ic, pval = stats.spearmanr(fv[mask], r20[mask])
            records.append({
                "month": str(ym),
                "factor": fc,
                "IC_20d": round(float(ic), 5),
                "pval": round(float(pval), 4),
                "N": int(mask.sum()),
            })

    result = pd.DataFrame(records)
    log.info("Phase 1 完成: %d 条 (IC, 因子=%d, 月=%d)",
             len(result), len(factor_cols), len(months))
    return result


# ── Phase 2: 组合回测 ─────────────────────────────────────────────────

def score_week(
    df_week: pd.DataFrame,
    factor_cols: list[str],
    matrix: dict,
    version_flags: dict,
) -> pd.Series:
    """对某周截面打分, 返回 {ts_code: layered_score} Series."""
    from src.stockagent_analysis.sparse_layered_score import (
        compute_sparse_layered_score, bucket_mv, bucket_pe,
    )
    scores = {}
    for _, row in df_week.iterrows():
        features = {}
        for fc in factor_cols:
            v = row.get(fc)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                features[fc] = float(v)

        mv = row.get("total_mv")
        pe = row.get("pe_ttm") or row.get("pe")
        context = {
            "mv_seg": bucket_mv(float(mv) if mv else None),
            "pe_seg": bucket_pe(float(pe) if pe else None),
            "industry": str(row.get("industry", "") or ""),
            "etf_held": False,
        }
        mf = mf_state_from_row(dict(row))
        ms = row.get("market_score_adj")
        regime = regime_from_score(float(ms) if ms else None)

        result = compute_sparse_layered_score(
            features, context,
            matrix=matrix,
            regime=regime,
            mf_state=mf,
            **version_flags,
        )
        scores[row["ts_code"]] = result["layered_score"]
    return pd.Series(scores)


def run_phase2(
    df: pd.DataFrame,
    factor_cols: list[str],
    weekly_dates: list[str],
    matrix: dict,
    ckpt: dict,
) -> pd.DataFrame:
    log.info("=== Phase 2: 周频组合回测 (%d 周) ===", len(weekly_dates))
    done_weeks = set(ckpt.get("phase2_weeks_done", []))

    out_path = OUTPUT_DIR / "portfolio_returns.parquet"
    existing = []
    if out_path.exists():
        existing = [pd.read_parquet(out_path)]

    records = []
    t0 = time.time()

    for i, wdate in enumerate(weekly_dates):
        if wdate in done_weeks:
            continue

        df_week = df[df["trade_date"] == wdate].copy()
        df_week = df_week[df_week[RET_COL].notna()]
        if len(df_week) < MIN_STOCKS:
            log.warning("跳过 %s: 股票数 %d < %d", wdate, len(df_week), MIN_STOCKS)
            continue

        week_rows = []
        for vname, vflags in VERSIONS.items():
            scores = score_week(df_week, factor_cols, matrix, vflags)
            # 按分位数分组
            n = len(scores)
            top_thresh = scores.quantile(1 - TOP_PCT)
            bot_thresh = scores.quantile(BOT_PCT)

            long_ret  = float(df_week.loc[df_week["ts_code"].isin(
                scores[scores >= top_thresh].index), RET_COL].mean())
            short_ret = float(df_week.loc[df_week["ts_code"].isin(
                scores[scores <= bot_thresh].index), RET_COL].mean())
            mkt_ret   = float(df_week[RET_COL].mean())
            ls_ret    = long_ret - short_ret

            # IC (本截面因子均值 vs r5)
            score_series = scores.reindex(df_week["ts_code"]).values
            ret_series   = df_week.set_index("ts_code").reindex(scores.index)[RET_COL].values
            valid = ~(np.isnan(score_series) | np.isnan(ret_series))
            ic5 = float(stats.spearmanr(score_series[valid], ret_series[valid])[0]) if valid.sum() > 10 else np.nan

            week_rows.append({
                "date": wdate,
                "version": vname,
                "long_ret": round(long_ret, 4),
                "short_ret": round(short_ret, 4),
                "ls_ret": round(ls_ret, 4),
                "mkt_ret": round(mkt_ret, 4),
                "long_excess": round(long_ret - mkt_ret, 4),
                "ic5": round(ic5, 5) if not np.isnan(ic5) else None,
                "n_stocks": n,
                "n_long": int((scores >= top_thresh).sum()),
                "n_short": int((scores <= bot_thresh).sum()),
            })

        records.extend(week_rows)
        done_weeks.add(wdate)
        ckpt["phase2_weeks_done"] = sorted(done_weeks)

        elapsed = time.time() - t0
        remaining = elapsed / (i + 1) * (len(weekly_dates) - i - 1)
        log.info("[%d/%d] %s | 已%ds 预计剩余%ds",
                 i + 1, len(weekly_dates), wdate,
                 int(elapsed), int(remaining))

        # checkpoint 每10周保存一次
        if (i + 1) % 10 == 0:
            pieces = existing + ([pd.DataFrame(records)] if records else [])
            df_partial = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]
            df_partial.to_parquet(out_path, index=False)
            save_checkpoint(ckpt)
            records = []
            existing = [df_partial]

    # 最终保存
    all_dfs = existing
    if records:
        all_dfs.append(pd.DataFrame(records))
    final = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 1 else all_dfs[0]
    final = final.drop_duplicates(["date", "version"]).sort_values(["version", "date"])
    final.to_parquet(out_path, index=False)
    save_checkpoint(ckpt)
    log.info("Phase 2 完成: %d 行 → %s", len(final), out_path)
    return final


# ── 汇总报告 ──────────────────────────────────────────────────────────

def build_perf_summary(ret_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vname in VERSIONS:
        v = ret_df[ret_df["version"] == vname].sort_values("date")
        if len(v) < 5:
            continue
        ls = v["ls_ret"].values / 100        # 百分比→小数
        ex = v["long_excess"].values / 100
        ic = v["ic5"].dropna().values

        # 累积收益
        cum_ls = np.cumprod(1 + ls) - 1
        # Sharpe (周频 × sqrt(52))
        sharpe = float(np.mean(ls) / np.std(ls) * np.sqrt(52)) if np.std(ls) > 0 else 0
        # Max drawdown
        peak = np.maximum.accumulate(1 + np.concatenate([[0], ls]))
        dd = (1 + np.concatenate([[0], np.cumprod(1 + ls)]) - peak) / peak
        max_dd = float(dd.min())
        # Annual return
        n_weeks = len(ls)
        ann_ret = float((1 + cum_ls[-1]) ** (52 / n_weeks) - 1) if n_weeks > 0 else 0
        # IC IR
        ic_ir = float(np.mean(ic) / np.std(ic)) if len(ic) > 2 and np.std(ic) > 0 else 0

        rows.append({
            "version": vname,
            "annual_ret_pct": round(ann_ret * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_dd_pct": round(max_dd * 100, 2),
            "ic5_mean": round(float(np.mean(ic)), 5) if len(ic) > 0 else None,
            "ic5_ir": round(ic_ir, 3),
            "long_excess_ann_pct": round(float(np.mean(ex)) * 52 * 100, 2),
            "n_weeks": n_weeks,
        })
    return pd.DataFrame(rows)


def build_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ic_df.groupby("factor")["IC_20d"]
        .agg(IC_mean="mean", IC_std="std", IC_IR=lambda x: x.mean() / x.std() if x.std() > 0 else 0,
             N_months="count")
        .reset_index()
        .sort_values("IC_IR", key=abs, ascending=False)
    )
    return summary


# ── 主入口 ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--reset", action="store_true", help="清除 checkpoint 重新跑")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset and CKPT_FILE.exists():
        CKPT_FILE.unlink()
        log.info("已清除 checkpoint")

    ckpt = load_checkpoint()

    # 加载数据
    df = load_all_factors()
    factor_cols = get_factor_cols(df)
    log.info("因子列数: %d", len(factor_cols))

    # 加载 validity matrix
    with open(MATRIX_PATH, encoding="utf-8") as f:
        matrix = json.load(f)

    run_p1 = args.phase in ("1", "all") and not ckpt.get("phase1_done")
    run_p2 = args.phase in ("2", "all")

    # Phase 1
    if run_p1:
        ic_df = run_phase1(df, factor_cols)
        ic_df.to_parquet(OUTPUT_DIR / "ic_monthly.parquet", index=False)
        ic_summary = build_ic_summary(ic_df)
        ic_summary.to_csv(OUTPUT_DIR / "ic_summary.csv", index=False, encoding="utf-8-sig")
        ckpt["phase1_done"] = True
        save_checkpoint(ckpt)
        log.info("Phase 1 IC 汇总 Top10:")
        print(ic_summary.head(10).to_string(index=False))
    elif args.phase in ("1", "all"):
        log.info("Phase 1 已完成 (checkpoint), 跳过. 用 --reset 重跑")

    # Phase 2
    if run_p2:
        weekly_dates = get_weekly_dates(df)
        ret_df = run_phase2(df, factor_cols, weekly_dates, matrix, ckpt)
        perf = build_perf_summary(ret_df)
        perf.to_csv(OUTPUT_DIR / "perf_summary.csv", index=False, encoding="utf-8-sig")
        log.info("\n=== 4版本性能对比 ===")
        print(perf.to_string(index=False))
        ckpt["phase2_done"] = True
        save_checkpoint(ckpt)

    log.info("回测完成. 结果目录: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
