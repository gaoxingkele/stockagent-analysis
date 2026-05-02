#!/usr/bin/env python3
"""OOS validity_matrix 构建脚本

从已有的 factor_lab_3y parquet 中取训练期数据(2023-01 → 2024-05),
用完全相同的 phase3 逻辑重算 validity_matrix.json,
输出到 output/factor_lab_oos/ 供真正 OOS 回测使用。

训练/测试划分:
  训练期: 20230103 → 20240531  (约 17 个月)
  测试期: 20240701 → 20260501  (约 18 个月, OOS)

用法:
  python build_oos_matrix.py
  python build_oos_matrix.py --train-end 20231231  # 更严格的截止
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_oos_matrix")

# ── 路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SRC_PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OOS_DIR = ROOT / "output" / "factor_lab_oos"
OOS_DIR.mkdir(parents=True, exist_ok=True)

VALIDITY_FILE = OOS_DIR / "validity_matrix.json"

# ── 参数 (与 factor_lab.py phase3 完全一致) ──────────────────────────
EXCESS_THRESHOLD = 0.05
ABS_MIN = 0.50
MIN_SAMPLES_MV  = 5000
MIN_SAMPLES_PE  = 5000
MIN_SAMPLES_IND = 10000
MIN_BUCKET_SAMPLES = 200

MV_BUCKETS = [(0, 50), (50, 100), (100, 300), (300, 1000), (1000, 1e9)]
MV_LABELS  = ["20-50亿", "50-100亿", "100-300亿", "300-1000亿", "1000亿+"]
PE_BUCKETS = [(-1e9, 0), (0, 15), (15, 30), (30, 50), (50, 100), (100, 1e9)]
PE_LABELS  = ["亏损", "0-15", "15-30", "30-50", "50-100", "100+"]

EXCLUDED_COLS = (
    {"ts_code", "trade_date", "industry", "name",
     "total_mv", "pe", "pe_ttm", "pb",
     "mv_bucket", "pe_bucket",
     "market_score_adj", "adx", "winner_rate", "main_net", "holder_pct",
     "mf_divergence", "mf_strength", "mf_consecutive"}
    | {f"r{h}"  for h in (5, 10, 20, 30, 40)}
    | {f"dd{h}" for h in (5, 10, 20, 30, 40)}
)


# ── 工具函数 ──────────────────────────────────────────────────────────

def bucket_mv(v):
    if pd.isna(v):
        return None
    bil = v / 10000
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


def _q_bucket_stats(sub: pd.DataFrame, factor: str, n_buckets: int = 5) -> dict | None:
    """5分位胜率统计 — 与 factor_lab.py phase3 完全一致。"""
    x = sub[factor]
    valid = sub[x.notna() & sub["r20"].notna()]
    if len(valid) < MIN_BUCKET_SAMPLES * n_buckets:
        return None
    try:
        ranks = valid[factor].rank(pct=True, method="first")
        bucket = pd.cut(ranks, bins=n_buckets,
                        labels=list(range(1, n_buckets + 1)), include_lowest=True)
        bucket = bucket.astype(float)
    except Exception:
        return None

    q_thresholds = np.percentile(valid[factor].values, [20, 40, 60, 80]).tolist()

    wins, avgs, ns = [], [], []
    for q in range(1, n_buckets + 1):
        mask = bucket == q
        if mask.sum() < MIN_BUCKET_SAMPLES:
            wins.append(None); avgs.append(None); ns.append(int(mask.sum()))
            continue
        r = valid.loc[mask, "r20"]
        wins.append(round(float((r > 0).mean()), 4))
        avgs.append(round(float(r.mean()), 4))
        ns.append(int(mask.sum()))

    valid_wins = [w for w in wins if w is not None]
    if not valid_wins:
        return None

    best_idx = int(np.argmax(valid_wins))
    worst_idx = int(np.argmin(valid_wins))
    best_win  = valid_wins[best_idx]
    worst_win = valid_wins[worst_idx]

    q3_win = wins[2] if wins[2] is not None else float(np.mean(valid_wins))
    excess = round(best_win - q3_win, 4)
    active = (excess >= EXCESS_THRESHOLD) and (best_win >= ABS_MIN)

    return {
        "q_thresholds": [round(t, 6) for t in q_thresholds],
        "q_wins": [round(w, 4) if w is not None else None for w in wins],
        "q_avgs": [round(a, 4) if a is not None else None for a in avgs],
        "q_ns":   ns,
        "best_q": best_idx + 1,
        "best_win": round(best_win, 4),
        "q3_win":  round(q3_win, 4),
        "excess":  excess,
        "worst_q": worst_idx + 1,
        "worst_win": round(worst_win, 4),
        "spread":  round(best_win - worst_win, 4),
        "sign":    1 if best_idx > worst_idx else -1,
        "active":  active,
    }


# ── 主流程 ────────────────────────────────────────────────────────────

def build(train_end: str = "20240531"):
    t_total = time.time()

    # 1. 加载 parquet 并过滤训练期
    log.info("加载 parquets from %s ...", SRC_PARQUET_DIR)
    files = sorted(SRC_PARQUET_DIR.glob("group_???.parquet"))
    if not files:
        sys.exit(f"找不到 parquet: {SRC_PARQUET_DIR}")

    parts = []
    for f in files:
        df = pd.read_parquet(f)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[df["trade_date"] <= train_end]
        if not df.empty:
            parts.append(df)

    full = pd.concat(parts, ignore_index=True)
    log.info("训练期样本: %d 行, %d 支股票, 日期 %s → %s",
             len(full), full["ts_code"].nunique(),
             full["trade_date"].min(), full["trade_date"].max())

    # 2. 分桶 + 因子列
    full["mv_bucket"] = full["total_mv"].apply(bucket_mv)
    full["pe_bucket"] = full["pe"].apply(bucket_pe)
    factor_cols = [c for c in full.columns
                   if c not in EXCLUDED_COLS and pd.api.types.is_numeric_dtype(full[c])]
    log.info("因子数: %d", len(factor_cols))

    base_win = float((full["r20"] > 0).mean())
    log.info("基准 D+20 胜率: %.4f", base_win)

    # 主要行业
    ind_counts = full["industry"].value_counts()
    main_inds = ind_counts[ind_counts >= MIN_SAMPLES_IND].index.tolist()[:30]
    log.info("主要行业: %d 个 (阈值 N≥%d)", len(main_inds), MIN_SAMPLES_IND)

    # 3. 计算每个因子的分层胜率
    matrix = {}
    n_active_total = 0
    n_factor_active = 0

    for i, fc in enumerate(factor_cols):
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_total
            eta = elapsed / (i + 1) * (len(factor_cols) - i - 1)
            log.info("  [%d/%d] 已%ds 预计剩%ds", i + 1, len(factor_cols),
                     int(elapsed), int(eta))

        entry = {"global": None, "mv": {}, "pe": {}, "industry": {}}

        entry["global"] = _q_bucket_stats(full, fc)

        for seg in MV_LABELS:
            sub = full[full["mv_bucket"] == seg]
            if len(sub) >= MIN_SAMPLES_MV:
                s = _q_bucket_stats(sub, fc)
                if s:
                    entry["mv"][seg] = s
                    if s["active"]:
                        n_active_total += 1

        for seg in PE_LABELS:
            sub = full[full["pe_bucket"] == seg]
            if len(sub) >= MIN_SAMPLES_PE:
                s = _q_bucket_stats(sub, fc)
                if s:
                    entry["pe"][seg] = s
                    if s["active"]:
                        n_active_total += 1

        for ind in main_inds:
            sub = full[full["industry"] == ind]
            if len(sub) >= MIN_SAMPLES_IND:
                s = _q_bucket_stats(sub, fc)
                if s:
                    entry["industry"][ind] = s
                    if s["active"]:
                        n_active_total += 1

        any_active = (
            (entry["global"] and entry["global"]["active"]) or
            any(v["active"] for v in entry["mv"].values()) or
            any(v["active"] for v in entry["pe"].values()) or
            any(v["active"] for v in entry["industry"].values())
        )
        if any_active:
            n_factor_active += 1

        matrix[fc] = entry

    # 4. 写 JSON
    output = {
        "meta": {
            "n_samples":         int(len(full)),
            "n_stocks":          int(full["ts_code"].nunique()),
            "n_factors":         len(factor_cols),
            "n_industries":      len(main_inds),
            "base_win_rate":     round(base_win, 4),
            "activation_rule":   "v2: (best_win - q3_win) >= excess_threshold AND best_win >= abs_min",
            "excess_threshold":  EXCESS_THRESHOLD,
            "abs_min":           ABS_MIN,
            "hold_period":       "D+20",
            "min_bucket_samples": MIN_BUCKET_SAMPLES,
            "data_period":       [str(full["trade_date"].min()), str(full["trade_date"].max())],
            "train_end":         train_end,
            "oos_start":         "20240701",
            "industries":        main_inds,
            "mv_labels":         MV_LABELS,
            "pe_labels":         PE_LABELS,
            "generated_at":      datetime.now().isoformat(),
        },
        "factors": matrix,
    }
    VALIDITY_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    elapsed = time.time() - t_total
    log.info("validity_matrix 写入 %s (%.1fs)", VALIDITY_FILE, elapsed)
    log.info("激活段总数 %d | 至少一段激活的因子 %d/%d",
             n_active_total, n_factor_active, len(factor_cols))
    log.info("训练期: %s → %s | OOS 测试期: 20240701 →",
             full["trade_date"].min(), train_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="20240531",
                        help="训练期截止日期 (含, 格式 YYYYMMDD, 默认 20240531)")
    args = parser.parse_args()
    build(train_end=args.train_end)


if __name__ == "__main__":
    main()
