# -*- coding: utf-8 -*-
"""FU-001 — Point-in-time 基本面面板 (fina_indicator_vip 拉长 + ann_date 对齐).

产出 research/features/fundamental_pit.parquet, 键 (ts_code, trade_date), trade_date
为每月最后一个交易日 (月度 rebalance 网格, 与 walk-forward 一致)。每行 as-of 取
ann_date<=trade_date 的最新报告期基本面 + 当日 size/value 控制列。

纪律:
  - SIGN-R04 泄漏前置闸: 基本面用 **ann_date(公告日)** 对齐, 绝不用 end_date(报告期末);
    财报重述用**首次披露 ann_date / 原值**, 不回填最新修正值 (playbook FU-001_data)。
    落盘前断言: 每行 fina_ann_date <= trade_date (无未来公告)。
  - SIGN-R06 ST 源头排除: stock_basic.name 含 'ST' 在加载即剔除。
  - SIGN-R08 长任务 checkpoint: fina/daily_basic 拉取分块缓存, 已完成跳过。

控制列 (R12++ 消融用): size = total_mv/circ_mv; value = pe/pe_ttm/pb/ps_ttm。
基本面列: roe / grossprofit_margin / netprofit_yoy / or_yoy / debt_to_assets。
输出: research/features/fundamental_pit.parquet + research/verdicts/FU-001.json (覆盖率分市值档)。
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
CACHE = ROOT / "research" / "cache"
FINA_CACHE = CACHE / "fu001_fina_raw.parquet"
DB_CACHE = CACHE / "fu001_daily_basic.parquet"
OUT_DIR = ROOT / "research" / "features"
OUT_P = OUT_DIR / "fundamental_pit.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "FU-001.json"

KEY = ["ts_code", "trade_date"]
FINA_FIELDS = ("ts_code,ann_date,end_date,roe,grossprofit_margin,"
               "netprofit_yoy,or_yoy,debt_to_assets,update_flag")
FINA_COLS = ["roe", "grossprofit_margin", "netprofit_yoy", "or_yoy", "debt_to_assets"]
DB_FIELDS = "ts_code,trade_date,close,total_mv,circ_mv,pe,pe_ttm,pb,ps_ttm"
DB_COLS = ["total_mv", "circ_mv", "pe", "pe_ttm", "pb", "ps_ttm"]

# 报告期: 2019Q4 .. 2025Q3 (24 期), 兜 2022 起的月度网格 as-of
PERIODS = [f"{y}{q}" for y in range(2019, 2026)
           for q in ("0331", "0630", "0930", "1231")]
PERIODS = [p for p in PERIODS if "20191231" <= p <= "20250930"]

FORWARD_TOKENS = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                 {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                 {"max_gain", "maxgain", "future_ret", "fwd_ret"}


def _pro():
    from dotenv import load_dotenv
    load_dotenv()
    import tushare as ts
    tok = os.getenv("TUSHARE_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("missing TUSHARE_TOKEN")
    ts.set_token(tok)
    return ts.pro_api(timeout=30)


def load_st_set() -> set:
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def rebalance_dates() -> list[str]:
    """每月最后一个交易日 (月度 rebalance 网格)。"""
    dates = sorted(f.stem for f in DAILY_DIR.glob("*.parquet"))
    s = pd.Series(dates)
    last = s.groupby(s.str[:6]).max()
    return sorted(last.tolist())


def fetch_fina() -> pd.DataFrame:
    """bulk 拉 fina_indicator_vip 各报告期; 首次披露(min ann_date)去重防回填泄漏。"""
    if FINA_CACHE.exists():
        print(f"[fina] 命中缓存 {FINA_CACHE.name}", flush=True)
        return pd.read_parquet(FINA_CACHE)
    pro = _pro()
    parts = []
    for p in PERIODS:
        for attempt in range(3):
            try:
                df = pro.fina_indicator_vip(period=p, fields=FINA_FIELDS)
                parts.append(df)
                print(f"  fina {p}: {len(df)} 行", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  fina {p} 重试{attempt}: {e}", flush=True)
                time.sleep(15)
        time.sleep(0.9)
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.dropna(subset=["ann_date", "end_date"])
    raw["ann_date"] = raw["ann_date"].astype(str)
    raw["end_date"] = raw["end_date"].astype(str)
    # 首次披露: 每 (ts_code,end_date) 取最早 ann_date (原值, 不用重述修正值回填)
    raw = (raw.sort_values(["ts_code", "end_date", "ann_date"])
              .drop_duplicates(["ts_code", "end_date"], keep="first")
              .reset_index(drop=True))
    CACHE.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(FINA_CACHE, index=False)
    print(f"[fina] 去重后 {len(raw)} 行 -> {FINA_CACHE.name}", flush=True)
    return raw


def fetch_daily_basic(dates: list[str]) -> pd.DataFrame:
    """逐 rebalance 日 bulk 拉 daily_basic (size/value), checkpoint 已完成跳过。"""
    done = pd.DataFrame()
    if DB_CACHE.exists():
        done = pd.read_parquet(DB_CACHE)
        done["trade_date"] = done["trade_date"].astype(str)
    have = set(done["trade_date"].unique()) if len(done) else set()
    todo = [d for d in dates if d not in have]
    if not todo:
        print(f"[daily_basic] 全部 {len(dates)} 日命中缓存", flush=True)
        return done
    pro = _pro()
    CACHE.mkdir(parents=True, exist_ok=True)
    acc = [done] if len(done) else []
    for i, d in enumerate(todo):
        for attempt in range(3):
            try:
                df = pro.daily_basic(trade_date=d, fields=DB_FIELDS)
                df["trade_date"] = df["trade_date"].astype(str)
                acc.append(df)
                print(f"  db {d} ({i+1}/{len(todo)}): {len(df)} 行", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  db {d} 重试{attempt}: {e}", flush=True)
                time.sleep(15)
        time.sleep(0.6)
        if (i + 1) % 10 == 0:  # 周期性落盘 checkpoint
            pd.concat(acc, ignore_index=True).to_parquet(DB_CACHE, index=False)
    out = pd.concat(acc, ignore_index=True).drop_duplicates(KEY, keep="last")
    out.to_parquet(DB_CACHE, index=False)
    print(f"[daily_basic] 累计 {out['trade_date'].nunique()} 日 / {len(out)} 行", flush=True)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)

    dates = rebalance_dates()
    print(f"[grid] 月度 rebalance {len(dates)} 日 ({dates[0]}~{dates[-1]})", flush=True)

    fina = fetch_fina()
    db = fetch_daily_basic(dates)
    db = db[db["trade_date"].isin(dates)].copy()

    st = load_st_set()
    # 宇宙 = 各 rebalance 日 daily_basic 出现的股票 (ST 源头排除)
    panel = db[~db["ts_code"].isin(st)].copy()
    panel["trade_date"] = panel["trade_date"].astype(str)
    print(f"[universe] ST 排除 {db['ts_code'].isin(st).sum()} 行; "
          f"面板 {len(panel)} 行 / {panel['ts_code'].nunique()} 股", flush=True)

    # ── as-of merge: 取 ann_date<=trade_date 的最新报告期 (merge_asof 需 datetime 键) ──
    fina_a = fina.copy()
    fina_a["ann_date"] = fina_a["ann_date"].astype(str)
    fina_a["_ann_dt"] = pd.to_datetime(fina_a["ann_date"], format="%Y%m%d")
    fina_a = fina_a.sort_values("_ann_dt").reset_index(drop=True)
    panel["_td_dt"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")
    panel = panel.sort_values("_td_dt").reset_index(drop=True)
    merged = pd.merge_asof(
        panel, fina_a[["ts_code", "_ann_dt", "ann_date", "end_date"] + FINA_COLS],
        left_on="_td_dt", right_on="_ann_dt", by="ts_code",
        direction="backward",
    )
    merged = merged.rename(columns={"ann_date": "fina_ann_date",
                                    "end_date": "fina_end_date"})
    merged = merged.drop(columns=["_td_dt", "_ann_dt"])

    # ── 泄漏自查: 无未来 ann_date (every fina_ann_date <= trade_date) ──
    has_fina = merged["fina_ann_date"].notna()
    future = merged.loc[has_fina, "fina_ann_date"].astype(str) > merged.loc[has_fina, "trade_date"]
    assert not future.any(), f"泄漏: {int(future.sum())} 行 fina_ann_date > trade_date"

    # ── 列名泄漏自查 (forward 黑名单) ──
    bad = [c for c in merged.columns if c.lower() in FORWARD_TOKENS
           or c.lower().startswith(("fwd_", "future_", "next_", "r1_next"))
           or c.lower().endswith("_forward")]
    assert not bad, f"输出含 forward 字段: {bad}"

    # 报告滞后 (天) — sanity: 应为正且通常 30~120 天
    md = pd.to_datetime(merged.loc[has_fina, "trade_date"], format="%Y%m%d")
    ad = pd.to_datetime(merged.loc[has_fina, "fina_ann_date"], format="%Y%m%d")
    lag_days = (md - ad).dt.days
    assert (lag_days >= 0).all(), "存在负报告滞后 (泄漏)"

    out_cols = KEY + ["close"] + DB_COLS + ["fina_ann_date", "fina_end_date"] + FINA_COLS
    out = merged[out_cols].sort_values(KEY).reset_index(drop=True)
    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)}  ({len(out):,} 行 × {len(out.columns)} 列)", flush=True)

    # ── 覆盖率分市值档 (size = total_mv 三分位) ──
    cov_total = float(out["or_yoy"].notna().mean())
    cov_by_bucket = {}
    valid_mv = out["total_mv"].notna()
    if valid_mv.any():
        q = out.loc[valid_mv, "total_mv"].rank(pct=True)
        bucket = pd.cut(q, [0, 1 / 3, 2 / 3, 1.0], labels=["small", "mid", "large"],
                        include_lowest=True)
        tmp = out.loc[valid_mv].assign(_b=bucket.values)
        for b, g in tmp.groupby("_b", observed=True):
            cov_by_bucket[str(b)] = round(float(g["or_yoy"].notna().mean()), 4)
    low_cov = {b: c for b, c in cov_by_bucket.items() if c < 0.10}

    n_periods_used = int(out["fina_end_date"].nunique())
    verdict = {
        "task": "FU-001",
        "status": "built",
        "conclusion": (
            f"Point-in-time 基本面面板落盘 {len(out):,} 行 × {out['ts_code'].nunique()} 股 × "
            f"{out['trade_date'].nunique()} 月度网格 ({out['trade_date'].min()}~{out['trade_date'].max()}); "
            f"ann_date 对齐(首次披露不回填), 无未来公告(泄漏自查通过), or_yoy 覆盖率 {cov_total:.1%}; "
            f"size/value 控制列齐全。命中 playbook FU-001_data (重述用首次ann_date)。"),
        "artifact": str(OUT_P.relative_to(ROOT)),
        "responsePlaybook_branch": "FU-001_data: 财报重述用首次披露 ann_date 原值不回填",
        "metrics": {
            "rows": int(len(out)),
            "n_stocks": int(out["ts_code"].nunique()),
            "n_rebalance_dates": int(out["trade_date"].nunique()),
            "date_range": [out["trade_date"].min(), out["trade_date"].max()],
            "n_report_periods_pulled": len(PERIODS),
            "n_report_periods_used": n_periods_used,
            "fina_coverage_or_yoy": round(cov_total, 4),
            "coverage_by_mv_bucket": cov_by_bucket,
            "low_coverage_buckets(<10%)": low_cov,
            "report_lag_days_median": int(lag_days.median()),
            "report_lag_days_min": int(lag_days.min()),
            "report_lag_days_max": int(lag_days.max()),
            "leakage_free": True,
            "restatement_handling": "first-disclosure (min ann_date) per (ts_code,end_date)",
        },
        "guardrails": ["SIGN-R04 ann_date 对齐+无未来公告", "SIGN-R06 ST 源头排除",
                       "SIGN-R08 fina/daily_basic checkpoint"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"  覆盖率 or_yoy={cov_total:.1%}, 分档={cov_by_bucket}, 滞后中位 {int(lag_days.median())}d", flush=True)
    if low_cov:
        print(f"  ⚠️ 低覆盖档(<10%): {low_cov} — 该档单列评估 (playbook 覆盖不足)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
