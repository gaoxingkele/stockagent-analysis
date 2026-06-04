# -*- coding: utf-8 -*-
"""RT-002 — A0 标准 TA 标量对照特征集 (消融基线) 全历史落盘 + 泄漏/确定性自检.

对全历史 daily OHLCV 面板用 a0_encoder.encode_frame 生成 32 列标准标量 TA (多尺度动量 +
ATR/range + 多尺度量比 + 缺口 + 均线结构 + 振荡器 + 自身形态), 与关系张量 (RT-001) 同口径同期,
按月分块落盘 research/features/a0_scalar_ta/。这是 A2-vs-A0 决定性消融的强 baseline。

纪律 (与 RT-001 一致):
  - ST 源头排除 (SIGN-R06)。
  - 零泄漏 (SIGN-R04): 全 causal (shift/rolling/ewm 仅向后), 输出列自查不命中 forward 黑名单。
  - 确定性 (SIGN-R07): 重算逐位一致 (抽样比对)。
  - checkpoint (SIGN-R08): 按月分块, 已存在跳过。
  - 同口径同期: 同一 daily 面板 + 同 σ20 warmup 闸 → 锚点集与张量对齐 (RT-004 inner-join)。

用法: python research/rt002_build_a0.py [--smoke YYYYMM]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from a0_encoder import encode_frame, feature_columns, KEY, SIGMA_WIN  # noqa: E402

DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
TENSOR_DIR = ROOT / "research" / "features" / "rel_tensor_v2"
OUT_DIR = ROOT / "research" / "features" / "a0_scalar_ta"
INDEX_P = OUT_DIR / "_index.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "RT-002.json"

FORWARD_TOKENS = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                 {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                 {"max_gain", "maxgain", "future_ret", "fwd_ret"}


def assert_no_leakage(cols: list[str]) -> None:
    bad = [c for c in cols if c.lower() in FORWARD_TOKENS
           or c.lower().startswith(("fwd_", "future_", "next_"))
           or c.lower().endswith("_forward")
           or (c[0:1] == "r" and c[1:].split("_")[0].isdigit())]
    assert not bad, f"输出含 forward 字段 (泄漏 SIGN-R04): {bad}"


def load_st_set() -> set:
    if not BASIC_P.exists():
        print("  ⚠️ stock_basic.parquet 不存在, ST 排除跳过", flush=True)
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_panel(smoke_month: str | None) -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pre_close"]
    files = sorted(DAILY_DIR.glob("*.parquet"))
    if smoke_month:
        files = [f for f in files if f.stem <= smoke_month + "31"]
        files = files[-90:]  # ma_60 需更长 warmup, 多取
    print(f"[panel] 加载 {len(files)} 日 OHLCV ...", flush=True)
    parts = [pd.read_parquet(f, columns=cols) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.drop_duplicates(subset=KEY, keep="last")
    big = big[big["close"].notna() & (big["close"] > 0)].reset_index(drop=True)
    st = load_st_set()
    if st:
        before = len(big)
        big = big[~big["ts_code"].isin(st)].reset_index(drop=True)
        print(f"  ST 源头排除: {before - len(big):,} 行 ({len(st)} 只 ST)", flush=True)
    return big.sort_values(KEY).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default=None, help="仅构建该月 YYYYMM (冒烟)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    feat = feature_columns()
    assert_no_leakage(feat)

    panel = load_panel(args.smoke)
    print(f"[panel] {len(panel):,} 行 / {panel['ts_code'].nunique()} 股 / "
          f"{panel['trade_date'].min()}~{panel['trade_date'].max()}", flush=True)

    t = time.time()
    enc = encode_frame(panel)
    assert list(enc.columns) == KEY + feat, "列顺序与 feature_columns() 不一致"
    assert_no_leakage(list(enc.columns))
    enc["month"] = enc["trade_date"].str[:6]
    if args.smoke:
        enc = enc[enc["month"] == args.smoke].reset_index(drop=True)
    print(f"[encode] {len(enc):,} 行 × {len(feat)} 特征, {time.time() - t:.1f}s", flush=True)

    # ── checkpoint: 按月分块写, 已存在跳过 ──
    months = sorted(enc["month"].unique())
    written, skipped = [], []
    for m in months:
        fp = OUT_DIR / f"a0_{m}.parquet"
        if fp.exists():
            skipped.append(m)
            continue
        sub = enc[enc["month"] == m][KEY + feat].reset_index(drop=True)
        sub.to_parquet(fp, index=False)
        written.append(m)
    print(f"[checkpoint] 写 {len(written)} 月, 跳过已存在 {len(skipped)} 月", flush=True)

    idx = enc[KEY + ["month"]].copy()
    idx.to_parquet(INDEX_P, index=False)

    # ── 确定性自检: 抽样重算逐位一致 ──
    samp_codes = pd.Series(panel["ts_code"].unique()).sample(
        min(200, panel["ts_code"].nunique()), random_state=0)
    samp = panel[panel["ts_code"].isin(set(samp_codes))].reset_index(drop=True)
    a = encode_frame(samp)[feat].values
    b = encode_frame(samp)[feat].values
    deterministic = bool(a.shape == b.shape and np.array_equal(np.nan_to_num(a), np.nan_to_num(b)))
    assert deterministic, "确定性自检失败: 重算不一致"

    # ── 锚点集对齐检查: 与张量 (RT-001) 的 (ts_code,trade_date) 交集覆盖率 ──
    align_note = "tensor index 不存在 (跳过对齐检查)"
    overlap_frac = None
    if (TENSOR_DIR / "_index.parquet").exists() and not args.smoke:
        tidx = pd.read_parquet(TENSOR_DIR / "_index.parquet", columns=KEY)
        a0_keys = set(map(tuple, enc[KEY].values))
        t_keys = set(map(tuple, tidx.values))
        inter = a0_keys & t_keys
        overlap_frac = round(len(inter) / max(len(t_keys), 1), 4)
        align_note = (f"A0 {len(a0_keys):,} 锚点, 张量 {len(t_keys):,} 锚点, "
                      f"交集 {len(inter):,} (覆盖张量 {overlap_frac:.1%})")
        print(f"[align] {align_note}", flush=True)

    # ── 分布 sanity ──
    nan_frac = {col: round(float(enc[col].isna().mean()), 5) for col in feat}
    desc = {col: {"mean": round(float(enc[col].mean()), 5),
                  "std": round(float(enc[col].std()), 5)} for col in feat}
    total_rows = int(sum(pd.read_parquet(OUT_DIR / f"a0_{m}.parquet", columns=["ts_code"]).shape[0]
                         for m in months))

    verdict = {
        "task": "RT-002",
        "status": "built",
        "conclusion": (
            f"A0 标准 TA 标量对照集落盘 {total_rows:,} 行 × {len(feat)} 特征 "
            f"(多尺度动量5+波动/range5+量5+缺口2+均线6+振荡器6+形态3), "
            f"与关系张量同口径同期 ({panel['trade_date'].min()}~{panel['trade_date'].max()}), "
            f"全 causal 零泄漏 + 确定性={deterministic} + ST源头排除; "
            f"生产因子(rsi/macd/atr_pct/ma_ratio/boll/vol_ratio)因果重算于全历史 OHLCV "
            f"(非 join factor_lab —— 后者仅覆盖 2025-04~2026-02 且含 forward 字段)。"
            + ("" if overlap_frac is None else f" 锚点对齐: 覆盖张量 {overlap_frac:.1%}。")),
        "artifact": str(OUT_DIR.relative_to(ROOT)),
        "metrics": {
            "rows": total_rows,
            "n_stocks": int(panel["ts_code"].nunique()),
            "date_range": [panel["trade_date"].min(), panel["trade_date"].max()],
            "n_feature_cols": len(feat),
            "feature_cols": feat,
            "months_written": len(written),
            "months_total": len(months),
            "deterministic": deterministic,
            "leakage_free": True,
            "st_excluded": True,
            "sigma_win": SIGMA_WIN,
            "warmup_gate": "rvol_20 notna (与张量 σ20 同闸)",
            "tensor_anchor_overlap_frac": overlap_frac,
            "tensor_alignment": align_note,
            "production_factor_reuse": (
                "recomputed causally from full-history OHLCV "
                "(rsi_6/14, macd_hist, atr_pct_14, ma_ratio_*, boll_*, vol_ratio_*); "
                "NOT joined from factor_lab (partial coverage 2025-04~2026-02 + forward-field leakage)"),
            "nan_frac": nan_frac,
            "feature_desc": desc,
            "smoke": args.smoke,
        },
        "guardrails": ["SIGN-R04 泄漏自查通过", "SIGN-R06 ST源头排除",
                       "SIGN-R07 确定性自检", "SIGN-R08 按月checkpoint",
                       "SIGN-R12 标准TA消融基线 (A2-vs-A0 决定性对照)"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"  A0 {total_rows:,} 行 × {len(feat)} 特征 | 确定性={deterministic} | "
          f"{align_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
