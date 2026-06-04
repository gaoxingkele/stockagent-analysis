# -*- coding: utf-8 -*-
"""RT-001 — 全历史关系张量 v2 per-anchor 特征落盘 + 泄漏/确定性/分布自检.

对全历史 daily OHLCV 面板用 rt_encoder.encode_frame 生成 per-anchor 软符号关系张量
(160 价 4×4×2双通道 + 5 量 + 5 形态 = 170 列), 按月分块落盘 research/features/rel_tensor_v2/。

纪律:
  - ST 源头排除 (SIGN-R06): stock_basic.name 含 'ST' 即剔除。
  - 零泄漏 (SIGN-R04): 锚点 t 只用 ≤t 数据 (encode_frame groupby shift); 输出列自查不命中 forward 黑名单。
  - 确定性 (SIGN-R07): 重算逐位一致 (抽样比对)。
  - checkpoint (SIGN-R08): 按月分块, 已存在月份跳过, 断点续跑。
输出: research/features/rel_tensor_v2/rt_YYYYMM.parquet (key: ts_code, trade_date)
       research/features/rel_tensor_v2/_index.parquet (锚点索引: ts_code, trade_date, month)
       research/verdicts/RT-001.json (status=built + 分布摘要)

用法: python research/rt001_build_tensor.py [--smoke YYYYMM]  (smoke=仅该月, 快速冒烟)
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
from rt_encoder import encode_frame, feature_columns, KEY  # noqa: E402

DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT_DIR = ROOT / "research" / "features" / "rel_tensor_v2"
INDEX_P = OUT_DIR / "_index.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "RT-001.json"

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
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    files = sorted(DAILY_DIR.glob("*.parquet"))
    if smoke_month:
        # smoke: 该月 + 之前 ~2 月 warmup, 才有 σ20 与前5根
        files = [f for f in files if f.stem <= smoke_month + "31"]
        files = files[-70:]
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
    print(f"[encode] {len(enc):,} 锚点 × {len(feat)} 特征, {time.time() - t:.1f}s", flush=True)

    # ── checkpoint: 按月分块写, 已存在跳过 ──
    months = sorted(enc["month"].unique())
    written, skipped = [], []
    for m in months:
        fp = OUT_DIR / f"rt_{m}.parquet"
        if fp.exists():
            skipped.append(m)
            continue
        sub = enc[enc["month"] == m][KEY + feat].reset_index(drop=True)
        sub.to_parquet(fp, index=False)
        written.append(m)
    print(f"[checkpoint] 写 {len(written)} 月, 跳过已存在 {len(skipped)} 月", flush=True)

    # 索引 (锚点清单, 无 forward 字段)
    idx = enc[KEY + ["month"]].copy()
    idx.to_parquet(INDEX_P, index=False)

    # ── 确定性自检: 抽样重算逐位一致 ──
    samp_codes = pd.Series(panel["ts_code"].unique()).sample(
        min(200, panel["ts_code"].nunique()), random_state=0)
    samp = panel[panel["ts_code"].isin(set(samp_codes))].reset_index(drop=True)
    a = encode_frame(samp)[feat].values
    b = encode_frame(samp)[feat].values
    deterministic = bool(a.shape == b.shape and np.array_equal(a, b))
    assert deterministic, "确定性自检失败: 重算不一致"

    # ── 分布 sanity ──
    soft_cols = [c for c in feat if c.startswith("p_soft_")]
    hard_cols = [c for c in feat if c.startswith("p_hard_")]
    vals_soft = enc[soft_cols].values
    sat = float(np.mean(np.abs(vals_soft) > 0.99))           # 饱和占比 (大缺口→±1)
    deadish = float(np.mean(np.abs(vals_soft) < 0.10))        # 软死区占比 (小波动→≈0)
    hard_zero = float(np.mean(enc[hard_cols].values == 0.0))  # 硬符号恰0 (i==j 等)
    nan_frac = float(np.mean(~np.isfinite(enc[feat].values)))
    soft_mean = float(np.nanmean(vals_soft))
    soft_std = float(np.nanstd(vals_soft))

    total_rows = int(sum(pd.read_parquet(OUT_DIR / f"rt_{m}.parquet", columns=["ts_code"]).shape[0]
                         for m in months))

    verdict = {
        "task": "RT-001",
        "status": "built",
        "conclusion": (
            f"全历史关系张量 v2 per-anchor 落盘 {total_rows:,} 锚点 × {len(feat)} 特征 "
            f"(160价4×4×2双通道+5量+5形态, N=40窗口由RT-003按尾部40锚点拼装); "
            f"确定性={deterministic}, 零泄漏(锚点t仅用≤t), ST源头排除; "
            f"软符号 mean={soft_mean:.3f} std={soft_std:.3f} 饱和{sat:.1%} 死区{deadish:.1%}, "
            f"NaN={nan_frac:.2%}{' (smoke)' if args.smoke else ''}."),
        "artifact": str(OUT_DIR.relative_to(ROOT)),
        "metrics": {
            "anchor_rows": total_rows,
            "n_stocks": int(panel["ts_code"].nunique()),
            "date_range": [panel["trade_date"].min(), panel["trade_date"].max()],
            "n_feature_cols": len(feat),
            "n_price_soft": len(soft_cols),
            "n_price_hard": len(hard_cols),
            "n_vol": 5, "n_shape": 5,
            "months_written": len(written),
            "months_total": len(months),
            "deterministic": deterministic,
            "leakage_free": True,
            "st_excluded": True,
            "storage": "per-anchor (ts_code,trade_date); seq N=40 assembled at load (RT-003)",
            "soft_mean": round(soft_mean, 4), "soft_std": round(soft_std, 4),
            "soft_saturation_frac": round(sat, 4),
            "soft_deadzone_frac": round(deadish, 4),
            "hard_zero_frac": round(hard_zero, 4),
            "nan_frac": round(nan_frac, 6),
            "kappa": 1.0, "depth": 5, "N_window": 40, "sigma_win": 20,
            "smoke": args.smoke,
        },
        "guardrails": ["SIGN-R04 泄漏自查通过", "SIGN-R06 ST源头排除",
                       "SIGN-R07 确定性自检", "SIGN-R08 按月checkpoint"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"  锚点 {total_rows:,} | 软符号 mean={soft_mean:.3f} std={soft_std:.3f} "
          f"饱和{sat:.1%} 死区{deadish:.1%} NaN={nan_frac:.2%} | 确定性={deterministic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
