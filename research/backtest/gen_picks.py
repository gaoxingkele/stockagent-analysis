# -*- coding: utf-8 -*-
"""BT-001 — 生成 V12.31 基线 book 的每日持仓清单 (foundational asset, BT-002/003 复用).

复刻 V12.31 生产 book 的选股侧 = t005 baseline 臂 (V7c dual-track 池, 池内按 ratio_s5 排序;
ratio_s5 = 生产 v3c 同档 H5/g10% 3way pump 模型的 P_up/P_down)。**全部复用 t005 已缓存的月度
s5 模型 + 生产 r20 池模型, 不重训** (SIGN-R05 生产只读; 模型缓存于 research/cache/t005_wf_models/)。

产出 research/cache/bt001/picks_v1231_daily.parquet:
    [trade_date, ts_code, industry, alloc_pct, r20_fresh, month]
  每个交易日的 V12.31 dual-track 目标持仓 (A 70% 集中 + B 20% 分散, 行业 cap, 余 10% 现金)。
  alloc_pct = 该股目标权重 (A 段 0.70/n_a, B 段 0.20/n_b); r20_fresh = next_open→close_20d 前向 (供 t005 复刻校验)。

checkpoint (SIGN-R08): 每月落盘, 已完成月跳过 -> 可断点续跑。
ST 源头排除 (load_window exclude_st=True 默认, SIGN-R06)。前向列 r20_fresh 只留 cache, 非 features (SIGN-R04)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT))

from train_v15_refresh import load_window
from walk_forward_validation import compute_r20_label, compute_ind_mom
from t005_walk_forward_gate import build_dual, TEST_MONTHS, EPS  # build_dual 池逻辑 + 全局参数

PROD = ROOT / "output" / "production"
WF_MODELS = ROOT / "research" / "cache" / "t005_wf_models"
OUT_DIR = ROOT / "research" / "cache" / "bt001"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PICKS = OUT_DIR / "picks_v1231_daily.parquet"
CKPT_DIR = OUT_DIR / "picks_by_month"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DATA_START, DATA_END = "20240901", "20260601"   # 仅覆盖测试月 (202410+), 无需训练窗


def main():
    t0 = time.time()
    print("\n=== BT-001 gen_picks: 生成 V12.31 基线每日持仓 (复用缓存模型, 不重训) ===\n", flush=True)

    # 特征列 (v3c pump 模型 + r20 池模型)
    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]
    ind_map = meta_p.get("industry_map", {})
    meta_r20 = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    r20_fc = meta_r20["feature_cols"]
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if ind_map:
        daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in set(fc) | set(r20_fc):
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[data] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股", flush=True)

    print("[label] r20_fresh 前向 (next_open→close_20d, 供 t005 复刻校验) ...", flush=True)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    print("[ind] 行业 60d 动量 rank ...", flush=True)
    ind_mom = compute_ind_mom(daily)

    done_months = {p.stem for p in CKPT_DIR.glob("*.parquet")}
    print(f"[ckpt] 已完成 {len(done_months)} 月: {sorted(done_months)}\n", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done_months:
            continue
        s5dir = WF_MODELS / m_ / "pump_scale_5"
        s5file = s5dir / "classifier.txt"
        if not s5file.exists():
            print(f"  {m_}: 缺缓存 s5 模型, 跳过", flush=True)
            continue
        df = daily[daily["trade_date"].str.startswith(m_)].copy()
        if len(df) < 100:
            print(f"  {m_}: 测试样本不足 ({len(df)}), 跳过", flush=True)
            continue

        # s5 pump (ratio_s5 + pump_down_s5)
        b5 = lgb.Booster(model_str=s5file.read_text(encoding="utf-8"))
        Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = b5.predict(Xf)
        df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
        df["pump_down_s5"] = proba[:, 1]

        # r20 池排序 (生产模型)
        Xr = df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        df["pred_r20"] = b20.predict(Xr)

        # r20_fresh 前向 (校验用)
        df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

        # V7c dual-track 池, 池内按 ratio_s5 排序 (= V12.31 baseline 臂)
        hold = build_dual(df, ind_mom, sort_col="ratio_s5")
        if hold is None or hold.empty:
            print(f"  {m_}: 无持仓, 跳过", flush=True)
            continue
        hold = hold.rename(columns={"entry_date": "trade_date"})
        hold["month"] = m_
        hold.to_parquet(CKPT_DIR / f"{m_}.parquet", index=False)
        print(f"  {m_}: {len(hold):,} 持仓行 / {hold['trade_date'].nunique()} 日 / "
              f"{hold['ts_code'].nunique()} 股  (累计 {time.time()-t0:.0f}s)", flush=True)
        del df, b5
        gc.collect()

    # 合并全部月
    parts = [pd.read_parquet(p) for p in sorted(CKPT_DIR.glob("*.parquet"))]
    if not parts:
        print("[done] 无任何月完成", flush=True)
        return
    allh = pd.concat(parts, ignore_index=True)
    allh = allh[allh["month"].isin(TEST_MONTHS)].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    allh.to_parquet(PICKS, index=False)
    print(f"\n[done] picks -> {PICKS.relative_to(ROOT)}  "
          f"{len(allh):,} 行 / {allh['trade_date'].nunique()} 日 / {allh['month'].nunique()} 月 / "
          f"{allh['ts_code'].nunique()} 股  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
