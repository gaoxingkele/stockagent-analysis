# -*- coding: utf-8 -*-
"""STAB-001 gen_pool — 复用 DEP-001(剔holder, clean) + WFE-001(含holder) 缓存 r20 模型 + t005 缓存 s5,
重建每个交易日的 **V7c 候选池** (深度 ≥ baseline top20, 供低换手/N30/稳定性构建), 不重训不改选股逻辑.

北极星 (A1 低换手/稳定化组合构建):
  DEP-001 picks_oos_daily 只存了**最终入选 ~20 只 + alloc_pct**, 无法支撑 top30/粘性(名次跌出top30)
  这类**构建层**实验。本脚本把 dep001/wfe001 的逐月评分**原样重放** (NO retrain, 缓存模型直接 load),
  发出每日**完整 V7c 候选池** (过完 r20top/pyr/行业动量排除/pump_down 过滤后按 ratio_s5 排序的全量名单 + pool_rank)。
  这是 run_stab001 构建 baseline/V_stick/V_N30/V_combo 四变体的唯一新增数据, **选股分数/池构造逻辑逐字不动**,
  只是把"已选 top20"扩成"完整排序候选池"。

两条扰动臂 (接 DIAG-001):
  - clean  臂: pred_r20 = DEP-001 缓存 r20 模型 (剔 holder_pct, PIT-clean) → 主回测用
  - holder 臂: pred_r20 = WFE-001 缓存 r20 模型 (含 holder_pct) → 仅用于"扰动重排率"(变体对 holder 扰动的稳定性)
  ratio_s5 / pump_down_s5 / pyr 两臂共享 (t005 缓存 s5), 唯一差异 = pred_r20 → 池成员不同 = DIAG-001 的扰动机制。

冻结 (R01): 唯一改动是构建层 (下游 run_stab001), 本脚本不改任何选股阈值/特征/模型。
R05 生产只读 (缓存模型直接 load, 不碰生产文件); R06 ST 源头排除 (load_window 默认);
R08 每月候选池 checkpoint; R04 大缓存 research/cache/stab001/ gitignored, 前向 r20_fresh 不写 features/。
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
sys.path.insert(0, str(ROOT / "research" / "backtest"))

from train_v15_refresh import load_window
from walk_forward_validation import compute_r20_label, compute_ind_mom
from t005_walk_forward_gate import (
    TEST_MONTHS, EPS,
    P_BUY, PYR_Q, M_EXCL, PUMP_DOWN_THRESHOLD,
)
from wf001_gen_picks import build_r20_feat_cols
from wfe001_gen_picks import DATA_START, DATA_END, train_r20_month_embargo
from dep001_gen_picks import train_r20_month_clean, DROP_LEAK

PROD = ROOT / "output" / "production"
T005_MODELS = ROOT / "research" / "cache" / "t005_wf_models"
OUT_DIR = ROOT / "research" / "cache" / "stab001"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT = OUT_DIR / "pool_by_month"
CKPT.mkdir(parents=True, exist_ok=True)
POOL_CLEAN = OUT_DIR / "candidate_pool_clean.parquet"
POOL_HOLDER = OUT_DIR / "candidate_pool_holder.parquet"


def build_pool(df_day: pd.DataFrame, ind_mom: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """复刻 build_dual 的 V7c 过滤 (r20top/pyr/行业动量排除/pump_down), 但发出**完整候选池** (非 head(N)).

    与 t005_walk_forward_gate.build_dual 逐行同口径, 唯一区别: 不做 head(MAX_A/MAX_B) 截断与 sizing,
    而是返回过完所有过滤后按 ratio_s5 降序的全量名单 + pool_rank (1-based)。pred_col 切换 clean/holder 臂。
    """
    daily = df_day.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 100:
            continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 50:
            continue
        g["r20_rank"] = g[pred_col].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0:
            continue
        v7c = v7c[v7c["pump_down_s5"] < PUMP_DOWN_THRESHOLD]
        if len(v7c) == 0:
            continue
        v7c = v7c.sort_values("ratio_s5", ascending=False).reset_index(drop=True)
        for i, row in v7c.iterrows():
            rows.append({
                "trade_date": d_, "ts_code": row["ts_code"],
                "industry": row.get("industry", "") or "",
                "ratio_s5": float(row["ratio_s5"]),
                "pump_down_s5": float(row["pump_down_s5"]),
                "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                "pool_rank": int(i + 1),
            })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("\n=== STAB-001 gen_pool: 重建每日 V7c 候选池 (clean+holder 双臂, 复用缓存模型不重训) ===\n", flush=True)

    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]
    meta_r20p = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    r20p_fc = meta_r20p["feature_cols"]

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    industries = pd.Categorical(daily["industry"].fillna("unknown"))
    daily["industry_id"] = industries.codes.astype(int)
    for c in set(fc) | set(r20p_fc):
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    r20_fc_full = build_r20_feat_cols(daily)
    r20_fc_clean = [c for c in r20_fc_full if c != DROP_LEAK]     # DEP-001 剔 holder_pct
    r20_fc_holder = r20_fc_full                                   # WFE-001 含 holder_pct
    cal_all = sorted(daily["trade_date"].unique())
    print(f"[data] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 / "
          f"r20 特征 clean {len(r20_fc_clean)} / holder {len(r20_fc_holder)} / {len(cal_all)} 交易日", flush=True)

    print("[label] r20_fresh 前向 (评测/IC 用, 不进选股) ...", flush=True)
    r20_lab = compute_r20_label(); r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    print("[ind] 行业 60d 动量 rank ...", flush=True)
    ind_mom = compute_ind_mom(daily)

    done = {p.stem.replace("_clean", "") for p in CKPT.glob("*_clean.parquet")}
    print(f"[ckpt] 已完成 {len(done)} 月: {sorted(done)}\n", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        s5file = T005_MODELS / m_ / "pump_scale_5" / "classifier.txt"
        if not s5file.exists():
            print(f"  {m_}: 缺缓存 s5 模型, 跳过", flush=True)
            continue
        df = daily[daily["trade_date"].str.startswith(m_)].copy()
        if len(df) < 100:
            print(f"  {m_}: 测试样本不足 ({len(df)}), 跳过", flush=True)
            continue

        # ratio_s5 / pump_down_s5 (t005 缓存, 两臂共享)
        b5 = lgb.Booster(model_str=s5file.read_text(encoding="utf-8"))
        Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = b5.predict(Xf)
        df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
        df["pump_down_s5"] = proba[:, 1]

        # pred_r20: clean (dep001 缓存) + holder (wfe001 缓存) — 直接 load 缓存, NO retrain
        b20_clean = train_r20_month_clean(daily, r20_fc_clean, m_, cal_all)
        b20_holder = train_r20_month_embargo(daily, r20_fc_holder, m_, cal_all)
        df["pred_r20_clean"] = b20_clean.predict(
            df[r20_fc_clean].astype("float32").replace([np.inf, -np.inf], np.nan))
        df["pred_r20_holder"] = b20_holder.predict(
            df[r20_fc_holder].astype("float32").replace([np.inf, -np.inf], np.nan))

        df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

        pool_clean = build_pool(df, ind_mom, "pred_r20_clean")
        pool_holder = build_pool(df, ind_mom, "pred_r20_holder")
        pool_clean.to_parquet(CKPT / f"{m_}_clean.parquet", index=False)
        pool_holder.to_parquet(CKPT / f"{m_}_holder.parquet", index=False)
        print(f"  {m_}: clean 候选 {len(pool_clean):,} 行 / {pool_clean['trade_date'].nunique()} 日 "
              f"(中位 {pool_clean.groupby('trade_date').size().median():.0f} 只/日)  "
              f"holder {len(pool_holder):,} 行  (累计 {time.time()-t0:.0f}s)", flush=True)
        del df, b5, b20_clean, b20_holder, pool_clean, pool_holder
        gc.collect()

    parts_c = [pd.read_parquet(p) for p in sorted(CKPT.glob("*_clean.parquet"))]
    parts_h = [pd.read_parquet(p) for p in sorted(CKPT.glob("*_holder.parquet"))]
    if not parts_c:
        print("[done] 无任何月完成", flush=True)
        return
    allc = pd.concat(parts_c, ignore_index=True).sort_values(["trade_date", "pool_rank"]).reset_index(drop=True)
    allh = pd.concat(parts_h, ignore_index=True).sort_values(["trade_date", "pool_rank"]).reset_index(drop=True)
    allc.to_parquet(POOL_CLEAN, index=False)
    allh.to_parquet(POOL_HOLDER, index=False)
    print(f"\n[done] clean 候选池 -> {POOL_CLEAN.relative_to(ROOT)}  {len(allc):,} 行 / "
          f"{allc['trade_date'].nunique()} 日 / {allc['ts_code'].nunique()} 股", flush=True)
    print(f"       holder 候选池 -> {POOL_HOLDER.relative_to(ROOT)}  {len(allh):,} 行  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
