# -*- coding: utf-8 -*-
"""DEP-001 — 剔 holder_pct 泄漏因子, PIT-clean r20 label-embargo walk-forward → V12.31 OOS 持仓.

北极星 (剔泄漏因子, 清 codex 唯一硬红线):
  FIN-002 部署前 checklist 定位到 r20 池排序模型 r20_v16_long_nost 的 236 特征里**唯一 MEDIUM PIT 风险 =
  holder_pct** (Tushare stk_holdernumber 季频股东户数; backtest_new_factors.compute_holder_pct_at 按
  end_date<=target_date 选数, 未取 ann_date → Q1 报告 end_date 0331 在 ~0430 公告前即被用 = 真前视)。
  但其重要度极低 (r20 gain 排 118/236 占 0.005%)。用户决策 = 低影响则剔最干净 (零 ann_date 工程, 隐患彻底消除)。

DEP-001 = WFE-001 (label-embargo r20 月度 walk-forward) **唯一改动 = r20 训练特征集严格剔除 holder_pct**:
  build_r20_feat_cols(daily) 后去掉 holder_pct (235 -> 234 特征), 其余口径 (24m lookback / embargo
  P_start-21交易日 / 固定120树 / 双轨 / 引擎 / 成本 / 再平衡) 与 WFE-001 逐字一致 → apples-to-apples,
  唯一差异 = 去 holder_pct。

下游 run_dep001.py 把 PIT-clean OOS picks 过同一 book 引擎, 报 PIT-clean Sharpe/年化/maxDD + per-cohort
bootstrap ΔvsWFE-001(1.31) 判 clean确认 / 材料性下降 (gate_dep)。

checkpoint (SIGN-R08): 每月 r20 模型落 research/cache/dep001/r20_models/{月}/; picks 落 picks_by_month/。
ST 源头排除 (load_window exclude_st=True, R06)。前向列只留 research/cache/ (R04)。生产线只读 (R05)。
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
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom,
    get_train_end_for_test_month, get_train_start,
)
from t005_walk_forward_gate import build_dual, TEST_MONTHS, EPS, TRAIN_LOOKBACK_MONTHS
from wf001_gen_picks import build_r20_feat_cols
from wfe001_gen_picks import (
    embargo_cut_for_month, DATA_START, DATA_END, N_TRAIN, N_TREES_FIXED, SEED,
    R20_HORIZON, EMBARGO_TDAYS,
)

PROD = ROOT / "output" / "production"
T005_MODELS = ROOT / "research" / "cache" / "t005_wf_models"
OUT_DIR = ROOT / "research" / "cache" / "dep001"
OUT_DIR.mkdir(parents=True, exist_ok=True)
R20_MODELS = OUT_DIR / "r20_models"
R20_MODELS.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "picks_by_month"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PICKS = OUT_DIR / "picks_oos_daily.parquet"
DIAG = OUT_DIR / "r20_oos_diagnostics.csv"

# ── DEP-001 唯一改动: r20 训练特征集剔除的泄漏因子 (PIT 前视) ──
DROP_LEAK = "holder_pct"


def train_r20_month_clean(daily_full: pd.DataFrame, feat_cols: list, test_month: str,
                          cal_all: list):
    """月度 walk-forward 重训 r20 回归器 + label-availability embargo, 特征集已剔 holder_pct。

    与 WFE-001 train_r20_month_embargo 唯一差异 = 入参 feat_cols 已去 holder_pct (在 main 里统一剔)。
    截止口径/树数/超参逐字一致。
    """
    mdir = R20_MODELS / test_month
    mfile = mdir / "classifier.txt"
    if mfile.exists():
        return lgb.Booster(model_str=mfile.read_text(encoding="utf-8"))
    mdir.mkdir(parents=True, exist_ok=True)

    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    embargo_cut = embargo_cut_for_month(cal_all, test_month)
    if embargo_cut is None:
        raise RuntimeError(f"{test_month}: 无法定位 embargo 截止日")

    df_tr = daily_full[(daily_full["trade_date"] >= train_start) &
                       (daily_full["trade_date"] <= embargo_cut)]
    sub = df_tr.dropna(subset=["r20"]).copy()
    if len(sub) < 100_000:
        raise RuntimeError(f"{test_month}: r20 训练样本不足 {len(sub)} (embargo后)")
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)

    def X(d):
        return d[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    Xtr, ytr = X(sub), sub["r20"].astype("float32")

    clf = lgb.LGBMRegressor(
        n_estimators=N_TREES_FIXED, learning_rate=0.04, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1, max_bin=127, force_col_wise=True,
        random_state=SEED, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(Xtr, ytr, categorical_feature=["industry_id"])

    clf.booster_.save_model(str(mfile))
    (mdir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": "r20", "model_type": "regressor",
        "dropped_leak_factor": DROP_LEAK,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (mdir / "meta.json").write_text(json.dumps({
        "test_month": test_month,
        "train_window_wf001": [train_start, train_end],
        "embargo_cut": embargo_cut, "embargo_tdays": EMBARGO_TDAYS,
        "n_trees_fixed": N_TREES_FIXED, "n_train": int(len(sub)),
        "n_features": len(feat_cols), "dropped_leak_factor": DROP_LEAK,
        "note": "DEP-001 PIT-clean r20 (剔 holder_pct) + label-availability embargo (同 WFE-001 口径)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [r20 clean+embargo] {test_month}: {train_start}~{embargo_cut} "
          f"(剔 {DROP_LEAK}, n_feat={len(feat_cols)}), n_tr={len(sub):,} trees={N_TREES_FIXED}",
          flush=True)
    return clf.booster_


def main():
    t0 = time.time()
    print("\n=== DEP-001 gen_picks: PIT-clean r20 (剔 holder_pct) embargo walk-forward → OOS 持仓 ===\n",
          flush=True)

    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]
    meta_r20p = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    r20p_fc = meta_r20p["feature_cols"]
    b20_prod = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))

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
    # ── DEP-001 唯一改动: 严格剔 holder_pct (PIT 前视) ──
    r20_fc = [c for c in r20_fc_full if c != DROP_LEAK]
    assert DROP_LEAK not in r20_fc, f"{DROP_LEAK} 未被剔除"
    dropped = DROP_LEAK in r20_fc_full
    print(f"[clean] r20 特征 {len(r20_fc_full)} -> {len(r20_fc)} "
          f"(剔 {DROP_LEAK}: {'命中并剔除' if dropped else '原本不在特征集!'})", flush=True)

    cal_all = sorted(daily["trade_date"].unique())
    print(f"[data] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 / r20 特征(clean) {len(r20_fc)} / "
          f"{len(cal_all)} 交易日 / embargo={EMBARGO_TDAYS}交易日", flush=True)

    print("[label] r20_fresh 前向 (next_open->close_20d, 评测/IC 用) ...", flush=True)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    print("[ind] 行业 60d 动量 rank ...", flush=True)
    ind_mom = compute_ind_mom(daily)

    done_months = {p.stem for p in CKPT_DIR.glob("*.parquet")}
    print(f"[ckpt] 已完成 {len(done_months)} 月: {sorted(done_months)}\n", flush=True)

    diag_rows = []
    if DIAG.exists():
        diag_rows = pd.read_csv(DIAG, dtype={"month": str}).to_dict("records")
    diag_done = {str(r["month"]) for r in diag_rows}

    for m_ in TEST_MONTHS:
        if m_ in done_months and m_ in diag_done:
            continue
        s5file = T005_MODELS / m_ / "pump_scale_5" / "classifier.txt"
        if not s5file.exists():
            print(f"  {m_}: 缺缓存 s5 模型, 跳过", flush=True)
            continue
        df = daily[daily["trade_date"].str.startswith(m_)].copy()
        if len(df) < 100:
            print(f"  {m_}: 测试样本不足 ({len(df)}), 跳过", flush=True)
            continue

        b5 = lgb.Booster(model_str=s5file.read_text(encoding="utf-8"))
        Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = b5.predict(Xf)
        df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
        df["pump_down_s5"] = proba[:, 1]

        # PIT-clean (剔 holder_pct) + EMBARGO r20 (本月 walk-forward 重训)
        b20_oos = train_r20_month_clean(daily, r20_fc, m_, cal_all)
        Xr_oos = df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan)
        df["pred_r20"] = b20_oos.predict(Xr_oos)

        # in-sample 生产 r20 (对照, 含 holder_pct) — 不进 picks
        Xr_p = df[r20p_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        pred_r20_prod = b20_prod.predict(Xr_p)

        df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

        msk = df["r20_fresh"].notna()
        if msk.sum() > 50:
            fresh = df.loc[msk, "r20_fresh"].clip(-30, 30)
            ic_oos = pd.Series(df.loc[msk, "pred_r20"].values).corr(
                pd.Series(fresh.values), method="spearman")
            ic_prod = pd.Series(pred_r20_prod[msk.values]).corr(
                pd.Series(fresh.values), method="spearman")
        else:
            ic_oos = ic_prod = np.nan

        hold = build_dual(df, ind_mom, sort_col="ratio_s5")
        if hold is None or hold.empty:
            print(f"  {m_}: 无持仓, 跳过", flush=True)
            continue
        hold = hold.rename(columns={"entry_date": "trade_date"})
        hold["month"] = m_
        hold.to_parquet(CKPT_DIR / f"{m_}.parquet", index=False)

        in_sample = m_ <= "202509"
        diag_rows = [r for r in diag_rows if str(r["month"]) != m_]
        diag_rows.append({"month": m_, "r20_oos_rankic": float(ic_oos) if pd.notna(ic_oos) else np.nan,
                          "r20_prod_rankic": float(ic_prod) if pd.notna(ic_prod) else np.nan,
                          "prod_in_train_window": bool(in_sample),
                          "n_picks_rows": int(len(hold)), "n_days": int(hold["trade_date"].nunique())})
        pd.DataFrame(diag_rows).to_csv(DIAG, index=False)
        print(f"  {m_}: {len(hold):,} 持仓行 / {hold['trade_date'].nunique()} 日 | "
              f"r20 IC OOS(clean)={ic_oos:+.4f} vs 生产(含holder)={ic_prod:+.4f} "
              f"({'in-sample' if in_sample else 'true-OOS'})  (累计 {time.time()-t0:.0f}s)", flush=True)
        del df, b5, b20_oos
        gc.collect()

    parts = [pd.read_parquet(p) for p in sorted(CKPT_DIR.glob("*.parquet"))]
    if not parts:
        print("[done] 无任何月完成", flush=True)
        return
    allh = pd.concat(parts, ignore_index=True)
    allh = allh[allh["month"].isin(TEST_MONTHS)].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    allh.to_parquet(PICKS, index=False)
    print(f"\n[done] PIT-clean OOS picks -> {PICKS.relative_to(ROOT)}  "
          f"{len(allh):,} 行 / {allh['trade_date'].nunique()} 日 / {allh['month'].nunique()} 月 / "
          f"{allh['ts_code'].nunique()} 股  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
