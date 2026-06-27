# -*- coding: utf-8 -*-
"""WF-001 — de-lookahead r20 真实 walk-forward → 生成 V12.31 OOS 每日持仓.

诊断: BT-001/002/003 的 book 绝对量级 (+140% 年化 / Sharpe 2.78) 被注水, 唯一注水源 =
**r20 池排序模型 r20_v16_long_nost 是单一生产模型, 训练窗 < 20250930 却跨整个测试期 (202410-202604)
做预测** → 测试期前半段 (202410-202506) 在模型训练窗内 = 共模 lookahead。
(pump s5 排序模型在 t005 里已是月度 walk-forward 重训, 无 lookahead; 故唯一要去 lookahead 的是 r20。)

本脚本: 对 r20 池模型做**严格月度 walk-forward 重训** (复刻 train_daily_long_oos 的 LGBMRegressor 配置,
24m lookback, 每预测月只用该月之前数据, time-split val 防 OOS 偷看), 生成 OOS pred_r20, 重建 V12.31
dual-track 池 (池内仍按已 walk-forward 的 ratio_s5 排序) → 产出 OOS 每日持仓 (与 picks_v1231_daily 同 schema)。

下游 run_wf001.py 把 OOS picks 过同一 book 引擎 (同 BT-002 配置), 报真实 Sharpe/年化/maxDD/换手 +
vs BT-002 注水版的缩水幅度 + 分 regime + 因果自检 (OOS r20 IC 应低于 in-sample 生产模型)。

checkpoint (SIGN-R08): 每月 r20 模型落 research/cache/wf001/r20_models/{月}/; 每月 picks 落 picks_by_month/。
ST 源头排除 (load_window exclude_st=True, R06)。前向列 (r20_fresh) 只留 research/cache/ (R04)。生产线只读 (R05)。
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

from train_v15_refresh import load_window, spearman_ic, EXCLUDE
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom,
    get_train_end_for_test_month, get_train_start,
)
from t005_walk_forward_gate import build_dual, TEST_MONTHS, EPS, TRAIN_LOOKBACK_MONTHS

PROD = ROOT / "output" / "production"
T005_MODELS = ROOT / "research" / "cache" / "t005_wf_models"   # 已 walk-forward 的 s5 pump 模型
OUT_DIR = ROOT / "research" / "cache" / "wf001"
OUT_DIR.mkdir(parents=True, exist_ok=True)
R20_MODELS = OUT_DIR / "r20_models"
R20_MODELS.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "picks_by_month"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PICKS = OUT_DIR / "picks_oos_daily.parquet"
DIAG = OUT_DIR / "r20_oos_diagnostics.csv"

# 24m lookback 需要测试月前 24 月数据; 202410 前推 24m → 2022-08
DATA_START, DATA_END = "20220801", "20260601"
N_TRAIN = 900_000    # 子采样上限 (生产用 1.8M; 此处为 walk-forward 19 次, 取折中保真度)
N_TREES_FIXED = 120  # 固定树数 (生产自选 best_iter=87; 见 train_r20_month 注释)
SEED = 42


def build_r20_feat_cols(daily: pd.DataFrame) -> list:
    """复刻 train_daily_long_oos: 数值列去 EXCLUDE (前向字段) + industry_id。"""
    fc = [c for c in daily.columns
          if c not in EXCLUDE and pd.api.types.is_numeric_dtype(daily[c])]
    if "industry_id" not in fc:
        fc.append("industry_id")
    return fc


def train_r20_month(daily_full: pd.DataFrame, feat_cols: list, test_month: str):
    """月度 walk-forward 重训 r20 回归器 (24m lookback, time-split val 防 OOS 偷看)。checkpoint。"""
    mdir = R20_MODELS / test_month
    mfile = mdir / "classifier.txt"
    if mfile.exists():
        return lgb.Booster(model_str=mfile.read_text(encoding="utf-8"))
    mdir.mkdir(parents=True, exist_ok=True)

    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    df_tr = daily_full[(daily_full["trade_date"] >= train_start) &
                       (daily_full["trade_date"] < train_end)]
    sub = df_tr.dropna(subset=["r20"]).copy()
    if len(sub) < 100_000:
        raise RuntimeError(f"{test_month}: r20 训练样本不足 {len(sub)}")
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)

    # de-lookahead 模型复杂度: 固定 N_TREES_FIXED 棵树 (无早停)。
    # 理由 (R01 前置, 看 book P&L 之前定): 生产 r20 模型自身在 5-7 月 future val 上早停于 best_iter=87 →
    # ~100 棵是该信号的恰当复杂度。walk-forward 下两种 val 都退化: 单一时间切片 (2.4m) 对 r20 的 IC 估计噪声
    # 过大, 早停到 2 棵 (degenerate, 高估缩水); 随机 holdout 与 train 共享日期 → val IC 单调升, 训到 2000 cap
    # (过拟合, 低估缩水)。固定 120 棵 (略高于生产 87, 补偿 24m 较短窗) + 强正则 (min_child_samples=300,
    # ff=0.7, bagging) 避免噪声选择, 是最忠实的"只用过去数据的生产风格 r20"复刻。
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
    best = N_TREES_FIXED

    clf.booster_.save_model(str(mfile))
    (mdir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": "r20", "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (mdir / "meta.json").write_text(json.dumps({
        "test_month": test_month, "train_window": [train_start, train_end],
        "best_iter": best, "n_train": int(len(sub)), "n_trees_fixed": N_TREES_FIXED,
        "note": "WF-001 月度 walk-forward r20 回归器 (de-lookahead, 固定树数无早停), 非生产",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [r20 train] {test_month}: train {train_start}-{train_end}, "
          f"n_tr={len(sub):,} trees={best}", flush=True)
    return clf.booster_


def main():
    t0 = time.time()
    print("\n=== WF-001 gen_picks: de-lookahead r20 月度 walk-forward → OOS 持仓 ===\n", flush=True)

    # v3c pump fc (ratio_s5) + 生产 r20 fc (仅供 in-sample 对照 IC)
    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]
    ind_map_p = meta_p.get("industry_map", {})
    meta_r20p = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    r20p_fc = meta_r20p["feature_cols"]
    b20_prod = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)

    # 全局 industry 编码 (行业分类是同期元数据, 非 lookahead; 跨月一致)
    industries = pd.Categorical(daily["industry"].fillna("unknown"))
    daily["industry_id"] = industries.codes.astype(int)

    for c in set(fc) | set(r20p_fc):
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    r20_fc = build_r20_feat_cols(daily)
    print(f"[data] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 / r20 特征 {len(r20_fc)}", flush=True)

    print("[label] r20_fresh 前向 (next_open→close_20d, 评测/IC 用) ...", flush=True)
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

        # ratio_s5 + pump_down_s5 (已 walk-forward, 复用 t005 缓存)
        b5 = lgb.Booster(model_str=s5file.read_text(encoding="utf-8"))
        Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = b5.predict(Xf)
        df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
        df["pump_down_s5"] = proba[:, 1]

        # de-lookahead r20 (本月 walk-forward 重训)
        b20_oos = train_r20_month(daily, r20_fc, m_)
        Xr_oos = df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan)
        df["pred_r20"] = b20_oos.predict(Xr_oos)

        # in-sample 生产 r20 (对照, 因果自检) — 不进 picks, 仅 IC 对比
        Xr_p = df[r20p_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        pred_r20_prod = b20_prod.predict(Xr_p)

        # r20_fresh 前向 (评测 + parity)
        df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

        # 因果自检: OOS vs 生产 r20 的截面 rank-IC vs r20_fresh
        msk = df["r20_fresh"].notna()
        if msk.sum() > 50:
            fresh = df.loc[msk, "r20_fresh"].clip(-30, 30)
            ic_oos = pd.Series(df.loc[msk, "pred_r20"].values).corr(
                pd.Series(fresh.values), method="spearman")
            ic_prod = pd.Series(pred_r20_prod[msk.values]).corr(
                pd.Series(fresh.values), method="spearman")
        else:
            ic_oos = ic_prod = np.nan

        # V7c dual-track, 池内按 ratio_s5 排序 (= V12.31), pred_r20 = OOS
        hold = build_dual(df, ind_mom, sort_col="ratio_s5")
        if hold is None or hold.empty:
            print(f"  {m_}: 无持仓, 跳过", flush=True)
            continue
        hold = hold.rename(columns={"entry_date": "trade_date"})
        hold["month"] = m_
        hold.to_parquet(CKPT_DIR / f"{m_}.parquet", index=False)

        # 生产 r20_v16_long_nost 训练窗 < 20250930 → 月 ≤ 202509 在训练窗内 (in-sample, lookahead)
        in_sample = m_ <= "202509"
        diag_rows = [r for r in diag_rows if str(r["month"]) != m_]
        diag_rows.append({"month": m_, "r20_oos_rankic": float(ic_oos) if pd.notna(ic_oos) else np.nan,
                          "r20_prod_rankic": float(ic_prod) if pd.notna(ic_prod) else np.nan,
                          "prod_in_train_window": bool(in_sample),
                          "n_picks_rows": int(len(hold)), "n_days": int(hold["trade_date"].nunique())})
        pd.DataFrame(diag_rows).to_csv(DIAG, index=False)
        print(f"  {m_}: {len(hold):,} 持仓行 / {hold['trade_date'].nunique()} 日 | "
              f"r20 IC OOS={ic_oos:+.4f} vs 生产={ic_prod:+.4f} "
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
    print(f"\n[done] OOS picks -> {PICKS.relative_to(ROOT)}  "
          f"{len(allh):,} 行 / {allh['trade_date'].nunique()} 日 / {allh['month'].nunique()} 月 / "
          f"{allh['ts_code'].nunique()} 股  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
