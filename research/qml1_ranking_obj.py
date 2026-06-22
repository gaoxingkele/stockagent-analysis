# -*- coding: utf-8 -*-
"""QML-1 Phase-1 — 排序目标函数 vs 回归 (LGBM r20), walk-forward OOS RankIC.

QuantML 文章33精神: 生产 r20 用 objective=regression(MSE) + spearman_ic 早停 → 训练梯度优化
MSE 但选股用排序 = 错配。测: 同特征同超参, 仅换 objective ∈ {regression(base), lambdarank,
rank_xendcg}, walk-forward(24月lookback, 19测试月)重训, 比测试月 OOS RankIC(pred vs r20 因果)。

Phase-1 廉价闸(SIGN-R16): 若排序目标 mean OOS RankIC 不超 regression(差异 NW-t<2) → 廉价 REJECT
(目标函数不漏信号), 不进 Phase-2 book gate。只认 walk-forward(R03); 分regime(R11)。

apples-to-apples: 唯一差异=objective; 同 load_window 特征(ST源头排除+forward label已剔), 同超参,
同 subsample, 同 train/test 月窗。checkpoint(R08): 月度 RankIC append, 跳过已完成。生产线冻结(R05)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from train_v15_refresh import load_window, EXCLUDE
from walk_forward_validation import (
    get_train_end_for_test_month, get_train_start, TEST_MONTHS, TRAIN_LOOKBACK_MONTHS,
)
import oa_screen  # NW-t

CACHE = ROOT / "research" / "cache"
MONTHLY = CACHE / "qml1_monthly_rankic.csv"
VERDICT = ROOT / "research" / "verdicts" / "QML-1.json"

DATA_START = "20220801"
DATA_END = "20260525"
N_EST = 400
SUBSAMPLE = 900_000
N_DECILE = 10          # 排序相关性 label 分桶 (per day)
OBJECTIVES = ["regression", "lambdarank", "rank_xendcg"]

PARAMS = dict(n_estimators=N_EST, learning_rate=0.05, num_leaves=63,
              min_child_samples=300, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1,
              max_bin=127, force_col_wise=True, random_state=42, n_jobs=4, verbose=-1)


def daily_rankic(df_pred: pd.DataFrame) -> float:
    ics = []
    for d, g in df_pred.groupby("trade_date"):
        g = g.dropna(subset=["pred", "r20"])
        if len(g) < 30:
            continue
        ic = stats.spearmanr(g["pred"], g["r20"])[0]
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def train_predict(obj: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
                  feat_cols: list) -> np.ndarray:
    Xtr = df_train[feat_cols].astype("float32")
    if obj == "regression":
        m = lgb.LGBMRegressor(objective="regression", **PARAMS)
        m.fit(Xtr, df_train["r20"].astype("float32"), categorical_feature=["industry_id"])
        return m.predict(df_test[feat_cols].astype("float32"))
    # ranking: per-day decile relevance + group
    dtr = df_train.sort_values("trade_date").reset_index(drop=True)
    Xtr = dtr[feat_cols].astype("float32")
    rel = dtr.groupby("trade_date")["r20"].transform(
        lambda s: pd.qcut(s.rank(method="first"), N_DECILE, labels=False, duplicates="drop"))
    rel = rel.fillna(0).astype(int).to_numpy()
    grp = dtr.groupby("trade_date").size().to_numpy()
    m = lgb.LGBMRanker(objective=obj, label_gain=[i for i in range(N_DECILE)], **PARAMS)
    m.fit(Xtr, rel, group=grp, categorical_feature=["industry_id"])
    return m.predict(df_test[feat_cols].astype("float32"))


def main():
    t0 = time.time()
    print("\n=== QML-1 Phase-1 排序目标 vs 回归 (walk-forward OOS RankIC) ===\n", flush=True)
    print(f"[data] load_window {DATA_START}-{DATA_END} (ST源头排除) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    meta_r20 = json.loads((ROOT / "output/production/r20_v16_long_nost/feature_meta.json")
                          .read_text(encoding="utf-8"))
    ind_map = meta_r20.get("industry_map", {})
    daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    feat_cols = [c for c in daily.columns
                 if c not in set(EXCLUDE) and pd.api.types.is_numeric_dtype(daily[c])
                 and c != "r20"]
    if "industry_id" not in feat_cols:
        feat_cols.append("industry_id")
    for c in feat_cols:
        daily[c] = daily[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    print(f"[data] {len(daily):,} 行 / {len(feat_cols)} 特征 / r20 label 非空 {daily['r20'].notna().mean():.2%}", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records"); done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月", flush=True)

    for tm in TEST_MONTHS:
        if tm in done:
            continue
        train_end = get_train_end_for_test_month(tm)
        train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
        dtr = daily[(daily["trade_date"] >= train_start) & (daily["trade_date"] < train_end)]
        dtr = dtr.dropna(subset=["r20"]).copy()
        dte = daily[daily["trade_date"].str.startswith(tm)].copy()
        dte = dte.dropna(subset=["r20"])
        if len(dtr) < 100_000 or len(dte) < 100:
            print(f"  {tm}: 样本不足 (train {len(dtr)}, test {len(dte)})", flush=True)
            continue
        if len(dtr) > SUBSAMPLE:
            dtr = dtr.sample(n=SUBSAMPLE, random_state=42).reset_index(drop=True)
        row = {"test_month": tm, "n_train": len(dtr), "n_test": len(dte)}
        for obj in OBJECTIVES:
            pred = train_predict(obj, dtr, dte, feat_cols)
            dp = pd.DataFrame({"trade_date": dte["trade_date"].to_numpy(),
                               "pred": pred, "r20": dte["r20"].to_numpy()})
            row[f"rankic_{obj}"] = daily_rankic(dp)
        results.append(row)
        pd.DataFrame(results).to_csv(MONTHLY, index=False)
        print(f"  {tm}: reg={row['rankic_regression']:+.4f} "
              f"lambdarank={row['rankic_lambdarank']:+.4f} "
              f"xendcg={row['rankic_rank_xendcg']:+.4f}", flush=True)
        gc.collect()

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].sort_values("test_month").reset_index(drop=True)
    base = df["rankic_regression"].to_numpy()
    summary = {"test_months": int(len(df)), "baseline_regression": {}}
    print("\n=== QML-1 汇总 (OOS RankIC, walk-forward) ===", flush=True)
    print(f"  {'objective':16s} {'mean RankIC':>12s} {'vs base Δ':>10s} {'Δ NW-t':>8s}", flush=True)
    res_obj = {}
    for obj in OBJECTIVES:
        v = df[f"rankic_{obj}"].to_numpy()
        mean_ic = float(np.nanmean(v))
        delta = v - base
        d_mean = float(np.nanmean(delta))
        d_t = oa_screen.tstat_nw(delta[~np.isnan(delta)]) if obj != "regression" else 0.0
        res_obj[obj] = {"mean_rankic": round(mean_ic, 4), "delta_vs_base": round(d_mean, 4),
                        "delta_nw_t": round(float(d_t), 2)}
        print(f"  {obj:16s} {mean_ic:>+12.4f} {d_mean:>+10.4f} {d_t:>+8.2f}", flush=True)
    summary["by_objective"] = res_obj

    # Phase-1 gate: 任一排序目标 Δ>0 且 NW-t>=2 → 进 Phase-2
    survivors = [o for o in ("lambdarank", "rank_xendcg")
                 if res_obj[o]["delta_vs_base"] > 0 and res_obj[o]["delta_nw_t"] >= 2.0]
    status = "PROCEED_phase2" if survivors else "REJECT_cheap"
    best = max(("lambdarank", "rank_xendcg"), key=lambda o: res_obj[o]["mean_rankic"])
    conclusion = (
        f"walk-forward {len(df)}月 OOS RankIC: regression(base)={res_obj['regression']['mean_rankic']:+.4f}, "
        f"lambdarank={res_obj['lambdarank']['mean_rankic']:+.4f}(Δ{res_obj['lambdarank']['delta_vs_base']:+.4f} "
        f"NW-t{res_obj['lambdarank']['delta_nw_t']:+.2f}), rank_xendcg={res_obj['rank_xendcg']['mean_rankic']:+.4f}"
        f"(Δ{res_obj['rank_xendcg']['delta_vs_base']:+.4f} NW-t{res_obj['rank_xendcg']['delta_nw_t']:+.2f}). "
        + (f"存活 {survivors} → Phase-2 book Δα gate。" if survivors
           else "排序目标未显著超回归(NW-t<2) → 廉价REJECT: 目标函数不漏信号, V12.31 r20 用回归+IC早停已足。"))
    summary.update({"status": status, "survivors": survivors, "best_ranker": best,
                    "conclusion": conclusion,
                    "guardrails": ["SIGN-R01 gate冻结", "SIGN-R03 walk-forward OOS RankIC",
                                   "SIGN-R16 廉价闸先行+NW-t", "apples同特征同超参唯一差异objective",
                                   "SIGN-R05 生产线冻结"]})
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[决策] {status}; {conclusion}", flush=True)
    print(f"[done] 耗时 {(time.time()-t0)/60:.1f} min -> {VERDICT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
