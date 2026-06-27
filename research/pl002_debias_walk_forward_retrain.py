# -*- coding: utf-8 -*-
"""PL-002 去偏 pump 启动子 walk-forward 月度重训.

承 PL-001 诊断 (收紧后的北极星):
  - 去偏 = 唯一改动 label 的 past_r5 约束, 且该约束**只活在 down 类** (past_r5<=8% 把高涨幅
    伪跌启动踢成 neutral); up 类两版逐位相同 (横盘/微涨启动早已是 up 正样本)。
  - PL-001 已预警: 纯 label 重训杠杆仅在 down 约束, up 端零增量 → 增益若有只能来自 down 约束
    放松对 3way softmax 的间接影响。本任务把这个假设送进真 walk-forward 重训 (中间产物),
    捕捉验证 (PL-003) 与 ship gate (PL-004) 在后续任务。

本任务 (PL-002) 做的事:
  复用 t005 walk-forward 口径 (19 月 202410-202604, 24 月 lookback 月度重训), 训**两臂**
  3 分类 LGBM (apples-to-apples, 唯一差异 = label 的 past_r5 约束):
    arm "v3c"    : label = PL-001 的 v3c_label    (down 带 past_r5<=8% 约束, = 生产口径)
    arm "debias" : label = PL-001 的 debias_label (down 去 past_r5 约束, 纳入横盘/微涨伪跌)
  两臂同生产 pump 特征 (247 列)、同窗口、同超参、同种子 → 任何共模偏差在 Δ 中相消。
  每月每臂模型 checkpoint; 测试月用当月新训模型推理 → pump_up/pump_down/ratio。

为何两臂都在 PL-002 训 (而非只训 debias):
  生产 v3c 模型是一次性训到 20250930 的, 与 walk-forward 月度重训窗口不可逐位比 (训练窗/样本
  量/早停都不同)。PL-004 的 apples-to-apples gate 要求"两臂仅 label 不同", 故 baseline 也必须
  是 walk-forward 重训的 v3c (同 harness)。两臂在同一脚本同窗口产出, 保证逐位可比。

零泄漏 / 红线纪律 (SIGN-R03/R04/R05/R06/R08):
  - 特征全 causal (load_window backward); label 来自 PL-001 cache (前向 high/low 仅训练目标),
    输出落 research/cache/ (非 features), 不进泄漏黑名单闸。
  - ST 源头排除 (load_window exclude_st=True 默认)。
  - 完全不碰生产模型/文件 (只读 feature_meta + production v3c 仅取 feature_cols/industry_map)。
  - 中间指标 (正例率/类分布) 仅训练摘要; ship 由 PL-004 walk-forward α 定 (SIGN-R03)。
  - checkpoint: 每月每臂模型 research/cache/pl002_wf_models/{month}/{arm}/ 存在即跳过;
    pump 分数逐月 append research/cache/pl002_pump_scores.parquet 分片 → 断点续跑 (SIGN-R08)。

产出:
  - research/cache/pl002_wf_models/{month}/{v3c,debias}/  (月度两臂模型 checkpoint)
  - research/cache/pl002_pump_scores/{month}.parquet       (逐月分片, 两臂 pump 分数)
  - research/cache/pl002_pump_scores.parquet               (合并; ts_code,trade_date,past_r5,
                                                            两版 label, 两臂 P_up/P_down/ratio)
  - research/cache/pl002_results.json                       (训练摘要 + 正例率 vs v3c)
  - research/verdicts/PL-002.json                           (status=built)
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window
from walk_forward_validation import get_train_end_for_test_month, get_train_start

CACHE = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
WF_MODELS = CACHE / "pl002_wf_models"
WF_MODELS.mkdir(parents=True, exist_ok=True)
SCORE_DIR = CACHE / "pl002_pump_scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_CACHE = CACHE / "pump_debias_label.parquet"     # PL-001 产出
SCORES_OUT = CACHE / "pl002_pump_scores.parquet"
RESULTS = CACHE / "pl002_results.json"
VERDICT = ROOT / "research" / "verdicts" / "PL-002.json"
PROD = ROOT / "output" / "production"

# walk-forward 口径 (与 t005 / V11 walk-forward 完全一致)
TEST_MONTHS = ["202410", "202411", "202412",
               "202501", "202502", "202503", "202504", "202505", "202506",
               "202507", "202508", "202509",
               "202510", "202511", "202512",
               "202601", "202602", "202603", "202604"]
TRAIN_LOOKBACK_MONTHS = 24
DATA_START, DATA_END = "20220801", "20260601"

N_TRAIN = 400_000
SEED = 20260612
EPS = 0.01
ARMS = ["v3c", "debias"]
LABEL_COL = {"v3c": "v3c_label", "debias": "debias_label"}


# ───────────────────── 训练 (3way, 月度, 单臂, checkpoint) ─────────────────────
def train_arm_month(df_train: pd.DataFrame, fc: list, label_col: str,
                    test_month: str, arm: str) -> tuple[lgb.Booster, dict]:
    mdir = WF_MODELS / test_month / arm
    mfile = mdir / "classifier.txt"
    metafile = mdir / "meta.json"
    if mfile.exists() and metafile.exists():
        booster = lgb.Booster(model_str=mfile.read_text(encoding="utf-8"))
        return booster, json.loads(metafile.read_text(encoding="utf-8"))
    mdir.mkdir(parents=True, exist_ok=True)

    sub = df_train.dropna(subset=[label_col]).copy()
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=SEED).reset_index(drop=True)
    # 时间切分早停 (与 t005 同: 排序位置 0.92 分位日期, 字典序==时序)
    if len(sub):
        _sd = np.sort(sub["trade_date"].astype(str).to_numpy())
        cut = str(_sd[min(int(len(_sd) * 0.92), len(_sd) - 1)])
    else:
        cut = "0"
    tr = sub[sub["trade_date"] < cut]
    va = sub[sub["trade_date"] >= cut]
    if len(va) < 5000 or len(tr) < 20000:
        tr, va = sub, sub.iloc[:0]

    def Xy(d):
        X = d[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        y = d[label_col].astype("int")
        return X, y

    Xtr, ytr = Xy(tr)
    cls_dist = ytr.value_counts(normalize=True).sort_index().round(4).to_dict()
    params = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                  num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_child_samples=80,
                  seed=SEED, verbose=-1, num_threads=0)
    dtr = lgb.Dataset(Xtr, ytr)
    if len(va):
        Xva, yva = Xy(va)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False),
                                       lgb.log_evaluation(0)])
        best = booster.best_iteration
    else:
        booster = lgb.train(params, dtr, num_boost_round=150)
        best = 150
    n_up = int((ytr == 2).sum()); n_dn = int((ytr == 1).sum()); n_ne = int((ytr == 0).sum())
    meta = {
        "test_month": test_month, "arm": arm, "label_col": label_col,
        "best_iter": int(best), "n_train": int(len(tr)), "n_val": int(len(va)),
        "class_dist": {str(k): v for k, v in cls_dist.items()},
        "n_up": n_up, "n_down": n_dn, "n_neutral": n_ne,
        "up_rate": round(n_up / max(len(ytr), 1), 5),
        "down_rate": round(n_dn / max(len(ytr), 1), 5),
        "note": "PL-002 walk-forward 月度去偏 pump 模型, 非生产 (apples-to-apples 两臂仅 label 不同)",
    }
    mfile.write_text(booster.model_to_string(), encoding="utf-8")
    metafile.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return booster, meta


# ───────────────────────── 单月 (训两臂 + 推理测试月) ─────────────────────────
def run_month(test_month: str, daily_full: pd.DataFrame, fc: list) -> dict | None:
    spath = SCORE_DIR / f"{test_month}.parquet"
    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    print(f"\n--- {test_month}: train {train_start}-{train_end} ---", flush=True)

    df_train = daily_full[(daily_full["trade_date"] >= train_start) &
                          (daily_full["trade_date"] < train_end)].copy()
    if len(df_train) < 100_000:
        print(f"  跳过 (训练样本不足 {len(df_train)})", flush=True)
        return None

    metas = {}
    boosters = {}
    for arm in ARMS:
        npos = int((df_train[LABEL_COL[arm]] == 2).sum())
        nneg = int((df_train[LABEL_COL[arm]] == 1).sum())
        print(f"  训 {arm} (up={npos:,} down={nneg:,}) ...", flush=True)
        boosters[arm], metas[arm] = train_arm_month(
            df_train, fc, LABEL_COL[arm], test_month, arm)

    # 测试月推理 (分片 checkpoint 可能已存在)
    if spath.exists():
        print(f"  [ckpt] {spath.name} 命中, 跳过推理", flush=True)
        scored = pd.read_parquet(spath)
    else:
        df_test = daily_full[daily_full["trade_date"].str.startswith(test_month)].copy()
        if len(df_test) < 100:
            print(f"  跳过 (测试样本不足 {len(df_test)})", flush=True)
            return None
        Xf = df_test[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        scored = df_test[["ts_code", "trade_date", "past_r5",
                          "v3c_label", "debias_label"]].copy()
        for arm in ARMS:
            proba = boosters[arm].predict(Xf)
            scored[f"{arm}_pump_up"] = proba[:, 2]
            scored[f"{arm}_pump_down"] = proba[:, 1]
            scored[f"{arm}_ratio"] = proba[:, 2] / (proba[:, 1] + EPS)
        scored.to_parquet(spath, index=False)
        print(f"  scored {len(scored):,} 行 -> {spath.name}", flush=True)
        del df_test, Xf

    # 月度训练摘要 (正例率 vs v3c, SIGN-R03 仅描述)
    out = {"test_month": test_month, "train_start": train_start, "train_end": train_end,
           "n_train_rows": int(len(df_train))}
    for arm in ARMS:
        out[f"{arm}_up_rate"] = metas[arm]["up_rate"]
        out[f"{arm}_down_rate"] = metas[arm]["down_rate"]
        out[f"{arm}_n_up"] = metas[arm]["n_up"]
        out[f"{arm}_n_down"] = metas[arm]["n_down"]
        out[f"{arm}_best_iter"] = metas[arm]["best_iter"]
    out["down_rate_delta_debias_minus_v3c"] = round(
        metas["debias"]["down_rate"] - metas["v3c"]["down_rate"], 5)
    out["up_rate_delta_debias_minus_v3c"] = round(
        metas["debias"]["up_rate"] - metas["v3c"]["up_rate"], 5)
    print(f"  v3c up={out['v3c_up_rate']:.4f} down={out['v3c_down_rate']:.4f} | "
          f"debias up={out['debias_up_rate']:.4f} down={out['debias_down_rate']:.4f} | "
          f"Δdown={out['down_rate_delta_debias_minus_v3c']:+.4f}", flush=True)

    del df_train, boosters
    gc.collect()
    return out


def main():
    t0 = time.time()
    print("\n=== PL-002 去偏 pump 启动子 walk-forward 月度重训 (两臂) ===\n", flush=True)

    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json")
                      .read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})
    print(f"[meta] 生产 pump 特征 {len(fc)} 列 (apples-to-apples 两臂共用)", flush=True)

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map:
        daily_full["industry_id"] = (daily_full["industry"].fillna("unknown")
                                     .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily_full.columns:
            daily_full[c] = 0.0

    # 合并 PL-001 两版 label (前向 outcome 仅训练目标, 来自 cache)
    if not LABEL_CACHE.exists():
        raise FileNotFoundError(f"缺 PL-001 label cache: {LABEL_CACHE} (先跑 pl001_debias_label.py)")
    lab = pd.read_parquet(LABEL_CACHE)
    lab["trade_date"] = lab["trade_date"].astype(str)
    n_before = len(daily_full)
    daily_full = daily_full.merge(
        lab[["ts_code", "trade_date", "past_r5", "v3c_label", "debias_label"]],
        on=["ts_code", "trade_date"], how="left")
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    cov = float(daily_full["v3c_label"].notna().mean())
    print(f"[data] daily_full {len(daily_full):,} 行 / {daily_full['ts_code'].nunique()} 股; "
          f"label 覆盖 {cov:.1%} (未覆盖=窗口末端无前向 outcome)", flush=True)
    assert len(daily_full) == n_before, "merge 改变行数 (重复键?)"

    # checkpoint: 已完成月跳过
    results = []
    done = set()
    summ_csv = CACHE / "pl002_monthly_summary.csv"
    if summ_csv.exists():
        prev = pd.read_csv(summ_csv, dtype={"test_month": str})
        results = prev.to_dict("records")
        done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月: {sorted(done)}", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = run_month(m_, daily_full, fc)
        if r:
            results.append(r)
            pd.DataFrame(results).to_csv(summ_csv, index=False)

    # 合并逐月 pump 分数分片
    parts = [pd.read_parquet(SCORE_DIR / f"{m}.parquet") for m in TEST_MONTHS
             if (SCORE_DIR / f"{m}.parquet").exists()]
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(SCORES_OUT, index=False)
    print(f"\n[out] 合并 pump 分数 -> {SCORES_OUT.name} ({len(scores):,} 行)", flush=True)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].copy()
    df["test_month"] = df["test_month"].astype(str)
    df = df.sort_values("test_month").reset_index(drop=True)

    # ── 汇总训练摘要 (正例率 vs v3c) ──
    summary = {
        "n_months": int(len(df)),
        "v3c_up_rate_mean": round(float(df["v3c_up_rate"].mean()), 5),
        "debias_up_rate_mean": round(float(df["debias_up_rate"].mean()), 5),
        "v3c_down_rate_mean": round(float(df["v3c_down_rate"].mean()), 5),
        "debias_down_rate_mean": round(float(df["debias_down_rate"].mean()), 5),
        "down_rate_delta_mean": round(float(df["down_rate_delta_debias_minus_v3c"].mean()), 5),
        "up_rate_delta_mean": round(float(df["up_rate_delta_debias_minus_v3c"].mean()), 5),
        "down_rate_delta_min": round(float(df["down_rate_delta_debias_minus_v3c"].min()), 5),
        "down_rate_delta_max": round(float(df["down_rate_delta_debias_minus_v3c"].max()), 5),
    }
    # 推理摘要 (两臂 ratio/P_up 相关, 描述去偏对排序的撬动度)
    pup_corr = round(float(scores["v3c_pump_up"].corr(scores["debias_pump_up"])), 5)
    ratio_corr = round(float(scores["v3c_ratio"].corr(scores["debias_ratio"])), 5)
    pdown_corr = round(float(scores["v3c_pump_down"].corr(scores["debias_pump_down"])), 5)
    infer_summary = {
        "n_scored_rows": int(len(scores)),
        "P_up_corr_v3c_vs_debias": pup_corr,
        "P_down_corr_v3c_vs_debias": pdown_corr,
        "ratio_corr_v3c_vs_debias": ratio_corr,
        "note": "P_up/ratio 相关越接近 1 → 去偏对池内排序的撬动越小 (印证 PL-001: up 端零增量, "
                "杠杆仅在 down 约束的间接影响)。",
    }

    res = {
        "task": "PL-002",
        "protocol": f"walk-forward {TRAIN_LOOKBACK_MONTHS}m lookback, monthly retrain, "
                    f"3way LGBM, {len(fc)} prod pump features, apples-to-apples (label past_r5 only diff)",
        "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
        "arms": {"v3c": "down 带 past_r5<=8% (生产口径)",
                 "debias": "down 去 past_r5 约束 (纳入横盘/微涨伪跌启动)"},
        "train_summary_pos_rate": summary,
        "infer_summary": infer_summary,
        "monthly": df.to_dict("records"),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "guardrails": ["SIGN-R03 中间指标(正例率/相关)仅训练摘要, ship 由 PL-004 walk-forward 定",
                       "SIGN-R04 label/分数留 cache 不进 features", "SIGN-R05 生产模型只读",
                       "SIGN-R06 ST 源头排除", "SIGN-R08 月度模型+分片 checkpoint"],
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n\n=== PL-002 训练摘要 (正例率 vs v3c) ===\n", flush=True)
    print(f"  月数: {len(df)}  ({TEST_MONTHS[0]}-{TEST_MONTHS[-1]})", flush=True)
    print(f"  {'指标':24s} {'v3c':>10s} {'debias':>10s} {'Δ(deb-v3c)':>12s}", flush=True)
    print(f"  {'up 正例率(均)':24s} {summary['v3c_up_rate_mean']:>10.4f} "
          f"{summary['debias_up_rate_mean']:>10.4f} {summary['up_rate_delta_mean']:>+12.4f}", flush=True)
    print(f"  {'down 正例率(均)':24s} {summary['v3c_down_rate_mean']:>10.4f} "
          f"{summary['debias_down_rate_mean']:>10.4f} {summary['down_rate_delta_mean']:>+12.4f}", flush=True)
    print(f"\n  推理: P_up 相关={pup_corr} | P_down 相关={pdown_corr} | ratio 相关={ratio_corr}", flush=True)
    print("\n  月度 down 正例率 Δ (debias - v3c):", flush=True)
    for _, r in df.iterrows():
        print(f"    {r['test_month']}  v3c_down {r['v3c_down_rate']:.4f}  "
              f"debias_down {r['debias_down_rate']:.4f}  "
              f"(Δ {r['down_rate_delta_debias_minus_v3c']:+.4f})", flush=True)

    # ── verdict ──
    conclusion = (
        f"已 walk-forward 月度重训两臂去偏 pump 启动子 ({len(df)} 月 {TEST_MONTHS[0]}-{TEST_MONTHS[-1]}, "
        f"24 月 lookback, 3 分类 LGBM, {len(fc)} 生产特征, apples-to-apples 唯一差异=label 的 past_r5 约束)。"
        f"训练摘要印证 PL-001 诊断: **up 正例率两臂几乎相同** (v3c {summary['v3c_up_rate_mean']:.4f} vs "
        f"debias {summary['debias_up_rate_mean']:.4f}, Δ={summary['up_rate_delta_mean']:+.4f}), 去偏的全部"
        f"杠杆在 **down 类** (down 正例率 v3c {summary['v3c_down_rate_mean']:.4f} → debias "
        f"{summary['debias_down_rate_mean']:.4f}, Δ={summary['down_rate_delta_mean']:+.4f}/月, 去 past_r5<=8% "
        f"约束把高涨幅伪跌启动收回 down 正样本)。推理端 P_up 相关 {pup_corr} / ratio 相关 {ratio_corr} → "
        f"去偏对池内排序的撬动主要经 down 约束放松对 3way softmax 的间接影响 "
        f"({'撬动有限' if ratio_corr > 0.9 else '撬动可观'}, "
        f"印证 PL-001 '修盲点能力存疑' 预警)。pump 分数已逐月落 cache 供 PL-003 捕捉验证 / PL-004 gate。"
        f"中间指标仅训练摘要, ship 由 PL-004 walk-forward α 定 (SIGN-R03)。")
    VERDICT.write_text(json.dumps({
        "id": "PL-002", "status": "built", "conclusion": conclusion,
        "protocol": res["protocol"], "window": res["window"], "arms": res["arms"],
        "train_summary_pos_rate": summary,
        "infer_summary": infer_summary,
        "artifacts": ["research/cache/pl002_wf_models/",
                      "research/cache/pl002_pump_scores/",
                      "research/cache/pl002_pump_scores.parquet",
                      "research/cache/pl002_monthly_summary.csv",
                      "research/cache/pl002_results.json"],
        "note": "apples-to-apples 两臂仅 label 的 past_r5 约束不同 (up 逐位相同); pump 分数留 cache "
                "非 features (SIGN-R04); 生产模型只读 (SIGN-R05); 正例率/相关仅训练摘要, ship 由 "
                "PL-004 walk-forward α 定 (SIGN-R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] PL-002 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
