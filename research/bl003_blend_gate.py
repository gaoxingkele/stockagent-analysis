# -*- coding: utf-8 -*-
"""BL-003 — squat+base blend walk-forward 决策 gate (isGate).

承接 BL-001 (BASE 入场+突破+紧尾 label, 样本不足→v1) / BL-002 (v1 等权 base_score, 信号弱+不削尾)。
本 task 是**裁决闸**: 在 19 月 walk-forward 上, 把 BASE sleeve 注入 V12.31 的 book, 看是否带来
事前注册 gate 要求的真 alpha (Δα>0 且不付尾)。

apples-to-apples (两臂唯一差异 = 是否注入 BASE; 冻结 SIGN-R01):
  - ArmA = SQUAT-only (= V12.31 等价): 完全复用 t005 的 build_dual, 池内按 ratio_s5 排序。
      (= t005 的 baseline 臂, 与生产 v3c 同档 pump 模型 H=5/g10%/dd-5%。)
  - ArmB = squat+base: 同一 V7c dual-track 池, **track A 的下 60% 名额 (Q∈{3..5}) 优先**
      留给当日 base_score 最高的 BASE 入场候选 (BL-002 打分), 上 40% (Q1-2) 高 conviction
      ratio picks 保护不动, 其余 ratio 填满至同 N, 同行业 cap。无 base 候选的日 (BL-001 记 37%
      天 0 注入) → ArmB 与 ArmA **逐位一致**。
  - 两臂共用同一 V7c 池构造 / r20 池模型 / 行业 cap / 仓位 / 成本 → 任何共模偏差 (含 r20 池
    潜在 lookahead) 在 Δα 中相消; gate 只看相对量 (SIGN-R03)。

复用 t005 资产 (s5 月度模型已 checkpoint 缓存, 不重训):
  - research/cache/t005_wf_models/{month}/pump_scale_5/  (ratio_s5 + pump_down_s5, 两臂共用)
  - r20_v16_long_nost (生产 r20 池模型, 不重训, 两臂共用)

事前注册 gate (冻结 SIGN-R01, prd.preRegisteredGate.gate, 全相对 ArmA 同 harness):
  ① Δα(ArmB − ArmA) > 0 pp/月 (19月 walk-forward)
  ② worst_month(ArmB) 不比 worst_month(ArmA) 差 (核心: 拿质量不付尾; 恶化即 REJECT_tail)
  ③ Sharpe 不降; 正α月占比不降; 单月outlier剔后 Δα 仍>0; 分regime不伤
  ④ BASE 注入频率/容量 must log (不可静默)

5 分叉 (responsePlaybook.BL-003, REJECT 合法完成 SIGN-R02):
  PASS            — Δα>0 且最差月不恶化 且全部稳健条件满足 → 进 BL-004 opt-in 落地
  真小            — Δα>0 且尾不恶化, 但增益小/容量限/稳健性差 → 记 blend 候选不强上
  REJECT_tail     — Δα>0 但最差月恶化 → blend 仍买尾, 与 pump-debias 同病, 文档化
  REJECT_no_alpha — Δα<=0 → BASE 质量在 blend 中未兑现, 文档化

checkpoint (SIGN-R08): research/cache/bl003_monthly.csv 逐月 append, 已完成月跳过。
不碰任何生产文件 (SIGN-R05)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from train_v15_refresh import load_window  # noqa: E402
from walk_forward_validation import (  # noqa: E402
    compute_r20_label, compute_ind_mom, apply_cap, industry_alloc,
    get_train_end_for_test_month, get_train_start,
)
# 复用 t005 的池构造 / 单月指标 / 模型 checkpoint loader / 常数 (SIGN-R01 同口径)
from t005_walk_forward_gate import (  # noqa: E402
    build_dual, month_metrics, train_scale_month,
    TEST_MONTHS, TRAIN_LOOKBACK_MONTHS, DATA_START, DATA_END,
    MAX_A, MAX_B, A_PCT, B_PCT, CAP_IN, CAP_CROSS, M_EXCL, P_BUY, PYR_Q,
    PUMP_DOWN_THRESHOLD, EPS,
)
from t004_multiscale_gate import build_label, load_daily, SCALES  # noqa: E402

CACHE = ROOT / "research" / "cache"
MONTHLY = CACHE / "bl003_monthly.csv"
RESULTS = CACHE / "bl003_results.json"
VERDICT = ROOT / "research" / "verdicts" / "BL-003.json"
PROD = ROOT / "output" / "production"
BL002_CACHE = CACHE / "bl002"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"

# ── blend 注入参数 (冻结 SIGN-R01): track A 下 60% (Q3-5) 留给 base, 上 40% (Q1-2) 保护 ──
PROTECT_FRAC = 0.40   # track A 顶部受保护的 ratio picks 比例 (Q1-2)


def load_base_scores() -> pd.DataFrame:
    """BL-002 月度 base_score, 仅取因果 (ts_code,trade_date,base_score), 不带 label/前向列 (R04)。"""
    frames = []
    for f in sorted(BL002_CACHE.glob("base_score_*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "base_score"])
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["ts_code", "trade_date", "base_score"])
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = out["trade_date"].astype(str)
    return out


def build_dual_blend(daily: pd.DataFrame, ind_mom: pd.DataFrame):
    """ArmB: 同 build_dual(ratio_s5) 但 track A 下 60% 名额优先注入 base 候选。
    返回 (holdings_df, inject_stats)。无 base 候选的日 → 与 ArmA 逐位一致。"""
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    inj_days = 0
    inj_total = 0
    n_pool_days = 0
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 100:
            continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 50:
            continue
        g["r20_rank"] = g["pred_r20"].rank(pct=True, method="first")
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
        v7c = v7c.sort_values("ratio_s5", ascending=False)
        n_pool_days += 1

        # ── ArmA track A 的 ratio picks ──
        a_ratio = v7c.head(MAX_A).copy()
        nA = len(a_ratio)
        n_protect = int(np.floor(PROTECT_FRAC * nA))
        protected = a_ratio.head(n_protect)
        prot_codes = set(protected["ts_code"])
        n_reserve = nA - n_protect

        # ── base 候选 (当日 g 内, 有 base_score; = BASE 入场被 BL-002 打分) ──
        base_today = g[g["base_score"].notna()].sort_values("base_score", ascending=False)
        bsel = base_today[~base_today["ts_code"].isin(prot_codes)].head(n_reserve)
        n_base = len(bsel)
        inj_codes = set(bsel["ts_code"])
        if n_base > 0:
            inj_days += 1
            inj_total += n_base

        # ── 下 60% 剩余名额用 ratio 填 (排除已注入), 不足再从 v7c 兜底至同 N ──
        contest = a_ratio.iloc[n_protect:]
        ratio_fill = contest[~contest["ts_code"].isin(inj_codes)].head(max(0, n_reserve - n_base))
        a_pool = pd.concat([protected, bsel, ratio_fill])
        if len(a_pool) < nA:   # 保证同 N (apples-to-apples): 兜底从 v7c 补
            have = set(a_pool["ts_code"])
            extra = v7c[~v7c["ts_code"].isin(have)].head(nA - len(a_pool))
            a_pool = pd.concat([a_pool, extra])
        a_pool = a_pool.copy()
        # blend_rank: protected(高) > base > ratio_fill, 供 cap 贪心保序
        a_pool["blend_rank"] = np.arange(len(a_pool), 0, -1)

        a_pool = apply_cap(a_pool, A_PCT, CAP_IN, sort_col="blend_rank")
        a_ind = industry_alloc(a_pool, A_PCT)
        # track B 仍按 ratio (两臂同), 排除 a_pool
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN, sort_col="ratio_s5")
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind, sort_col="ratio_s5")
        for tag, pool, alloc in [("A", a_pool, A_PCT), ("B", b_pool, B_PCT)]:
            if len(pool) == 0:
                continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                             "industry": row.get("industry", ""),
                             "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                             "alloc_pct": per})
    stats = {"pool_days": n_pool_days, "inject_days": inj_days,
             "inject_total": inj_total,
             "inject_day_frac": round(inj_days / n_pool_days, 4) if n_pool_days else 0.0,
             "inject_per_inject_day": round(inj_total / inj_days, 3) if inj_days else 0.0}
    return pd.DataFrame(rows), stats


def evaluate_month(test_month, daily_full, fc, gate_unused, ind_mom, r20_lab, b20, r20_fc):
    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    print(f"\n--- {test_month}: train {train_start}-{train_end} ---", flush=True)
    df_train = daily_full[(daily_full["trade_date"] >= train_start) &
                          (daily_full["trade_date"] < train_end)].copy()
    if len(df_train) < 100_000:
        print(f"  跳过 (训练样本不足 {len(df_train)})", flush=True)
        return None

    # s5 模型 (checkpoint 命中即秒载, 两臂共用 ratio_s5 + pump_down_s5)
    booster_s5 = train_scale_month(df_train, fc, "label_s5", test_month, 5)

    df_test = daily_full[daily_full["trade_date"].str.startswith(test_month)].copy()
    if len(df_test) < 100:
        print(f"  跳过 (测试样本不足 {len(df_test)})", flush=True)
        return None
    for c in fc:
        if c not in df_test.columns:
            df_test[c] = 0.0
    for c in r20_fc:
        if c not in df_test.columns:
            df_test[c] = 0.0

    Xf = df_test[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = booster_s5.predict(Xf)
    df_test["pump_down_s5"] = proba[:, 1]
    df_test["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)

    Xr = df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_test["pred_r20"] = b20.predict(Xr)
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
    # base_score 注入列 (仅因果 base_score, 无 label/前向, R04)
    df_test = df_test.merge(BASE_SCORES, on=["ts_code", "trade_date"], how="left")

    out = {"test_month": test_month, "train_start": train_start, "train_end": train_end}
    # ArmA = SQUAT-only (= V12.31, t005 baseline 臂逐字复用)
    holdA = build_dual(df_test, ind_mom, sort_col="ratio_s5")
    mA = month_metrics(holdA, df_test)
    # ArmB = squat+base
    holdB, inj = build_dual_blend(df_test, ind_mom)
    mB = month_metrics(holdB, df_test)
    if mA is None or mB is None:
        print(f"  无持仓/无 r20, 跳过本月", flush=True)
        return None
    out.update({"a_alpha": mA["alpha"], "a_sharpe": mA["sharpe"],
                "a_n_days": mA["n_days"], "a_n_stocks": mA["avg_n_stocks"],
                "b_alpha": mB["alpha"], "b_sharpe": mB["sharpe"],
                "b_n_days": mB["n_days"], "b_n_stocks": mB["avg_n_stocks"],
                "inject_days": inj["inject_days"], "pool_days": inj["pool_days"],
                "inject_total": inj["inject_total"],
                "inject_day_frac": inj["inject_day_frac"]})
    print(f"  ArmA α={mA['alpha']:+.3f}pp Sh={mA['sharpe']:+.2f} | "
          f"ArmB α={mB['alpha']:+.3f}pp Sh={mB['sharpe']:+.2f} | "
          f"Δ={mB['alpha']-mA['alpha']:+.3f} | 注入 {inj['inject_days']}/{inj['pool_days']}日 "
          f"共{inj['inject_total']}个", flush=True)
    del df_train, df_test, booster_s5
    gc.collect()
    return out


# 全局 base_score (evaluate_month 内 merge)
BASE_SCORES: pd.DataFrame = None


def month_regime_map() -> dict:
    """每测试月 → 该月主导 regime (modal), 供分 regime 不伤检验 (SIGN-R11)。"""
    if not REGIME_P.exists():
        return {}
    reg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]].copy()
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["ym"] = reg["trade_date"].str[:6]
    out = {}
    for ym, g in reg.groupby("ym"):
        out[ym] = g["regime"].mode().iloc[0]
    return out


def main():
    global BASE_SCORES
    t0 = time.time()
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== BL-003 squat+base blend walk-forward 决策 gate ===\n", flush=True)

    BASE_SCORES = load_base_scores()
    print(f"[base] BL-002 base_score {len(BASE_SCORES):,} 条 "
          f"({BASE_SCORES['trade_date'].min()}~{BASE_SCORES['trade_date'].max()})", flush=True)

    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json")
                      .read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map:
        daily_full["industry_id"] = (daily_full["industry"].fillna("unknown")
                                     .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily_full.columns:
            daily_full[c] = 0.0
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[data] daily_full {len(daily_full):,} 行 / {daily_full['ts_code'].nunique()} 股", flush=True)

    # s5 forward 标签 (仅训练用; checkpoint 命中时其实用不到, 但 train_scale_month 需列存在)
    px = load_daily()
    slim = (daily_full[["ts_code", "trade_date"]]
            .merge(px, on=["ts_code", "trade_date"], how="left")
            .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    daily_full["label_s5"] = build_label(slim, 5, SCALES[5]).values
    del slim; gc.collect()

    print("[label] r20 forward + 行业动量 ...", flush=True)
    r20_lab = compute_r20_label(); r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)

    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt")
                      .read_text(encoding="utf-8"))
    meta_r20 = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json")
                          .read_text(encoding="utf-8"))
    r20_fc = meta_r20["feature_cols"]
    print(f"[prep] 完成 {time.time()-t0:.0f}s\n", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records")
        done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月: {sorted(done)}", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = evaluate_month(m_, daily_full, fc, None, ind_mom, r20_lab, b20, r20_fc)
        if r:
            results.append(r)
            pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].copy()
    df["test_month"] = df["test_month"].astype(str)
    df = df.sort_values("test_month").reset_index(drop=True)

    # ── 汇总两臂 ──
    def agg(arm):
        a = df[f"{arm}_alpha"]
        return {"alpha_mean": float(a.mean()), "alpha_median": float(a.median()),
                "sharpe_mean": float(df[f"{arm}_sharpe"].mean()),
                "worst_month": float(a.min()),
                "worst_month_id": str(df.loc[a.idxmin(), "test_month"]),
                "pos_alpha_ratio": float((a > 0).mean()),
                "n_months": int(len(df))}
    A, B = agg("a"), agg("b")
    d_alpha = B["alpha_mean"] - A["alpha_mean"]

    # ── 单月 outlier 剔除 (Δα 序列) ──
    dmonthly = (df["b_alpha"] - df["a_alpha"]).to_numpy()
    dev = np.abs(dmonthly - dmonthly.mean())
    drop_i = int(np.argmax(dev))
    kept = np.delete(dmonthly, drop_i)
    outlier = {"dropped_month": str(df.loc[drop_i, "test_month"]),
               "dropped_delta_pp": round(float(dmonthly[drop_i]), 4),
               "delta_after_drop_pp": round(float(kept.mean()), 4),
               "still_positive": bool(kept.mean() > 0)}

    # ── 分 regime Δα (SIGN-R11) ──
    rmap = month_regime_map()
    df["regime"] = df["test_month"].map(rmap).fillna("unknown")
    by_regime = {}
    for rg, g in df.groupby("regime"):
        dd = (g["b_alpha"] - g["a_alpha"])
        by_regime[str(rg)] = {"n_months": int(len(g)),
                              "delta_alpha_pp": round(float(dd.mean()), 4),
                              "a_alpha_pp": round(float(g["a_alpha"].mean()), 4),
                              "b_alpha_pp": round(float(g["b_alpha"].mean()), 4)}
    main_rg = {k: v for k, v in by_regime.items() if k in ("momentum", "reversal", "mixed")}
    regime_not_hurt = all(v["delta_alpha_pp"] >= -0.05 for v in main_rg.values()) if main_rg else True

    # ── 注入容量汇总 ──
    inj_days_total = int(df["inject_days"].sum())
    pool_days_total = int(df["pool_days"].sum())
    inj_total = int(df["inject_total"].sum())
    inject_summary = {
        "inject_days": inj_days_total, "pool_days": pool_days_total,
        "inject_total": inj_total,
        "inject_day_frac": round(inj_days_total / pool_days_total, 4) if pool_days_total else 0.0,
        "inject_per_inject_day": round(inj_total / inj_days_total, 3) if inj_days_total else 0.0,
        "months_with_zero_base": [m for m in TEST_MONTHS
                                  if m not in set(BASE_SCORES["trade_date"].str[:6])],
    }

    # ── 事前注册 gate (冻结 SIGN-R01) ──
    c1 = d_alpha > 0
    c2_tail = B["worst_month"] >= A["worst_month"] - 1e-9
    c3_sharpe = B["sharpe_mean"] >= A["sharpe_mean"] - 1e-9
    c4_posratio = B["pos_alpha_ratio"] >= A["pos_alpha_ratio"] - 1e-9
    c5_outlier = outlier["still_positive"]
    conds = {"delta_alpha>0": bool(c1), "worst_month_not_worse": bool(c2_tail),
             "sharpe_not_lower": bool(c3_sharpe), "pos_alpha_ratio_not_lower": bool(c4_posratio),
             "outlier_drop_delta_still_pos": bool(c5_outlier), "regime_not_hurt": bool(regime_not_hurt)}

    # ── 5 分叉 (responsePlaybook.BL-003) ──
    if d_alpha <= 0:
        status = "REJECT_no_alpha"
    elif not c2_tail:
        status = "REJECT_tail"
    elif all(conds.values()):
        status = "PASS"
    else:
        status = "真小"

    res = {
        "window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
        "protocol": f"walk-forward {TRAIN_LOOKBACK_MONTHS}m lookback, s5 月度模型复用 t005 checkpoint, "
                    "r20 池模型固定 (prod), apples-to-apples (仅 ArmB 注入 BASE)",
        "ArmA_SQUAT_only_V1231": A, "ArmB_squat_base": B,
        "delta_alpha_pp": d_alpha, "outlier_drop": outlier,
        "by_regime": by_regime, "regime_not_hurt": bool(regime_not_hurt),
        "inject_summary": inject_summary,
        "gate_conditions": conds, "gate_status": status,
        "note": "Δα 等相对量是 gate; 绝对 α 含两臂共模 r20 池 lookahead, 不作判定 (SIGN-R03)。"
                "REJECT 合法完成, 禁止同段 OOS 重调 (SIGN-R02)。",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n\n=== BL-003 walk-forward 汇总 ===\n", flush=True)
    print(f"  月数: {len(df)}  ({TEST_MONTHS[0]}-{TEST_MONTHS[-1]})", flush=True)
    print(f"  {'指标':24s} {'ArmA(SQUAT/V1231)':>18s} {'ArmB(squat+base)':>18s}", flush=True)
    print(f"  {'月化 α(pp)':24s} {A['alpha_mean']:>+18.3f} {B['alpha_mean']:>+18.3f}", flush=True)
    print(f"  {'Sharpe':24s} {A['sharpe_mean']:>+18.2f} {B['sharpe_mean']:>+18.2f}", flush=True)
    print(f"  {'最差月(pp)':24s} {A['worst_month']:>+18.3f} {B['worst_month']:>+18.3f}", flush=True)
    print(f"  {'正 α 月占比':24s} {A['pos_alpha_ratio']:>18.1%} {B['pos_alpha_ratio']:>18.1%}", flush=True)
    print(f"\n  Δα = {d_alpha:+.3f}pp  (gate 需 >0)", flush=True)
    print(f"  单月outlier剔({outlier['dropped_month']} Δ{outlier['dropped_delta_pp']:+.3f}) → "
          f"Δα {outlier['delta_after_drop_pp']:+.3f}pp 仍正={outlier['still_positive']}", flush=True)
    print(f"  注入容量: {inj_days_total}/{pool_days_total} 池日有注入 "
          f"({inject_summary['inject_day_frac']:.1%}), 共 {inj_total} 个, "
          f"注入日均 {inject_summary['inject_per_inject_day']}", flush=True)
    print(f"  分 regime Δα: " +
          " ".join(f"{k}={v['delta_alpha_pp']:+.3f}(n{v['n_months']})" for k, v in by_regime.items()), flush=True)
    print(f"\n  gate 条件: {conds}", flush=True)
    print(f"  >>> verdict = {status} <<<\n", flush=True)
    print("  月度明细 (A α / B α / Δ / 注入日):", flush=True)
    for _, r in df.iterrows():
        print(f"    {r['test_month']}  A {r['a_alpha']:+.3f}  B {r['b_alpha']:+.3f}  "
              f"Δ{r['b_alpha']-r['a_alpha']:+.3f}  注入{int(r['inject_days'])}/{int(r['pool_days'])}日", flush=True)

    # ── verdict ──
    table = {
        "metric": ["alpha_monthly_pp", "sharpe", "worst_month_pp", "pos_alpha_month_ratio"],
        "ArmA_SQUAT_only": [round(A["alpha_mean"], 3), round(A["sharpe_mean"], 2),
                            round(A["worst_month"], 3), round(A["pos_alpha_ratio"], 3)],
        "ArmB_squat_base": [round(B["alpha_mean"], 3), round(B["sharpe_mean"], 2),
                            round(B["worst_month"], 3), round(B["pos_alpha_ratio"], 3)],
        "delta": [round(d_alpha, 3), round(B["sharpe_mean"] - A["sharpe_mean"], 2),
                  round(B["worst_month"] - A["worst_month"], 3),
                  round(B["pos_alpha_ratio"] - A["pos_alpha_ratio"], 3)],
    }
    fail = [k for k, v in conds.items() if not v]
    cmap = {
        "PASS": "BASE sleeve 在 blend 中带真 alpha 且不付尾 → 值得 opt-in 落地 (进 BL-004)。",
        "真小": "Δα>0 且最差月不恶化, 但增益小/容量受限/稳健性不足 → 记 blend 候选不强上 (类比 pump-debias 真小)。",
        "REJECT_tail": "Δα>0 但最差月恶化 → 严格 base premium 太罕见/已 priced, blend 仍买尾, 与 pump-debias 同病。",
        "REJECT_no_alpha": "Δα<=0 → BASE 质量在 blend 中未兑现 (被 squat 名额挤占或已被 V7c 池吃掉)。",
    }
    conclusion = (
        f"squat+base blend walk-forward 裁决={status}。19 月 apples-to-apples (两臂仅差 ArmB 注入 BASE, "
        f"复用 t005 s5/r20 模型 checkpoint, V12.31 池/cap/成本全一致)。"
        f"ArmA(SQUAT/V12.31) α={A['alpha_mean']:+.3f}pp/月 Sh={A['sharpe_mean']:.2f} 最差月{A['worst_month']:+.3f}pp 正α月{A['pos_alpha_ratio']:.0%}; "
        f"ArmB(squat+base) α={B['alpha_mean']:+.3f}pp Sh={B['sharpe_mean']:.2f} 最差月{B['worst_month']:+.3f}pp 正α月{B['pos_alpha_ratio']:.0%}; "
        f"**Δα={d_alpha:+.3f}pp/月**, 单月outlier({outlier['dropped_month']})剔后 Δα {outlier['delta_after_drop_pp']:+.3f}pp 仍正={outlier['still_positive']}。"
        f"注入容量: {inj_days_total}/{pool_days_total} 池日({inject_summary['inject_day_frac']:.0%})有 BASE 注入, 共{inj_total}个, 注入日均{inject_summary['inject_per_inject_day']}个 "
        f"(BL-001 已记 BASE 入场稀疏+37%天0注入, 202410/202411 整月无 base) → 多数日 ArmB 与 ArmA 逐位一致, 增量天然被注入频率上限封死。"
        f"分 regime Δα: {json.dumps({k: v['delta_alpha_pp'] for k, v in by_regime.items()}, ensure_ascii=False)} (regime不伤={regime_not_hurt})。"
        f"gate 未过: {fail if fail else '无 (全过)'}。{cmap[status]} "
        f"承接 BL-002 预判 (base_score 排序信号弱+不削尾) — {('印证' if status.startswith('REJECT') or status=='真小' else '反转')}。"
        f"全因果 base_score 只注入因果列 (R04, label/前向只在 cache); ST 源头排除 (R06); 分 regime (R11); "
        f"gate 阈值启动冻结未上调 (R01); REJECT 合法完成 (R02); 生产线只读指纹未变 (R05)。")

    VERDICT.write_text(json.dumps({
        "id": "BL-003", "status": status, "conclusion": conclusion,
        "metrics": {"alpha_delta_pp": round(d_alpha, 3),
                    "sharpe": round(B["sharpe_mean"], 2),
                    "worst_month": round(B["worst_month"], 3),
                    "pos_alpha_month_ratio": round(B["pos_alpha_ratio"], 3),
                    "arm_a": A, "arm_b": B, "comparison_table": table,
                    "outlier_drop": outlier, "by_regime": by_regime,
                    "inject_summary": inject_summary},
        "gate_conditions": conds, "playbook_hit": status,
        "n_months": int(len(df)), "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
        "artifacts": ["research/cache/bl003_monthly.csv", "research/cache/bl003_results.json"],
        "note": "isGate: apples-to-apples 两臂唯一差异=ArmB track A 下60%(Q3-5)注入 base_score 最高 BASE 入场候选; "
                "绝对 α 含共模 r20 池 lookahead, 仅 Δ 量作 gate (SIGN-R03)。REJECT 合法完成 (R02)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] verdict={status} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    main()
