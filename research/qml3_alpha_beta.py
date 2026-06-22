# -*- coding: utf-8 -*-
"""QML-3 — Alpha/Beta 解耦 (ABCM/FactorVAE 精神) book walk-forward Δα.

QuantML 文章46(ABCM)/44(FactorVAE)核心: 端到端分离 alpha(特质)与 beta(市场), 选股用 beta-中性
alpha。轻量版测: V7c 池选股时把 pred_r20 对 beta_60 横截面残差化 → beta-中性 r20 选池, vs 原始
pred_r20 选池, 19月 book walk-forward Δα + skeptic。接上发现: 我们 book beta主导/alpha bull-loaded
([[project_market_state_exposure_reject_0618]]) → 去 beta 选股理论上提纯 alpha, 但先验低(QML-1/2
已证模型轴非杠杆, 且我们 edge 与 beta 耦合)。

apples-to-apples: 复用 oa_gate/t005 harness, 两臂唯一差异=池选择键(raw pred_r20 vs beta-中性 pred_r20)。
beta_60 从共享面板算(cov/var)。Δα 相消共模 lookahead(R03)。gate: c1 Δα>=+0.30pp + c2-c6 + c7 skeptic。
生产冻结(R05); ST源头排除; checkpoint(R08)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
from train_v15_refresh import load_window
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom, apply_cap, industry_alloc,
    get_train_end_for_test_month, get_train_start,
)
from t004_multiscale_gate import build_label, load_daily, SCALES, EPS
from t005_walk_forward_gate import train_scale_month, TEST_MONTHS, TRAIN_LOOKBACK_MONTHS
import oa_vol_factors as ov
import research_env as renv

CACHE = ROOT / "research" / "cache"
MONTHLY = CACHE / "qml3_monthly.csv"
VERDICT = ROOT / "research" / "verdicts" / "QML-3.json"
PROD = ROOT / "output" / "production"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
DATA_START, DATA_END = "20220801", "20260601"
SCALE_LIST = [3, 5, 10]
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15
PUMP_DOWN_THRESHOLD = 0.60
GATE_DELTA = 0.30


def compute_beta_daily(panel):
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    mkt = panel.groupby("trade_date")["ret"].mean().rename("mkt")
    panel = panel.merge(mkt, on="trade_date", how="left")
    panel["_rm"] = panel["ret"] * panel["mkt"]
    g = panel.groupby("ts_code", sort=False)
    rb = lambda c: g[c].transform(lambda s: s.rolling(60, min_periods=40).mean())
    m_r, m_m, m_rm = rb("ret"), rb("mkt"), rb("_rm")
    v_m = g["mkt"].transform(lambda s: s.rolling(60, min_periods=40).var())
    panel["beta_60"] = (m_rm - m_r * m_m) / v_m.replace(0, np.nan)
    return panel[["ts_code", "trade_date", "beta_60"]].replace([np.inf, -np.inf], np.nan)


def _resid_vs_beta(g):
    """单日横截面: pred_r20 对 beta_60 OLS 残差 (去 beta 成分)。"""
    y = g["pred_r20"].to_numpy(float)
    b = g["beta_60"].to_numpy(float)
    ok = np.isfinite(b)
    if ok.sum() < 30:
        return pd.Series(y, index=g.index)
    bb = b.copy()
    bb[~ok] = np.nanmean(b[ok])
    X = np.column_stack([np.ones(len(bb)), (bb - bb.mean()) / (bb.std() + 1e-9)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(y - X @ beta, index=g.index)


def build_dual(daily, ind_mom, beta_neutral: bool):
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 100:
            continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 50:
            continue
        # 池选择键: raw pred_r20 vs beta-中性 pred_r20
        sel = _resid_vs_beta(g) if beta_neutral else g["pred_r20"]
        g["_sel"] = sel
        g["r20_rank"] = g["_sel"].rank(pct=True, method="first")
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
        a_pool = v7c.head(MAX_A).copy()
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN, sort_col="ratio_s5")
        a_ind = industry_alloc(a_pool, A_PCT)
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
                             "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                             "alloc_pct": per})
    return pd.DataFrame(rows)


def month_metrics(hold, df_test):
    if hold is None or hold.empty:
        return None
    h = hold.dropna(subset=["r20_fresh"]).copy()
    if h.empty:
        return None
    h["r20_fresh"] = h["r20_fresh"].clip(-30, 30)
    h["w"] = h["alloc_pct"] * h["r20_fresh"]
    total = A_PCT + B_PCT
    pnl = h.groupby("entry_date").agg(gross=("w", "sum")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = df_test.groupby("trade_date")["r20_fresh"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]
    pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt"] * total
    a = float(pnl["alpha"].mean())
    return {"alpha": a, "sharpe": float(a / (pnl["alpha"].std() + 1e-9) * np.sqrt(12))}


def evaluate_month(tm, daily_full, fc, beta_daily, ind_mom, r20_lab, b20, r20_fc):
    te = get_train_end_for_test_month(tm); ts = get_train_start(te, TRAIN_LOOKBACK_MONTHS)
    df_train = daily_full[(daily_full["trade_date"] >= ts) & (daily_full["trade_date"] < te)].copy()
    if len(df_train) < 100_000:
        return None
    boosters = {H: train_scale_month(df_train, fc, f"label_s{H}", tm, H) for H in SCALE_LIST}
    df_test = daily_full[daily_full["trade_date"].str.startswith(tm)].copy()
    if len(df_test) < 100:
        return None
    for c in fc:
        if c not in df_test.columns: df_test[c] = 0.0
    for c in r20_fc:
        if c not in df_test.columns: df_test[c] = 0.0
    Xf = df_test[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    for H in SCALE_LIST:
        proba = boosters[H].predict(Xf)
        if H == 5:
            df_test["pump_down_s5"] = proba[:, 1]; df_test["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
    df_test["pred_r20"] = b20.predict(df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0))
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
    df_test = df_test.merge(beta_daily, on=["ts_code", "trade_date"], how="left")
    out = {"test_month": tm}
    for arm, bn in [("base", False), ("beta_neutral", True)]:
        m = month_metrics(build_dual(df_test, ind_mom, bn), df_test)
        if m is None:
            return None
        out[f"{arm}_alpha"] = m["alpha"]; out[f"{arm}_sharpe"] = m["sharpe"]
    print(f"  {tm}: base α={out['base_alpha']:+.3f} | beta_neutral α={out['beta_neutral_alpha']:+.3f} "
          f"(Δ{out['beta_neutral_alpha']-out['base_alpha']:+.3f})", flush=True)
    del df_train, df_test, boosters; gc.collect()
    return out


def main():
    t0 = time.time()
    print("\n=== QML-3 Alpha/Beta 解耦 book walk-forward ===\n", flush=True)
    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]; ind_map = meta.get("industry_map", {})
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map:
        daily_full["industry_id"] = daily_full["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in fc:
        if c not in daily_full.columns: daily_full[c] = 0.0
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    px = load_daily()
    slim = (daily_full[["ts_code", "trade_date"]].merge(px, on=["ts_code", "trade_date"], how="left")
            .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    for H, g in SCALES.items():
        daily_full[f"label_s{H}"] = build_label(slim, H, g).values
    del slim; gc.collect()
    r20_lab = compute_r20_label(); r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)
    print("[beta] 日级 beta_60 ...", flush=True)
    beta_daily = compute_beta_daily(ov.build_panel())
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))
    r20_fc = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))["feature_cols"]
    print(f"[prep] {time.time()-t0:.0f}s\n", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records"); done = set(prev["test_month"].astype(str))
    for tm in TEST_MONTHS:
        if tm in done:
            continue
        r = evaluate_month(tm, daily_full, fc, beta_daily, ind_mom, r20_lab, b20, r20_fc)
        if r:
            results.append(r); pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].sort_values("test_month").reset_index(drop=True)
    d_alpha = float((df["beta_neutral_alpha"] - df["base_alpha"]).mean())
    base = {"alpha": float(df["base_alpha"].mean()), "sharpe": float(df["base_sharpe"].mean()),
            "worst": float(df["base_alpha"].min()), "posr": float((df["base_alpha"] > 0).mean())}
    cand = {"alpha": float(df["beta_neutral_alpha"].mean()), "sharpe": float(df["beta_neutral_sharpe"].mean()),
            "worst": float(df["beta_neutral_alpha"].min()), "posr": float((df["beta_neutral_alpha"] > 0).mean())}
    df["d"] = df["beta_neutral_alpha"] - df["base_alpha"]
    d_ex = float(df.sort_values("d").iloc[:-1]["d"].mean()) if len(df) > 1 else d_alpha
    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]].copy()
    rg["trade_date"] = rg["trade_date"].astype(str); rg["m"] = rg["trade_date"].str[:6]
    rmap = (rg[rg["m"].isin(TEST_MONTHS)].groupby("m")["regime"].agg(lambda x: x.value_counts().index[0])).to_dict()
    df["regime"] = df["test_month"].map(rmap).fillna("unknown")
    regime_delta = {str(r_): round(float(gg["d"].mean()), 3) for r_, gg in df.groupby("regime")}
    try:
        bb = renv.block_bootstrap_ci(df["d"].to_numpy(float), cohort_len=2, n_boot=2000, ppy=12)
        boot_pass = bool(bb["p_sharpe_gt_0"] >= 0.95 and bb["sharpe_ci95"][0] > 0)
    except Exception:
        bb = {}; boot_pass = False
    conds = {"c1_delta>=0.30": bool(d_alpha >= GATE_DELTA), "c2_sharpe>=base": bool(cand["sharpe"] >= base["sharpe"]),
             "c3_worst>=base": bool(cand["worst"] >= base["worst"]), "c4_posr>=base": bool(cand["posr"] >= base["posr"]),
             "c5_ex_outlier>0": bool(d_ex > 0), "c6_regime_not_hurt": all(v >= 0 for v in regime_delta.values()),
             "c7_bootstrap": boot_pass}
    status = "PASS" if all(conds.values()) else ("真小" if (0 < d_alpha < GATE_DELTA and conds["c2_sharpe>=base"]
             and conds["c6_regime_not_hurt"] and conds["c5_ex_outlier>0"]) else "REJECT")
    concl = (f"beta-中性选股 vs 原始 pred_r20 选股, {len(df)}月 walk-forward Δα={d_alpha:+.3f}pp/月 "
             f"(剔outlier{d_ex:+.3f}, regime{regime_delta}, bootstrap_pass={boot_pass}) → {status}. "
             + ("alpha/beta解耦在选股层有增量。" if status == "PASS"
                else "去beta选股未过+0.30pp门槛: 我们edge与beta耦合(alpha bull-loaded), 解耦不增量。"))
    VERDICT.write_text(json.dumps({
        "id": "QML-3", "status": status, "delta_alpha_pp": round(d_alpha, 4), "delta_ex_outlier": round(d_ex, 4),
        "base": base, "beta_neutral": cand, "regime_delta": regime_delta, "gate_conditions": conds,
        "bootstrap": bb, "n_months": int(len(df)), "conclusion": concl,
        "guardrails": ["R01 gate冻结", "R02 负/真小=合法完成", "R03 walk-forward α", "R05 生产冻结",
                       "R11 分regime", "R13 bootstrap负控"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== QML-3 汇总 ===", flush=True)
    print(f"  base α={base['alpha']:+.3f} Sharpe={base['sharpe']:+.2f} | beta_neutral α={cand['alpha']:+.3f} "
          f"Sharpe={cand['sharpe']:+.2f} | Δα={d_alpha:+.3f}pp", flush=True)
    print(f"  gate: {conds}", flush=True)
    print(f"  [决策] {status}; {concl}", flush=True)
    print(f"[done] {(time.time()-t0)/60:.1f} min -> {VERDICT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
