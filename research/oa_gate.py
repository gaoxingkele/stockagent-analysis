# -*- coding: utf-8 -*-
"""OA-*-GATE — 正交存活因子 walk-forward 决策 gate + skeptic_report.

承接 OA-VOL (ivol_60 orth IC=-0.0811) + OA-VP (on_minus_in_20 orth IC=+0.0483) 两族 Phase-1
存活。回答唯一 ship 问题 (SIGN-R03 IC≠落地, 只认 walk-forward α): 把候选正交因子接进生产
V12.31 池排序, 19月 walk-forward 下相对纯 V12.31 的 Δα 是否过事前注册 gate?

apples-to-apples (完全复用 gate001/t005 harness, 两臂唯一差异=池内排序键):
  - 共用 V7c dual-track 池 (r20 top + pyr_velocity 低 + 行业动量排除 + s5 pump_down 过滤),
    生产 r20 模型 + 复用 t005_wf_models 尺度模型 (不重训) → 与 t005/gate001 baseline 逐位可比。
  - base 臂 = V12.31: 池内按 ratio_s5 排序。
  - cand 臂 = V12.31 + 候选: 池内按 z_pool(ratio_s5) + sign·z_pool(候选) 等权混合 (无可调λ防p-hack)。
  共模 lookahead 在 Δα 相消; gate 只看相对量 (SIGN-R03)。

候选 (冻结 SIGN-R01, 从 oa_daily_panel 全窗口日级算, 每日有值无需carry):
  vol 臂: ivol_60,  sign=-1 (低波动异象, 低 ivol → 高预期收益)
  vp  臂: on_minus_in_20, sign=+1 (隔夜-日内分解, orth IC 正)

事前注册 gate (冻结 R01; 全相对 base; 分regime R11; 单月outlier; 多重检验 R15):
  PASS 当且仅当 (某候选臂):
    c1 Δα(月化)>=+0.30pp   c2 Sharpe>=base   c3 最差月>=base   c4 正α月占比>=base
    c5 剔单月outlier后 Δα>0   c6 各regime Δα不为负
    c7 skeptic_report: bootstrap P(ΔSharpe>0)>=0.95 (block, ppy=12); PBO<=0.5 (arm矩阵); DSR 报告(n_trials=manifest×3)
  0<Δα<+0.30 且不伤 → 真小; 否则 REJECT。负/真小=合法完成(R02), 禁止同段OOS重调。
  ⚠ SIGN-R16 caveat: Phase-1 在全54月(含walk-forward 19月)选因子=选择层泄漏; 任何PASS须独立OOS/前向复核。

checkpoint (R08): 复用 t005_wf_models; 逐月 append research/cache/oa_gate_monthly.csv 跳过已完成。
不碰生产文件 (R05); ST 源头排除 (load_window)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from train_v15_refresh import load_window
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom, apply_cap, industry_alloc,
    get_train_end_for_test_month, get_train_start,
)
from t004_multiscale_gate import build_label, load_daily, SCALES, EPS
from t005_walk_forward_gate import train_scale_month, TEST_MONTHS, TRAIN_LOOKBACK_MONTHS
import oa_vol_factors as ov
import oa_screen
import research_env as renv

CACHE = ROOT / "research" / "cache"
WF_MODELS = CACHE / "t005_wf_models"
MONTHLY = CACHE / "oa_gate_monthly.csv"
RESULTS = CACHE / "oa_gate_results.json"
VERDICT_VOL = ROOT / "research" / "verdicts" / "OA-VOL-GATE.json"
VERDICT_VP = ROOT / "research" / "verdicts" / "OA-VP-GATE.json"
PROD = ROOT / "output" / "production"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
MANIFEST_P = CACHE / "oa_manifest.json"

DATA_START, DATA_END = "20220801", "20260601"
SCALE_LIST = [3, 5, 10]
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15
PUMP_DOWN_THRESHOLD = 0.60
GATE_DELTA = 0.30

# 候选臂 (冻结)
CANDIDATES = [("vol", "ivol_60", -1.0), ("vp", "on_minus_in_20", 1.0)]


def zscore_pool(s: pd.Series) -> pd.Series:
    v = s.astype(float)
    present = v.dropna()
    if len(present) == 0:
        return pd.Series(0.0, index=s.index)
    mu, sd = present.mean(), present.std()
    v = v.fillna(mu)
    if sd is None or not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (v - mu) / sd


def compute_candidates_daily(panel: pd.DataFrame) -> pd.DataFrame:
    """从共享面板全窗口日级算 ivol_60 + on_minus_in_20 (每日有值)。复用 oa_vol/oa_vp 向量逻辑。"""
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    mkt = panel.groupby("trade_date")["ret"].mean().rename("mkt")
    panel = panel.merge(mkt, on="trade_date", how="left")
    panel["_rm"] = panel["ret"] * panel["mkt"]
    g = panel.groupby("ts_code", sort=False)
    rb = lambda c: g[c].transform(lambda s: s.rolling(60, min_periods=40).mean())
    m_r, m_m, m_rm = rb("ret"), rb("mkt"), rb("_rm")
    v_m = g["mkt"].transform(lambda s: s.rolling(60, min_periods=40).var())
    v_r = g["ret"].transform(lambda s: s.rolling(60, min_periods=40).var())
    cov = m_rm - m_r * m_m
    panel["ivol_60"] = np.sqrt((v_r - cov * cov / v_m.replace(0, np.nan)).clip(lower=0))
    on20 = g["overnight"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    in20 = g["intraday"].transform(lambda s: s.rolling(20, min_periods=10).sum())
    panel["on_minus_in_20"] = on20 - in20
    out = panel[["ts_code", "trade_date", "ivol_60", "on_minus_in_20"]].copy()
    return out.replace([np.inf, -np.inf], np.nan)


def build_dual(daily: pd.DataFrame, ind_mom: pd.DataFrame, sort_col: str,
               cand_col: str | None = None, sign: float = 1.0) -> pd.DataFrame:
    """V7c dual-track。cand_col=None → 纯 sort_col (base); 否则 z(sort_col)+sign·z(cand_col)。"""
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
        if cand_col is None:
            use_col = sort_col
        else:
            v7c["_blend"] = (zscore_pool(v7c[sort_col]).to_numpy()
                             + sign * zscore_pool(v7c[cand_col]).to_numpy())
            use_col = "_blend"
        v7c = v7c.sort_values(use_col, ascending=False)
        a_pool = v7c.head(MAX_A).copy()
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN, sort_col=use_col)
        a_ind = industry_alloc(a_pool, A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN, sort_col=use_col)
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind, sort_col=use_col)
        for tag, pool, alloc in [("A", a_pool, A_PCT), ("B", b_pool, B_PCT)]:
            if len(pool) == 0:
                continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                             "industry": row.get("industry", ""),
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
    pnl = h.groupby("entry_date").agg(gross=("w", "sum"), n=("ts_code", "count")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = df_test.groupby("trade_date")["r20_fresh"].apply(
        lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]
    pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt"] * total
    alpha_avg = float(pnl["alpha"].mean())
    sharpe = float(alpha_avg / (pnl["alpha"].std() + 1e-9) * np.sqrt(12))
    return {"alpha": alpha_avg, "sharpe": sharpe, "n_days": int(pnl["entry_date"].nunique())}


def evaluate_month(test_month, daily_full, fc, cand_daily, ind_mom, r20_lab, b20, r20_fc):
    train_end = get_train_end_for_test_month(test_month)
    train_start = get_train_start(train_end, TRAIN_LOOKBACK_MONTHS)
    print(f"\n--- {test_month}: train {train_start}-{train_end} (复用 t005 尺度模型) ---", flush=True)
    df_train = daily_full[(daily_full["trade_date"] >= train_start) &
                          (daily_full["trade_date"] < train_end)].copy()
    if len(df_train) < 100_000:
        print(f"  跳过 (训练样本不足 {len(df_train)})", flush=True)
        return None
    boosters = {}
    for H in SCALE_LIST:
        boosters[H] = train_scale_month(df_train, fc, f"label_s{H}", test_month, H)
    df_test = daily_full[daily_full["trade_date"].str.startswith(test_month)].copy()
    if len(df_test) < 100:
        return None
    for c in fc:
        if c not in df_test.columns:
            df_test[c] = 0.0
    for c in r20_fc:
        if c not in df_test.columns:
            df_test[c] = 0.0
    Xf = df_test[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    for H in SCALE_LIST:
        proba = boosters[H].predict(Xf)
        if H == 5:
            df_test["pump_down_s5"] = proba[:, 1]
            df_test["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
    Xr = df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_test["pred_r20"] = b20.predict(Xr)
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
    df_test = df_test.merge(cand_daily, on=["ts_code", "trade_date"], how="left")
    cov = {c: float(df_test[c].notna().mean()) for _, c, _ in CANDIDATES}

    out = {"test_month": test_month}
    arms = [("base", None, 1.0)] + [(name, col, sign) for name, col, sign in CANDIDATES]
    for name, col, sign in arms:
        hold = build_dual(df_test, ind_mom, sort_col="ratio_s5", cand_col=col, sign=sign)
        m = month_metrics(hold, df_test)
        if m is None:
            print(f"  {name}: 无持仓, 跳过本月", flush=True)
            return None
        out[f"{name}_alpha"] = m["alpha"]
        out[f"{name}_sharpe"] = m["sharpe"]
    out["cov_ivol"] = cov.get("ivol_60", 0.0)
    out["cov_onin"] = cov.get("on_minus_in_20", 0.0)
    print(f"  base α={out['base_alpha']:+.3f} | vol α={out['vol_alpha']:+.3f} "
          f"(Δ{out['vol_alpha']-out['base_alpha']:+.3f}) | vp α={out['vp_alpha']:+.3f} "
          f"(Δ{out['vp_alpha']-out['base_alpha']:+.3f})", flush=True)
    del df_train, df_test, boosters
    gc.collect()
    return out


def agg(df, arm):
    a = df[f"{arm}_alpha"]
    return {"alpha_mean": float(a.mean()), "alpha_std": float(a.std()),
            "sharpe_mean": float(df[f"{arm}_sharpe"].mean()),
            "worst_month": float(a.min()),
            "pos_alpha_ratio": float((a > 0).mean()), "n_months": int(len(df))}


def skeptic(df, arm, n_trials):
    """c7: bootstrap P(ΔSharpe>0) on monthly Δα + PBO on arm matrix + DSR 报告。"""
    delta = (df[f"{arm}_alpha"] - df["base_alpha"]).to_numpy(float)
    res = {}
    try:
        bb = renv.block_bootstrap_ci(delta, cohort_len=2, n_boot=2000, ppy=12)
        res["bootstrap_delta"] = {"p_sharpe_gt_0": bb["p_sharpe_gt_0"],
                                  "sharpe_ci95": bb["sharpe_ci95"],
                                  "point_sharpe": bb["point_sharpe"]}
        res["bootstrap_pass"] = bool(bb["p_sharpe_gt_0"] >= 0.95 and bb["sharpe_ci95"][0] > 0)
    except Exception as e:
        res["bootstrap_delta"] = f"err:{repr(e)[:80]}"; res["bootstrap_pass"] = False
    try:
        mat = np.column_stack([df["base_alpha"].to_numpy(float),
                               df["vol_alpha"].to_numpy(float),
                               df["vp_alpha"].to_numpy(float)])
        pbo = renv.pbo_cscv(mat, n_splits=6, ppy=12)
        res["pbo"] = {"pbo": pbo["pbo"], "n_combinations": pbo["n_combinations"]}
        res["pbo_pass"] = bool(pbo["pbo"] <= 0.5)
    except Exception as e:
        res["pbo"] = f"err:{repr(e)[:80]}"; res["pbo_pass"] = False
    try:
        a = df[f"{arm}_alpha"].to_numpy(float) / 100.0   # pp → 比例 月收益近似
        sr_po = renv.annualized_to_per_obs(renv.sharpe(a, 12), 12)
        # sr_variance 保守代理: 三臂 per-obs Sharpe 的方差 (有限, 偏小→DSR偏乐观, 已文档化)
        arm_sr = [renv.annualized_to_per_obs(renv.sharpe(df[f"{x}_alpha"].to_numpy(float)/100.0, 12), 12)
                  for x in ("base", "vol", "vp")]
        var = float(np.var(arm_sr, ddof=1)) if len(arm_sr) > 1 else 0.01
        dsr = renv.deflated_sharpe_ratio(sr_po, n_trials, len(a), max(var, 1e-6))
        res["dsr"] = dsr; res["dsr_pass"] = bool(dsr["passes"])
        res["dsr_note"] = "sr_variance 用3臂Sharpe方差代理(偏小→DSR偏乐观); n_trials=manifest×3 多重检验基数"
    except Exception as e:
        res["dsr"] = f"err:{repr(e)[:80]}"; res["dsr_pass"] = False
    res["c7_pass"] = bool(res.get("bootstrap_pass") and res.get("pbo_pass"))
    return res


def decide(df, arm, base, n_trials, regime_map):
    a = agg(df, arm)
    d_alpha = a["alpha_mean"] - base["alpha_mean"]
    df = df.copy()
    df["d"] = df[f"{arm}_alpha"] - df["base_alpha"]
    d_ex_outlier = float(df.sort_values("d").iloc[:-1]["d"].mean()) if len(df) > 1 else d_alpha
    df["regime"] = df["test_month"].map(regime_map).fillna("unknown")
    regime_delta = {}
    for r_, gg in df.groupby("regime"):
        regime_delta[str(r_)] = round(float(gg["d"].mean()), 3)
    regime_not_hurt = all(v >= 0 for v in regime_delta.values())
    sk = skeptic(df, arm, n_trials)
    c1 = d_alpha >= GATE_DELTA
    c2 = a["sharpe_mean"] >= base["sharpe_mean"]
    c3 = a["worst_month"] >= base["worst_month"]
    c4 = a["pos_alpha_ratio"] >= base["pos_alpha_ratio"]
    c5 = d_ex_outlier > 0
    c6 = regime_not_hurt
    c7 = sk["c7_pass"]
    conds = {"c1_delta>=+0.30pp": bool(c1), "c2_sharpe>=base": bool(c2),
             "c3_worst>=base": bool(c3), "c4_posratio>=base": bool(c4),
             "c5_ex_outlier>0": bool(c5), "c6_regime_not_hurt": bool(c6),
             "c7_skeptic": bool(c7)}
    not_hurt = c2 and c3 and c4 and c6
    if all(conds.values()):
        status = "PASS"
    elif 0 < d_alpha < GATE_DELTA and not_hurt and c5:
        status = "真小"
    else:
        status = "REJECT"
    return {"arm": arm, "status": status, "delta_alpha_pp": round(d_alpha, 4),
            "delta_ex_outlier": round(d_ex_outlier, 4), "agg": a,
            "regime_delta": regime_delta, "gate_conditions": conds, "skeptic": sk}


def main():
    t0 = time.time()
    print("\n=== OA-*-GATE walk-forward (V12.31 + ivol/on_minus_in vs 纯 V12.31) ===\n", flush=True)
    n_trials = 3 * json.loads(MANIFEST_P.read_text(encoding="utf-8"))["n_trials_total"] \
        if MANIFEST_P.exists() else 72
    print(f"[mult-test] DSR n_trials = manifest×3 = {n_trials}", flush=True)

    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]; ind_map = meta.get("industry_map", {})
    print(f"[data] load_window {DATA_START}-{DATA_END} (ST源头排除, +mfk) ...", flush=True)
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

    px = load_daily()
    slim = (daily_full[["ts_code", "trade_date"]].merge(px, on=["ts_code", "trade_date"], how="left")
            .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    for H, g in SCALES.items():
        daily_full[f"label_s{H}"] = build_label(slim, H, g).values
    del slim; gc.collect()
    r20_lab = compute_r20_label(); r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)

    print("[cand] 候选日级因子 (ivol_60 + on_minus_in_20) 从共享面板算 ...", flush=True)
    cand_daily = compute_candidates_daily(ov.build_panel())
    print(f"[cand] {len(cand_daily):,} 行", flush=True)

    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))
    r20_fc = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))["feature_cols"]
    print(f"[prep] 完成 {time.time()-t0:.0f}s\n", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records"); done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月", flush=True)
    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = evaluate_month(m_, daily_full, fc, cand_daily, ind_mom, r20_lab, b20, r20_fc)
        if r:
            results.append(r); pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].copy()
    df["test_month"] = df["test_month"].astype(str)
    df = df.sort_values("test_month").reset_index(drop=True)

    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]].copy()
    rg["trade_date"] = rg["trade_date"].astype(str); rg["m"] = rg["trade_date"].str[:6]
    regime_map = (rg[rg["m"].isin(TEST_MONTHS)].groupby("m")["regime"]
                  .agg(lambda x: x.value_counts().index[0])).to_dict()

    base = agg(df, "base")
    dec_vol = decide(df, "vol", base, n_trials, regime_map)
    dec_vp = decide(df, "vp", base, n_trials, regime_map)

    res = {"window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
           "n_trials_dsr": n_trials, "baseline_v1231": base,
           "vol_ivol_60": dec_vol, "vp_on_minus_in_20": dec_vp,
           "note": "Δα相对量是gate(共模lookahead相消, SIGN-R03); 绝对α含r20池lookahead不作判定。"
                   "SIGN-R16: Phase-1在含walk-forward的全54月选因子=选择层泄漏, PASS须独立OOS/前向复核。"}
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n\n=== OA-GATE 汇总 ===", flush=True)
    print(f"  月数 {len(df)} | base α={base['alpha_mean']:+.3f} Sharpe={base['sharpe_mean']:+.2f}", flush=True)
    for name, dec in [("VOL/ivol_60", dec_vol), ("VP/on_minus_in_20", dec_vp)]:
        print(f"\n  [{name}] Δα={dec['delta_alpha_pp']:+.3f}pp 剔outlier={dec['delta_ex_outlier']:+.3f} "
              f"→ {dec['status']}", flush=True)
        print(f"    regime Δ: {dec['regime_delta']}", flush=True)
        print(f"    gate: {dec['gate_conditions']}", flush=True)
        print(f"    skeptic: bootstrap_pass={dec['skeptic'].get('bootstrap_pass')} "
              f"pbo={dec['skeptic'].get('pbo')} dsr_pass={dec['skeptic'].get('dsr_pass')}", flush=True)

    # verdict 双写
    for vp_path, dec, fam, col in [(VERDICT_VOL, dec_vol, "OA-VOL-GATE", "ivol_60"),
                                   (VERDICT_VP, dec_vp, "OA-VP-GATE", "on_minus_in_20")]:
        concl = (f"{fam}: 候选 {col} 接 V12.31 池排序, {len(df)}月 walk-forward Δα="
                 f"{dec['delta_alpha_pp']:+.3f}pp/月 → {dec['status']}. "
                 f"gate={dec['gate_conditions']}. SIGN-R16: 选择层泄漏 caveat, PASS须独立OOS复核。"
                 f" 负/真小=合法完成(R02)。")
        vp_path.write_text(json.dumps({
            "id": fam, "status": dec["status"], "candidate": col,
            "delta_alpha_pp": dec["delta_alpha_pp"], "delta_ex_outlier": dec["delta_ex_outlier"],
            "baseline_v1231": base, "candidate_agg": dec["agg"],
            "regime_delta": dec["regime_delta"], "gate_conditions": dec["gate_conditions"],
            "skeptic_report": dec["skeptic"], "n_trials_dsr": n_trials,
            "n_months": int(len(df)), "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
            "conclusion": concl,
            "guardrails": ["SIGN-R01 gate冻结", "SIGN-R02 负/真小=合法完成", "SIGN-R03 只认walk-forward α",
                           "SIGN-R05 生产线冻结", "SIGN-R06 ST源头排除", "SIGN-R11 分regime",
                           "SIGN-R15 DSR n_trials含DoF", "SIGN-R16 选择层泄漏caveat"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[verdict] {fam} = {dec['status']} -> {vp_path.relative_to(ROOT)}", flush=True)
    print(f"\n[done] 耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    main()
