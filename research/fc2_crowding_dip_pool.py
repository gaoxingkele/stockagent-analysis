# -*- coding: utf-8 -*-
"""FC-2 — 机构抱团∩回调 池 walk-forward Δα (crowding_delta 入 V7c 池排序).

FC-1 唯一存活=crowding_delta(B). V7c 池本就选回调(pump_down低/pyr低)=我们回调DNA, 把 PIT 抱团流
信号 z 混入池排序 = "机构抱团信号 ∩ 回调池"。复用 oa_gate harness, 两臂唯一差异=排序键。
PIT: crowding 按 ann_date 前向填充到日级(零前视)。报告池内机构持有覆盖率(抱团∩回调重叠)。

gate(冻结R01): Δα>=+0.30pp + c2-c6 + skeptic(bootstrap/PBO). 数据窗受持仓限(202410+), 标低功率。
生产冻结(R05); ST源头排除。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
from train_v15_refresh import load_window
from walk_forward_validation import (compute_r20_label, compute_ind_mom, apply_cap, industry_alloc,
    get_train_end_for_test_month, get_train_start)
from t004_multiscale_gate import build_label, load_daily, SCALES, EPS
from t005_walk_forward_gate import train_scale_month, TEST_MONTHS, TRAIN_LOOKBACK_MONTHS
import research_env as renv

CACHE = ROOT / "research/cache"
HOLD = CACHE / "fc_holdings_broad.parquet"
MONTHLY = CACHE / "fc2_monthly.csv"
VERDICT = ROOT / "research/verdicts/FC-2.json"
PROD = ROOT / "output/production"
REGIME_P = ROOT / "research/features/regime_timeline.parquet"
DATA_START, DATA_END = "20220801", "20260601"
SCALE_LIST = [3, 5, 10]
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT, MAX_A, MAX_B, PUMP_DOWN_THRESHOLD = 0.70, 0.20, 8, 15, 0.60
GATE_DELTA = 0.30


def build_crowding_daily(trade_dates):
    """PIT 日级抱团: 每交易日用 ann_date<=d 最近季度, crowding_level + crowding_delta(环比季度)。"""
    h = pd.read_parquet(HOLD)[["fund", "ann_date", "end_date", "symbol"]].copy()
    h["ann_date"] = h["ann_date"].astype(str); h["symbol"] = h["symbol"].astype(str)
    anns = sorted(h["ann_date"].unique())
    td = sorted(set(trade_dates))
    # 每交易日映射到 active ann (max ann<=d)
    seg = {}
    for d in td:
        prev = [a for a in anns if a <= d]
        seg[d] = prev[-1] if prev else None
    # 每个 active ann 段算 crowding (cl) + 上一段(delta)
    ann_cl = {}
    for i, a in enumerate(anns):
        cur = h[h["ann_date"] <= a]
        idx = cur.groupby("fund")["ann_date"].transform("max")
        cur = cur[cur["ann_date"] == idx]
        ann_cl[a] = cur.groupby("symbol")["fund"].nunique()
    rows = []
    for d in td:
        a = seg[d]
        if a is None:
            continue
        cl = ann_cl[a]
        ai = anns.index(a)
        prev_cl = ann_cl[anns[ai - 1]] if ai > 0 else pd.Series(dtype=float)
        delta = cl.subtract(prev_cl.reindex(cl.index).fillna(0))
        for s in cl.index:
            rows.append((s, d, float(cl[s]), float(delta.get(s, 0.0))))
    df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "crowding_level", "crowding_delta"])
    return df


def zpool(s):
    v = s.astype(float); p = v.dropna()
    if len(p) == 0: return pd.Series(0.0, index=s.index)
    mu, sd = p.mean(), p.std(); v = v.fillna(mu)
    if not np.isfinite(sd) or sd == 0: return pd.Series(0.0, index=s.index)
    return (v - mu) / sd


def build_dual(daily, ind_mom, cand_col):
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows, cov = [], []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 100: continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 50: continue
        g["r20_rank"] = g["pred_r20"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        m_pyr = (g["pyr_velocity_20_60"] < g["pyr_velocity_20_60"].quantile(PYR_Q)) if "pyr_velocity_20_60" in g else pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue
        v7c = v7c[v7c["pump_down_s5"] < PUMP_DOWN_THRESHOLD]
        if len(v7c) == 0: continue
        if cand_col is not None:
            cov.append(float((v7c["crowding_level"].fillna(0) > 0).mean()))
            v7c["_k"] = zpool(v7c["ratio_s5"]).to_numpy() + zpool(v7c[cand_col]).to_numpy()
            use = "_k"
        else:
            use = "ratio_s5"
        v7c = v7c.sort_values(use, ascending=False)
        a_pool = apply_cap(v7c.head(MAX_A).copy(), A_PCT, CAP_IN, sort_col="ratio_s5")
        a_ind = industry_alloc(a_pool, A_PCT)
        b_pool = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])].head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN, sort_col="ratio_s5")
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind, sort_col="ratio_s5")
        for pool, alloc in [(a_pool, A_PCT), (b_pool, B_PCT)]:
            if len(pool) == 0: continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                             "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                             "alloc_pct": per})
    return pd.DataFrame(rows), (float(np.mean(cov)) if cov else 0.0)


def mmetrics(hold, df_test):
    if hold is None or hold.empty: return None
    h = hold.dropna(subset=["r20_fresh"]).copy()
    if h.empty: return None
    h["r20_fresh"] = h["r20_fresh"].clip(-30, 30); h["w"] = h["alloc_pct"] * h["r20_fresh"]
    total = A_PCT + B_PCT
    pnl = h.groupby("entry_date").agg(gross=("w", "sum")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = df_test.groupby("trade_date")["r20_fresh"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]; pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt"] * total
    a = float(pnl["alpha"].mean())
    return {"alpha": a, "sharpe": float(a / (pnl["alpha"].std() + 1e-9) * np.sqrt(12))}


def evaluate_month(tm, daily_full, fc, crowd, ind_mom, r20_lab, b20, r20_fc):
    te = get_train_end_for_test_month(tm); ts = get_train_start(te, TRAIN_LOOKBACK_MONTHS)
    dtr = daily_full[(daily_full["trade_date"] >= ts) & (daily_full["trade_date"] < te)]
    if len(dtr) < 100_000: return None
    boosters = {H: train_scale_month(dtr, fc, f"label_s{H}", tm, H) for H in SCALE_LIST}
    df_test = daily_full[daily_full["trade_date"].str.startswith(tm)].copy()
    if len(df_test) < 100: return None
    for c in fc:
        if c not in df_test.columns: df_test[c] = 0.0
    for c in r20_fc:
        if c not in df_test.columns: df_test[c] = 0.0
    Xf = df_test[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    for H in SCALE_LIST:
        pr = boosters[H].predict(Xf)
        if H == 5: df_test["pump_down_s5"] = pr[:, 1]; df_test["ratio_s5"] = pr[:, 2] / (pr[:, 1] + EPS)
    df_test["pred_r20"] = b20.predict(df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0))
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
    df_test = df_test.merge(crowd, on=["ts_code", "trade_date"], how="left")
    out = {"test_month": tm}
    hb, _ = build_dual(df_test, ind_mom, None); mb = mmetrics(hb, df_test)
    hc, cov = build_dual(df_test, ind_mom, "crowding_delta"); mc = mmetrics(hc, df_test)
    if mb is None or mc is None: return None
    out.update({"base_alpha": mb["alpha"], "base_sharpe": mb["sharpe"],
                "cand_alpha": mc["alpha"], "cand_sharpe": mc["sharpe"], "pool_crowd_cov": cov})
    print(f"  {tm}: base α={mb['alpha']:+.3f} | crowd_delta α={mc['alpha']:+.3f} "
          f"(Δ{mc['alpha']-mb['alpha']:+.3f}) | 池内机构持有 {cov:.0%}", flush=True)
    del dtr, df_test, boosters; gc.collect()
    return out


def main():
    t0 = time.time()
    print("\n=== FC-2 机构抱团∩回调 池 walk-forward ===\n", flush=True)
    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c/feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]; ind_map = meta.get("industry_map", {})
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map: daily_full["industry_id"] = daily_full["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in fc:
        if c not in daily_full.columns: daily_full[c] = 0.0
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    px = load_daily()
    slim = daily_full[["ts_code", "trade_date"]].merge(px, on=["ts_code", "trade_date"], how="left").sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for H, g in SCALES.items(): daily_full[f"label_s{H}"] = build_label(slim, H, g).values
    del slim; gc.collect()
    r20_lab = compute_r20_label(); r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)
    print("[crowd] PIT 日级抱团 ...", flush=True)
    tdays = daily_full[daily_full["trade_date"] >= "20241001"]["trade_date"].unique()
    crowd = build_crowding_daily(tdays)
    print(f"[crowd] {len(crowd):,}行 / {crowd['trade_date'].nunique()}日", flush=True)
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost/classifier.txt").read_text(encoding="utf-8"))
    r20_fc = json.loads((PROD / "r20_v16_long_nost/feature_meta.json").read_text(encoding="utf-8"))["feature_cols"]
    print(f"[prep] {time.time()-t0:.0f}s\n", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str}); results = prev.to_dict("records"); done = set(prev["test_month"].astype(str))
    for tm in TEST_MONTHS:
        if tm in done: continue
        r = evaluate_month(tm, daily_full, fc, crowd, ind_mom, r20_lab, b20, r20_fc)
        if r: results.append(r); pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results); df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].sort_values("test_month").reset_index(drop=True)
    d_alpha = float((df["cand_alpha"] - df["base_alpha"]).mean())
    df["d"] = df["cand_alpha"] - df["base_alpha"]
    d_ex = float(df.sort_values("d").iloc[:-1]["d"].mean()) if len(df) > 1 else d_alpha
    base = {"alpha": float(df["base_alpha"].mean()), "sharpe": float(df["base_sharpe"].mean()), "worst": float(df["base_alpha"].min()), "posr": float((df["base_alpha"] > 0).mean())}
    cand = {"alpha": float(df["cand_alpha"].mean()), "sharpe": float(df["cand_sharpe"].mean()), "worst": float(df["cand_alpha"].min()), "posr": float((df["cand_alpha"] > 0).mean())}
    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]].copy(); rg["trade_date"] = rg["trade_date"].astype(str); rg["m"] = rg["trade_date"].str[:6]
    rmap = (rg[rg["m"].isin(TEST_MONTHS)].groupby("m")["regime"].agg(lambda x: x.value_counts().index[0])).to_dict()
    df["regime"] = df["test_month"].map(rmap).fillna("unknown")
    regime_delta = {str(r_): round(float(gg["d"].mean()), 3) for r_, gg in df.groupby("regime")}
    try:
        bb = renv.block_bootstrap_ci(df["d"].to_numpy(float), cohort_len=2, n_boot=2000, ppy=12)
        boot_pass = bool(bb["p_sharpe_gt_0"] >= 0.95 and bb["sharpe_ci95"][0] > 0)
    except Exception: bb = {}; boot_pass = False
    conds = {"c1_delta>=0.30": bool(d_alpha >= GATE_DELTA), "c2_sharpe>=base": bool(cand["sharpe"] >= base["sharpe"]),
             "c3_worst>=base": bool(cand["worst"] >= base["worst"]), "c4_posr>=base": bool(cand["posr"] >= base["posr"]),
             "c5_ex_outlier>0": bool(d_ex > 0), "c6_regime_not_hurt": all(v >= 0 for v in regime_delta.values()), "c7_bootstrap": boot_pass}
    status = "PASS" if all(conds.values()) else ("真小" if (0 < d_alpha < GATE_DELTA and conds["c2_sharpe>=base"] and conds["c6_regime_not_hurt"] and conds["c5_ex_outlier>0"]) else "REJECT")
    cov_mean = float(df["pool_crowd_cov"].mean())
    concl = (f"crowding_delta入V7c回调池排序, {len(df)}月walk-forward(数据限202410+低功率) Δα={d_alpha:+.3f}pp/月 "
             f"(剔outlier{d_ex:+.3f}, regime{regime_delta}, bootstrap_pass={boot_pass}); 池内机构持有覆盖均值{cov_mean:.0%} → {status}. "
             + ("机构抱团信号∩回调池有增量。" if status == "PASS"
                else f"机构抱团∩回调增量未过门槛: V7c池(小盘回调)与机构抱团(大盘龙头)重叠仅{cov_mean:.0%}, 风格正交+抱团流信号弱(FC-1仅B). 不落地。"))
    VERDICT.write_text(json.dumps({"id": "FC-2", "status": status, "delta_alpha_pp": round(d_alpha, 4), "delta_ex_outlier": round(d_ex, 4),
        "base": base, "cand": cand, "regime_delta": regime_delta, "gate_conditions": conds, "pool_crowd_coverage": round(cov_mean, 4),
        "bootstrap": bb, "n_months": int(len(df)), "conclusion": concl,
        "guardrails": ["R01冻结", "R02负/真小=合法完成", "R03 walk-forward α", "R04 PIT ann_date", "R05生产冻结", "R11分regime", "R13 bootstrap"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== FC-2 汇总 ===\n  base α={base['alpha']:+.3f} | crowd_delta α={cand['alpha']:+.3f} | Δα={d_alpha:+.3f}pp | 池内机构覆盖{cov_mean:.0%}", flush=True)
    print(f"  gate: {conds}\n  [决策] {status}; {concl}", flush=True)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
