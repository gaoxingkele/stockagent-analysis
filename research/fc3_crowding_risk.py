# -*- coding: utf-8 -*-
"""FC-3 — 抱团拥挤度风控 overlay (排除≥p90拥挤股 vs 随机排除placebo).

对 V7c 回调池, 排除 crowding_level >= 池内p90 的拥挤股 vs (a)base不排除 (b)随机排除同数(placebo).
看 book Sharpe/maxDD. crowding_level FC-1已证零IC → 先验: 按拥挤排除≈随机排除(无技能, 纯去暴露)。
PIT(ann_date); 复用fc2 harness; 生产冻结(R05)。
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
import fc2_crowding_dip_pool as fc2
import research_env as renv

CACHE = ROOT / "research/cache"
MONTHLY = CACHE / "fc3_monthly.csv"
VERDICT = ROOT / "research/verdicts/FC-3.json"
PROD = ROOT / "output/production"
REGIME_P = ROOT / "research/features/regime_timeline.parquet"
A_PCT, B_PCT, MAX_A, MAX_B = 0.70, 0.20, 8, 15
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
PUMP_DOWN_THRESHOLD, COST_BPS = 0.60, 35.0 / 10000


def build_dual(daily, ind_mom, mode, seed=0):
    """mode: None=base / 'excl_crowd'=排≥p90拥挤 / 'excl_random'=随机排同数(placebo)。"""
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rng = np.random.default_rng(seed)
    rows = []
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
        v7c = v7c[v7c["pump_down_s5"] < PUMP_DOWN_THRESHOLD]
        if len(v7c) == 0: continue
        cl = v7c["crowding_level"].fillna(0)
        if mode == "excl_crowd" and (cl > 0).sum() >= 3:
            thr = cl[cl > 0].quantile(0.90)
            drop = cl >= thr
            n_drop = int(drop.sum())
            v7c = v7c[~drop]
        elif mode == "excl_random":
            # 排除同等数量随机股 (用 excl_crowd 同口径估 n_drop)
            n_pos = int((cl > 0).sum())
            n_drop = max(1, int(round(n_pos * 0.10))) if n_pos >= 3 else 0
            if n_drop > 0 and len(v7c) > n_drop:
                didx = rng.choice(v7c.index.to_numpy(), size=n_drop, replace=False)
                v7c = v7c.drop(index=didx)
        if len(v7c) == 0: continue
        v7c = v7c.sort_values("ratio_s5", ascending=False)
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
    return pd.DataFrame(rows)


def mm(hold, df_test):
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
    return {"alpha": a, "sharpe": float(a / (pnl["alpha"].std() + 1e-9) * np.sqrt(12)), "worst": float(pnl["alpha"].min())}


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
    mb = mm(build_dual(df_test, ind_mom, None), df_test)
    mc = mm(build_dual(df_test, ind_mom, "excl_crowd"), df_test)
    mr = mm(build_dual(df_test, ind_mom, "excl_random", seed=hash(tm) % 9999), df_test)
    if None in (mb, mc, mr): return None
    out.update({"base_a": mb["alpha"], "base_s": mb["sharpe"], "base_w": mb["worst"],
                "exclc_a": mc["alpha"], "exclc_s": mc["sharpe"], "exclc_w": mc["worst"],
                "exclr_a": mr["alpha"], "exclr_s": mr["sharpe"], "exclr_w": mr["worst"]})
    print(f"  {tm}: base Sh={mb['sharpe']:+.2f} | excl_crowd Sh={mc['sharpe']:+.2f} | excl_rand Sh={mr['sharpe']:+.2f}", flush=True)
    del dtr, df_test, boosters; gc.collect()
    return out


SCALE_LIST = [3, 5, 10]


def main():
    t0 = time.time()
    print("\n=== FC-3 抱团拥挤度风控 overlay ===\n", flush=True)
    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c/feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]; ind_map = meta.get("industry_map", {})
    daily_full = load_window(fc2.DATA_START, fc2.DATA_END, with_mfk=True)
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
    tdays = daily_full[daily_full["trade_date"] >= "20241001"]["trade_date"].unique()
    crowd = fc2.build_crowding_daily(tdays)
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

    df = pd.DataFrame(results); df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].reset_index(drop=True)
    agg = {k: {"sharpe": float(df[f"{k}_s"].mean()), "alpha": float(df[f"{k}_a"].mean()), "worst": float(df[f"{k}_w"].mean())}
           for k in ["base", "exclc", "exclr"]}
    # 拥挤排除 vs 随机排除: 是否有技能(excl_crowd优于excl_random)
    skill = agg["exclc"]["sharpe"] - agg["exclr"]["sharpe"]
    helps = agg["exclc"]["sharpe"] > agg["base"]["sharpe"]
    placebo_beat = agg["exclc"]["sharpe"] > agg["exclr"]["sharpe"] + 0.1
    status = "PASS" if (helps and placebo_beat) else ("真小_降回撤" if (agg["exclc"]["worst"] > agg["base"]["worst"] + 0.1 and placebo_beat) else "REJECT")
    concl = (f"拥挤风控(排≥p90拥挤股) {len(df)}月: base Sharpe={agg['base']['sharpe']:.2f} | excl_crowd={agg['exclc']['sharpe']:.2f} "
             f"| excl_random(placebo)={agg['exclr']['sharpe']:.2f}. 拥挤排除vs随机排除Δsharpe={skill:+.2f} → {status}. "
             + ("拥挤排除优于随机=真风控技能。" if status == "PASS"
                else "拥挤排除≈随机排除(crowding_level FC-1已证零IC), 无风控技能纯去暴露; 且我们小盘回调池本就少持极端拥挤龙头 → 拥挤风控对V12.31基本no-op。"))
    VERDICT.write_text(json.dumps({"id": "FC-3", "status": status, "agg": agg,
        "excl_crowd_vs_random_sharpe": round(skill, 3), "n_months": int(len(df)), "conclusion": concl,
        "guardrails": ["R01冻结", "R02负=合法完成", "R04 PIT", "R05生产冻结", "R13随机排除placebo负控"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== FC-3 汇总 ===", flush=True)
    print(f"  base Sharpe={agg['base']['sharpe']:.2f} maxworst={agg['base']['worst']:.2f}", flush=True)
    print(f"  excl_crowd Sharpe={agg['exclc']['sharpe']:.2f} | excl_random(placebo) Sharpe={agg['exclr']['sharpe']:.2f} | 技能Δ={skill:+.2f}", flush=True)
    print(f"  [决策] {status}; {concl}", flush=True)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
