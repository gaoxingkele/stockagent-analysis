# -*- coding: utf-8 -*-
"""GATE-001 — walk-forward 决策 gate: V12.31 + 席位印记信号 vs 纯 V12.31.

承接 SEAT-002 (residual_signal, 本循环首个 go 信号: sf_edge/winrate 扣[自身动量+市场]后
仍有独立横截面预测, orth_full IC≈+0.05 |t|>6)。NET-002 = no_residual 不入闸。本任务回答
唯一 ship 问题 (SIGN-R03 IC≠落地, 只认 walk-forward α): 把已验证的席位印记信号接进生产
V12.31 池排序, 19 月严格 walk-forward 下相对纯 V12.31 的 Δα 是否过事前注册 gate?

apples-to-apples 设计 (两臂唯一差异 = 池内排序列, 复用 T-005 harness 完全同口径):
  - 两臂共用同一 V7c 池构造 (r20 top + pyr_velocity 低 + 行业动量排除 + s5 pump_down 过滤),
    同 r20 池模型 (生产 r20_v16_long_nost, 不重训), 同 3way pump 尺度模型 (复用 T-005 月度
    checkpoint research/cache/t005_wf_models/, **不重训** → gate 与 T-005 baseline 逐位可比)。
  - baseline 臂 (= V12.31): 池内按 ratio_s5 排序 (生产 v3c 同档固定窗口 pump ratio)。
  - seat 臂 (= V12.31 + 席位信号): 池内按 z_pool(ratio_s5) + z_pool(sf_filled) 等权混合排序。
    任何共模偏差 (含 r20 池/尺度模型潜在 lookahead) 在 Δα 中相消; gate 只看相对量 (SIGN-R03)。

席位信号接入 (事前注册, 冻结 SIGN-R01; 不调不试第二版 SIGN-R02):
  - 信号 = sf_edge_r20 (PRIMARY gate, 与 r20-held 池**同 horizon**; SEAT-002 四对全过线,
    sf_edge_r20→fr20 orth_full IC 亦 +0.04~0.05)。另跑 sf_edge_r5 作 sensitivity (不决定裁决)。
  - 可用窗口: 龙虎榜收盘后公告 → 席位印记在 D+1 揭示 (SEAT-001 已 D+1 生效防泄漏); 揭示后
    沿 5 个交易日 entry 窗口 carry-forward (常数, 新事件覆盖旧), 窗外 = 中性 (池均值, z≈0)。
    5 日 = 行为信号"新鲜可交易"窗口 (与 SEAT-002 事件对齐口径一致, 非 r20 持有期)。
  - 混合: 池内 z_pool(ratio_s5) + z_pool(sf_filled), 等权 z 空间 (无可调 λ, 防 p-hack)。
    无信号股 z(sf)=0 → 排序与 baseline 一致; 有新鲜印记股按 sf 上抬/下压 = 纯增量 tilt。

事前注册 gate (冻结 SIGN-R01; 全相对 baseline 臂同 harness; 分 regime SIGN-R11; 单月 outlier):
  PASS 当且仅当 (PRIMARY seat 臂):
    c1 Δα(月化) >= +0.30pp           c2 Sharpe 不低于 baseline
    c3 最差月不低于 baseline 最差月    c4 正 α 月占比不低于 baseline
    c5 剔单月 outlier (Δα 最大的月) 后 Δα 仍 > 0
    c6 分 regime (momentum/reversal/mixed) 任一 regime Δα 不为负 (不伤)
  0 < Δα < +0.30 且不伤 baseline → status="真小" (记 blend 候选不强上, responsePlaybook)。
  否则 status=REJECT。REJECT/真小 都是合法完成 (SIGN-R02), 禁止同段 OOS 重调再试。

checkpoint (SIGN-R08): 复用 t005_wf_models (已全 19 月); 每月三臂结果 append
  research/cache/gate001_monthly.csv, 已完成月跳过 → 断点续跑。

产出:
  research/cache/gate001_monthly.csv      (逐月三臂 α/Sharpe/...)
  research/cache/gate001_results.json     (汇总 + gate 判定 + 分 regime)
  research/verdicts/GATE-001.json         (status in {PASS, 真小, REJECT} + 四项指标对照)
不碰任何生产文件 (SIGN-R05); ST 源头排除 (load_window 默认, SIGN-R06)。
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
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom, apply_cap, industry_alloc,
    get_train_end_for_test_month, get_train_start,
)
from t004_multiscale_gate import build_label, load_daily, SCALES, EPS
from t005_walk_forward_gate import train_scale_month, TEST_MONTHS, TRAIN_LOOKBACK_MONTHS

CACHE = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
WF_MODELS = CACHE / "t005_wf_models"            # 复用 T-005 月度尺度模型 (不重训)
MONTHLY = CACHE / "gate001_monthly.csv"
RESULTS = CACHE / "gate001_results.json"
VERDICT = ROOT / "research" / "verdicts" / "GATE-001.json"
PROD = ROOT / "output" / "production"
SF_P = ROOT / "research" / "features" / "seat_footprint.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"

DATA_START, DATA_END = "20220801", "20260601"
SCALE_LIST = [3, 5, 10]

# V7c / dual-track 池构造 (与 T-005 / walk_forward_validation.py 同口径)
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15
PUMP_DOWN_THRESHOLD = 0.60

# 席位信号接入 (冻结)
SEAT_CARRY_DAYS = 5                 # 揭示后 entry 窗口 (交易日)
SEAT_PRIMARY = "sf_edge_r20"        # PRIMARY gate 信号 (horizon-matched r20 池)
SEAT_SENS = "sf_edge_r5"            # sensitivity 信号 (不决定裁决)
GATE_DELTA = 0.30                   # Δα 门槛 (pp/月, 冻结 R01)


def zscore_pool(s: pd.Series) -> pd.Series:
    """池内标准化; 缺失填池内均值 (z≈0 中性)。全缺则返回全 0。"""
    v = s.astype(float)
    present = v.dropna()
    if len(present) == 0:
        return pd.Series(0.0, index=s.index)
    mu, sd = present.mean(), present.std()
    v = v.fillna(mu)
    if sd is None or not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (v - mu) / sd


def build_seat_active(daily_full: pd.DataFrame) -> pd.DataFrame:
    """把 D+1 揭示的席位印记沿 SEAT_CARRY_DAYS 个交易日 carry-forward (新事件覆盖旧)。

    返回 (ts_code, trade_date, sf_edge_r5, sf_edge_r20): 该 trade_date 上"新鲜可用"的席位印记。
    严格因果: 仅向 >= 揭示日的交易日扩散 (信号已是 D+1, 不回填过去)。
    """
    sf = pd.read_parquet(SF_P)[["ts_code", "trade_date", "sf_edge_r5", "sf_edge_r20"]].copy()
    sf["trade_date"] = sf["trade_date"].astype(str)
    tdays = np.array(sorted(daily_full["trade_date"].unique()))
    rows = []
    for _, r in sf.iterrows():
        t0 = r["trade_date"]
        idx = int(np.searchsorted(tdays, t0, side="left"))   # 第一个 >= 揭示日的交易日
        for j in range(idx, min(idx + SEAT_CARRY_DAYS, len(tdays))):
            rows.append((r["ts_code"], tdays[j], t0, r["sf_edge_r5"], r["sf_edge_r20"]))
    act = pd.DataFrame(rows, columns=["ts_code", "trade_date", "src_date",
                                      "sf_edge_r5", "sf_edge_r20"])
    # 同 (股, 交易日) 多事件 → 保留最新揭示 (src_date 最大)
    act = (act.sort_values(["ts_code", "trade_date", "src_date"])
              .drop_duplicates(["ts_code", "trade_date"], keep="last")
              .drop(columns=["src_date"]).reset_index(drop=True))
    return act


def build_dual(daily: pd.DataFrame, ind_mom: pd.DataFrame, sort_col: str,
               seat_col: str | None = None) -> pd.DataFrame:
    """V7c dual-track。seat_col=None → 按 sort_col (baseline=ratio_s5);
    seat_col 给定 → 池内按 z_pool(sort_col)+z_pool(seat_col) 等权混合排序 (seat 臂)。
    其余 (池/过滤/行业cap/仓位/成本) 与 baseline 完全一致 → apples-to-apples。"""
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
        # ── 排序键 ──
        if seat_col is None:
            use_col = sort_col
        else:
            v7c["_blend"] = (zscore_pool(v7c[sort_col]).to_numpy()
                             + zscore_pool(v7c[seat_col]).to_numpy())
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


def month_metrics(hold: pd.DataFrame, df_test: pd.DataFrame) -> dict | None:
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
    return {"alpha": alpha_avg, "sharpe": sharpe,
            "n_days": int(pnl["entry_date"].nunique()),
            "avg_n_stocks": float(pnl["n"].mean())}


# ───────────────────────── 单月评估 (三臂) ─────────────────────────
def evaluate_month(test_month, daily_full, fc, seat_act, ind_mom,
                   r20_lab, b20, r20_fc) -> dict | None:
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
    # 席位印记 (D+1 揭示 + carry; 缺=中性)
    df_test = df_test.merge(seat_act, on=["ts_code", "trade_date"], how="left")
    seat_cov = float(df_test[SEAT_PRIMARY].notna().mean())

    out = {"test_month": test_month, "train_start": train_start, "train_end": train_end,
           "seat_cov_all": seat_cov}
    arms = [("base", "ratio_s5", None),
            ("seat", "ratio_s5", SEAT_PRIMARY),
            ("seat5", "ratio_s5", SEAT_SENS)]
    for arm, col, scol in arms:
        hold = build_dual(df_test, ind_mom, sort_col=col, seat_col=scol)
        m = month_metrics(hold, df_test)
        if m is None:
            print(f"  {arm}: 无持仓/无 r20, 跳过本月", flush=True)
            return None
        out[f"{arm}_alpha"] = m["alpha"]
        out[f"{arm}_sharpe"] = m["sharpe"]
        out[f"{arm}_n_days"] = m["n_days"]
        out[f"{arm}_n_stocks"] = m["avg_n_stocks"]
    print(f"  base α={out['base_alpha']:+.3f} | seat α={out['seat_alpha']:+.3f} "
          f"(Δ {out['seat_alpha']-out['base_alpha']:+.3f}) | seat5 α={out['seat5_alpha']:+.3f} "
          f"| 席位覆盖 {seat_cov:.3%}", flush=True)

    del df_train, df_test, boosters
    gc.collect()
    return out


def agg(df, arm):
    a = df[f"{arm}_alpha"]
    return {"alpha_mean": float(a.mean()), "alpha_median": float(a.median()),
            "alpha_std": float(a.std()), "sharpe_mean": float(df[f"{arm}_sharpe"].mean()),
            "worst_month": float(a.min()),
            "worst_month_id": str(df.loc[a.idxmin(), "test_month"]),
            "pos_alpha_ratio": float((a > 0).mean()),
            "disaster_months": int((a < -1).sum()), "n_months": int(len(df))}


def main():
    t0 = time.time()
    print("\n=== GATE-001 walk-forward (V12.31 + 席位印记 vs 纯 V12.31) ===\n", flush=True)

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

    px = load_daily()
    slim = (daily_full[["ts_code", "trade_date"]]
            .merge(px, on=["ts_code", "trade_date"], how="left")
            .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    print("[label] 构建 3 尺度 forward 标签 (仅训练用; 模型已 checkpoint 故仅占位) ...", flush=True)
    for H, g in SCALES.items():
        daily_full[f"label_s{H}"] = build_label(slim, H, g).values
    del slim
    gc.collect()

    print("[label] r20 forward + 行业动量 ...", flush=True)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)

    print("[seat] 席位印记 D+1 揭示 + carry (因果) ...", flush=True)
    seat_act = build_seat_active(daily_full)
    print(f"[seat] active 表 {len(seat_act):,} 行 / {seat_act['ts_code'].nunique()} 股 / "
          f"{seat_act['trade_date'].nunique()} 日 (carry {SEAT_CARRY_DAYS}d, primary={SEAT_PRIMARY})", flush=True)

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
        r = evaluate_month(m_, daily_full, fc, seat_act, ind_mom, r20_lab, b20, r20_fc)
        if r:
            results.append(r)
            pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].copy()
    df["test_month"] = df["test_month"].astype(str)
    df = df.sort_values("test_month").reset_index(drop=True)

    base, seat, seat5 = agg(df, "base"), agg(df, "seat"), agg(df, "seat5")
    d_alpha = seat["alpha_mean"] - base["alpha_mean"]
    d_alpha5 = seat5["alpha_mean"] - base["alpha_mean"]

    # ── 单月 outlier: 剔 Δα 最大的月后 Δα 仍 >0 ──
    df["d"] = df["seat_alpha"] - df["base_alpha"]
    if len(df) > 1:
        d_ex_outlier = float(df.sort_values("d").iloc[:-1]["d"].mean())
    else:
        d_ex_outlier = d_alpha

    # ── 分 regime (R11): 各 regime Δα 不为负 ──
    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]].copy()
    rg["trade_date"] = rg["trade_date"].astype(str)
    rg["m"] = rg["trade_date"].str[:6]
    mon_regime = (rg[rg["m"].isin(TEST_MONTHS)].groupby("m")["regime"]
                  .agg(lambda x: x.value_counts().index[0]))
    df["regime"] = df["test_month"].map(mon_regime).fillna("unknown")
    regime_delta = {}
    for r_, gg in df.groupby("regime"):
        regime_delta[str(r_)] = {"d_alpha": round(float(gg["d"].mean()), 3),
                                 "n_months": int(len(gg)),
                                 "base_alpha": round(float(gg["base_alpha"].mean()), 3),
                                 "seat_alpha": round(float(gg["seat_alpha"].mean()), 3)}
    regime_not_hurt = all(v["d_alpha"] >= 0 for v in regime_delta.values())

    # ── 事前注册 gate (冻结 R01) ──
    c1 = d_alpha >= GATE_DELTA
    c2 = seat["sharpe_mean"] >= base["sharpe_mean"]
    c3 = seat["worst_month"] >= base["worst_month"]
    c4 = seat["pos_alpha_ratio"] >= base["pos_alpha_ratio"]
    c5 = d_ex_outlier > 0
    c6 = regime_not_hurt
    conds = {"delta_alpha>=+0.30pp": bool(c1), "sharpe>=baseline": bool(c2),
             "worst_month>=baseline": bool(c3), "pos_alpha_ratio>=baseline": bool(c4),
             "single_month_outlier_excl_delta>0": bool(c5),
             "all_regime_not_hurt": bool(c6)}
    passed = all(conds.values())
    not_hurt = c2 and c3 and c4 and c6
    if passed:
        status = "PASS"
    elif 0 < d_alpha < GATE_DELTA and not_hurt and c5:
        status = "真小"
    else:
        status = "REJECT"

    seat_cov_mean = float(df["seat_cov_all"].mean())
    res = {
        "window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
        "protocol": f"walk-forward {TRAIN_LOOKBACK_MONTHS}m lookback, scale models reused from "
                    "t005_wf_models (no retrain), r20 pool model fixed (prod), "
                    "apples-to-apples (pool sort col only diff)",
        "seat_integration": {
            "primary_signal": SEAT_PRIMARY, "sensitivity_signal": SEAT_SENS,
            "carry_days": SEAT_CARRY_DAYS,
            "blend": "z_pool(ratio_s5) + z_pool(sf_filled), equal-weight, neutral fill = pool mean",
            "avail": "D+1 reveal (SEAT-001 causal) carried forward 5 trading days, newer overrides"},
        "baseline_v1231": base, "seat_primary": seat, "seat_sensitivity_r5": seat5,
        "delta_alpha_pp": round(d_alpha, 4),
        "delta_alpha_pp_sensitivity_r5": round(d_alpha5, 4),
        "delta_alpha_excl_outlier": round(d_ex_outlier, 4),
        "regime_delta": regime_delta, "regime_not_hurt": regime_not_hurt,
        "seat_pool_coverage_mean_all": round(seat_cov_mean, 5),
        "gate_conditions": conds, "gate_status": status,
        "note": "Δα 等相对量是 gate; 绝对 α 含两臂共模 r20/尺度池 lookahead, 不作判定 (SIGN-R03)。"
                "REJECT/真小 合法完成, 禁止同段 OOS 重调 (SIGN-R02)。",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n\n=== GATE-001 walk-forward 汇总 ===\n", flush=True)
    print(f"  月数: {len(df)}  ({TEST_MONTHS[0]}-{TEST_MONTHS[-1]})", flush=True)
    print(f"  {'指标':22s} {'baseline':>12s} {'seat(r20)':>12s} {'seat5(r5)':>12s}", flush=True)
    print(f"  {'月化 α(pp)':22s} {base['alpha_mean']:>+12.3f} {seat['alpha_mean']:>+12.3f} {seat5['alpha_mean']:>+12.3f}", flush=True)
    print(f"  {'Sharpe':22s} {base['sharpe_mean']:>+12.2f} {seat['sharpe_mean']:>+12.2f} {seat5['sharpe_mean']:>+12.2f}", flush=True)
    print(f"  {'最差月(pp)':22s} {base['worst_month']:>+12.3f} {seat['worst_month']:>+12.3f} {seat5['worst_month']:>+12.3f}", flush=True)
    print(f"  {'正 α 月占比':22s} {base['pos_alpha_ratio']:>12.1%} {seat['pos_alpha_ratio']:>12.1%} {seat5['pos_alpha_ratio']:>12.1%}", flush=True)
    print(f"\n  Δα(seat-base) = {d_alpha:+.3f}pp  (gate 需 >= +{GATE_DELTA:.2f}pp); 剔单月outlier后 {d_ex_outlier:+.3f}pp", flush=True)
    print(f"  席位池覆盖均值 {seat_cov_mean:.3%} (全市场); 分 regime Δα: "
          f"{ {k: v['d_alpha'] for k, v in regime_delta.items()} }", flush=True)
    print(f"\n  gate 条件: {conds}", flush=True)
    print(f"  >>> verdict = {status} <<<\n", flush=True)
    for _, r in df.iterrows():
        print(f"    {r['test_month']} [{r['regime'][:4]}]  base {r['base_alpha']:+.3f}  "
              f"seat {r['seat_alpha']:+.3f}  (Δ {r['d']:+.3f})  cov {r['seat_cov_all']:.2%}", flush=True)

    # ── verdict ──
    table = {
        "metric": ["alpha_monthly_pp", "sharpe", "worst_month_pp", "pos_alpha_month_ratio"],
        "baseline_v1231": [round(base["alpha_mean"], 3), round(base["sharpe_mean"], 2),
                           round(base["worst_month"], 3), round(base["pos_alpha_ratio"], 3)],
        "v1231_plus_seat": [round(seat["alpha_mean"], 3), round(seat["sharpe_mean"], 2),
                            round(seat["worst_month"], 3), round(seat["pos_alpha_ratio"], 3)],
        "delta": [round(d_alpha, 3),
                  round(seat["sharpe_mean"] - base["sharpe_mean"], 2),
                  round(seat["worst_month"] - base["worst_month"], 3),
                  round(seat["pos_alpha_ratio"] - base["pos_alpha_ratio"], 3)],
    }
    if status == "PASS":
        conclusion = (f"PASS: V12.31 池排序混入席位印记 (sf_edge_r20, D+1+5d carry) 在 19 月 "
                      f"walk-forward 上 Δα={d_alpha:+.3f}pp/月 (>=+{GATE_DELTA}pp) 且 Sharpe/最差月/"
                      f"正α月占比不低于纯 V12.31, 剔单月 outlier 后 Δα={d_ex_outlier:+.3f}pp>0, "
                      f"分 regime 不伤 → 真 alpha 成立, 解除 LAND-001 落地门。")
    elif status == "真小":
        conclusion = (f"真小: 席位印记 Δα={d_alpha:+.3f}pp/月 (0<Δα<+{GATE_DELTA}, 不伤 baseline) — "
                      f"SEAT-002 验证的横截面残差 (orth_full IC≈+0.05) 真实, 但接进 V12.31 r20 池后"
                      f"增量未过 +0.30pp 门槛 (席位信号稀疏: 池内平均仅 {seat_cov_mean:.2%} 股有新鲜印记, "
                      f"且短线 r5/r20 行为信号被 V12.31 启动子部分吃掉)。记 blend 候选, 不强上生产 "
                      f"(responsePlaybook GATE 真小分叉)。")
    else:
        fail = [k for k, v in conds.items() if not v]
        conclusion = (f"REJECT: 席位印记接进 V12.31 r20 池排序, 19 月 walk-forward Δα={d_alpha:+.3f}pp/月, "
                      f"未满足事前注册 gate (未过: {fail})。SEAT-002 横截面残差 IC 真实但池稀疏 "
                      f"(平均 {seat_cov_mean:.2%} 池股有新鲜印记) + 短线行为信号与 V12.31 r20 持有期错配 → "
                      f"无可落地真 alpha, 生产线不动 (SIGN-R02 负结果=合法完成, R05 冻结)。")

    VERDICT.write_text(json.dumps({
        "id": "GATE-001", "status": status, "conclusion": conclusion,
        "metrics": {"alpha_delta_pp": round(d_alpha, 3),
                    "sharpe": round(seat["sharpe_mean"], 2),
                    "worst_month": round(seat["worst_month"], 3),
                    "pos_alpha_month_ratio": round(seat["pos_alpha_ratio"], 3)},
        "comparison_table": table,
        "gate_conditions": conds,
        "delta_alpha_excl_single_month_outlier": round(d_ex_outlier, 3),
        "sensitivity_seat_r5_delta_alpha_pp": round(d_alpha5, 3),
        "regime_delta": regime_delta, "regime_not_hurt": regime_not_hurt,
        "seat_pool_coverage_mean": round(seat_cov_mean, 5),
        "n_months": int(len(df)), "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
        "arms_evaluated": "NET-002=no_residual 不入闸; 仅 SEAT 轨 (SEAT-002=residual_signal) 入 walk-forward",
        "artifacts": ["research/cache/gate001_monthly.csv", "research/cache/gate001_results.json"],
        "note": "apples-to-apples: 三臂唯一差异=池内排序键 (ratio_s5 vs +z(sf_edge_r20) vs +z(sf_edge_r5)); "
                "复用 t005_wf_models 尺度模型 + 生产 r20 池模型 (两臂共模, 仅 Δ 量作 gate, SIGN-R03); "
                "seat_r5 仅 sensitivity 不决定裁决。REJECT/真小 是合法完成。",
        "guardrails": ["SIGN-R01 gate 阈值冻结", "SIGN-R02 负/真小结果=合法完成不重调",
                       "SIGN-R03 只认 walk-forward α", "SIGN-R05 生产线冻结未动",
                       "SIGN-R06 ST 源头排除 (load_window)", "SIGN-R11 分 regime 评估",
                       "SIGN-R08 复用 t005 月度模型 checkpoint"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] verdict={status} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    main()
