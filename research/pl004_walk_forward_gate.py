# -*- coding: utf-8 -*-
"""PL-004 walk-forward 决策 gate — debias pump vs v3c pump (同池同排序, apples-to-apples).

研究问题 (承 PL-001/PL-002/PL-003 北极星的最后一问):
  PL-003 已证去偏 pump **捕捉成功 (对精确盲点 flat-mid/flat_base)**: debias 把 PL-001 锁定的
  flat-mid(-5%~+5%) 启动 ratio 排名分位抬升 +0.034 / recall@20% +0.060 / precision@10% 4.9%→9.0%
  (近翻倍)。本任务回答唯一的 ship 问题: **捕捉了那段横盘 base 启动, 在严格 walk-forward 上能否增 α?**
  还是横盘启动质量低/已 priced — **盲点真实但修了不赚 = past_r5 偏好其实是隐性质量过滤** (REJECT_捕捉无α)。

apples-to-apples 设计 (两臂唯一差异 = pump label 的 past_r5 约束):
  - 两臂共用同一 V7c dual-track 池骨架 (r20 top + pyr_velocity 低 + 行业动量排除 + pump_down 过滤),
    同 r20 池模型 (生产 r20_v16_long_nost, 不重训, 两臂共用)、同行业 cap、同仓位分配、同成本。
  - v3c 臂 (= V12.31 等价): 池内 pump_down 过滤 + ratio 排序均来自 **v3c pump** (down 带 past_r5<=8%)。
  - debias 臂: 池内 pump_down 过滤 + ratio 排序均来自 **debias pump** (down 去 past_r5 约束)。
  两臂 pump 分数来自 PL-002 月度 walk-forward 重训 (同窗口/超参/种子, 仅 label 的 past_r5 约束不同),
  → 任何共模偏差 (含 r20 池模型潜在 lookahead) 在 Δα 中相消; gate 只看相对量 (SIGN-R03)。

  注: 采用"全 label 互换" (filter+sort 都用本臂 pump), 而非仅换 sort 列 —— 因为生产 V12.31 的
  pump_down 过滤与 ratio 排序本就来自同一个 3way pump 模型, 换 label 自然同时改变两者。这才是
  "两臂仅 pump label 不同" 的诚实落地 (PL-003 发现的高 past_r5 股 P_down 抬高沉底效应会经过滤+排序双通道流过)。

walk-forward 协议 (与 PL-002 / t005 / V11 walk-forward 完全一致):
  - 测试月 202410-202604 共 19 月; pump 分数直接复用 PL-002 月度重训产出 (cache/pl002_pump_scores.parquet)。
  - r20 池模型沿用生产 r20_v16_long_nost (不重训, 两臂共用)。
  - 特征全 causal; ST 已在 PL-002 源头排除 (load_window exclude_st=True 默认, SIGN-R06)。

事前注册 gate (冻结 SIGN-R01; 全相对 v3c 臂同 harness):
  c1 Δα(debias−v3c, 月化) >= +0.30pp
  c2 Sharpe 不低于 v3c           c3 最差月不低于 v3c           c4 正 α 月占比不低于 v3c
  c5 单月 outlier 剔后 Δα 仍 > 0   c6 分 regime (动量/反转月, RG-001) debias 不伤 (每 regime Δα >= -tol)
  全过 → PASS。playbook 分叉 (SIGN-R02 REJECT 合法完成, 禁止同段 OOS 重调):
    PASS            : 全 gate 通过 → 解除 PL-005 落地条件门。
    真小            : 捕捉了但 0<Δα 但未满足全 gate → 记 blend 候选不强上。
    REJECT_捕捉无α  : 捕捉了横盘启动但 Δα<=0 → **重要发现**: 盲点真实但修了不赚 = past_r5 偏好
                      其实是隐性质量过滤 (横盘启动质量低/已 priced)。
    REJECT_未捕捉   : PL-003 已失败的延续 (本轮 PL-003=捕捉成功, 不触发)。

产出:
  - research/cache/pl004_monthly.csv        (逐月两臂 α/Sharpe + regime 标注)
  - research/cache/pl004_results.json       (汇总 + gate 判定 + 分 regime + 单月 outlier)
  - research/verdicts/PL-004.json           (status + 四项指标对照 + 三/双臂对照)
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
from train_v15_refresh import load_window
from walk_forward_validation import (
    compute_r20_label, compute_ind_mom, apply_cap, industry_alloc,
)

CACHE = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
SCORES = CACHE / "pl002_pump_scores.parquet"          # PL-002 两臂 walk-forward pump 分数
RG_RESULTS = CACHE / "rg001_results.json"             # RG-001 月度 regime
MONTHLY = CACHE / "pl004_monthly.csv"
RESULTS = CACHE / "pl004_results.json"
VERDICT = ROOT / "research" / "verdicts" / "PL-004.json"
PROD = ROOT / "output" / "production"

TEST_MONTHS = ["202410", "202411", "202412",
               "202501", "202502", "202503", "202504", "202505", "202506",
               "202507", "202508", "202509",
               "202510", "202511", "202512",
               "202601", "202602", "202603", "202604"]
# 评估只需测试月特征 (pump 分数已含历史信息; r20/ind_mom 从全量 tushare_cache 独立算)
DATA_START, DATA_END = "20241001", "20260601"
EPS = 0.01

# V7c / dual-track 池构造 (与 walk_forward_validation.py / t005 同口径)
COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15
PUMP_DOWN_THRESHOLD = 0.60

ARMS = ["v3c", "debias"]
REGIME_HURT_TOL = 0.0   # 分 regime "不伤" 容差: 每 regime debias 月均 α 不低于 v3c (>= -tol)


# ───────────────────────── 池构造 (两臂共用骨架, pump 列可配) ─────────────────────────
def build_dual(daily: pd.DataFrame, ind_mom: pd.DataFrame,
               sort_col: str, pdown_col: str) -> pd.DataFrame:
    """V7c dual-track, 池内 pump_down 过滤 (pdown_col) + ratio 排序 (sort_col)。
    两臂仅这两列 (来自同一臂 pump) 不同, 其余完全一致 (= 仅 pump label 不同)。"""
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
        v7c = v7c[v7c[pdown_col] < PUMP_DOWN_THRESHOLD]    # 本臂 pump_down 过滤
        if len(v7c) == 0:
            continue
        v7c = v7c.sort_values(sort_col, ascending=False)    # 本臂 ratio 排序
        a_pool = v7c.head(MAX_A).copy()
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN, sort_col=sort_col)
        a_ind = industry_alloc(a_pool, A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN, sort_col=sort_col)
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind, sort_col=sort_col)
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


# ───────────────────────── 单月评估 (两臂) ─────────────────────────
def evaluate_month(test_month, daily_full, scores, ind_mom, r20_lab, b20, r20_fc) -> dict | None:
    print(f"\n--- {test_month} ---", flush=True)
    df_test = daily_full[daily_full["trade_date"].str.startswith(test_month)].copy()
    if len(df_test) < 100:
        print(f"  跳过 (测试样本不足 {len(df_test)})", flush=True)
        return None
    for c in r20_fc:
        if c not in df_test.columns:
            df_test[c] = 0.0

    # r20 池排序 (生产模型, 两臂共用)
    Xr = df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_test["pred_r20"] = b20.predict(Xr)

    # 合并两臂 pump 分数 (PL-002 月度 walk-forward 重训, apples-to-apples)
    sc = scores[scores["trade_date"].str.startswith(test_month)]
    pump_cols = ["ts_code", "trade_date",
                 "v3c_pump_down", "v3c_ratio", "debias_pump_down", "debias_ratio"]
    n_before = len(df_test)
    df_test = df_test.merge(sc[pump_cols], on=["ts_code", "trade_date"], how="inner")
    if len(df_test) < 100:
        print(f"  跳过 (pump 分数 merge 后样本不足 {len(df_test)})", flush=True)
        return None

    # r20 forward label
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

    out = {"test_month": test_month, "n_scored": int(len(df_test)), "n_daily": int(n_before)}
    for arm in ARMS:
        hold = build_dual(df_test, ind_mom,
                          sort_col=f"{arm}_ratio", pdown_col=f"{arm}_pump_down")
        m = month_metrics(hold, df_test)
        if m is None:
            print(f"  {arm}: 无持仓/无 r20, 跳过本月", flush=True)
            return None
        out[f"{arm}_alpha"] = m["alpha"]
        out[f"{arm}_sharpe"] = m["sharpe"]
        out[f"{arm}_n_days"] = m["n_days"]
        out[f"{arm}_n_stocks"] = m["avg_n_stocks"]
    out["delta_alpha"] = out["debias_alpha"] - out["v3c_alpha"]
    print(f"  v3c α={out['v3c_alpha']:+.3f}pp Sh={out['v3c_sharpe']:+.2f} | "
          f"debias α={out['debias_alpha']:+.3f}pp Sh={out['debias_sharpe']:+.2f} | "
          f"Δα={out['delta_alpha']:+.3f}pp", flush=True)
    del df_test
    gc.collect()
    return out


# ───────────────────────── 主流程 ─────────────────────────
def main():
    t0 = time.time()
    print("\n=== PL-004 walk-forward 决策 gate (debias pump vs v3c, 同池同排序) ===\n", flush=True)

    if not SCORES.exists():
        raise FileNotFoundError(f"缺 PL-002 pump 分数: {SCORES} (先跑 pl002)")
    scores = pd.read_parquet(SCORES)
    scores["trade_date"] = scores["trade_date"].astype(str)
    print(f"[scores] PL-002 两臂 pump 分数 {len(scores):,} 行 / {scores['ts_code'].nunique()} 股", flush=True)

    # 月度 regime (RG-001 因果检测器, 用于分 regime gate SIGN-R11)
    import io
    rg = json.load(io.open(RG_RESULTS, encoding="utf-8"))
    month_regime = {r["ym"]: r["dominant"] for r in rg.get("monthly_dominant_regime", [])}
    print(f"[regime] RG-001 月度 regime 覆盖 {len(month_regime)} 月", flush=True)

    # r20 池模型特征 (生产, 不重训)
    meta_r20 = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json")
                          .read_text(encoding="utf-8"))
    r20_fc = meta_r20["feature_cols"]
    ind_map = meta_r20.get("industry_map", {})
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt")
                      .read_text(encoding="utf-8"))

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map and "industry" in daily_full.columns:
        daily_full["industry_id"] = (daily_full["industry"].fillna("unknown")
                                     .map(ind_map).fillna(-1).astype(int))
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[data] daily_full {len(daily_full):,} 行 / {daily_full['ts_code'].nunique()} 股", flush=True)

    print("[label] r20 forward + 行业动量 (全量 tushare_cache, causal) ...", flush=True)
    r20_lab = compute_r20_label()
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)
    print(f"[prep] 完成 {time.time()-t0:.0f}s\n", flush=True)

    # checkpoint: 已完成月跳过 (SIGN-R08)
    done = set()
    results = []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records")
        done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月: {sorted(done)}", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = evaluate_month(m_, daily_full, scores, ind_mom, r20_lab, b20, r20_fc)
        if r:
            results.append(r)
            pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].copy()
    df["test_month"] = df["test_month"].astype(str)
    df = df.sort_values("test_month").reset_index(drop=True)
    df["regime"] = df["test_month"].map(month_regime).fillna("unknown")

    # ── 汇总两臂 ──
    def agg(arm):
        a = df[f"{arm}_alpha"]
        return {"alpha_mean": float(a.mean()), "alpha_median": float(a.median()),
                "alpha_std": float(a.std()), "sharpe_mean": float(df[f"{arm}_sharpe"].mean()),
                "worst_month": float(a.min()),
                "worst_month_id": str(df.loc[a.idxmin(), "test_month"]),
                "pos_alpha_ratio": float((a > 0).mean()),
                "disaster_months": int((a < -1).sum()), "n_months": int(len(df))}
    v3c, deb = agg("v3c"), agg("debias")
    d_alpha = deb["alpha_mean"] - v3c["alpha_mean"]

    # ── 单月 outlier 剔除后 Δα (SIGN: 单月outlier) ──
    pm_delta = (df["debias_alpha"] - df["v3c_alpha"]).to_numpy()
    out_idx = int(np.argmax(np.abs(pm_delta)))
    out_month = str(df.loc[out_idx, "test_month"])
    d_alpha_trim = float(np.delete(pm_delta, out_idx).mean())

    # ── 分 regime (SIGN-R11): 每 regime debias 不伤 ──
    regime_tbl = {}
    for rg_name in ["momentum", "reversal", "mixed"]:
        sub = df[df["regime"] == rg_name]
        if len(sub) == 0:
            continue
        v3c_m = float(sub["v3c_alpha"].mean())
        deb_m = float(sub["debias_alpha"].mean())
        regime_tbl[rg_name] = {"n_months": int(len(sub)),
                               "v3c_alpha": round(v3c_m, 3), "debias_alpha": round(deb_m, 3),
                               "delta": round(deb_m - v3c_m, 3)}
    # "不伤" = 动量月 + 反转月 (真 regime) 每个 Δα >= -tol
    regime_ok = all(regime_tbl.get(r, {"delta": 0.0})["delta"] >= -REGIME_HURT_TOL
                    for r in ["momentum", "reversal"] if r in regime_tbl)

    # ── 事前注册 gate (SIGN-R01, 全相对 v3c 臂) ──
    c1 = d_alpha >= 0.30
    c2 = deb["sharpe_mean"] >= v3c["sharpe_mean"]
    c3 = deb["worst_month"] >= v3c["worst_month"]
    c4 = deb["pos_alpha_ratio"] >= v3c["pos_alpha_ratio"]
    c5 = d_alpha_trim > 0
    c6 = bool(regime_ok)
    conds = {"delta_alpha>=+0.30pp": bool(c1), "sharpe>=v3c": bool(c2),
             "worst_month>=v3c": bool(c3), "pos_alpha_ratio>=v3c": bool(c4),
             "outlier_trim_delta>0": bool(c5), "regime_not_hurt": bool(c6)}
    passed = all(conds.values())

    # ── playbook 分叉 (PL-003=捕捉成功, 不触发 REJECT_未捕捉) ──
    if passed:
        status = "PASS"
    elif d_alpha <= 0:
        status = "REJECT_捕捉无α"
    else:
        status = "真小"

    res = {
        "task": "PL-004", "window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
        "protocol": "walk-forward 19m 202410-202604, pump 分数复用 PL-002 月度重训, "
                    "r20 池模型固定 (prod), apples-to-apples (两臂仅 pump label 的 past_r5 约束不同, "
                    "filter+sort 全用本臂 pump)",
        "v3c": v3c, "debias": deb,
        "delta_alpha_pp": d_alpha,
        "outlier_trim": {"dropped_month": out_month,
                         "dropped_delta": round(float(pm_delta[out_idx]), 3),
                         "delta_alpha_trimmed": round(d_alpha_trim, 3)},
        "regime_table": regime_tbl,
        "gate_conditions": conds, "gate_status": status,
        "note": "Δα 等相对量是 gate; 绝对 α 含两臂共模 r20 池 lookahead, 不作判定 (SIGN-R03)。"
                "REJECT 合法完成, 禁止同段 OOS 重调 (SIGN-R02)。",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n\n=== PL-004 walk-forward 汇总 ===\n", flush=True)
    print(f"  月数: {len(df)}  ({TEST_MONTHS[0]}-{TEST_MONTHS[-1]})", flush=True)
    print(f"  {'指标':22s} {'v3c(baseline)':>14s} {'debias':>14s}", flush=True)
    print(f"  {'月化 α(pp)':22s} {v3c['alpha_mean']:>+14.3f} {deb['alpha_mean']:>+14.3f}", flush=True)
    print(f"  {'Sharpe':22s} {v3c['sharpe_mean']:>+14.2f} {deb['sharpe_mean']:>+14.2f}", flush=True)
    print(f"  {'最差月(pp)':22s} {v3c['worst_month']:>+14.3f} {deb['worst_month']:>+14.3f}", flush=True)
    print(f"  {'正 α 月占比':22s} {v3c['pos_alpha_ratio']:>14.1%} {deb['pos_alpha_ratio']:>14.1%}", flush=True)
    print(f"\n  Δα = {d_alpha:+.3f}pp  (gate 需 >= +0.30pp)", flush=True)
    print(f"  单月 outlier 剔 {out_month} (Δ {pm_delta[out_idx]:+.3f}) 后 Δα = {d_alpha_trim:+.3f}pp", flush=True)
    print("  分 regime Δα:", flush=True)
    for rg_name, v in regime_tbl.items():
        print(f"    {rg_name:10s} n={v['n_months']:2d}  v3c {v['v3c_alpha']:+.3f}  "
              f"debias {v['debias_alpha']:+.3f}  (Δ {v['delta']:+.3f})", flush=True)
    print(f"\n  gate 六条件: {conds}", flush=True)
    print(f"  >>> verdict = {status} <<<\n", flush=True)
    print("  月度明细 (v3c α / debias α):", flush=True)
    for _, r in df.iterrows():
        print(f"    {r['test_month']} [{r['regime'][:3]}]  v3c {r['v3c_alpha']:+.3f}  "
              f"debias {r['debias_alpha']:+.3f}  (Δ {r['debias_alpha']-r['v3c_alpha']:+.3f})", flush=True)

    # ── verdict ──
    table = {
        "metric": ["alpha_monthly_pp", "sharpe", "worst_month_pp", "pos_alpha_month_ratio"],
        "v3c_baseline": [round(v3c["alpha_mean"], 3), round(v3c["sharpe_mean"], 2),
                         round(v3c["worst_month"], 3), round(v3c["pos_alpha_ratio"], 3)],
        "debias": [round(deb["alpha_mean"], 3), round(deb["sharpe_mean"], 2),
                   round(deb["worst_month"], 3), round(deb["pos_alpha_ratio"], 3)],
        "delta": [round(d_alpha, 3),
                  round(deb["sharpe_mean"] - v3c["sharpe_mean"], 2),
                  round(deb["worst_month"] - v3c["worst_month"], 3),
                  round(deb["pos_alpha_ratio"] - v3c["pos_alpha_ratio"], 3)],
    }
    fail = [k for k, v in conds.items() if not v]
    if status == "PASS":
        conclusion = (
            f"PASS: 去偏 pump 在 19 月 walk-forward 上 Δα={d_alpha:+.3f}pp/月 (>=+0.30pp), 且 Sharpe/"
            f"最差月/正α月占比/单月outlier剔后/分regime 全不伤 → 捕捉那段横盘 base 启动确带可落地真 alpha, "
            f"解除 PL-005 落地条件门 (opt-in)。")
    elif status == "REJECT_捕捉无α":
        conclusion = (
            f"REJECT_捕捉无α (重要发现): 去偏 pump 在 PL-003 已证捕捉成功 (flat-mid 启动 precision@10% "
            f"4.9%→9.0%), 但 19 月 walk-forward Δα={d_alpha:+.3f}pp/月 <= 0 → **盲点真实但修了不赚**。"
            f"印证北极星核心张力: pump 启动子 v3c label 的 past_r5 偏好其实是**隐性质量过滤** —— 横盘/微涨 "
            f"base 启动 (MA20 走平形态一族) 虽启动率更高, 但启动质量低/已 priced, 把它们排上来反而稀释 α。"
            f"去偏=把高 past_r5 趋势启动股 (P_down 抬高) 挤出池顶、换入低质横盘启动, 净 α 不增。"
            f"未过: {fail}。REJECT 合法完成 (SIGN-R02), 不落地, 生产 V12.31 v3c 不动。")
    else:  # 真小
        conclusion = (
            f"真小 (0<Δα 但未满全 gate): 去偏 pump 19 月 walk-forward Δα={d_alpha:+.3f}pp/月, 捕捉了 "
            f"flat-mid 横盘启动但增益不足以单独 ship (未过: {fail})。记 blend 候选不强上 (SIGN-R02/R03), "
            f"生产 V12.31 v3c 不动。")
    VERDICT.write_text(json.dumps({
        "id": "PL-004", "status": status, "conclusion": conclusion,
        "metrics": {"alpha_delta_pp": round(d_alpha, 3),
                    "sharpe": round(deb["sharpe_mean"], 2),
                    "worst_month": round(deb["worst_month"], 3),
                    "pos_alpha_month_ratio": round(deb["pos_alpha_ratio"], 3)},
        "comparison_table": table,
        "gate_conditions": conds,
        "outlier_trim": res["outlier_trim"],
        "regime_table": regime_tbl,
        "n_months": int(len(df)), "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
        "artifacts": ["research/cache/pl004_monthly.csv", "research/cache/pl004_results.json"],
        "note": "apples-to-apples: 两臂仅 pump label 的 past_r5 约束不同 (filter+sort 全用本臂 pump, "
                "pump 分数复用 PL-002 月度 walk-forward 重训); 绝对 α 含共模 r20 池 lookahead, 仅 Δ 量作 "
                "gate (SIGN-R03)。捕捉验证已在 PL-003 完成 (捕捉成功)。REJECT 是合法完成 (SIGN-R02)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] verdict={status} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    main()
