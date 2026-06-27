# -*- coding: utf-8 -*-
"""BT-003 — 因子/铁律 leave-one-out 归因 (谁在 book 层拖后腿).

对 V12.31 生产 book 的各**评分组件 + V7c 铁律**逐个 leave-one-out, 消融 arm 过 BT-001 引擎,
报 book 层 Δ(净Sharpe / 年化 / 最大回撤) vs 完整 V12.31 基线, 分 regime, 标正/负贡献。

设计 (冻结, prd.preRegisteredGate.attribution):
  baseline arm = 完整 V12.31 研究 book (t005 build_dual 全过滤 + 双轨 + ratio_s5 排序)。
  每个消融 arm = baseline 去掉**单一**组件 (leave-one-out), 其余完全一致 →
  Δ 是受控差分 (两臂共享同一 r20 池 lookahead / 同成本 / 同引擎), 共模偏差在 Δ 中相消,
  **部分抵消** BT-002 的绝对量级 caveat (SIGN-R03: 只看相对 Δ, 不看绝对 α)。

消融组件 (本研究 book 实际实例化的 7 个; 见下方 ARMS):
  评分组件: r20池filter / pump_up排序(ratio_s5) / pump_down过滤 / 行业cap / 双轨sizing
  V7c 铁律 (研究 book 中存在的): pyr_velocity 力竭过滤 / 行业60d动量排除
  注: 生产推理铁律"双静默(±1%)"与"非僵尸"不在 t005 研究池 harness 中 (推理时过滤, 非池构造),
      故其 book 层归因不在本 task 范围 (诚实标注, 不杜撰消融)。

成本: 基线滑点 0.10%/边 (BT-002 同档, 冻结 R01); 真实成本 ON。
评估: 全 history (21 月) 过引擎; 净 Sharpe/年化/maxDD/月换手; 分 regime 月度超额 (R11)。
checkpoint (R08): 评分按月落盘 (scored_by_month/), 已完成跳过 → 可断点续跑。
ST 源头排除 (R06); 生产线只读未碰 (R05); 大缓存 research/cache/bt003/ gitignored (R04)。
描述性归因 = 合法完成 (R02), 非 ship gate (BT-004 才是)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT / "research" / "backtest"))

from engine import PortfolioBacktester, CostModel
from train_v15_refresh import load_window
from walk_forward_validation import (compute_r20_label, compute_ind_mom,
                                     apply_cap, industry_alloc)
from t005_walk_forward_gate import TEST_MONTHS, EPS
# 池/sizing 参数 (与 t005 build_dual 逐字一致, 冻结)
from t005_walk_forward_gate import (P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS,
                                    A_PCT, B_PCT, MAX_A, MAX_B, PUMP_DOWN_THRESHOLD)
# BT-002 复用: 价表加载 / 目标构建 / 净值统计 / regime
import run_bt002 as bt2

PROD = ROOT / "output" / "production"
WF_MODELS = ROOT / "research" / "cache" / "t005_wf_models"
CACHE = ROOT / "research" / "cache" / "bt003"
CACHE.mkdir(parents=True, exist_ok=True)
SCORED_DIR = CACHE / "scored_by_month"
SCORED_DIR.mkdir(parents=True, exist_ok=True)
SCORED = CACHE / "scored_daily.parquet"
INDMOM = CACHE / "ind_mom.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "BT-003.json"

DATA_START, DATA_END = "20240901", "20260601"
BASELINE_SLIP = 0.0010      # 与 BT-002 基线同档 (冻结 R01)

# leave-one-out arms: baseline + 7 单组件消融。flags 描述"保留哪些组件"。
ARMS = {
    "baseline":         dict(),                       # 完整 V12.31
    "no_r20_pool":      dict(r20_pool=False),         # 评分: 去 r20 top5% filter
    "no_pump_up_sort":  dict(pump_up_sort=False),     # 评分: 排序 ratio_s5 -> pred_r20
    "no_pump_down":     dict(pump_down=False),        # 评分: 去跌启动子过滤
    "no_industry_cap":  dict(industry_cap=False),     # 评分: 去行业 cap
    "no_dual_track":    dict(dual_track=False),        # 评分: 双轨 -> 单轨等权
    "no_pyr_velocity":  dict(pyr_velocity=False),     # 铁律: 去力竭速度过滤
    "no_ind_mom_excl":  dict(ind_mom_excl=False),     # 铁律: 去行业动量排除
}
ARM_KIND = {
    "no_r20_pool": "评分组件", "no_pump_up_sort": "评分组件", "no_pump_down": "评分组件",
    "no_industry_cap": "评分组件", "no_dual_track": "评分组件",
    "no_pyr_velocity": "V7c铁律", "no_ind_mom_excl": "V7c铁律",
}


# ───────────────────────── 评分 (复用缓存模型, 不重训; 月度 checkpoint) ─────────────────────────
def score_daily() -> pd.DataFrame:
    done = {p.stem for p in SCORED_DIR.glob("*.parquet")}
    todo = [m for m in TEST_MONTHS if m not in done]
    if not todo and SCORED.exists():
        df = pd.read_parquet(SCORED)
        df["trade_date"] = df["trade_date"].astype(str)
        return df
    print(f"[score] 已完成 {len(done)} 月, 待评分 {len(todo)} 月", flush=True)

    meta_p = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta_p["feature_cols"]
    meta_r20 = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    r20_fc = meta_r20["feature_cols"]
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))

    if todo:
        print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除, +mfk) ...", flush=True)
        daily = load_window(DATA_START, DATA_END, with_mfk=True)
        daily["trade_date"] = daily["trade_date"].astype(str)
        for c in set(fc) | set(r20_fc):
            if c not in daily.columns:
                daily[c] = 0.0
        daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        r20_lab = compute_r20_label()
        r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)

        keep = ["ts_code", "trade_date", "industry", "pred_r20", "ratio_s5",
                "pump_down_s5", "pyr_velocity_20_60", "r20_fresh", "month"]
        for m_ in todo:
            s5file = WF_MODELS / m_ / "pump_scale_5" / "classifier.txt"
            if not s5file.exists():
                print(f"  {m_}: 缺缓存 s5 模型, 跳过", flush=True)
                continue
            df = daily[daily["trade_date"].str.startswith(m_)].copy()
            if len(df) < 100:
                continue
            b5 = lgb.Booster(model_str=s5file.read_text(encoding="utf-8"))
            Xf = df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
            proba = b5.predict(Xf)
            df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
            df["pump_down_s5"] = proba[:, 1]
            Xr = df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
            df["pred_r20"] = b20.predict(Xr)
            df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
            df["month"] = m_
            if "pyr_velocity_20_60" not in df.columns:
                df["pyr_velocity_20_60"] = np.nan
            df[keep].to_parquet(SCORED_DIR / f"{m_}.parquet", index=False)
            print(f"  {m_}: {len(df):,} 行 评分落盘 (累计 {time.time()-_t0:.0f}s)", flush=True)
            del df, b5
            gc.collect()
        del daily
        gc.collect()

    parts = [pd.read_parquet(p) for p in sorted(SCORED_DIR.glob("*.parquet"))]
    alld = pd.concat(parts, ignore_index=True)
    alld = alld[alld["month"].isin(TEST_MONTHS)].reset_index(drop=True)
    alld["trade_date"] = alld["trade_date"].astype(str)
    alld.to_parquet(SCORED, index=False)
    print(f"[score] 合并 -> {len(alld):,} 行 / {alld['trade_date'].nunique()} 日 / {alld['month'].nunique()} 月", flush=True)
    return alld


# ───────────────────────── 参数化池构造 (leave-one-out) ─────────────────────────
def build_book(daily: pd.DataFrame, ind_mom: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """复刻 t005 build_dual, 但可单组件 leave-one-out。flags 缺省=保留 (= baseline)。"""
    r20_pool = flags.get("r20_pool", True)
    pump_up_sort = flags.get("pump_up_sort", True)
    pump_down = flags.get("pump_down", True)
    industry_cap = flags.get("industry_cap", True)
    dual_track = flags.get("dual_track", True)
    pyr_velocity = flags.get("pyr_velocity", True)
    ind_mom_excl = flags.get("ind_mom_excl", True)

    sort_col = "ratio_s5" if pump_up_sort else "pred_r20"
    cap_in = CAP_IN if industry_cap else 1.0
    cap_cross = CAP_CROSS if industry_cap else 1.0

    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 100:
            continue
        g = g.copy()
        if ind_mom_excl:
            ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
            g = g[ind_ok]
        if len(g) < 50:
            continue
        g["r20_rank"] = g["pred_r20"].rank(pct=True, method="first")
        if r20_pool:
            m_buy = g["r20_rank"] >= (1 - P_BUY)
        else:
            m_buy = pd.Series(True, index=g.index)
        if pyr_velocity and g["pyr_velocity_20_60"].notna().any():
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0:
            continue
        if pump_down:
            v7c = v7c[v7c["pump_down_s5"] < PUMP_DOWN_THRESHOLD]
        if len(v7c) == 0:
            continue
        v7c = v7c.sort_values(sort_col, ascending=False)

        if dual_track:
            a_pool = apply_cap(v7c.head(MAX_A).copy(), A_PCT, cap_in, sort_col=sort_col)
            a_ind = industry_alloc(a_pool, A_PCT)
            b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
            b_pool = apply_cap(b_cand.head(MAX_B).copy(), B_PCT, cap_in, sort_col=sort_col)
            b_pool = apply_cap(b_pool, B_PCT, cap_cross, prior=a_ind, sort_col=sort_col)
            segs = [(a_pool, A_PCT), (b_pool, B_PCT)]
        else:
            # 单轨等权: top (MAX_A+MAX_B) 等权, 总仓位 = A_PCT+B_PCT (与双轨同暴露)
            s_pool = apply_cap(v7c.head(MAX_A + MAX_B).copy(), A_PCT + B_PCT, cap_in, sort_col=sort_col)
            segs = [(s_pool, A_PCT + B_PCT)]

        for pool, alloc in segs:
            if len(pool) == 0:
                continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"trade_date": d_, "ts_code": row["ts_code"],
                             "industry": row.get("industry", ""), "alloc_pct": per})
    return pd.DataFrame(rows)


# ───────────────────────── 单 arm 过引擎 ─────────────────────────
def run_arm(holds: pd.DataFrame, px: pd.DataFrame, cal: list) -> dict:
    tgt = bt2.build_dual_alloc_targets(holds, cal)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    res = bt.run(tgt, start=holds["trade_date"].min())
    summ = res.summary()
    nav = res.timeseries["nav"]
    return {"summary": summ, "nav": nav, "n_rebal": len(tgt)}


_t0 = time.time()


def main():
    print("\n=== BT-003: 因子/铁律 leave-one-out 归因 (谁拖后腿) ===\n", flush=True)

    scored = score_daily()
    print(f"[scored] {len(scored):,} 行 / {scored['trade_date'].nunique()} 日 / {scored['month'].nunique()} 月", flush=True)

    if INDMOM.exists():
        ind_mom = pd.read_parquet(INDMOM)
        ind_mom["trade_date"] = ind_mom["trade_date"].astype(str)
    else:
        print("[ind_mom] 行业 60d 动量 rank ...", flush=True)
        ind_mom = compute_ind_mom(scored)
        ind_mom["trade_date"] = ind_mom["trade_date"].astype(str)
        ind_mom.to_parquet(INDMOM, index=False)

    start = scored["trade_date"].min()
    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = bt2.load_prices(start)
    cal = sorted(px["trade_date"].unique())
    print(f"[prices] {len(px):,} 行 / {px['ts_code'].nunique()} 股 / {len(cal)} 交易日", flush=True)

    # regime (R11)
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())

    # ── 跑各 arm ──
    print("\n[arms] leave-one-out 消融 (基线滑点 0.10%, 真实成本) ...", flush=True)
    arm_res = {}
    arm_monthly = {}
    for arm, flags in ARMS.items():
        holds = build_book(scored, ind_mom, flags)
        r = run_arm(holds, px, cal)
        s = r["summary"]
        arm_res[arm] = s
        arm_monthly[arm] = bt2.monthly_returns(r["nav"])
        kind = ARM_KIND.get(arm, "baseline")
        print(f"  [{kind:8s}] {arm:16s}: 年化 {s['ann_return']:+.1%}  净Sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s['max_drawdown']:+.1%}  月换手 {s['avg_monthly_turnover']:.2f}  "
              f"持仓 {s['avg_n_positions']:.0f}  (累计 {time.time()-_t0:.0f}s)", flush=True)

    base = arm_res["baseline"]

    # ── Δ 归因表 (vs baseline) + regime ──
    print("\n[attribution] Δ vs 完整 V12.31 baseline (负 Δ = 该组件正贡献; 正 Δ = 拖后腿) ...", flush=True)
    base_monthly = arm_monthly["baseline"].rename("base")
    attribution = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        s = arm_res[arm]
        d_sharpe = round(s["sharpe"] - base["sharpe"], 3)
        d_ann = round(s["ann_return"] - base["ann_return"], 4)
        d_dd = round(s["max_drawdown"] - base["max_drawdown"], 4)

        # regime 分解: 月度收益差 (arm - baseline), 按 regime 取均
        m = pd.DataFrame({"arm": arm_monthly[arm], "base": base_monthly}).dropna()
        m["regime"] = m.index.map(month_regime)
        m["d_ret_pp"] = (m["arm"] - m["base"]) * 100
        reg_delta = {rg: round(float(g["d_ret_pp"].mean()), 3)
                     for rg, g in m.groupby("regime")}

        # 完整组件的贡献 = -(消融 Δ): 消融后 Sharpe 降 (d_sharpe<0) => 组件正贡献
        contribution = "正贡献" if d_sharpe < -0.02 else ("拖后腿" if d_sharpe > 0.02 else "中性")
        attribution[arm] = {
            "kind": ARM_KIND.get(arm, "?"),
            "ablated_ann_return": round(s["ann_return"], 4),
            "ablated_net_sharpe": round(s["sharpe"], 3),
            "ablated_max_drawdown": round(s["max_drawdown"], 4),
            "ablated_monthly_turnover": round(s["avg_monthly_turnover"], 3),
            "ablated_n_positions": round(s["avg_n_positions"], 1),
            "delta_net_sharpe": d_sharpe,
            "delta_ann_return": d_ann,
            "delta_max_drawdown": d_dd,
            "component_contribution": contribution,
            "regime_delta_ret_pp": reg_delta,   # arm - baseline 月均 (负=该组件在此 regime 有用)
        }
        rg_str = "  ".join(f"{k}:{v:+.2f}" for k, v in reg_delta.items())
        print(f"  {arm:16s} [{ARM_KIND.get(arm,'?'):8s}]: ΔSharpe {d_sharpe:+.2f}  Δ年化 {d_ann:+.1%}  "
              f"ΔmaxDD {d_dd:+.1%}  => 组件 {contribution}  | regimeΔ(arm-base) {rg_str}", flush=True)

    # ── 排序: 谁拖后腿 (消融后 Sharpe 反升最多 = 拖后腿最重) ──
    laggards = sorted(attribution.items(), key=lambda kv: -kv[1]["delta_net_sharpe"])
    core = sorted(attribution.items(), key=lambda kv: kv[1]["delta_net_sharpe"])

    results = {
        "config": {"baseline_slippage": BASELINE_SLIP, "rebalance_every_tdays": bt2.REBAL_EVERY,
                   "total_invested": bt2.TOTAL, "period": [start, cal[-1]],
                   "arms": list(ARMS.keys())},
        "baseline_summary": {k: (round(base[k], 4) if isinstance(base[k], float) else base[k])
                             for k in base},
        "attribution": attribution,
        "ranking": {
            "biggest_laggard": laggards[0][0] if laggards else None,
            "biggest_laggard_delta_sharpe": laggards[0][1]["delta_net_sharpe"] if laggards else None,
            "most_core": core[0][0] if core else None,
            "most_core_delta_sharpe": core[0][1]["delta_net_sharpe"] if core else None,
        },
        "caveat": "Δ 是受控 leave-one-out 差分 (各 arm 共享同一 r20 池 lookahead + 同成本 + 同引擎), "
                  "共模偏差相消, 故归因 (谁正贡献/谁拖后腿) 比 BT-002 绝对量级更可信; "
                  "但绝对 Sharpe 仍含共模 lookahead, 非可交易声明。"
                  "双静默/非僵尸生产推理铁律不在 t005 研究池 harness, 其 book 归因不在本 task 范围。",
    }
    (CACHE / "bt003_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ── 结论句 ──
    pos = [a for a, v in attribution.items() if v["component_contribution"] == "正贡献"]
    neg = [a for a, v in attribution.items() if v["component_contribution"] == "拖后腿"]
    neu = [a for a, v in attribution.items() if v["component_contribution"] == "中性"]
    lag_name, lag_v = (laggards[0] if laggards else (None, {}))
    core_name, core_v = (core[0] if core else (None, {}))

    conclusion = (
        f"对 V12.31 研究 book 的 7 个组件逐个 leave-one-out 过引擎 (基线滑点 {BASELINE_SLIP*100:.2f}%, "
        f"{base['n_months']} 月)。完整基线: 净Sharpe {base['sharpe']:+.2f} 年化 {base['ann_return']:+.1%} "
        f"maxDD {base['max_drawdown']:+.1%}。"
        f"正贡献 (去掉伤 book): {', '.join(pos) if pos else '无'}; "
        f"拖后腿 (去掉反而改善 book): {', '.join(neg) if neg else '无'}; "
        f"中性: {', '.join(neu) if neu else '无'}。"
        f"最核心 = {core_name} (去掉 ΔSharpe {core_v.get('delta_net_sharpe'):+.2f}); "
        f"最拖后腿 = {lag_name} (去掉 ΔSharpe {lag_v.get('delta_net_sharpe'):+.2f})。"
        f"Δ 为受控差分 (共模 lookahead 相消, 归因比绝对量级可信); 生产线冻结, 仅作 future opt-in 候选依据 (SIGN-R02/R05)。"
    )
    VERDICT.write_text(json.dumps({
        "id": "BT-003", "status": "built", "conclusion": conclusion,
        "baseline_metrics": {
            "net_sharpe": round(base["sharpe"], 3),
            "ann_return": round(base["ann_return"], 4),
            "max_drawdown": round(base["max_drawdown"], 4),
            "avg_monthly_turnover": round(base["avg_monthly_turnover"], 3),
            "n_months": base["n_months"],
        },
        "attribution_table": {a: {
            "kind": v["kind"],
            "delta_net_sharpe": v["delta_net_sharpe"],
            "delta_ann_return": v["delta_ann_return"],
            "delta_max_drawdown": v["delta_max_drawdown"],
            "contribution": v["component_contribution"],
            "regime_delta_ret_pp": v["regime_delta_ret_pp"],
        } for a, v in attribution.items()},
        "ranking": results["ranking"],
        "positive_contributors": pos, "laggards": neg, "neutral": neu,
        "cost_model": CostModel(slippage=BASELINE_SLIP, enabled=True).describe(),
        "caveat": results["caveat"],
        "artifacts": [
            "research/backtest/run_bt003.py",
            "research/cache/bt003/scored_daily.parquet",
            "research/cache/bt003/bt003_results.json",
        ],
        "note": "leave-one-out 受控差分 (R03 看相对不看绝对); ST 源头排除 (R06); regime 分层必带 (R11); "
                "成本/池参数逐字冻结复刻 t005 build_dual (R01); 复用缓存 s5+r20 模型不重训, 生产只读 (R05); "
                "评分按月 checkpoint (R08); 大缓存 gitignored, 无 features 写入 (R04); 描述性归因非 ship gate (R02)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-_t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
