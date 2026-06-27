# -*- coding: utf-8 -*-
"""止损 / 尾risk 叠加回测 — V12.31 picks 加止损 vs 全程持有 (apples-to-apples)。

动机 (承 RG-002): 实盘亏损死法 = 接飞刀 (买 past_r5<0 回调, 继续跌)。RG-002 否定了
"因果动量 regime 预测" 的修法 (动量态之后平均均值回归), 但**止损是事后纪律, 不需要预测
regime** —— 直接打在机理上: 持仓后破止损就砍。本脚本测它在 19 月 walk-forward 上是否
以可接受的平均代价砍掉左尾 (尤其 202602 极端动量月)。

apples-to-apples: 两臂同 picks (V12.31 = 池内 ratio_s5 排序), 唯一差异 = 出场规则。
  - full 臂:  持有 20 交易日 (close[entry] → close[entry+20])
  - stop 臂:  逐日 close, 首日 close 相对 entry 跌破 -S → 当日 close 出场; 否则持满 20d
  月度组合收益 = alloc 加权 picks 收益; Δ_month = stop - full = 止损净效应 (市场共模相消)。

事前注册 gate (冻结, SIGN-R01; 主变体 STOP_PRIMARY=-12% close 触发):
  PASS 当且仅当 4 条全满足:
    (1) 最差月改善:  worst(stop) - worst(full) >= +0.50pp
    (2) 动量月有效:  动量-regime 月 mean(Δ) >= +0.30pp     (血洗发生处)
    (3) 平均代价可接受: 全期 mean(Δ) >= -0.30pp
    (4) 不新增灾难月: disaster(stop) <= disaster(full)        (灾难=月收益<-1pp)
  REJECT 是合法完成 (SIGN-R02), 禁止同段 OOS 重调。按 regime 分层评估 (SIGN-R11)。
  其余 stop 档 {-8,-15} 仅描述, 不作 gate (避免择优 p-hack)。

诚实限制: 回测窗 202410-202604 不含实盘血洗窗 0508-0603 (需 6 月下旬前向数据, 未到);
  202602 (RG-001 判极端动量月) 是窗内代理。完整复检该 episode 须 2026-07 后另注册 gate。

复用 t005 缓存月度 s5 模型 (免重训) + load_daily 路径。不碰生产文件 (SIGN-R05)。
checkpoint: 逐月 append research/cache/rg_stoploss_monthly.csv, 已完成跳过 (SIGN-R08)。
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
    compute_r20_label, compute_ind_mom, get_train_end_for_test_month, get_train_start,
)
from t004_multiscale_gate import load_daily, EPS
from t005_walk_forward_gate import build_dual, TEST_MONTHS, TRAIN_LOOKBACK_MONTHS, DATA_START, DATA_END

CACHE = ROOT / "research" / "cache"
WF_MODELS = CACHE / "t005_wf_models"
MONTHLY = CACHE / "rg_stoploss_monthly.csv"
RESULTS = CACHE / "rg_stoploss_results.json"
VERDICT = ROOT / "research" / "verdicts" / "RG-STOP.json"
PROD = ROOT / "output" / "production"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"

HOLD = 20
STOP_LEVELS = [8.0, 12.0, 15.0]      # 百分比
STOP_PRIMARY = 12.0


def simulate_stops(holds: pd.DataFrame, close_wide: pd.DataFrame,
                   td_list: list, td_idx: dict) -> pd.DataFrame:
    """对每个持仓算 full-hold 与各止损档收益 (close 基准, 收盘触发)。"""
    out = []
    for _, h in holds.iterrows():
        ts, ed, w = h["ts_code"], h["entry_date"], h["alloc_pct"]
        i = td_idx.get(ed)
        if i is None or ts not in close_wide.columns:
            continue
        path = close_wide[ts].to_numpy()
        buy = path[i]
        if not np.isfinite(buy) or buy <= 0:
            continue
        j_end = min(i + HOLD, len(td_list) - 1)
        seg = path[i + 1:j_end + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            continue
        rets = seg / buy - 1.0
        full = float(rets[-1]) * 100
        rec = {"entry_date": ed, "ts_code": ts, "w": w, "full": full}
        for S in STOP_LEVELS:
            hit = np.where(rets <= -S / 100)[0]
            rec[f"stop{int(S)}"] = float(rets[hit[0]]) * 100 if len(hit) else full
        out.append(rec)
    return pd.DataFrame(out)


def month_port(sim: pd.DataFrame, col: str) -> float:
    """alloc 加权月度组合收益 (按 entry_date 等权再跨日平均)."""
    if sim.empty:
        return np.nan
    g = sim.groupby("entry_date").apply(
        lambda d: np.average(d[col].clip(-30, 30), weights=d["w"]))
    return float(g.mean())


def evaluate_month(test_month, daily_full, fc, r20_fc, b20, ind_mom,
                   r20_lab, close_wide, td_list, td_idx) -> dict | None:
    s5_file = WF_MODELS / test_month / "pump_scale_5" / "classifier.txt"
    if not s5_file.exists():
        print(f"  {test_month}: 缺缓存 s5 模型, 跳过", flush=True)
        return None
    b5 = lgb.Booster(model_str=s5_file.read_text(encoding="utf-8"))

    df_test = daily_full[daily_full["trade_date"].str.startswith(test_month)].copy()
    if len(df_test) < 100:
        return None
    for c in fc + r20_fc:
        if c not in df_test.columns:
            df_test[c] = 0.0

    Xf = df_test[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = b5.predict(Xf)
    df_test["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
    df_test["pump_down_s5"] = proba[:, 1]
    Xr = df_test[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    df_test["pred_r20"] = b20.predict(Xr)
    df_test = df_test.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

    holds = build_dual(df_test, ind_mom, sort_col="ratio_s5")
    if holds is None or holds.empty:
        return None
    sim = simulate_stops(holds, close_wide, td_list, td_idx)
    if sim.empty:
        return None

    row = {"test_month": test_month, "n_pos": int(len(sim)),
           "full": month_port(sim, "full")}
    for S in STOP_LEVELS:
        row[f"stop{int(S)}"] = month_port(sim, f"stop{int(S)}")
    print(f"  {test_month}: full {row['full']:+.3f}  "
          + "  ".join(f"s{int(S)} {row[f'stop{int(S)}']:+.3f}" for S in STOP_LEVELS)
          + f"  (n={row['n_pos']})", flush=True)
    del df_test, holds, sim, b5
    gc.collect()
    return row


def main():
    t0 = time.time()
    print("\n=== 止损/尾risk 叠加回测 (V12.31 picks + 止损 vs 全程持有) ===\n", flush=True)

    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]; ind_map = meta.get("industry_map", {})

    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 排除, +mfk) ...", flush=True)
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map:
        daily_full["industry_id"] = (daily_full["industry"].fillna("unknown")
                                     .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily_full.columns:
            daily_full[c] = 0.0
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[data] {len(daily_full):,} 行 / {daily_full['ts_code'].nunique()} 股", flush=True)

    print("[price] close 路径透视 ...", flush=True)
    px = load_daily()
    close_wide = px.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    td_list = list(close_wide.index)
    td_idx = {d: i for i, d in enumerate(td_list)}

    r20_lab = compute_r20_label(); r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))
    r20_fc = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))["feature_cols"]
    print(f"[prep] {time.time()-t0:.0f}s\n", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records"); done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = evaluate_month(m_, daily_full, fc, r20_fc, b20, ind_mom, r20_lab,
                           close_wide, td_list, td_idx)
        if r:
            results.append(r)
            pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].sort_values("test_month").reset_index(drop=True)

    # regime 分层 (RG-001 timeline -> 每月主导 regime)
    rt = pd.read_parquet(REGIME)[["trade_date", "regime"]].copy()
    rt["test_month"] = rt["trade_date"].astype(str).str[:6]
    mon_reg = (rt.groupby("test_month")["regime"]
               .agg(lambda s: s.value_counts().idxmax()).rename("regime").reset_index())
    df = df.merge(mon_reg, on="test_month", how="left")

    sp = f"stop{int(STOP_PRIMARY)}"
    df["delta"] = df[sp] - df["full"]

    def summ(col):
        a = df[col]
        return {"mean": float(a.mean()), "worst": float(a.min()),
                "worst_id": str(df.loc[a.idxmin(), "test_month"]),
                "disaster": int((a < -1).sum()), "pos_ratio": float((a > 0).mean())}
    full_s, stop_s = summ("full"), summ(sp)
    mom = df[df["regime"] == "momentum"]
    rev = df[df["regime"] == "reversal"]

    # 事前注册 gate (冻结)
    c1 = (stop_s["worst"] - full_s["worst"]) >= 0.50
    c2 = (float(mom["delta"].mean()) if len(mom) else -9) >= 0.30
    c3 = float(df["delta"].mean()) >= -0.30
    c4 = stop_s["disaster"] <= full_s["disaster"]
    conds = {"worst_month改善>=+0.5pp": bool(c1), "动量月Δ>=+0.3pp": bool(c2),
             "平均代价Δ>=-0.3pp": bool(c3), "不新增灾难月": bool(c4)}
    status = "PASS" if all(conds.values()) else "REJECT"

    res = {"window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
           "stop_primary_pct": STOP_PRIMARY, "hold_days": HOLD,
           "full": full_s, f"{sp}": stop_s,
           "delta_mean_all": float(df["delta"].mean()),
           "delta_mean_momentum": float(mom["delta"].mean()) if len(mom) else None,
           "delta_mean_reversal": float(rev["delta"].mean()) if len(rev) else None,
           "n_momentum_months": int(len(mom)), "n_reversal_months": int(len(rev)),
           "other_stops": {f"stop{int(S)}": {"mean": float(df[f'stop{int(S)}'].mean()),
                                             "worst": float(df[f'stop{int(S)}'].min()),
                                             "delta_mean": float((df[f'stop{int(S)}']-df['full']).mean())}
                           for S in STOP_LEVELS},
           "gate_conditions": conds, "gate_status": status,
           "note": "apples: 两臂同 picks 仅出场规则不同; Δ=止损净效应。窗内不含实盘血洗 0508-0603 "
                   "(需7月数据), 202602 极端动量月作代理。REJECT 合法完成 (SIGN-R02)。"}
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n\n=== 止损回测汇总 ===\n", flush=True)
    print(f"  月数 {len(df)}  动量月 {len(mom)}  反转月 {len(rev)}  (主止损 -{STOP_PRIMARY:.0f}%)", flush=True)
    print(f"  {'指标':16s} {'full持有':>12s} {sp+'止损':>12s}", flush=True)
    print(f"  {'月化均值(pp)':16s} {full_s['mean']:>+12.3f} {stop_s['mean']:>+12.3f}", flush=True)
    print(f"  {'最差月(pp)':16s} {full_s['worst']:>+12.3f} {stop_s['worst']:>+12.3f}  "
          f"({full_s['worst_id']} / {stop_s['worst_id']})", flush=True)
    print(f"  {'灾难月数(<-1)':16s} {full_s['disaster']:>12d} {stop_s['disaster']:>12d}", flush=True)
    print(f"\n  Δ(stop-full): 全期 {df['delta'].mean():+.3f}  "
          f"动量月 {mom['delta'].mean() if len(mom) else float('nan'):+.3f}  "
          f"反转月 {rev['delta'].mean() if len(rev) else float('nan'):+.3f}", flush=True)
    print(f"\n  其余止损档: " + "  ".join(
        f"-{int(S)}%(Δ{(df[f'stop{int(S)}']-df['full']).mean():+.2f})" for S in STOP_LEVELS), flush=True)
    print(f"\n  gate: {conds}\n  >>> verdict = {status} <<<\n", flush=True)
    print("  逐月 (full / stop / regime):", flush=True)
    for _, r in df.iterrows():
        print(f"    {r['test_month']} [{str(r.get('regime',''))[:8]:8s}] "
              f"full {r['full']:+.3f}  {sp} {r[sp]:+.3f}  Δ{r['delta']:+.3f}", flush=True)

    concl = (f"{status}: V12.31 picks 加 -{STOP_PRIMARY:.0f}% 收盘止损, 19 月 walk-forward "
             f"全期 Δ={df['delta'].mean():+.3f}pp 最差月 {full_s['worst']:+.2f}→{stop_s['worst']:+.2f} "
             f"动量月 Δ={mom['delta'].mean() if len(mom) else float('nan'):+.3f}; gate 未过={[k for k,v in conds.items() if not v]}. "
             f"窗内不含实盘血洗窗(7月数据后复检), 202602 代理。") if status == "REJECT" else (
             f"PASS: -{STOP_PRIMARY:.0f}% 止损砍左尾 (最差月 {full_s['worst']:+.2f}→{stop_s['worst']:+.2f}) "
             f"动量月 Δ={mom['delta'].mean():+.3f} 平均代价 {df['delta'].mean():+.3f}pp, 4 条 gate 全过。")
    VERDICT.write_text(json.dumps({
        "id": "RG-STOP", "status": status, "conclusion": concl,
        "metrics": {"delta_mean_all_pp": round(float(df["delta"].mean()), 3),
                    "delta_mean_momentum_pp": round(float(mom["delta"].mean()), 3) if len(mom) else None,
                    "worst_full_pp": round(full_s["worst"], 3), "worst_stop_pp": round(stop_s["worst"], 3),
                    "disaster_full": full_s["disaster"], "disaster_stop": stop_s["disaster"]},
        "gate_conditions": conds, "n_months": int(len(df)),
        "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
        "artifacts": ["research/cache/rg_stoploss_monthly.csv", "research/cache/rg_stoploss_results.json"],
        "note": "止损 tail risk 叠加; apples 两臂同 picks; 窗不含实盘血洗 0508-0603 (7月数据后另注册 gate 复检)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] verdict={status} 耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
