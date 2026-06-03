# -*- coding: utf-8 -*-
"""入场选择细化: 健康回调 vs 破位飞刀 — V12.31 池加"趋势质量"过滤 vs 原池。

动机 (承三连否 [[project_three_rejects_meanreversion_meta_0603]]): 动态窗口/动量overlay/止损
全 REJECT, 且后两个反指——它们都在**出场/择时**侧, 与均值回归 edge 反向。本实验换到**入场选择**
侧 (唯一未碰、不与 edge 打架的角度): 同样买回调, 区分"上升趋势中的 dip (健康)" vs "下降趋势中的
破位 (飞刀)", **只买前者**。这是 selection 细化, 不像止损砍在底部。

apples-to-apples: 两臂唯一差异 = 候选池是否加"趋势质量"过滤, 其余 (ratio_s5 排序/行业 cap/
仓位/持有 20d) 完全一致。
  - base 臂:  现 V7c 池 (= V12.31)
  - filt 臂:  池构造前先要求 close[entry] >= MA60[entry] (在 60 日均线上方 = 健康趋势)
  两臂持仓均持满 20 交易日 (close→close); Δ_month = filt - base = 趋势过滤净效应。

事前注册 gate (冻结, SIGN-R01; 按 regime 分层 SIGN-R11):
  PASS 当且仅当 4 条全满足:
    (1) 不显著伤平均: 全期 mean(Δ) >= -0.10pp
    (2) 砍飞刀尾部:   动量-regime 月 mean(Δ) >= +0.30pp
    (3) 最差月改善:   worst(filt) - worst(base) >= +0.30pp
    (4) 不新增灾难月: disaster(filt) <= disaster(base)
  REJECT 是合法完成 (SIGN-R02)。MA20 变体仅描述, 不作 gate。

诚实限制: 同 rg_stoploss — 窗 202410-202604 不含实盘血洗 0508-0603 (7月数据后另注册复检),
  202602 极端动量月作窗内代理。复用 t005 缓存 s5 模型, 不碰生产文件 (SIGN-R05)。
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
from walk_forward_validation import compute_r20_label, compute_ind_mom
from t004_multiscale_gate import load_daily, EPS
from t005_walk_forward_gate import build_dual, TEST_MONTHS, DATA_START, DATA_END

CACHE = ROOT / "research" / "cache"
WF_MODELS = CACHE / "t005_wf_models"
MONTHLY = CACHE / "rg_trendq_monthly.csv"
RESULTS = CACHE / "rg_trendq_results.json"
VERDICT = ROOT / "research" / "verdicts" / "RG-TRENDQ.json"
PROD = ROOT / "output" / "production"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"

HOLD = 20
MA_PRIMARY = 60
MA_LIST = [20, 60]


def full_hold_returns(holds, close_wide, td_list, td_idx) -> pd.DataFrame:
    out = []
    for _, h in holds.iterrows():
        ts, ed, w = h["ts_code"], h["entry_date"], h["alloc_pct"]
        i = td_idx.get(ed)
        if i is None or ts not in close_wide.columns:
            continue
        path = close_wide[ts].to_numpy()
        buy = path[i]
        j = min(i + HOLD, len(td_list) - 1)
        exitp = path[j]
        if not (np.isfinite(buy) and buy > 0 and np.isfinite(exitp)):
            continue
        out.append({"entry_date": ed, "w": w, "ret": (exitp / buy - 1.0) * 100})
    return pd.DataFrame(out)


def month_port(sim) -> float:
    if sim is None or sim.empty:
        return np.nan
    g = sim.groupby("entry_date").apply(lambda d: np.average(d["ret"].clip(-30, 30), weights=d["w"]))
    return float(g.mean())


def evaluate_month(tm, daily_full, fc, r20_fc, b20, ind_mom, r20_lab,
                   close_wide, ma_long, td_list, td_idx) -> dict | None:
    s5f = WF_MODELS / tm / "pump_scale_5" / "classifier.txt"
    if not s5f.exists():
        return None
    b5 = lgb.Booster(model_str=s5f.read_text(encoding="utf-8"))
    df = daily_full[daily_full["trade_date"].str.startswith(tm)].copy()
    if len(df) < 100:
        return None
    for c in fc + r20_fc:
        if c not in df.columns:
            df[c] = 0.0
    proba = b5.predict(df[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0))
    df["ratio_s5"] = proba[:, 2] / (proba[:, 1] + EPS)
    df["pump_down_s5"] = proba[:, 1]
    df["pred_r20"] = b20.predict(df[r20_fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0))
    df = df.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
    df = df.merge(ma_long, on=["ts_code", "trade_date"], how="left")

    row = {"test_month": tm}
    # base 臂
    hb = build_dual(df, ind_mom, sort_col="ratio_s5")
    row["base"] = month_port(full_hold_returns(hb, close_wide, td_list, td_idx)) if hb is not None and not hb.empty else np.nan
    row["base_n"] = 0 if hb is None else len(hb)
    # filt 臂 (各 MA 档)
    for M in MA_LIST:
        dff = df[df["close"] >= df[f"ma{M}"]].copy()
        hf = build_dual(dff, ind_mom, sort_col="ratio_s5")
        row[f"filt{M}"] = month_port(full_hold_returns(hf, close_wide, td_list, td_idx)) if hf is not None and not hf.empty else np.nan
        row[f"filt{M}_n"] = 0 if hf is None else len(hf)
    print(f"  {tm}: base {row['base']:+.3f}(n{row['base_n']})  "
          + "  ".join(f"ma{M} {row[f'filt{M}']:+.3f}(n{row[f'filt{M}_n']})" for M in MA_LIST), flush=True)
    del df, b5
    gc.collect()
    return row


def main():
    t0 = time.time()
    print("\n=== 趋势质量过滤 (健康回调 vs 破位飞刀) vs V12.31 ===\n", flush=True)
    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json").read_text(encoding="utf-8"))
    fc = meta["feature_cols"]; ind_map = meta.get("industry_map", {})

    print(f"[data] load_window {DATA_START}-{DATA_END} ...", flush=True)
    daily_full = load_window(DATA_START, DATA_END, with_mfk=True)
    daily_full["trade_date"] = daily_full["trade_date"].astype(str)
    if ind_map:
        daily_full["industry_id"] = (daily_full["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily_full.columns:
            daily_full[c] = 0.0
    daily_full = daily_full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[data] {len(daily_full):,} 行", flush=True)

    px = load_daily()
    close_wide = px.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    td_list = list(close_wide.index); td_idx = {d: i for i, d in enumerate(td_list)}
    # MA 长表 (close + ma20 + ma60)
    print("[ma] 计算 MA 长表 ...", flush=True)
    ma_parts = {"close": close_wide}
    for M in MA_LIST:
        ma_parts[f"ma{M}"] = close_wide.rolling(M, min_periods=M // 2).mean()
    ma_long = (pd.concat({k: v.stack() for k, v in ma_parts.items()}, axis=1)
               .reset_index().rename(columns={"level_0": "trade_date", "level_1": "ts_code"}))
    ma_long.columns = ["trade_date", "ts_code"] + list(ma_parts.keys())
    ma_long["trade_date"] = ma_long["trade_date"].astype(str)

    r20_lab = compute_r20_label(); r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    ind_mom = compute_ind_mom(daily_full)
    b20 = lgb.Booster(model_str=(PROD / "r20_v16_long_nost" / "classifier.txt").read_text(encoding="utf-8"))
    r20_fc = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))["feature_cols"]
    print(f"[prep] {time.time()-t0:.0f}s\n", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records"); done = set(prev["test_month"].astype(str))
    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = evaluate_month(m_, daily_full, fc, r20_fc, b20, ind_mom, r20_lab,
                           close_wide, ma_long, td_list, td_idx)
        if r:
            results.append(r); pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].sort_values("test_month").reset_index(drop=True)
    rt = pd.read_parquet(REGIME)[["trade_date", "regime"]].copy()
    rt["test_month"] = rt["trade_date"].astype(str).str[:6]
    mon_reg = rt.groupby("test_month")["regime"].agg(lambda s: s.value_counts().idxmax()).rename("regime").reset_index()
    df = df.merge(mon_reg, on="test_month", how="left")

    fp = f"filt{MA_PRIMARY}"
    df["delta"] = df[fp] - df["base"]

    def summ(col):
        a = df[col].dropna()
        return {"mean": float(a.mean()), "worst": float(a.min()),
                "worst_id": str(df.loc[df[col].idxmin(), "test_month"]),
                "disaster": int((a < -1).sum()), "pos_ratio": float((a > 0).mean())}
    base_s, filt_s = summ("base"), summ(fp)
    mom = df[df["regime"] == "momentum"]; rev = df[df["regime"] == "reversal"]

    c1 = float(df["delta"].mean()) >= -0.10
    c2 = (float(mom["delta"].mean()) if len(mom) else -9) >= 0.30
    c3 = (filt_s["worst"] - base_s["worst"]) >= 0.30
    c4 = filt_s["disaster"] <= base_s["disaster"]
    conds = {"不伤平均Δ>=-0.1pp": bool(c1), "动量月Δ>=+0.3pp": bool(c2),
             "最差月改善>=+0.3pp": bool(c3), "不新增灾难月": bool(c4)}
    status = "PASS" if all(conds.values()) else "REJECT"

    res = {"window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
           "ma_primary": MA_PRIMARY, "base": base_s, fp: filt_s,
           "delta_mean_all": float(df["delta"].mean()),
           "delta_mean_momentum": float(mom["delta"].mean()) if len(mom) else None,
           "delta_mean_reversal": float(rev["delta"].mean()) if len(rev) else None,
           "n_momentum_months": int(len(mom)), "n_reversal_months": int(len(rev)),
           "other_ma": {f"ma{M}": {"mean": float(df[f'filt{M}'].mean()),
                                   "delta_mean": float((df[f'filt{M}']-df['base']).mean())} for M in MA_LIST},
           "gate_conditions": conds, "gate_status": status,
           "note": "apples 两臂仅候选池趋势过滤不同; 窗不含实盘血洗 (7月复检), 202602 代理。"}
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n\n=== 趋势质量过滤汇总 ===\n", flush=True)
    print(f"  月数 {len(df)}  动量月 {len(mom)}  反转月 {len(rev)}  (MA{MA_PRIMARY})", flush=True)
    print(f"  {'指标':16s} {'base(V12.31)':>14s} {'filt(健康回调)':>16s}", flush=True)
    print(f"  {'月化均值(pp)':16s} {base_s['mean']:>+14.3f} {filt_s['mean']:>+16.3f}", flush=True)
    print(f"  {'最差月(pp)':16s} {base_s['worst']:>+14.3f} {filt_s['worst']:>+16.3f}  ({base_s['worst_id']}/{filt_s['worst_id']})", flush=True)
    print(f"  {'灾难月数(<-1)':16s} {base_s['disaster']:>14d} {filt_s['disaster']:>16d}", flush=True)
    print(f"\n  Δ(filt-base): 全期 {df['delta'].mean():+.3f}  "
          f"动量月 {mom['delta'].mean() if len(mom) else float('nan'):+.3f}  "
          f"反转月 {rev['delta'].mean() if len(rev) else float('nan'):+.3f}", flush=True)
    print(f"  MA20 变体: Δ {(df['filt20']-df['base']).mean():+.3f}", flush=True)
    print(f"\n  gate: {conds}\n  >>> verdict = {status} <<<\n", flush=True)
    for _, r in df.iterrows():
        print(f"    {r['test_month']} [{str(r.get('regime',''))[:8]:8s}] base {r['base']:+.3f}  "
              f"{fp} {r[fp]:+.3f}  Δ{r['delta']:+.3f}", flush=True)

    concl = (f"{status}: V12.31 池加 close>=MA{MA_PRIMARY} 健康回调过滤, 19 月 walk-forward 全期 "
             f"Δ={df['delta'].mean():+.3f}pp 动量月 Δ={mom['delta'].mean() if len(mom) else float('nan'):+.3f} "
             f"最差月 {base_s['worst']:+.2f}→{filt_s['worst']:+.2f}; "
             + ("4 条 gate 全过 → selection 侧趋势过滤有真增益。" if status == "PASS"
                else f"gate 未过={[k for k,v in conds.items() if not v]}。窗内不含实盘血洗 (7月复检)。"))
    VERDICT.write_text(json.dumps({
        "id": "RG-TRENDQ", "status": status, "conclusion": concl,
        "metrics": {"delta_mean_all_pp": round(float(df["delta"].mean()), 3),
                    "delta_mean_momentum_pp": round(float(mom["delta"].mean()), 3) if len(mom) else None,
                    "worst_base_pp": round(base_s["worst"], 3), "worst_filt_pp": round(filt_s["worst"], 3),
                    "disaster_base": base_s["disaster"], "disaster_filt": filt_s["disaster"]},
        "gate_conditions": conds, "n_months": int(len(df)), "window": [TEST_MONTHS[0], TEST_MONTHS[-1]],
        "artifacts": ["research/cache/rg_trendq_monthly.csv", "research/cache/rg_trendq_results.json"],
        "note": "入场选择细化 (不与均值回归 edge 反向); apples 两臂仅候选池趋势过滤不同。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] verdict={status} 耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
