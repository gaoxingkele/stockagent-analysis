# -*- coding: utf-8 -*-
"""MT-004 walk-forward — 梅花轨迹特征 apples-to-apples 增益闸 (Phase-2).

承接 mt004_compare_gate.py: 升级消融 (resid3 扣 线性动量 + 已实现波动率 + 多尺度MA堆叠)
后, B(mhB_upper) 与 C(mhC_move_std) 残差**仍线性存活** → 按 gate 必须跑 walk-forward 定 ship。

为什么还要 walk-forward (resid3 线性存活还不够):
  resid3 只扣了控制集的**线性**投影。mhB_upper=离散MA堆叠卦, mhC_move_std=|涨跌幅|桶的离散度,
  二者与 vol/MA堆叠 是**非线性**关系, 线性 OLS 残差化扣不干净。真正 apples-to-apples =
  让 baseline 用一个能**非线性**吃标准TA的模型 (GBDT), 看加梅花特征后 OOS 组合 α 是否真增。

apples-to-apples 设计 (两臂唯一差异 = 是否含梅花特征):
  baseline 臂: GBDT(r20) 特征 = 标准TA对同一走势的编码 (12 个: 多尺度动量/MA堆叠/已实现波动率/RSI)。
  meihua  臂: GBDT(r20) 特征 = 标准TA (同上) + 全部梅花轨迹特征 (mhB_* 10 + mhC_* 10)。
  两臂同 universe/同日期/同 r20 实现收益/同超参/同 seed → Δα 纯由梅花特征贡献。

walk-forward 协议 (同 T-005 / project_walk_forward_validation_0525):
  测试月 202410-202604 共 19 月; 每月用前 24 月训练 (1 月 gap 防 r20 forward 偷看), 独立重训。
  组合: 每个交易日按 pred_r20 横截面排序, 多头 top 5% 等权, 收益 = 实现 r20; 月 α = 组合 − universe 均值。
  ST 源头已在上游 panel 排除 (SIGN-R06)。分 regime 汇总 (SIGN-R11)。

事前注册 gate (冻结 SIGN-R01; phase2_gate + R11):
  Δα(月化, meihua − baseline) >= +0.30pp  且  Sharpe 不低于 baseline  且  最差月不低于 baseline
  且  动量月 Δα >= 0 (不伤动量) 且 反转月 Δα >= 0 (不伤反转)  → PASS, 否则 REJECT。
  REJECT 是合法完成 (SIGN-R02); 中间 IC 不作数, 只认 walk-forward α (SIGN-R03)。

checkpoint (SIGN-R08): 逐月结果 append research/cache/mt004_wf_monthly.csv, 已完成月跳过。
"""
from __future__ import annotations
import gc, json, sys, time
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache"
FEAT = ROOT / "research" / "features"
MT003_PANEL = CACHE / "mt003_panel.parquet"
CTRL = CACHE / "mt004_controls.parquet"
WF_PANEL = CACHE / "mt004_wf_panel.parquet"
MONTHLY = CACHE / "mt004_wf_monthly.csv"
RESULTS = CACHE / "mt004_wf_results.json"

KEY = ["ts_code", "trade_date"]
HORIZON = 20

STD_TA = ["mom_5", "mom_10", "mom_20", "mom_60", "ma_pos", "rsi_14",
          "vol_20", "atr_20", "r_c_ma5", "r_ma5_ma10", "r_ma10_ma20", "r_ma20_ma60"]
MHB = ["mhB_base_gua", "mhB_mutual_gua", "mhB_changed_gua", "mhB_upper", "mhB_lower",
       "mhB_moving", "mhB_ti_elem", "mhB_yong_elem", "mhB_relation", "mhB_yang_count"]
MHC = ["mhC_cum_base_gua", "mhC_cum_upper", "mhC_cum_lower", "mhC_cum_changed_gua",
       "mhC_move_last", "mhC_cum_yang", "mhC_move_drift", "mhC_move_std",
       "mhC_wuxing_net", "mhC_wuxing_net_recent"]
MH_FEATS = MHB + MHC
BASE_FEATS = STD_TA
MEIHUA_FEATS = STD_TA + MH_FEATS

TEST_MONTHS = ["202410", "202411", "202412",
               "202501", "202502", "202503", "202504", "202505", "202506",
               "202507", "202508", "202509",
               "202510", "202511", "202512",
               "202601", "202602", "202603", "202604"]
TRAIN_LOOKBACK_MONTHS = 24
N_TRAIN = 300_000
SEED = 20260604
TOP_PCT = 0.05
COST_BPS = 35.0 / 10000


def shift_ym(ym: str, months: int) -> str:
    y, m = int(ym[:4]), int(ym[4:6])
    idx = (y * 12 + (m - 1)) + months
    return f"{idx // 12:04d}{idx % 12 + 1:02d}"


def build_wf_panel() -> pd.DataFrame:
    if WF_PANEL.exists():
        print(f"[panel] checkpoint 命中 {WF_PANEL.name}", flush=True)
        return pd.read_parquet(WF_PANEL)
    print("[panel] 构建 WF 面板: mt003 spine + 升级控制集 + 全部 mhB/mhC ...", flush=True)
    c = pd.read_parquet(MT003_PANEL, columns=(
        KEY + ["mom_5", "mom_20", "ma_pos", "rsi_14", "r20", "regime"]))
    ctrl = pd.read_parquet(CTRL)            # vol_20, atr_20, mom_10, mom_60, MA堆叠
    b = pd.read_parquet(FEAT / "meihua_traj_B.parquet")
    cc = pd.read_parquet(FEAT / "meihua_traj_C.parquet")
    df = (c.merge(ctrl, on=KEY, how="inner")
          .merge(b, on=KEY, how="inner")
          .merge(cc, on=KEY, how="inner"))
    df = df[df["r20"].notna()].reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df["ym"] = df["trade_date"].str[:6]
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WF_PANEL, index=False)
    print(f"[panel] -> {WF_PANEL.relative_to(ROOT)} ({len(df):,} 行 × "
          f"{df['ts_code'].nunique()} 股 × {df['trade_date'].nunique()} 日)", flush=True)
    return df


def train_predict(df_train: pd.DataFrame, df_test: pd.DataFrame, feats: list[str],
                  seed: int) -> np.ndarray:
    sub = df_train.dropna(subset=["r20"]).copy()
    if len(sub) > N_TRAIN:
        sub = sub.sample(n=N_TRAIN, random_state=seed).reset_index(drop=True)
    # 时间切分 val 早停 (val < train_end, leak-free)
    sd = np.sort(sub["trade_date"].to_numpy())
    cut = str(sd[min(int(len(sd) * 0.9), len(sd) - 1)]) if len(sd) else "0"
    tr = sub[sub["trade_date"] < cut]
    va = sub[sub["trade_date"] >= cut]
    if len(va) < 5000 or len(tr) < 20000:
        tr, va = sub, sub.iloc[:0]

    def Xy(d):
        X = d[feats].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X, d["r20"].astype("float32").values
    Xtr, ytr = Xy(tr)
    params = dict(objective="regression", metric="l2", num_leaves=31,
                  learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, min_child_samples=80, seed=seed, verbose=-1,
                  num_threads=0)
    dtr = lgb.Dataset(Xtr, ytr)
    if len(va):
        Xva, yva = Xy(va)
        booster = lgb.train(params, dtr, num_boost_round=400,
                            valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)],
                            callbacks=[lgb.early_stopping(30, verbose=False),
                                       lgb.log_evaluation(0)])
    else:
        booster = lgb.train(params, dtr, num_boost_round=200)
    Xte = df_test[feats].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return booster.predict(Xte)


def portfolio_alpha(df_test: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """每日 top TOP_PCT 多头等权, α = 组合 r20 − universe r20; 返回逐日 (entry_date, alpha)."""
    g = df_test[["trade_date", pred_col, "r20"]].dropna().copy()
    g["r20"] = g["r20"].clip(-0.5, 0.5)
    rows = []
    for d_, sub in g.groupby("trade_date"):
        if len(sub) < 100:
            continue
        thr = sub[pred_col].quantile(1 - TOP_PCT)
        top = sub[sub[pred_col] >= thr]
        if len(top) < 3:
            continue
        port = float(top["r20"].mean()) - COST_BPS * 2     # 双边成本
        mkt = float(sub["r20"].mean())
        rows.append({"entry_date": d_, "alpha": (port - mkt) * 100, "n": len(top)})
    return pd.DataFrame(rows)


def evaluate_month(test_month: str, panel: pd.DataFrame) -> dict | None:
    test_start = f"{test_month}01"
    train_end = shift_ym(test_month, -1) + "01"            # 1 月 gap
    train_start = shift_ym(test_month, -1 - TRAIN_LOOKBACK_MONTHS) + "01"
    df_train = panel[(panel["trade_date"] >= train_start) &
                     (panel["trade_date"] < train_end)]
    df_test = panel[panel["ym"] == test_month]
    if len(df_train) < 100_000 or len(df_test) < 100:
        print(f"  {test_month}: 样本不足 (train {len(df_train)} / test {len(df_test)}), 跳过",
              flush=True)
        return None
    out = {"test_month": test_month, "train_start": train_start, "train_end": train_end}
    preds = {}
    for arm, feats in [("base", BASE_FEATS), ("mh", MEIHUA_FEATS)]:
        p = train_predict(df_train, df_test, feats, SEED)
        preds[arm] = p
    dft = df_test.copy()
    for arm in ("base", "mh"):
        dft[f"pred_{arm}"] = preds[arm]
    # regime 分层: 用测试月内每只票的 regime 众数? 用逐日 regime → 拆 α
    for arm in ("base", "mh"):
        pa = portfolio_alpha(dft, f"pred_{arm}")
        if pa.empty:
            print(f"  {test_month} {arm}: 无持仓, 跳过本月", flush=True)
            return None
        # 合 regime (按 entry_date)
        rg = dft.groupby("trade_date")["regime"].agg(lambda s: s.mode().iloc[0]
                                                     if len(s.mode()) else "unknown")
        pa["regime"] = pa["entry_date"].map(rg)
        a = pa["alpha"]
        out[f"{arm}_alpha"] = float(a.mean())
        out[f"{arm}_sharpe"] = float(a.mean() / (a.std() + 1e-9) * np.sqrt(12))
        for reg in ("momentum", "mixed", "reversal"):
            sub = pa[pa["regime"] == reg]["alpha"]
            out[f"{arm}_alpha_{reg}"] = float(sub.mean()) if len(sub) else np.nan
        out[f"{arm}_n_days"] = int(pa["entry_date"].nunique())
    print(f"  {test_month}: base α={out['base_alpha']:+.3f} Sh={out['base_sharpe']:+.2f} | "
          f"mh α={out['mh_alpha']:+.3f} Sh={out['mh_sharpe']:+.2f} | "
          f"Δα={out['mh_alpha']-out['base_alpha']:+.3f}", flush=True)
    del df_train, dft
    gc.collect()
    return out


def main():
    t0 = time.time()
    print("\n=== MT-004 walk-forward: 梅花轨迹特征 apples-to-apples 增益闸 ===\n", flush=True)
    panel = build_wf_panel()
    print(f"[panel] {len(panel):,} 行; regime "
          f"{panel['regime'].value_counts(normalize=True).round(3).to_dict()}\n", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records")
        done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月", flush=True)

    for m_ in TEST_MONTHS:
        if m_ in done:
            continue
        r = evaluate_month(m_, panel)
        if r:
            results.append(r)
            pd.DataFrame(results).to_csv(MONTHLY, index=False)

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].sort_values("test_month").reset_index(drop=True)

    def agg(arm):
        a = df[f"{arm}_alpha"]
        return {"alpha_mean": float(a.mean()), "alpha_median": float(a.median()),
                "sharpe_mean": float(df[f"{arm}_sharpe"].mean()),
                "worst_month": float(a.min()),
                "worst_month_id": str(df.loc[a.idxmin(), "test_month"]),
                "pos_alpha_ratio": float((a > 0).mean()),
                "alpha_momentum": float(df[f"{arm}_alpha_momentum"].mean()),
                "alpha_mixed": float(df[f"{arm}_alpha_mixed"].mean()),
                "alpha_reversal": float(df[f"{arm}_alpha_reversal"].mean()),
                "n_months": int(len(df))}
    base, mh = agg("base"), agg("mh")
    d_alpha = mh["alpha_mean"] - base["alpha_mean"]
    d_mom = mh["alpha_momentum"] - base["alpha_momentum"]
    d_rev = mh["alpha_reversal"] - base["alpha_reversal"]

    c1 = d_alpha >= 0.30
    c2 = mh["sharpe_mean"] >= base["sharpe_mean"]
    c3 = mh["worst_month"] >= base["worst_month"]
    c4 = d_mom >= 0      # 不伤动量月 (SIGN-R11)
    c5 = d_rev >= 0      # 不伤反转月 (SIGN-R11)
    conds = {"delta_alpha>=+0.30pp": bool(c1), "sharpe>=baseline": bool(c2),
             "worst_month>=baseline": bool(c3), "delta_alpha_momentum>=0": bool(c4),
             "delta_alpha_reversal>=0": bool(c5)}
    passed = all(conds.values())
    status = "PASS" if passed else "REJECT"

    res = {
        "window": [TEST_MONTHS[0], TEST_MONTHS[-1]], "n_months": int(len(df)),
        "protocol": (f"walk-forward {TRAIN_LOOKBACK_MONTHS}m lookback + 1m gap, monthly retrain, "
                     "GBDT(r20) top 5% long, apples-to-apples (feature set only diff: "
                     "baseline=standard TA / meihua=standard TA + mhB_* + mhC_*)"),
        "base_features": BASE_FEATS, "meihua_features": MEIHUA_FEATS,
        "baseline": base, "meihua": mh,
        "delta_alpha_pp": d_alpha, "delta_alpha_momentum": d_mom, "delta_alpha_reversal": d_rev,
        "gate_conditions": conds, "gate_status": status,
        "note": "Δα 是 gate (两臂共模偏差相消); GBDT 能非线性吃标准TA, 故 Δα>0 才是梅花超出标准TA "
                "对同一走势编码的真增益。REJECT 合法完成 (SIGN-R02), 中间 IC 不作数 (SIGN-R03)。",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n\n=== MT-004 walk-forward 汇总 ===\n", flush=True)
    print(f"  月数 {len(df)} ({TEST_MONTHS[0]}-{TEST_MONTHS[-1]})", flush=True)
    print(f"  {'指标':20s} {'baseline':>12s} {'meihua':>12s} {'Δ':>10s}", flush=True)
    print(f"  {'月化α(pp)':20s} {base['alpha_mean']:>+12.3f} {mh['alpha_mean']:>+12.3f} {d_alpha:>+10.3f}", flush=True)
    print(f"  {'Sharpe':20s} {base['sharpe_mean']:>+12.2f} {mh['sharpe_mean']:>+12.2f} {mh['sharpe_mean']-base['sharpe_mean']:>+10.2f}", flush=True)
    print(f"  {'最差月(pp)':20s} {base['worst_month']:>+12.3f} {mh['worst_month']:>+12.3f} {mh['worst_month']-base['worst_month']:>+10.3f}", flush=True)
    print(f"  {'正α月占比':20s} {base['pos_alpha_ratio']:>12.1%} {mh['pos_alpha_ratio']:>12.1%}", flush=True)
    print(f"  {'动量月α(pp)':20s} {base['alpha_momentum']:>+12.3f} {mh['alpha_momentum']:>+12.3f} {d_mom:>+10.3f}", flush=True)
    print(f"  {'反转月α(pp)':20s} {base['alpha_reversal']:>+12.3f} {mh['alpha_reversal']:>+12.3f} {d_rev:>+10.3f}", flush=True)
    print(f"\n  gate: {conds}", flush=True)
    print(f"  >>> verdict = {status} <<<", flush=True)
    print(f"[done] {(time.time()-t0)/60:.1f} min", flush=True)
    return res


if __name__ == "__main__":
    main()
