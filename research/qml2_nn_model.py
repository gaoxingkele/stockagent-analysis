# -*- coding: utf-8 -*-
"""QML-2 Phase-1 — NN 模型 class vs GBDT (r20 预测), walk-forward OOS RankIC.

QuantML 合集核心主张: 神经网络(LNN/iTransformer/FactorVAE/AlphaNet…)优于树模型。测唯一未测轴
"模型 class": 同 236 factor_lab 特征, 同 train/test 月窗, LGBM(regression base) vs torch MLP,
walk-forward(24月lookback, 19测试月) OOS RankIC。

范围诚实声明: 测"NN 非线性 在我们工程特征上 vs GBDT"。我们 236 特征已编码大量时序信息(mom/
vol/pyramid…), 序列-NN-on-raw-OHLCV 多半只重新推导这些 + CPU walk-forward 昂贵 → 作后续单列,
仅当 MLP 在此显出苗头才投。GBDT 对截面 tabular 是公认极强基线, MLP 先验低。

Phase-1 廉价闸(R16): MLP mean OOS RankIC 不超 LGBM(差异 NW-t<2) → 廉价 REJECT(模型 class 不漏
信号)。apples: 同特征同 subsample 同月窗, 唯一差异=模型。只认 walk-forward(R03); 生产冻结(R05)。
checkpoint(R08): 月度 RankIC append。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
from train_v15_refresh import load_window, EXCLUDE
from walk_forward_validation import (
    get_train_end_for_test_month, get_train_start, TEST_MONTHS, TRAIN_LOOKBACK_MONTHS,
)
import oa_screen

import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

CACHE = ROOT / "research" / "cache"
MONTHLY = CACHE / "qml2_monthly_rankic.csv"
VERDICT = ROOT / "research" / "verdicts" / "QML-2.json"
DATA_START, DATA_END = "20220801", "20260525"
SUBSAMPLE = 900_000
N_EST = 400

LGB_PARAMS = dict(n_estimators=N_EST, learning_rate=0.05, num_leaves=63, min_child_samples=300,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1,
                  reg_lambda=0.1, max_bin=127, force_col_wise=True, random_state=42, n_jobs=4,
                  verbose=-1, objective="regression")


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def daily_rankic(dates, pred, r20) -> float:
    dp = pd.DataFrame({"d": dates, "p": pred, "y": r20})
    ics = []
    for _, g in dp.groupby("d"):
        g = g.dropna()
        if len(g) < 30:
            continue
        ic = stats.spearmanr(g["p"], g["y"])[0]
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def train_mlp(Xtr, ytr, Xva, yva, n_in):
    dev = "cpu"
    m = MLP(n_in).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    n = len(Xtr_t); bs = 4096
    best_ic, best_state, patience, bad = -9, None, 6, 0
    for ep in range(40):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = m(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Xva_t).numpy()
        ic = stats.spearmanr(pv, yva)[0]
        ic = ic if np.isfinite(ic) else -9
        if ic > best_ic:
            best_ic, best_state, bad = ic, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    return m


def main():
    t0 = time.time()
    print("\n=== QML-2 Phase-1 NN(MLP) vs GBDT (walk-forward OOS RankIC) ===\n", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    meta_r20 = json.loads((ROOT / "output/production/r20_v16_long_nost/feature_meta.json").read_text(encoding="utf-8"))
    ind_map = meta_r20.get("industry_map", {})
    daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    feat_cols = [c for c in daily.columns if c not in set(EXCLUDE)
                 and pd.api.types.is_numeric_dtype(daily[c]) and c != "r20"]
    if "industry_id" not in feat_cols:
        feat_cols.append("industry_id")
    for c in feat_cols:
        daily[c] = daily[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    print(f"[data] {len(daily):,} 行 / {len(feat_cols)} 特征", flush=True)

    done, results = set(), []
    if MONTHLY.exists():
        prev = pd.read_csv(MONTHLY, dtype={"test_month": str})
        results = prev.to_dict("records"); done = set(prev["test_month"].astype(str))
        print(f"[ckpt] 已完成 {len(done)} 月", flush=True)

    for tm in TEST_MONTHS:
        if tm in done:
            continue
        te = get_train_end_for_test_month(tm); ts = get_train_start(te, TRAIN_LOOKBACK_MONTHS)
        dtr = daily[(daily["trade_date"] >= ts) & (daily["trade_date"] < te)].dropna(subset=["r20"]).copy()
        dte = daily[daily["trade_date"].str.startswith(tm)].dropna(subset=["r20"]).copy()
        if len(dtr) < 100_000 or len(dte) < 100:
            print(f"  {tm}: 样本不足", flush=True); continue
        if len(dtr) > SUBSAMPLE:
            dtr = dtr.sample(n=SUBSAMPLE, random_state=42).reset_index(drop=True)
        # 时序 val split (末10%交易日)
        udates = sorted(dtr["trade_date"].unique())
        cut = udates[int(len(udates) * 0.9)]
        tr = dtr[dtr["trade_date"] < cut]; va = dtr[dtr["trade_date"] >= cut]

        # LGBM baseline
        lgbm = lgb.LGBMRegressor(**LGB_PARAMS)
        lgbm.fit(tr[feat_cols].astype("float32"), tr["r20"].astype("float32"),
                 categorical_feature=["industry_id"])
        pred_lgb = lgbm.predict(dte[feat_cols].astype("float32"))
        ic_lgb = daily_rankic(dte["trade_date"].to_numpy(), pred_lgb, dte["r20"].to_numpy())

        # MLP: 标准化 (fit on train)
        mu = tr[feat_cols].mean(); sd = tr[feat_cols].std().replace(0, 1)
        Xtr = ((tr[feat_cols] - mu) / sd).fillna(0).to_numpy("float32")
        Xva = ((va[feat_cols] - mu) / sd).fillna(0).to_numpy("float32")
        Xte = ((dte[feat_cols] - mu) / sd).fillna(0).to_numpy("float32")
        ytr = tr["r20"].to_numpy("float32"); yva = va["r20"].to_numpy("float32")
        mlp = train_mlp(Xtr, ytr, Xva, yva, len(feat_cols))
        mlp.eval()
        with torch.no_grad():
            pred_mlp = mlp(torch.tensor(Xte)).numpy()
        ic_mlp = daily_rankic(dte["trade_date"].to_numpy(), pred_mlp, dte["r20"].to_numpy())

        results.append({"test_month": tm, "rankic_lgbm": ic_lgb, "rankic_mlp": ic_mlp})
        pd.DataFrame(results).to_csv(MONTHLY, index=False)
        print(f"  {tm}: lgbm={ic_lgb:+.4f} mlp={ic_mlp:+.4f} (Δ{ic_mlp-ic_lgb:+.4f})", flush=True)
        del dtr, dte, lgbm, mlp; gc.collect()

    df = pd.DataFrame(results)
    df = df[df["test_month"].astype(str).isin(TEST_MONTHS)].sort_values("test_month").reset_index(drop=True)
    base = df["rankic_lgbm"].to_numpy(); mlpv = df["rankic_mlp"].to_numpy()
    delta = mlpv - base
    d_t = oa_screen.tstat_nw(delta[~np.isnan(delta)])
    mean_lgb = float(np.nanmean(base)); mean_mlp = float(np.nanmean(mlpv))
    status = "PROCEED_phase2" if (np.nanmean(delta) > 0 and d_t >= 2.0) else "REJECT_cheap"
    conclusion = (
        f"walk-forward {len(df)}月 OOS RankIC: LGBM(base)={mean_lgb:+.4f}, MLP={mean_mlp:+.4f} "
        f"(Δ{np.nanmean(delta):+.4f} NW-t{d_t:+.2f}, MLP胜base {int((mlpv>base).sum())}/{len(df)}月). "
        + ("MLP显著超LGBM → Phase-2 book gate。" if status == "PROCEED_phase2"
           else "MLP未显著超LGBM(NW-t<2) → 廉价REJECT: 模型class(NN非线性)在我们236工程特征上不漏信号, "
                "GBDT对截面tabular仍是最优。序列-NN-on-raw另作CPU昂贵后续(本结论不覆盖)。"))
    summary = {"id": "QML-2", "status": status, "test_months": int(len(df)),
               "mean_rankic_lgbm": round(mean_lgb, 4), "mean_rankic_mlp": round(mean_mlp, 4),
               "delta_mean": round(float(np.nanmean(delta)), 4), "delta_nw_t": round(float(d_t), 2),
               "mlp_win_months": int((mlpv > base).sum()),
               "scope": "NN(MLP)非线性 vs GBDT 在236 factor_lab工程特征上; 序列-NN-on-raw未测(CPU昂贵)",
               "conclusion": conclusion,
               "guardrails": ["SIGN-R01 gate冻结", "SIGN-R03 walk-forward OOS RankIC", "SIGN-R16 廉价闸+NW-t",
                              "apples同特征同subsample同月窗唯一差异模型", "SIGN-R05 生产线冻结"]}
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== QML-2 汇总 ===", flush=True)
    print(f"  LGBM mean RankIC={mean_lgb:+.4f} | MLP mean RankIC={mean_mlp:+.4f} | "
          f"Δ={np.nanmean(delta):+.4f} NW-t={d_t:+.2f}", flush=True)
    print(f"  [决策] {status}; {conclusion}", flush=True)
    print(f"[done] 耗时 {(time.time()-t0)/60:.1f} min -> {VERDICT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
