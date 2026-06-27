# -*- coding: utf-8 -*-
"""RT-004 — Phase-1 廉价筛查: A2 张量 embedding 残差是否独立于 A0 (precondition go/no-go).

设计见 research/DESIGN_relation_tensor_v2.md §5/§10 第2步。这是**事前注册的廉价关卡**, 不是 gate:
只在 r20 主标签 + GBDT-on-embedding 上跑, 判断关系张量编码出的非线性形态信号, 在**扣除 A0 标量 TA
+ 月×板块朴素对照**后, 对 r20 是否还有独立残差。无残差 → 廉价 REJECT, 直接跳过昂贵的
RT-005 walk-forward (省 ~90% 算力)。

方法 (仿 MH-002 OOF target-encoding 消融 + SIGN-R12 朴素对照):
  对每个 OOF 折 (expanding 窗, **全部落在 gate 窗 202410-202604 之外**, 保持 RT-005 OOS 纯净):
    1. 训 FusionModel(NN end-to-end) 于 train 月 → 冻结 SeqEncoder → 抽 train/test 序列 embedding。
    2. emb_pred = LightGBM(embedding → r20z) 在 train 拟合, OOF 预测 test  —— 张量自身预测 (≈A1)。
    3. a0_pred  = LightGBM(A0 标量TA → r20z) 在 train 拟合, OOF 预测 test —— A0 baseline 预测。
    4. a2_pred  = LightGBM(embedding ⊕ A0 → r20z)                       —— 融合 (描述用)。
  汇总所有折的 OOF test 样本:
    - **残差 rank-IC** = 把 emb_pred 与 r20 同时对 [a0_pred + 月×板块 dummies] 做 OLS 残差化, 再逐日
      Spearman → 跨日均值 + t。这是**决策指标**: 张量在扣 A0+日历/板块后剩的独立预测力。
    - 按 regime (动量/反转) 分层 (SIGN-R11)。
  **裁决 (事前注册阈值, R01 不可改)**: 残差 |rank-IC|>=0.01 且 |t|>=3 → residual_signal, 否则 no_residual。

纪律:
  - SIGN-R03 中间指标≠落地: 这里产出的 IC 仅决定 go/no-go, 不是 ship 信号 (ship 看 RT-005 α)。
  - SIGN-R04 泄漏: r20 是 label, 全因果 (close shift(-HORIZON)), 落 research/cache 不进 features。
  - SIGN-R06 ST: rel_tensor/a0 已源头排除, inner-join 继承。
  - SIGN-R08 checkpoint: 每折 OOF 预测落 research/cache/rt004_fold_*.parquet, 已存在跳过。
  - SIGN-R11/R12: 分 regime + 扣 A0 标量TA + 月×板块朴素对照。

用法: python research/rt004_phase1_screen.py [--epochs 2] [--max-train 40000]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from rt_encoder import KEY  # noqa: E402
# 复用 RT-003 的低层件 (特征列表/序列拼装/逐日IC), 避免重复定义
from rt003_train import (  # noqa: E402
    RT_FEATS, A0_FEATS, A0_COLS, N_SEQ, HORIZON,
    RT_DIR, A0_DIR, DAILY_DIR, CACHE, gather_batch, xs_rank_ic,
)

REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "RT-004.json"
SEED = 0

# ── OOF 折 (expanding 窗, 全部 < 202410 gate 窗, 保持 RT-005 OOS 纯净; SIGN-R02) ──
# train 取 test 前的有界窗 (控内存/算力), 加载 = train..test 连续月 (供尾部 40 锚点拼装)。
FOLDS = [
    {"name": "f1", "train": ("202206", "202306"), "test": ("202307", "202309")},
    {"name": "f2", "train": ("202209", "202309"), "test": ("202310", "202312")},
    {"name": "f3", "train": ("202301", "202312"), "test": ("202401", "202403")},
]

# 决策阈值 (事前注册, R01 不可改)
GATE_IC = 0.01
GATE_T = 3.0


def month_range(a: str, b: str) -> list[str]:
    """'YYYYMM'..'YYYYMM' 含端点。"""
    ya, ma = int(a[:4]), int(a[4:]); yb, mb = int(b[:4]), int(b[4:])
    out = []
    y, m = ya, ma
    while (y, m) <= (yb, mb):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1; y += 1
    return out


def board_of(ts_code: str) -> str:
    """板块 (朴素对照, 由代码前缀派生; SIGN-R12)。"""
    p = ts_code[:2]
    return {"60": "main_sh", "00": "main_sz", "30": "chinext",
            "68": "star"}.get(p, "other")


def load_feat_months(months: list[str]) -> pd.DataFrame:
    """rel_tensor(170) ⊕ a0(32) inner join, 覆盖 months。"""
    rt_parts, a0_parts = [], []
    for m in months:
        rt_parts.append(pd.read_parquet(RT_DIR / f"rt_{m}.parquet"))
        a0_parts.append(pd.read_parquet(A0_DIR / f"a0_{m}.parquet"))
    rt = pd.concat(rt_parts, ignore_index=True)
    a0 = pd.concat(a0_parts, ignore_index=True)
    rt["trade_date"] = rt["trade_date"].astype(str)
    a0["trade_date"] = a0["trade_date"].astype(str)
    a0 = a0[KEY + A0_FEATS].rename(columns=dict(zip(A0_FEATS, A0_COLS)))
    df = rt.merge(a0, on=KEY, how="inner")
    df["month"] = df["trade_date"].str[:6]
    return df.sort_values(KEY).reset_index(drop=True)


def load_r20_label() -> pd.DataFrame:
    """前向 r20 = close[t+HORIZON]/close[t]-1 (因果, 落 cache 不进 features; SIGN-R04)。

    覆盖全 fold 跨度 (含 test 后 HORIZON 日), 一次算齐缓存供 RT-005 复用。
    """
    cp = CACHE / "rt004_r20_label.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    dmin = min(month_range(f["train"][0], f["test"][1])[0] for f in FOLDS) + "01"
    files = [f for f in sorted(DAILY_DIR.glob("*.parquet")) if f.stem >= dmin]
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"]) for f in files]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px[px["close"] > 0].drop_duplicates(KEY, keep="last")
    px = px.sort_values(KEY).reset_index(drop=True)
    fwd = px.groupby("ts_code")["close"].shift(-HORIZON)
    px["r20"] = (fwd / px["close"] - 1.0).astype(np.float32)
    out = px[KEY + ["r20"]].dropna(subset=["r20"]).reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cp, index=False)
    return out


def build_per_stock(df: pd.DataFrame, label: pd.DataFrame) -> dict:
    lab = label.set_index(KEY)["r20"]
    df = df.join(lab, on=KEY)
    per_stock = {}
    for code, sub in df.groupby("ts_code", sort=False):
        sub = sub.sort_values("trade_date")
        if len(sub) < N_SEQ:
            continue
        per_stock[code] = {
            "feat": sub[RT_FEATS].to_numpy(np.float32),
            "a0": sub[A0_COLS].to_numpy(np.float32),
            "y": sub["r20"].to_numpy(np.float32),
            "date": sub["trade_date"].to_numpy(),
            "month": sub["month"].to_numpy(),
        }
    return per_stock


def collect_idx(per_stock: dict, target_months: set) -> list:
    idx = []
    for code, rec in per_stock.items():
        n = len(rec["date"])
        for pos in range(N_SEQ - 1, n):
            if rec["month"][pos] in target_months and np.isfinite(rec["y"][pos]):
                idx.append((code, pos))
    return idx


def daily_zscore(y: np.ndarray, dates: np.ndarray) -> np.ndarray:
    s = pd.Series(y).groupby(pd.Series(dates)).transform(
        lambda v: (v - v.mean()) / (v.std() + 1e-9))
    return s.to_numpy(np.float32)


def run_fold(fold: dict, label: pd.DataFrame, args) -> pd.DataFrame:
    """训 + 抽 OOF emb_pred / a0_pred / a2_pred → 折级 parquet (checkpoint)。"""
    cp = CACHE / f"rt004_fold_{fold['name']}.parquet"
    if cp.exists():
        print(f"[{fold['name']}] checkpoint 命中, 跳过 -> {cp.name}", flush=True)
        return pd.read_parquet(cp)

    import torch
    import lightgbm as lgb
    from rt_models import FusionModel

    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    train_months = set(month_range(*fold["train"]))
    test_months = set(month_range(*fold["test"]))
    load_months = month_range(fold["train"][0], fold["test"][1])
    t0 = time.time()
    df = load_feat_months(load_months)
    per_stock = build_per_stock(df, label)
    tr_idx = collect_idx(per_stock, train_months)
    te_idx = collect_idx(per_stock, test_months)
    print(f"[{fold['name']}] load {load_months[0]}..{load_months[-1]} | "
          f"train {len(tr_idx):,} test {len(te_idx):,} | {len(per_stock)} 股 ({time.time()-t0:.0f}s)",
          flush=True)
    if not tr_idx or not te_idx:
        raise RuntimeError(f"{fold['name']} 样本不足")

    if len(tr_idx) > args.max_train:
        sel = rng.choice(len(tr_idx), args.max_train, replace=False)
        tr_idx = [tr_idx[i] for i in sorted(sel)]

    tr_X, tr_A, tr_y, tr_d, _ = gather_batch(per_stock, tr_idx)
    tr_yz = daily_zscore(tr_y, tr_d)

    # ── 训 FusionModel (NN end-to-end), 用于抽冻结序列 embedding ──
    model = FusionModel(n_a0=len(A0_FEATS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.MSELoss()
    n = len(tr_idx)
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        tot, seen = 0.0, 0
        for i in range(0, n, args.batch):
            b = order[i:i + args.batch]
            pred = model(torch.from_numpy(tr_X[b]), torch.from_numpy(tr_A[b]))
            loss = lossf(pred, torch.from_numpy(tr_yz[b]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b); seen += len(b)
        print(f"  [{fold['name']}] epoch {ep+1}/{args.epochs} MSE={tot/seen:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    model.eval()

    def infer(idx):
        embs, ys, ds, cs, a0s = [], [], [], [], []
        for i in range(0, len(idx), args.batch):
            X, A, y, d, c = gather_batch(per_stock, idx[i:i + args.batch])
            with torch.no_grad():
                _, em = model(torch.from_numpy(X), torch.from_numpy(A), return_embedding=True)
            embs.append(em.numpy()); ys.append(y); ds.append(d); cs.append(c); a0s.append(A)
        return (np.concatenate(embs), np.concatenate(ys), np.concatenate(ds),
                np.concatenate(cs), np.concatenate(a0s))

    tr_emb, _, _, _, tr_A2 = infer(tr_idx)
    te_emb, te_y, te_d, te_c, te_A = infer(te_idx)

    lgb_params = {"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
                  "seed": SEED, "verbose": -1, "num_threads": 0}

    def fit_pred(Xtr, Xte):
        gbm = lgb.train(lgb_params, lgb.Dataset(Xtr, label=tr_yz), num_boost_round=200)
        return gbm.predict(Xte).astype(np.float32)

    emb_pred = fit_pred(tr_emb, te_emb)                                   # 张量自身 (≈A1)
    a0_pred = fit_pred(tr_A2, te_A)                                       # A0 baseline
    a2_pred = fit_pred(np.concatenate([tr_emb, tr_A2], 1),
                       np.concatenate([te_emb, te_A], 1))                 # 融合 (描述)

    out = pd.DataFrame({
        "ts_code": te_c, "trade_date": te_d,
        "r20": te_y.astype(np.float32),
        "r20z": daily_zscore(te_y, te_d),
        "emb_pred": emb_pred, "a0_pred": a0_pred, "a2_pred": a2_pred,
        "fold": fold["name"],
    })
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cp, index=False)
    print(f"[{fold['name']}] OOF {len(out):,} 行 -> {cp.name} ({time.time()-t0:.0f}s)", flush=True)
    return out


def residualize(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS 残差 v - Z @ beta (Z 含截距)。"""
    beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
    return v - Z @ beta


def daily_ic(x: np.ndarray, y: np.ndarray, dates: np.ndarray):
    """逐日 Spearman → (mean, t, n_dates, ic_by_date dict)。"""
    d = pd.DataFrame({"d": dates, "x": x, "y": y})
    rows = []
    for dt, g in d.groupby("d"):
        if len(g) < 5:
            continue
        ic = g["x"].rank().corr(g["y"].rank())
        if np.isfinite(ic):
            rows.append((dt, ic))
    if len(rows) < 2:
        return None, None, len(rows), {}
    ics = np.array([r[1] for r in rows])
    m, s = float(ics.mean()), float(ics.std(ddof=1))
    t = m / s * np.sqrt(len(ics)) if s > 0 else None
    return (round(m, 5), round(t, 3) if t is not None else None, len(rows),
            {r[0]: r[1] for r in rows})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()
    print("=== RT-004 Phase-1 廉价筛查 (A2 残差是否独立于 A0) ===", flush=True)
    label = load_r20_label()
    print(f"[label] r20 因果标签 {len(label):,} 行 (cache)", flush=True)

    folds = [run_fold(f, label, args) for f in FOLDS]
    pool = pd.concat(folds, ignore_index=True)
    pool = pool.dropna(subset=["r20z", "emb_pred", "a0_pred"]).reset_index(drop=True)

    # 板块 + regime 标注 (朴素对照 R12 + 分层 R11)
    pool["board"] = pool["ts_code"].map(board_of)
    reg = pd.read_parquet(REGIME)[["trade_date", "regime"]]
    reg["trade_date"] = reg["trade_date"].astype(str)
    pool = pool.merge(reg, on="trade_date", how="left")
    pool["regime"] = pool["regime"].fillna("unknown")

    dates = pool["trade_date"].to_numpy()

    # ── 原始 (raw) 逐日 rank-IC (描述: A0 / A1张量 / A2融合) ──
    raw = {
        "A0_a0_pred": daily_ic(pool["a0_pred"].to_numpy(), pool["r20"].to_numpy(), dates)[:3],
        "A1_emb_pred": daily_ic(pool["emb_pred"].to_numpy(), pool["r20"].to_numpy(), dates)[:3],
        "A2_fused": daily_ic(pool["a2_pred"].to_numpy(), pool["r20"].to_numpy(), dates)[:3],
    }

    # ── 决策: 残差 rank-IC = emb_pred 与 r20 同对 [a0_pred + 月×板块 dummies] 残差化后逐日 IC ──
    mb = (pool["month"] if "month" in pool else pool["trade_date"].str[:6]).astype(str) \
        + "_" + pool["board"].astype(str)
    D = pd.get_dummies(mb, drop_first=True).to_numpy(np.float64)
    Z = np.column_stack([np.ones(len(pool)), pool["a0_pred"].to_numpy(np.float64), D])
    x_res = residualize(pool["emb_pred"].to_numpy(np.float64), Z)
    y_res = residualize(pool["r20z"].to_numpy(np.float64), Z)
    rm, rt_, rn, ic_by_date = daily_ic(x_res, y_res, dates)

    # 分 regime (SIGN-R11): 用残差按日分组
    res_df = pd.DataFrame({"d": dates, "x": x_res, "y": y_res, "regime": pool["regime"]})
    regime_ic = {}
    for rg, g in res_df.groupby("regime"):
        m, t, nd, _ = daily_ic(g["x"].to_numpy(), g["y"].to_numpy(), g["d"].to_numpy())
        regime_ic[rg] = {"resid_ic": m, "t": t, "n_dates": nd}

    passed = (rm is not None and rt_ is not None
              and abs(rm) >= GATE_IC and abs(rt_) >= GATE_T)
    status = "residual_signal" if passed else "no_residual"

    print(f"\n[raw] A0={raw['A0_a0_pred']} A1张量={raw['A1_emb_pred']} A2融合={raw['A2_fused']}",
          flush=True)
    print(f"[残差] 扣 A0+月×板块: rank-IC={rm} t={rt_} n_dates={rn} "
          f"(阈值 |IC|>={GATE_IC} 且 |t|>={GATE_T})", flush=True)
    print(f"[分regime] {json.dumps(regime_ic, ensure_ascii=False)}", flush=True)
    print(f"[裁决] status={status}", flush=True)

    verdict = {
        "id": "RT-004",
        "status": status,
        "conclusion": (
            f"Phase-1 廉价筛查: 关系张量 embedding 对 r20 在扣 A0 标量TA + 月×板块朴素对照后, "
            f"残差 rank-IC={rm} (t={rt_}, {rn} 交易日), "
            f"{'达' if passed else '未达'}事前注册阈值 |IC|>={GATE_IC} 且 |t|>={GATE_T} → {status}。"
            + ("有独立残差信号, 进 RT-005 walk-forward gate。"
               if passed else
               "无独立残差 = 张量≈标准TA换皮 (梅花教训重演), 廉价 REJECT, 跳过 RT-005/RT-006 (省~90%算力)。")
            + " 中间 IC 非落地信号 (SIGN-R03), ship 仅由 RT-005 α 决定。"),
        "artifact": "research/rt004_phase1_screen.py",
        "decision_metric": {
            "residual_rank_ic": rm, "residual_t": rt_, "n_dates": rn,
            "gate_threshold": {"abs_ic_min": GATE_IC, "abs_t_min": GATE_T},
            "method": ("emb_pred(GBDT 吃冻结张量 embedding) 与 r20 同对 [a0_pred + 月×板块 dummies] "
                       "OLS 残差化后逐日 Spearman 跨日均值; a0_pred=GBDT 吃 A0 标量TA"),
        },
        "raw_rank_ic": {
            "A0_scalar_TA": {"ic": raw["A0_a0_pred"][0], "t": raw["A0_a0_pred"][1]},
            "A1_tensor_only": {"ic": raw["A1_emb_pred"][0], "t": raw["A1_emb_pred"][1]},
            "A2_fused": {"ic": raw["A2_fused"][0], "t": raw["A2_fused"][1]},
        },
        "regime_stratified": regime_ic,
        "oof_folds": [{"name": f["name"], "train": f["train"], "test": f["test"]} for f in FOLDS],
        "n_oof_samples": int(len(pool)),
        "ablation_note": ("决策对照 = A1张量 在扣 A0(标准TA标量版)+月×板块后是否仍有残差; "
                          "防梅花式'标准TA换皮' (SIGN-R12); OOF 折全在 gate 窗 202410-202604 之外 (SIGN-R02)"),
        "guardrails": ["SIGN-R03 IC非落地", "SIGN-R04 r20留cache零泄漏", "SIGN-R06 ST继承源头排除",
                       "SIGN-R08 折级checkpoint", "SIGN-R11 分regime", "SIGN-R12 扣朴素对照",
                       "SIGN-R01 阈值事前注册不可改"],
        "next": ("RT-005 walk-forward gate (解除 skip)" if passed
                 else "RT-005 skip=true 写 REJECT; RT-006 保持 skipped (廉价 REJECT)"),
    }
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status} -> {VERDICT.relative_to(ROOT)} ({time.time()-t0:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
