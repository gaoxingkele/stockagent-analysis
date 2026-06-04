# -*- coding: utf-8 -*-
"""RT-003 — CNN+TCN 编码器 + 双 head 单 fold 训通 + 推理路径跑通 (status=built).

设计见 research/DESIGN_relation_tensor_v2.md §2。本脚本是**工程管线 build & smoke**, 不是 gate:
按设计把 (RT-001 关系张量 per-anchor) 拼成 N=40 序列, 过 per-anchor CNN → 沿 N 轴 TCN →
序列 embedding ⊕ (RT-002 A0 标量因子) → head, 单 fold 训通两条 head:

  (b) GBDT-on-embedding (先, 稳/便宜): 训好的 SeqEncoder 冻结 → 抽序列 embedding ⊕ A0 → LightGBM。
  (a) NN end-to-end:                  FusionModel 整体训练 → head 直接出分。

单 fold: 训 TRAIN_MONTHS, 测 TEST_MONTH (apples-to-apples 的 walk-forward 留给 RT-005)。
报 train loss + 两条 head 在 test 月的 rank-IC —— **仅作管线 sanity, 非落地信号** (SIGN-R03,
只有 RT-005 walk-forward α 决定 ship)。推理路径产出 test 月截面分数样本落 cache。

纪律:
  - SIGN-R04 泄漏: r20 是 label, 只用作被预测量, 落 research/cache/ (不进 research/features/),
    全因果 (close shift(-20) 仅定义未来收益, 不作特征)。
  - SIGN-R06 ST: rel_tensor / a0 已源头排除, inner-join 自然继承。
  - SIGN-R05 生产线: 不触碰任何生产文件; 模型/embedding 落 research/cache、research/models。
  - SIGN-R07 确定性: torch/np/lgbm 固定种子。
  - SIGN-R08 checkpoint: 拼好的 fold 张量 + r20 label 缓存, 已存在跳过。

用法: python research/rt003_train.py [--epochs 3] [--max-train 40000]
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
from rt_encoder import feature_columns as rt_feature_columns, KEY  # noqa: E402
from a0_encoder import feature_columns as a0_feature_columns        # noqa: E402

RT_DIR = ROOT / "research" / "features" / "rel_tensor_v2"
A0_DIR = ROOT / "research" / "features" / "a0_scalar_ta"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
CACHE = ROOT / "research" / "cache"
MODELS = ROOT / "research" / "models"
VERDICT = ROOT / "research" / "verdicts" / "RT-003.json"

# ── 单 fold (engineering smoke; 多 fold walk-forward 是 RT-005) ──
TEST_MONTH = "202312"
TRAIN_MONTHS = ["202309", "202310", "202311"]
LOOKBACK_MONTHS = ["202307", "202308"]   # 为 train 起始锚点凑够 40 个前置锚点
N_SEQ = 40
HORIZON = 20                              # r20 主标签 (前向 20 交易日)
SEED = 0

RT_FEATS = rt_feature_columns()           # 170
A0_FEATS = a0_feature_columns()           # 32
# a0 与张量都含 shape_body/upsh/dnsh → 合并前给 a0 列加前缀避免碰撞
A0_COLS = [f"a0_{c}" for c in A0_FEATS]


def _months_needed() -> list[str]:
    return sorted(set(LOOKBACK_MONTHS + TRAIN_MONTHS + [TEST_MONTH]))


def load_features() -> pd.DataFrame:
    """rel_tensor (170) ⊕ a0 (32), inner join on (ts_code, trade_date), 覆盖所需月份。"""
    months = _months_needed()
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


def load_r20_label(codes: set, dmin: str, dmax_fwd: str) -> pd.DataFrame:
    """前向 r20 = close[t+20]/close[t]-1 (因果 label, 落 cache 不进 features, SIGN-R04)。

    需读到 TEST_MONTH 之后 ~20 交易日的 close 才能算齐, 故 dmax_fwd 给到次年。
    """
    cp = CACHE / "rt003_r20_label.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    files = [f for f in sorted(DAILY_DIR.glob("*.parquet")) if dmin <= f.stem <= dmax_fwd]
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"]) for f in files]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px[px["close"] > 0].drop_duplicates(KEY, keep="last")
    px = px[px["ts_code"].isin(codes)].sort_values(KEY).reset_index(drop=True)
    fwd = px.groupby("ts_code")["close"].shift(-HORIZON)
    px["r20"] = (fwd / px["close"] - 1.0).astype(np.float32)
    out = px[KEY + ["r20"]].dropna(subset=["r20"]).reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cp, index=False)
    return out


def build_fold(df: pd.DataFrame, label: pd.DataFrame):
    """拼 per-stock 数组 + 采样索引。返回 (per_stock dict, train_idx, test_idx, meta)。

    per_stock[code] = {feat:(n,170), a0:(n,32), y:(n,), date:(n,), month:(n,)}, 按日期升序。
    采样: pos>=N_SEQ-1 (有 40 窗口) 且 y 有效 且 month 在目标集。
    """
    lab = label.set_index(KEY)["r20"]
    df = df.join(lab, on=KEY)            # 附 r20 (label, 仅训练用)
    target_train = set(TRAIN_MONTHS)
    per_stock, train_idx, test_idx = {}, [], []
    feat_arr_cols = RT_FEATS
    for code, sub in df.groupby("ts_code", sort=False):
        sub = sub.sort_values("trade_date")
        n = len(sub)
        if n < N_SEQ:
            continue
        rec = {
            "feat": sub[feat_arr_cols].to_numpy(np.float32),
            "a0": sub[A0_COLS].to_numpy(np.float32),
            "y": sub["r20"].to_numpy(np.float32),
            "date": sub["trade_date"].to_numpy(),
            "month": sub["month"].to_numpy(),
        }
        per_stock[code] = rec
        for pos in range(N_SEQ - 1, n):
            m = rec["month"][pos]
            if not np.isfinite(rec["y"][pos]):
                continue
            if m in target_train:
                train_idx.append((code, pos))
            elif m == TEST_MONTH:
                test_idx.append((code, pos))
    return per_stock, train_idx, test_idx


def gather_batch(per_stock, idx_batch):
    """(code,pos) 列表 → (X_seq (B,N,170), A0 (B,32), y (B,), dates, codes)。"""
    B = len(idx_batch)
    X = np.empty((B, N_SEQ, len(RT_FEATS)), np.float32)
    A = np.empty((B, len(A0_FEATS)), np.float32)
    y = np.empty(B, np.float32)
    dates = np.empty(B, dtype=object)
    codes = np.empty(B, dtype=object)
    for k, (code, pos) in enumerate(idx_batch):
        rec = per_stock[code]
        X[k] = rec["feat"][pos - N_SEQ + 1:pos + 1]
        A[k] = rec["a0"][pos]
        y[k] = rec["y"][pos]
        dates[k] = rec["date"][pos]
        codes[k] = code
    np.nan_to_num(X, copy=False)
    np.nan_to_num(A, copy=False)
    return X, A, y, dates, codes


def xs_rank_ic(scores: np.ndarray, y: np.ndarray, dates: np.ndarray) -> dict:
    """逐日 Spearman(score, r20) 跨日平均 + t (管线 sanity, 非 gate)。"""
    d = pd.DataFrame({"d": dates, "s": scores, "y": y})
    ics = []
    for _, g in d.groupby("d"):
        if len(g) < 5:
            continue
        ics.append(g["s"].rank().corr(g["y"].rank()))
    ics = np.array([x for x in ics if np.isfinite(x)])
    if len(ics) < 2:
        return {"ic_mean": None, "ic_t": None, "n_dates": int(len(ics))}
    m, s = float(ics.mean()), float(ics.std(ddof=1))
    return {"ic_mean": round(m, 5), "ic_t": round(m / s * np.sqrt(len(ics)), 3) if s > 0 else None,
            "n_dates": int(len(ics))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    import torch
    import lightgbm as lgb
    from rt_models import FusionModel, SEQ_EMB

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    print(f"=== RT-003 单 fold 训通: train={TRAIN_MONTHS} test={TEST_MONTH} ===", flush=True)
    df = load_features()
    codes = set(df["ts_code"].unique())
    print(f"[feat] {len(df):,} 锚点行 × ({len(RT_FEATS)} 张量 + {len(A0_FEATS)} A0), "
          f"{len(codes)} 股, {df['trade_date'].min()}~{df['trade_date'].max()}", flush=True)

    # r20 label: 读到 test 月 + 20 交易日后 (次年初足够)
    label = load_r20_label(codes, _months_needed()[0] + "01", "20260301")
    print(f"[label] r20 因果标签 {len(label):,} 行 (cache, 不进 features)", flush=True)

    per_stock, train_idx, test_idx = build_fold(df, label)
    print(f"[fold] train 样本 {len(train_idx):,} | test 样本 {len(test_idx):,} "
          f"| {len(per_stock)} 股有 ≥{N_SEQ} 锚点", flush=True)
    if not train_idx or not test_idx:
        print("[fold] 样本不足, 退出", flush=True)
        return 1

    if len(train_idx) > args.max_train:
        sel = rng.choice(len(train_idx), args.max_train, replace=False)
        train_idx = [train_idx[i] for i in sorted(sel)]
        print(f"[fold] train 下采样 → {len(train_idx):,} (CPU 控算力, smoke)", flush=True)

    # 训练目标: 逐日 z-score r20 (横截面, 去全市场漂移); 推理 IC 用秩, 尺度无关。
    tr_X, tr_A, tr_y, tr_d, _ = gather_batch(per_stock, train_idx)
    dts = pd.Series(tr_d)
    yz = pd.Series(tr_y).groupby(dts).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    tr_yz = yz.to_numpy(np.float32)

    model = FusionModel(n_a0=len(A0_FEATS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.MSELoss()
    n = len(train_idx)
    print(f"[NN] 端到端训练 {args.epochs} epoch × batch {args.batch} ...", flush=True)
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        tot, seen = 0.0, 0
        for i in range(0, n, args.batch):
            b = order[i:i + args.batch]
            xb = torch.from_numpy(tr_X[b]); ab = torch.from_numpy(tr_A[b])
            yb = torch.from_numpy(tr_yz[b])
            opt.zero_grad()
            pred = model(xb, ab)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b); seen += len(b)
        print(f"  epoch {ep+1}/{args.epochs}  train MSE={tot/seen:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ── 推理 + 抽冻结 embedding (train & test), 分批过 CPU ──
    model.eval()

    def infer(idx):
        scores, embs, ys, ds, cs = [], [], [], [], []
        for i in range(0, len(idx), args.batch):
            X, A, y, d, c = gather_batch(per_stock, idx[i:i + args.batch])
            with torch.no_grad():
                sc, em = model(torch.from_numpy(X), torch.from_numpy(A), return_embedding=True)
            scores.append(sc.numpy()); embs.append(em.numpy())
            ys.append(y); ds.append(d); cs.append(c)
        return (np.concatenate(scores), np.concatenate(embs), np.concatenate(ys),
                np.concatenate(ds), np.concatenate(cs))

    print("[infer] 抽 test 月 NN 分数 + 序列 embedding ...", flush=True)
    te_sc, te_emb, te_y, te_d, te_c = infer(test_idx)
    nn_ic = xs_rank_ic(te_sc, te_y, te_d)

    # ── (b) GBDT-on-embedding: 冻结 SeqEmb ⊕ A0 → LightGBM (先, 稳) ──
    print("[GBDT] 抽 train 月冻结 embedding → LightGBM(emb⊕A0) ...", flush=True)
    _, tr_emb, _, _, _ = infer(train_idx)
    te_A = gather_batch(per_stock, test_idx)[1]
    Xtr = np.concatenate([tr_emb, tr_A], axis=1)
    Xte = np.concatenate([te_emb, te_A], axis=1)
    # 用原生 API (lgb 4.5 + sklearn 1.8 的 sklearn 包装层有 force_all_finite 不兼容)
    dtrain = lgb.Dataset(Xtr, label=tr_yz)
    gbm = lgb.train(
        {"objective": "regression", "num_leaves": 31, "learning_rate": 0.05,
         "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
         "seed": SEED, "verbose": -1, "num_threads": 0},
        dtrain, num_boost_round=200)
    gb_pred = gbm.predict(Xte)
    gb_ic = xs_rank_ic(gb_pred, te_y, te_d)

    # ── 推理路径产出: test 月截面分数样本落 cache (无 forward 字段) ──
    CACHE.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame({"ts_code": te_c, "trade_date": te_d,
                              "nn_score": te_sc.astype(np.float32),
                              "gbdt_emb_score": gb_pred.astype(np.float32)})
    scores_df.to_parquet(CACHE / "rt003_test_scores.parquet", index=False)
    torch.save(model.state_dict(), MODELS / "rt003_fusion_smoke.pt")
    np.save(CACHE / "rt003_test_embedding.npy", te_emb)

    print(f"[IC] NN head test rank-IC={nn_ic['ic_mean']} (t={nn_ic['ic_t']}) | "
          f"GBDT-on-emb test rank-IC={gb_ic['ic_mean']} (t={gb_ic['ic_t']})", flush=True)

    verdict = {
        "id": "RT-003",
        "status": "built",
        "conclusion": (
            f"CNN+TCN 编码器 + 双 head 单 fold 训通 + 推理跑通 (train={TRAIN_MONTHS}, test={TEST_MONTH})。"
            f"per-anchor CNN(价 10ch×4×4 ⊕ 量5+形态5)→锚点 emb 32; 沿 N={N_SEQ} 因果 TCN→序列 emb {SEQ_EMB}; "
            f"融合 序列 emb ⊕ A0({len(A0_FEATS)} 标量)→head。两条 head 均跑通: "
            f"(b) GBDT-on-冻结embedding test rank-IC={gb_ic['ic_mean']}; "
            f"(a) NN end-to-end test rank-IC={nn_ic['ic_mean']}。"
            f"IC 仅管线 sanity 非落地信号 (SIGN-R03); 决策留 RT-005 walk-forward。"
            f"r20 label 留 cache 零泄漏, ST 源头排除继承, 生产线未触碰。"),
        "artifact": "research/rt_models.py + research/rt003_train.py",
        "metrics": {
            "fold": {"train_months": TRAIN_MONTHS, "test_month": TEST_MONTH,
                     "lookback_months": LOOKBACK_MONTHS},
            "n_train_samples": len(train_idx), "n_test_samples": len(test_idx),
            "N_seq": N_SEQ, "anchor_feat": len(RT_FEATS), "a0_feat": len(A0_FEATS),
            "anchor_emb": 32, "seq_emb": SEQ_EMB,
            "epochs": args.epochs,
            "head_b_gbdt_on_embedding": {"test_rank_ic": gb_ic["ic_mean"],
                                         "test_ic_t": gb_ic["ic_t"], "n_dates": gb_ic["n_dates"]},
            "head_a_nn_end_to_end": {"test_rank_ic": nn_ic["ic_mean"],
                                     "test_ic_t": nn_ic["ic_t"], "n_dates": nn_ic["n_dates"]},
            "note": "single-fold engineering smoke; IC 非 gate (SIGN-R03), gate=RT-005 walk-forward α",
            "device": "cpu",
        },
        "two_heads": {
            "(b) GBDT-on-embedding": "SeqEncoder 训后冻结 → 序列 embedding(64) ⊕ A0(32) → LightGBM (先/稳/便宜)",
            "(a) NN end-to-end": "AnchorEncoder→SeqEncoder→FusionHead 整体反传 (对比, 历史上月度 walk-forward 常输 GBDT)",
        },
        "artifacts_out": ["research/cache/rt003_test_scores.parquet",
                          "research/cache/rt003_test_embedding.npy",
                          "research/cache/rt003_r20_label.parquet",
                          "research/models/rt003_fusion_smoke.pt"],
        "guardrails": ["SIGN-R03 IC非gate只描述", "SIGN-R04 r20留cache零泄漏",
                       "SIGN-R05 生产线未触碰", "SIGN-R06 ST源头排除继承",
                       "SIGN-R07 种子固定确定性", "SIGN-R08 label缓存checkpoint"],
    }
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] status=built -> {VERDICT.relative_to(ROOT)}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
