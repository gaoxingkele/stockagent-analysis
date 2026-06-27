# -*- coding: utf-8 -*-
"""关系张量 v2 编码器 (RT-001 核心) — K 线相对位置软符号张量, per-anchor.

设计见 research/DESIGN_relation_tensor_v2.md §1。本模块实现**单锚点**编码:
对锚点 K_t 与其前 d (=1..5) 根 K_{t-d} 的 OCHL 4×4 软/硬符号关系矩阵 + 量关系 + 自身形态。

存储策略 (重要): 一行 = 一个锚点 (ts_code, trade_date)。序列张量 N×5×4×4×2 是**模型输入视图**,
由 RT-003 的 data loader 在加载时按 ts_code 取尾部 N=40 个锚点行拼装 —— 避免逐样本重复存
(6.4K 值/样本 → 170 值/锚点, 38x 省盘), 且语义等价 (锚点本就是 (ts_code, date))。

软符号 (设计 §1.1):
    ℓ_{ij}^(d) = ln( K_t.i / K_{t-d}.j )           i,j ∈ {O,C,H,L}
    z_{ij}^(d) = ℓ / ( σ_t · √d )                  σ_t = 近20日日对数收益 std (causal)
    soft       = tanh( z / κ )      κ=1             连续幅度通道, ±0.5% 死区自动涌现
    hard       = sign(z) = sign(ℓ)                  离散形态通道 (缺口/吞没本就是类别)

量关系 (§1.3): vol_d = tanh( ln(V_t/V_{t-d}) / (σ^V_t·√d) ), σ^V_t = 近20日 ln量比 std。
自身形态 (§1.3, lag-free): 实体/上影/下影/归一振幅/跳空。

因果硬约束 (§1.4, SIGN-R04): 锚点 t 只用 ≤t 的数据。未来 K 绝不进。
确定性: 纯算术, 重算逐位一致。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

KEY = ["ts_code", "trade_date"]
PRICES = ["O", "C", "H", "L"]          # 行=锚点当前价 i, 列=前 d 根价 j
_COL = {"O": "open", "C": "close", "H": "high", "L": "low"}
DEPTHS = (1, 2, 3, 4, 5)
KAPPA = 1.0
SIGMA_WIN = 20                          # 波动归一窗口 (日)

SHAPE_COLS = ["shape_body", "shape_upsh", "shape_dnsh", "shape_amp", "shape_gap"]


def price_soft_col(d: int, i: str, j: str) -> str:
    return f"p_soft_d{d}_{i}{j}"


def price_hard_col(d: int, i: str, j: str) -> str:
    return f"p_hard_d{d}_{i}{j}"


def vol_col(d: int) -> str:
    return f"vol_d{d}"


def feature_columns() -> list[str]:
    """确定性有序特征列清单 (160 价 + 5 量 + 5 形态 = 170)。"""
    cols: list[str] = []
    for d in DEPTHS:
        for i in PRICES:
            for j in PRICES:
                cols.append(price_soft_col(d, i, j))
                cols.append(price_hard_col(d, i, j))
    cols += [vol_col(d) for d in DEPTHS]
    cols += SHAPE_COLS
    return cols


def _safe_ln_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """ln(num/den), 任一 <=0 或缺失 → NaN (suspended/坏数据)。"""
    num = num.where(num > 0)
    den = den.where(den > 0)
    return np.log(num / den)


def encode_frame(panel: pd.DataFrame, kappa: float = KAPPA) -> pd.DataFrame:
    """对全历史面板 (ts_code, trade_date, open, high, low, close, vol) 生成 per-anchor 张量。

    输入需含列: ts_code, trade_date, open, high, low, close, vol。
    输出: KEY + feature_columns() (float32)。warmup 不足 (σ 或前5根缺失) 的锚点行剔除。
    全程 groupby(ts_code) shift —— 锚点 t 只看 ≤t, 因果零泄漏。
    """
    df = panel.sort_values(KEY).reset_index(drop=True).copy()
    df["trade_date"] = df["trade_date"].astype(str)
    g = df.groupby("ts_code", sort=False)

    # ── σ_t: 近20日日对数收益 std (causal, 含当日已知收盘) ──
    logret = np.log(df["close"].where(df["close"] > 0) /
                    g["close"].shift(1).where(lambda s: s > 0))
    sigma = logret.groupby(df["ts_code"], sort=False).transform(
        lambda s: s.rolling(SIGMA_WIN, min_periods=SIGMA_WIN).std())

    # ── σ^V_t: 近20日 ln量比 std ──
    logvol = _safe_ln_ratio(df["vol"], g["vol"].shift(1))
    sigma_v = logvol.groupby(df["ts_code"], sort=False).transform(
        lambda s: s.rolling(SIGMA_WIN, min_periods=SIGMA_WIN).std())

    out = {k: df[k].values for k in KEY}

    # 预取各价位的 shift(d) (前 d 根的 O/C/H/L)
    shifted = {(p, d): g[_COL[p]].shift(d) for p in PRICES for d in DEPTHS}

    sqrt_d = {d: np.sqrt(d) for d in DEPTHS}
    for d in DEPTHS:
        denom = sigma * sqrt_d[d]
        denom = denom.where(denom > 0)            # σ=0 (停牌长横) → NaN, 避免 0 除
        for i in PRICES:
            cur_i = df[_COL[i]]
            for j in PRICES:
                ell = _safe_ln_ratio(cur_i, shifted[(j, d)])
                z = ell / denom
                out[price_soft_col(d, i, j)] = np.tanh(z / kappa).astype(np.float32)
                out[price_hard_col(d, i, j)] = np.sign(ell).astype(np.float32)
        # 量关系
        vdenom = (sigma_v * sqrt_d[d]).where(lambda s: s > 0)
        vol_ell = _safe_ln_ratio(df["vol"], g["vol"].shift(d))
        out[vol_col(d)] = np.tanh((vol_ell / vdenom) / kappa).astype(np.float32)

    # ── 自身形态 (lag-free, 仅锚点 K_t 自身 + 前一根收盘用于跳空) ──
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    rng = (h - l).where(lambda s: s > 0)
    out["shape_body"] = ((c - o) / rng).astype(np.float32)
    out["shape_upsh"] = ((h - np.maximum(o, c)) / rng).astype(np.float32)
    out["shape_dnsh"] = ((np.minimum(o, c) - l) / rng).astype(np.float32)
    out["shape_amp"] = ((h - l) / (sigma * c).where(lambda s: s > 0)).astype(np.float32)
    prev_c = g["close"].shift(1)
    out["shape_gap"] = ((o - prev_c) / (sigma * prev_c).where(lambda s: s > 0)).astype(np.float32)

    res = pd.DataFrame(out)
    feat = feature_columns()

    # warmup 闸: σ 与前5根必须就位 (否则张量含 NaN → 非有效锚点)
    valid = sigma.notna().values & shifted[("C", 5)].notna().values
    res = res[valid].reset_index(drop=True)

    # 剩余的零星 NaN (停牌/坏数据致量比缺失等) → 0 (中性), 形态除零亦然
    res[feat] = res[feat].fillna(0.0).astype(np.float32)
    return res[KEY + feat]
