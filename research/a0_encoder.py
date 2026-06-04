# -*- coding: utf-8 -*-
"""A0 标准 TA 标量对照编码器 (RT-002 核心) — 关系张量内容的标量版.

设计见 research/DESIGN_relation_tensor_v2.md §4。本模块实现**消融基线 A0** 的标量特征:
把关系张量 v2 (RT-001) 编码的内容 (OCHL 多尺度相对位置 + 量关系 + 自身形态) 还原成
**标准标量 TA**, 与张量同口径同期、全因果无泄漏, 用 GBDT 作强 baseline。
A2-vs-A0 是决定性对照: 张量在 A0 已含标量 TA 之上还能否挤出非线性增量 (SIGN-R12)。

特征构成 (32 列, 全 causal, 仅用 ≤t 数据):
  多尺度动量 (5): mom_1/3/5/10/20            ← 张量 OCHL 跨阶对数比的标量版
  波动/range  (5): rvol_5/rvol_20/atr_pct_14/amplitude/amp_mean_5  ← 张量 σ 归一内容
  量          (5): vol_ratio_5/10/20/amount_ratio_20/vol_chg_1     ← 张量量关系通道
  缺口        (2): gap/gap_norm              ← 张量 shape_gap 内容
  均线结构    (6): ma_ratio_5/10/20/60/ma5_ma20/ma20_ma60          ← 生产标准 TA
  振荡器      (6): rsi_6/rsi_14/macd_hist/wr_14/boll_pct/boll_width ← 生产标准 TA
  自身形态    (3): shape_body/shape_upsh/shape_dnsh                 ← 张量 shape 通道标量版

"生产因子复用": factor_lab 的 153 因子 (ma_ratio/macd/rsi/atr_pct/vol_ratio/boll 等) 在此**因果重算于
全历史 OHLCV**, 而非 join factor_lab parquet —— 因为 factor_lab 仅覆盖 2025-04~2026-02 (非同期),
且其文件含 forward 字段 (r5/dd5/r20...) 有泄漏风险。重算保证同口径同期 + 零泄漏 + 确定性。

因果硬约束 (SIGN-R04): 全部 shift/rolling/ewm 仅向后 (≤t)。确定性: 纯算术, 重算逐位一致。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

KEY = ["ts_code", "trade_date"]
SIGMA_WIN = 20  # 与张量 σ20 同口径 (warmup 闸亦同)

FEATURE_COLS = [
    # 多尺度动量
    "mom_1", "mom_3", "mom_5", "mom_10", "mom_20",
    # 波动 / range
    "rvol_5", "rvol_20", "atr_pct_14", "amplitude", "amp_mean_5",
    # 量
    "vol_ratio_5", "vol_ratio_10", "vol_ratio_20", "amount_ratio_20", "vol_chg_1",
    # 缺口
    "gap", "gap_norm",
    # 均线结构
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60", "ma5_ma20", "ma20_ma60",
    # 振荡器
    "rsi_6", "rsi_14", "macd_hist", "wr_14", "boll_pct", "boll_width",
    # 自身形态
    "shape_body", "shape_upsh", "shape_dnsh",
]


def feature_columns() -> list[str]:
    return list(FEATURE_COLS)


def _rsi(close_grp, n: int) -> pd.Series:
    """RSI(n), 简单滚动均 (确定性)。close_grp = groupby('ts_code')['close']."""
    delta = close_grp.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(_GRP_KEYS, sort=False).transform(
        lambda s: s.rolling(n, min_periods=n).mean())
    avg_loss = loss.groupby(_GRP_KEYS, sort=False).transform(
        lambda s: s.rolling(n, min_periods=n).mean())
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss==0 (纯涨) → RSI=100
    rsi = rsi.where(avg_loss > 0, 100.0).where(avg_gain.notna())
    return rsi


_GRP_KEYS: pd.Series | None = None  # module-level groupby key, 设于 encode_frame


def encode_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """对全历史面板 (ts_code, trade_date, open, high, low, close, vol, amount[, pre_close])
    生成 A0 标量 TA 特征。

    输出: KEY + FEATURE_COLS (float32)。warmup 不足 (rvol_20 缺失, 与张量 σ20 同闸) 的行剔除。
    全程 groupby(ts_code) 仅向后 shift/rolling/ewm —— 因果零泄漏。
    """
    global _GRP_KEYS
    df = panel.sort_values(KEY).reset_index(drop=True).copy()
    df["trade_date"] = df["trade_date"].astype(str)
    _GRP_KEYS = df["ts_code"]
    g = df.groupby("ts_code", sort=False)

    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    v = df["vol"]
    amount = df["amount"] if "amount" in df.columns else (c * v)
    prev_c = g["close"].shift(1)
    if "pre_close" in df.columns:
        prev_c_amp = df["pre_close"].where(df["pre_close"] > 0, prev_c)
    else:
        prev_c_amp = prev_c

    c_pos = c.where(c > 0)
    logret = np.log(c_pos / prev_c.where(prev_c > 0))

    def roll(s: pd.Series, win: int, fn: str = "mean"):
        return s.groupby(_GRP_KEYS, sort=False).transform(
            lambda x: getattr(x.rolling(win, min_periods=win), fn)())

    out: dict[str, np.ndarray] = {k: df[k].values for k in KEY}

    # ── 多尺度动量 (对数比, 跨阶) ──
    for k in (1, 3, 5, 10, 20):
        ck = g["close"].shift(k)
        out[f"mom_{k}"] = np.log(c_pos / ck.where(ck > 0)).astype(np.float32)

    # ── 波动 / range ──
    out["rvol_5"] = roll(logret, 5, "std").astype(np.float32)
    rvol_20 = roll(logret, SIGMA_WIN, "std")
    out["rvol_20"] = rvol_20.astype(np.float32)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_14 = roll(tr, 14, "mean")
    out["atr_pct_14"] = (atr_14 / c.where(c > 0)).astype(np.float32)
    amplitude = (h - l) / prev_c_amp.where(prev_c_amp > 0)
    out["amplitude"] = amplitude.astype(np.float32)
    out["amp_mean_5"] = roll(amplitude, 5, "mean").astype(np.float32)

    # ── 量 (多尺度量比 + 量变 + 成交额比) ──
    for k in (5, 10, 20):
        vma = roll(v, k, "mean")
        out[f"vol_ratio_{k}"] = (v / vma.where(vma > 0)).astype(np.float32)
    amt_ma20 = roll(amount, 20, "mean")
    out["amount_ratio_20"] = (amount / amt_ma20.where(amt_ma20 > 0)).astype(np.float32)
    prev_v = g["vol"].shift(1)
    out["vol_chg_1"] = np.log(v.where(v > 0) / prev_v.where(prev_v > 0)).astype(np.float32)

    # ── 缺口 (O vs 前 C; 张量 shape_gap 标量版) ──
    gap = (o - prev_c) / prev_c.where(prev_c > 0)
    out["gap"] = gap.astype(np.float32)
    out["gap_norm"] = (gap / rvol_20.where(rvol_20 > 0)).astype(np.float32)

    # ── 均线结构 ──
    sma = {k: roll(c, k, "mean") for k in (5, 10, 20, 60)}
    for k in (5, 10, 20, 60):
        out[f"ma_ratio_{k}"] = (c / sma[k].where(sma[k] > 0) - 1.0).astype(np.float32)
    out["ma5_ma20"] = (sma[5] / sma[20].where(sma[20] > 0) - 1.0).astype(np.float32)
    out["ma20_ma60"] = (sma[20] / sma[60].where(sma[60] > 0) - 1.0).astype(np.float32)

    # ── 振荡器 ──
    out["rsi_6"] = _rsi(g["close"], 6).astype(np.float32)
    out["rsi_14"] = _rsi(g["close"], 14).astype(np.float32)
    ema12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    macd = ema12 - ema26
    signal = macd.groupby(_GRP_KEYS, sort=False).transform(
        lambda s: s.ewm(span=9, adjust=False).mean())
    out["macd_hist"] = (macd - signal).astype(np.float32)
    hmax14 = roll(h, 14, "max")
    lmin14 = roll(l, 14, "min")
    wr_rng = (hmax14 - lmin14).where(lambda s: s > 0)
    out["wr_14"] = ((hmax14 - c) / wr_rng * -100.0).astype(np.float32)
    std20 = roll(c, 20, "std")
    out["boll_pct"] = ((c - (sma[20] - 2 * std20)) /
                       (4 * std20).where(lambda s: s > 0)).astype(np.float32)
    out["boll_width"] = (4 * std20 / sma[20].where(sma[20] > 0)).astype(np.float32)

    # ── 自身形态 (lag-free; 张量 shape 通道标量版) ──
    rng = (h - l).where(lambda s: s > 0)
    out["shape_body"] = ((c - o) / rng).astype(np.float32)
    out["shape_upsh"] = ((h - np.maximum(o, c)) / rng).astype(np.float32)
    out["shape_dnsh"] = ((np.minimum(o, c) - l) / rng).astype(np.float32)

    res = pd.DataFrame(out)
    feat = feature_columns()

    # warmup 闸: rvol_20 (σ20) 必须就位 —— 与张量 σ20 同闸, 锚点集对齐
    valid = rvol_20.notna().values & (c > 0).values
    res = res[valid].reset_index(drop=True)

    # 剩余零星 NaN (停牌致量比/比率缺失等) → 中性: 比率类→0, ma_60 等长窗早期不足保留 NaN 由 GBDT 处理。
    # 与张量 fillna(0) 一致, 仅对短窗核心列填 0; 长窗 (ma_60/ma20_ma60) 保留 NaN。
    nan_keep = {"ma_ratio_60", "ma20_ma60"}
    fill_cols = [col for col in feat if col not in nan_keep]
    res[fill_cols] = res[fill_cols].fillna(0.0)
    res[feat] = res[feat].astype(np.float32)
    return res[KEY + feat]
