# -*- coding: utf-8 -*-
"""CB-001 — 蓄势候选特征 (MA20走平 + 收敛 + 量能) + 僵尸分解.

用户核心洞察: 上涨前形态是一族 (回调/宽幅横盘/微涨震荡/旗形三角楔形向上整理),
统一特征 = 上涨前 **MA20 走平 + 收敛蓄势**。系统只抓"下蹲后起跳"(past_r5<0),
漏掉 past_r5≥0 的横盘/三角启动; 第 6 铁律"横盘僵尸过滤"可能误杀蓄势基底。

本任务量化这族形态, 全历史因果落盘 (CB-003 算正交 IC, CB-004 做 walk-forward 臂)。

特征族 (全部 cs_ 前缀, 决策时刻 t 只用 ≤t 的 OHLCV, 零泄漏):
  MA20 走平: cs_ma20_slope (过去 SLOPE_WIN 日 MA20 变化率), cs_ma20_absslope (低=平)
  收敛(三角/楔): cs_boll_bw (布林相对带宽), cs_bw_squeeze (带宽/60d基线, <1收敛),
                cs_amplitude20 (20d振幅/MA20), cs_amp_squeeze (振幅/60d基线, <1收窄)
  量能: cs_vol_ratio (5d量/20d量, <1缩量), cs_vol_ratio60 (20d量/60d量)
  基底回看: cs_past_r5 (过去 5 日收益, backward; 用于 base_type 分类)
  候选合成: cs_is_consolidation (MA20平 & 收敛 & 缩量)

基底类型 base_type ∈ {pullback(past_r5<0), flat_base(past_r5≈0 & MA20平),
                      up_drift(微涨震荡), other}

僵尸分解 (复用 zombie_filter.compute_zombie_factors, MA60±5% 横盘 N≥20):
  cs_zombie          : 原 is_zombie
  cs_zombie_dead     : 死横盘 — is_zombie & 无收敛/无缩量 (阴跌死水, MA60 走平偏跌)
  cs_zombie_coiling  : 蓄势横盘 — is_zombie & 收敛(带宽squeeze) & 缩量 (蓄势基底, 疑被误杀)

纪律:
  - ST 源头排除 (SIGN-R06)。
  - 零泄漏 (SIGN-R04): 输出列仅 key + cs_* + base_type, 决策时刻已知量 (cs_past_r5 是
    backward 收益, 非 forward; 列名不命中 forward 黑名单)。
  - 确定性 (SIGN): 抽样股票重算逐位比对。
  - checkpoint (SIGN-R08): 重 rolling 结果缓存 research/cache/cb001/。

输出:
  research/features/consolidation.parquet (key: ts_code, trade_date; 列 cs_* + base_type)
  research/verdicts/CB-001.json (status=built + 分布)
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.zombie_filter import compute_zombie_factors  # noqa: E402

DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
CACHE_DIR = ROOT / "research" / "cache" / "cb001"
PANEL_CACHE = CACHE_DIR / "consolidation_panel.parquet"
OUT_DIR = ROOT / "research" / "features"
OUT_P = OUT_DIR / "consolidation.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "CB-001.json"

KEY = ["ts_code", "trade_date"]

# ── 形态参数 ──
SLOPE_WIN = 10        # MA20 走平用过去 10 日 MA20 变化率衡量
FLAT_THR = 0.02       # |MA20 10日变化率| < 2% = 走平
PULLBACK_THR = 0.03   # past_r5 < -3% = 下蹲(回调)
DRIFT_HI = 0.10       # past_r5 in (FLAT, +10%] = 微涨震荡; 更高=已发动归 other
SQUEEZE_THR = 1.0     # bw_squeeze / amp_squeeze < 1 = 收敛
VOL_SHRINK_THR = 1.0  # vol_ratio < 1 = 缩量
MIN_LEN = 80          # 起算所需最少历史长度

# forward 黑名单 (与 verify.py 一致), 落盘前自查输出列绝不命中
FORWARD_EXACT = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                {"max_gain", "maxgain", "future_ret", "fwd_ret"}
FORWARD_PATTERNS = [re.compile(p) for p in
                    (r"^r\d+_next", r"^fwd_", r"^future_", r"^next_",
                     r"_forward$", r"^label$", r"^target$")]


def load_st_set() -> set:
    if not BASIC_P.exists():
        print("  ⚠️ stock_basic.parquet 不存在, ST 排除跳过", flush=True)
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_panel(st: set) -> pd.DataFrame:
    print(f"[panel] 加载全历史 OHLCV ({DAILY_DIR}) ...", flush=True)
    cols = ["ts_code", "trade_date", "high", "low", "close", "vol"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted(DAILY_DIR.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.drop_duplicates(subset=KEY, keep="last")
    big = big[big["close"].notna() & (big["close"] > 0)]
    if st:
        before = len(big)
        big = big[~big["ts_code"].isin(st)]
        print(f"  ST 源头排除: {before - len(big):,} 行 ({len(st)} 只 ST)", flush=True)
    return big.sort_values(KEY).reset_index(drop=True)


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    """对每股按时间序列计算蓄势形态特征 + 僵尸分解 (全 backward, 因果)。"""
    out = panel.copy()
    g = out.groupby("ts_code", sort=False)
    close = out["close"]

    # MA20 / MA60 走平
    ma20 = g["close"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    out["cs_ma20_slope"] = (ma20 / g_shift(ma20, out, SLOPE_WIN) - 1.0)
    out["cs_ma20_absslope"] = out["cs_ma20_slope"].abs()

    # 收敛: 布林相对带宽 + 振幅, 各除 60d 基线 = squeeze
    std20 = g["close"].transform(lambda s: s.rolling(20, min_periods=15).std())
    bw = 2.0 * std20 / ma20
    out["cs_boll_bw"] = bw
    bw_ref = g_rollmean(bw, out, 60, 30)
    out["cs_bw_squeeze"] = bw / bw_ref

    hi20 = g["high"].transform(lambda s: s.rolling(20, min_periods=15).max())
    lo20 = g["low"].transform(lambda s: s.rolling(20, min_periods=15).min())
    amp = (hi20 - lo20) / ma20
    out["cs_amplitude20"] = amp
    amp_ref = g_rollmean(amp, out, 60, 30)
    out["cs_amp_squeeze"] = amp / amp_ref

    # 量能: 缩量蓄势
    vma5 = g["vol"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    vma20 = g["vol"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    vma60 = g["vol"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    out["cs_vol_ratio"] = vma5 / vma20
    out["cs_vol_ratio60"] = vma20 / vma60

    # 基底回看收益 (backward, 非 forward)
    out["cs_past_r5"] = close / g_shift(close, out, 5) - 1.0

    # 候选合成: MA20平 & 收敛 & 缩量
    flat = out["cs_ma20_absslope"] < FLAT_THR
    converging = (out["cs_bw_squeeze"] < SQUEEZE_THR) | (out["cs_amp_squeeze"] < SQUEEZE_THR)
    shrinking = out["cs_vol_ratio"] < VOL_SHRINK_THR
    out["cs_is_consolidation"] = (flat & converging & shrinking).astype("boolean")
    # 缺任一输入则候选标签设为缺失而非 False
    valid_cand = (out["cs_ma20_absslope"].notna() & out["cs_bw_squeeze"].notna()
                  & out["cs_amp_squeeze"].notna() & out["cs_vol_ratio"].notna())
    out.loc[~valid_cand, "cs_is_consolidation"] = pd.NA

    # base_type 分类
    out["base_type"] = classify_base_type(out, flat)

    return out, ma20


def g_shift(series: pd.Series, frame: pd.DataFrame, n: int) -> pd.Series:
    return series.groupby(frame["ts_code"], sort=False).shift(n)


def g_rollmean(series: pd.Series, frame: pd.DataFrame, win: int, minp: int) -> pd.Series:
    return series.groupby(frame["ts_code"], sort=False).transform(
        lambda s: s.rolling(win, min_periods=minp).mean())


def classify_base_type(out: pd.DataFrame, flat: pd.Series) -> pd.Series:
    r5 = out["cs_past_r5"]
    bt = pd.Series("other", index=out.index, dtype=object)
    pullback = r5 < -PULLBACK_THR
    flat_base = (r5.abs() <= PULLBACK_THR) & flat
    up_drift = (r5 > PULLBACK_THR) & (r5 <= DRIFT_HI)
    bt[up_drift] = "up_drift"
    bt[pullback] = "pullback"
    bt[flat_base] = "flat_base"   # flat_base 优先级最高 (横盘启动核心)
    bt[r5.isna()] = "na"
    return bt


def split_zombie(panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """复用 zombie_filter 计算 is_zombie, 再用收敛+缩量拆 死横盘 vs 蓄势横盘。"""
    parts = []
    for ts, d in panel.groupby("ts_code", sort=False):
        if len(d) < MIN_LEN:
            z = d[["ts_code", "trade_date"]].copy()
            z["cs_zombie"] = pd.NA
            z["cs_zombie_ma60_slope"] = np.nan
            parts.append(z)
            continue
        zf = compute_zombie_factors(d[["trade_date", "close"]])
        z = pd.DataFrame({
            "ts_code": ts,
            "trade_date": zf["trade_date"].values,
            "cs_zombie": zf["is_zombie"].astype("boolean").values,
            "cs_zombie_ma60_slope": zf["ma60_slope_short"].values,
        })
        parts.append(z)
    zdf = pd.concat(parts, ignore_index=True)
    m = feats.merge(zdf, on=KEY, how="left")

    is_z = m["cs_zombie"].fillna(False).astype(bool)
    # 蓄势横盘: 僵尸 & 收敛 & 缩量 (旗形/三角/楔蓄势, 疑被第6铁律误杀)
    coiling = is_z & (
        ((m["cs_bw_squeeze"] < SQUEEZE_THR) | (m["cs_amp_squeeze"] < SQUEEZE_THR))
        & (m["cs_vol_ratio"] < VOL_SHRINK_THR))
    # 死横盘: 僵尸 & 非蓄势 (无收敛或不缩量, MA60 走平偏跌的阴跌死水)
    dead = is_z & ~coiling
    m["cs_zombie_coiling"] = pd.Series(np.where(is_z, coiling, pd.NA),
                                       index=m.index).astype("boolean")
    m["cs_zombie_dead"] = pd.Series(np.where(is_z, dead, pd.NA),
                                    index=m.index).astype("boolean")
    return m


def leakage_selfcheck(cols) -> None:
    bad = [c for c in cols if c.lower() in FORWARD_EXACT
           or any(p.search(c.lower()) for p in FORWARD_PATTERNS)]
    assert not bad, f"输出含 forward 字段 (泄漏): {bad}"


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    st = load_st_set()
    print(f"[st] ST 源头排除集合 {len(st)} 只", flush=True)
    panel = load_panel(st)
    print(f"[panel] {len(panel):,} 行 / {panel['ts_code'].nunique()} 股 / "
          f"{panel['trade_date'].nunique()} 日 "
          f"({panel['trade_date'].min()}~{panel['trade_date'].max()})", flush=True)

    if PANEL_CACHE.exists():
        print(f"[feat] checkpoint 命中 {PANEL_CACHE.name}", flush=True)
        m = pd.read_parquet(PANEL_CACHE)
    else:
        t = time.time()
        feats, _ = compute_features(panel)
        print(f"[feat] 形态特征算完 {time.time()-t:.0f}s", flush=True)
        t = time.time()
        m = split_zombie(panel, feats)
        print(f"[feat] 僵尸分解算完 {time.time()-t:.0f}s", flush=True)
        m.to_parquet(PANEL_CACHE, index=False)
        print(f"[feat] checkpoint 落盘 {PANEL_CACHE.name}", flush=True)

    feat_cols = [c for c in m.columns if c.startswith("cs_")] + ["base_type"]
    out = m[KEY + feat_cols].copy()

    # ── 泄漏自查 (SIGN-R04) ──
    leakage_selfcheck(out.columns)

    # ── 确定性自检: 抽 30 只股票重算逐位比对 ──
    samp_codes = sorted(panel["ts_code"].unique())[::max(1, panel["ts_code"].nunique() // 30)][:30]
    sp = panel[panel["ts_code"].isin(samp_codes)].copy()
    rf, _ = compute_features(sp)
    rm = split_zombie(sp, rf)
    ra = rm[KEY + feat_cols].sort_values(KEY).reset_index(drop=True)
    rb = out[out["ts_code"].isin(samp_codes)].sort_values(KEY).reset_index(drop=True)
    num_cols = [c for c in feat_cols if ra[c].dtype.kind in "fc"]
    det_num = bool(np.allclose(ra[num_cols].fillna(-9.99).values,
                               rb[num_cols].fillna(-9.99).values, atol=1e-9))
    det_bt = bool((ra["base_type"].values == rb["base_type"].values).all())
    deterministic = det_num and det_bt
    assert deterministic, f"确定性自检失败 num={det_num} base_type={det_bt}"

    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)}  ({len(out):,} 行 × {len(out.columns)} 列)",
          flush=True)

    # ── 分布摘要 ──
    bt_dist = {k: int(v) for k, v in out["base_type"].value_counts().to_dict().items()}
    bt_frac = {k: round(v / len(out), 4) for k, v in bt_dist.items()}
    cont = ["cs_ma20_slope", "cs_ma20_absslope", "cs_boll_bw", "cs_bw_squeeze",
            "cs_amplitude20", "cs_amp_squeeze", "cs_vol_ratio", "cs_vol_ratio60",
            "cs_past_r5"]
    dist = {c: {"mean": round(float(out[c].mean()), 6),
                "std": round(float(out[c].std()), 6),
                "p5": round(float(out[c].quantile(0.05)), 6),
                "p50": round(float(out[c].quantile(0.50)), 6),
                "p95": round(float(out[c].quantile(0.95)), 6),
                "nonnull": round(float(out[c].notna().mean()), 4)}
            for c in cont}
    n_cand = int(out["cs_is_consolidation"].fillna(False).astype(bool).sum())
    n_z = int(out["cs_zombie"].fillna(False).astype(bool).sum())
    n_z_coil = int(out["cs_zombie_coiling"].fillna(False).astype(bool).sum())
    n_z_dead = int(out["cs_zombie_dead"].fillna(False).astype(bool).sum())

    verdict = {
        "task": "CB-001",
        "status": "built",
        "conclusion": (
            f"蓄势候选特征落盘 {len(out):,} 行 × {len(feat_cols)} 列 "
            f"({out['trade_date'].min()}~{out['trade_date'].max()}); "
            f"base_type 占比 {bt_frac}; "
            f"蓄势候选(cs_is_consolidation) {n_cand:,} 行 ({n_cand/len(out)*100:.1f}%); "
            f"僵尸 {n_z:,} 拆 蓄势横盘 {n_z_coil:,}/死横盘 {n_z_dead:,} "
            f"(蓄势占僵尸 {n_z_coil/max(n_z,1)*100:.1f}%); "
            f"确定性={deterministic}, 零泄漏(全 backward), ST 源头排除。"),
        "artifact": str(OUT_P.relative_to(ROOT)),
        "design": {
            "ma20_flat": f"cs_ma20_slope=MA20 过去{SLOPE_WIN}日变化率; |slope|<{FLAT_THR} 视为走平",
            "convergence": "cs_bw_squeeze=布林带宽/60d基线, cs_amp_squeeze=20d振幅/60d基线, <1 收敛(三角/楔)",
            "volume": f"cs_vol_ratio=5d量/20d量 <{VOL_SHRINK_THR} 缩量蓄势",
            "base_type": {"pullback": f"past_r5<-{PULLBACK_THR}", "flat_base": f"|past_r5|<={PULLBACK_THR} & MA20平",
                          "up_drift": f"{PULLBACK_THR}<past_r5<={DRIFT_HI}", "other": "其余(已发动/未走平)"},
            "zombie_split": "cs_zombie_coiling=僵尸&收敛&缩量(蓄势疑误杀); cs_zombie_dead=僵尸&非蓄势(阴跌死水)",
        },
        "metrics": {
            "rows": int(len(out)),
            "n_stocks": int(out["ts_code"].nunique()),
            "n_dates": int(out["trade_date"].nunique()),
            "date_range": [out["trade_date"].min(), out["trade_date"].max()],
            "feature_cols": feat_cols,
            "base_type_dist": bt_dist,
            "base_type_frac": bt_frac,
            "n_candidate": n_cand,
            "candidate_frac": round(n_cand / len(out), 4),
            "n_zombie": n_z,
            "n_zombie_coiling": n_z_coil,
            "n_zombie_dead": n_z_dead,
            "zombie_coiling_frac_of_zombie": round(n_z_coil / max(n_z, 1), 4),
            "dist": dist,
            "deterministic": deterministic,
            "leakage_free": True,
        },
        "guardrails": ["SIGN-R04 泄漏自查通过(全 backward)", "SIGN-R06 ST 源头排除",
                       "SIGN-R08 panel checkpoint", "确定性逐位比对"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"  base_type 占比: {bt_frac}")
    print(f"  候选 {n_cand:,} ({n_cand/len(out)*100:.1f}%); 僵尸 {n_z:,} "
          f"(蓄势 {n_z_coil:,}/死 {n_z_dead:,})")
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
