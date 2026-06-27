# -*- coding: utf-8 -*-
"""T-001 相位对齐事件研究 — ratio vs MA5 拐点 (regime 分层, 零前视).

研究问题: pump ratio = P_up/(P_down+eps) 相对价格 MA5 的"上拐点"(局部底)
          是 lead (领先, 拐点前已抬头) / sync (同步) / lag (滞后, 拐点后才抬头)?

零前视纪律 (SIGN-R04):
  - 拐点用 *滞后确认* 的 MA5 斜率变号定义: 在 t 日, 当 slope(t)>0 且 slope(t-1)<=0
    才确认一个上拐点, τ=0 = t。slope/MA5 仅用截至 t 的收盘价, 不看未来。
  - ratio 在 τ+k (k∈[-10,10]) 的取值是事件研究的"观察", 每个 ratio 值本身 causal
    (v3c 逐日独立推理, 只用 t 及之前特征)。这是描述性事件研究, 不是交易规则。
  - regime 用过去 60 日 MA5/MA60 上下穿次数 (ma5_60_cross_60d) 定义, 全 causal,
    与 analyze_choppy_regime_pump_consistency.py 一致。

checkpoint (SIGN-R08):
  - ratio 序列推理结果落 research/cache/ratio_series.parquet, 存在则跳过重算。

产出:
  - research/cache/ratio_series.parquet      (ts_code, trade_date, pump_up, pump_down, ratio)
  - research/cache/t001_results.json         (各 regime 相位曲线 + lead/lag 天数 + 建议 status)
  - 控制台打印相位平均曲线 (±10 日)
裁决文件 research/verdicts/T-001.json 由调用方依据本脚本真实数字写入。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

CACHE = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
RATIO_CACHE = CACHE / "ratio_series.parquet"
RESULTS = CACHE / "t001_results.json"
PROD = ROOT / "output" / "production"

WIN_START = "20240601"   # ~19 月窗口, 覆盖趋势/震荡多种 regime
WIN_END = "20260101"
EPS = 0.01
K_MIN, K_MAX = -10, 10   # 相位窗口 ±10 日


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def compute_ratio_series() -> pd.DataFrame:
    """v3c 逐日推理 -> pump_up/pump_down/ratio。带 checkpoint。"""
    if RATIO_CACHE.exists():
        print(f"[ratio] checkpoint 命中 {RATIO_CACHE.name}, 跳过推理", flush=True)
        return pd.read_parquet(RATIO_CACHE)

    print(f"[ratio] load_window {WIN_START}-{WIN_END} (含 mfk)...", flush=True)
    daily = load_window(WIN_START, WIN_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    print(f"[ratio] daily {len(daily):,} 行 / {daily['ts_code'].nunique()} 股", flush=True)

    b, fc, ind_map = load_model("r5_pump_3way_lgbm_v3c")
    if ind_map:
        daily["industry_id"] = (daily["industry"].fillna("unknown")
                                .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily.columns:
            daily[c] = 0.0
    # 分块推理控内存
    out = []
    n = len(daily)
    CHUNK = 300_000
    for s in range(0, n, CHUNK):
        sl = daily.iloc[s:s + CHUNK]
        X = sl[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = b.predict(X)
        out.append(pd.DataFrame({
            "ts_code": sl["ts_code"].values,
            "trade_date": sl["trade_date"].values,
            "pump_down": proba[:, 1],
            "pump_up": proba[:, 2],
        }))
        print(f"  推理 {min(s + CHUNK, n):,}/{n:,}", flush=True)
    res = pd.concat(out, ignore_index=True)
    res["ratio"] = res["pump_up"] / (res["pump_down"] + EPS)
    res.to_parquet(RATIO_CACHE, index=False)
    print(f"[ratio] 落盘 {RATIO_CACHE.name} ({len(res):,} 行)", flush=True)
    return res


def compute_ma_regime() -> pd.DataFrame:
    """从 daily OHLC 算 MA5 上拐点 (滞后确认) + regime (60d MA5/MA60 穿越数)。全 causal。"""
    print("[ma] 加载 daily 全历史算 MA/拐点/regime ...", flush=True)
    ddir = ROOT / "output" / "tushare_cache" / "daily"
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(ddir.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = big.groupby("ts_code")["close"]
    big["ma5"] = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    big["ma60"] = g.transform(lambda s: s.rolling(60, min_periods=60).mean())
    # MA5 斜率 (截至 t 的差分, causal)
    big["ma5_slope"] = big.groupby("ts_code")["ma5"].diff()
    big["ma5_slope_prev"] = big.groupby("ts_code")["ma5_slope"].shift(1)
    # 滞后确认上拐点: slope 由 <=0 翻 >0, 在 t 日确认 (只用截至 t 数据)
    big["upturn"] = (big["ma5_slope"] > 0) & (big["ma5_slope_prev"] <= 0)
    # regime: 过去 60 日 MA5/MA60 上下穿次数
    big["ma5_above"] = (big["ma5"] > big["ma60"]).astype("float")
    big["cross"] = big.groupby("ts_code")["ma5_above"].diff().abs()
    big["ma5_60_cross_60d"] = (big.groupby("ts_code")["cross"]
                               .transform(lambda s: s.rolling(60, min_periods=30).sum()))
    return big[["ts_code", "trade_date", "ma5", "ma60", "upturn", "ma5_60_cross_60d"]]


def event_study(ratio: pd.DataFrame, ma: pd.DataFrame) -> dict:
    print("[event] 合并 + 构建相位窗口 ...", flush=True)
    df = ma.merge(ratio[["ts_code", "trade_date", "pump_up", "pump_down", "ratio"]],
                  on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    # 每个 t0 的 τ+k ratio: groupby shift(-k)
    for k in range(K_MIN, K_MAX + 1):
        df[f"r_{k}"] = df.groupby("ts_code")["ratio"].shift(-k)
    cols = [f"r_{k}" for k in range(K_MIN, K_MAX + 1)]

    # 事件: 上拐点 + 窗口落在 ratio 计算区间内 (要求 21 点全非空)
    ev = df[df["upturn"]].copy()
    ev = ev[(ev["trade_date"] >= WIN_START) & (ev["trade_date"] <= WIN_END)]
    ev = ev.dropna(subset=cols + ["ma5_60_cross_60d"])
    print(f"[event] 有效上拐点事件: {len(ev):,} 个 / {ev['ts_code'].nunique()} 股", flush=True)

    # regime 分层 (全样本 p20/p80, 与既有口径一致)
    p20 = ev["ma5_60_cross_60d"].quantile(0.20)
    p80 = ev["ma5_60_cross_60d"].quantile(0.80)
    ev["regime"] = "mid"
    ev.loc[ev["ma5_60_cross_60d"] <= p20, "regime"] = "trend"
    ev.loc[ev["ma5_60_cross_60d"] >= p80, "regime"] = "choppy"

    offsets = list(range(K_MIN, K_MAX + 1))
    out = {"window": [WIN_START, WIN_END], "n_events_total": int(len(ev)),
           "n_stocks": int(ev["ts_code"].nunique()),
           "regime_cut": {"cross_p20": float(p20), "cross_p80": float(p80)},
           "offsets": offsets, "regimes": {}}

    def curve_stats(sub):
        mat = sub[cols].to_numpy(dtype=float)          # (n_ev, 21) 原始 ratio
        raw_mean = mat.mean(axis=0)
        # 事件内 demean (去量纲, 提取相位形状; 对 ratio 重尾用 log 稳健化)
        lm = np.log(mat + EPS)
        shape = (lm - lm.mean(axis=1, keepdims=True)).mean(axis=0)
        # 相位地标 (基于 raw 均值曲线, 仅在内区 [-8,+8] 搜索以剔除边界水平漂移伪峰):
        #   trough  = 曲线最低点 (信号底)
        #   peak    = 曲线最高点 (信号顶)
        #   ignition= 最大单日抬升所在偏移 (信号"起飞"瞬间, argmax 的 Δraw)
        lo, hi = offsets.index(-8), offsets.index(8)        # 内区切片
        seg = raw_mean[lo:hi + 1]
        seg_off = offsets[lo:hi + 1]
        peak_off = seg_off[int(np.argmax(seg))]
        trough_off = seg_off[int(np.argmin(seg))]
        d = np.diff(raw_mean)                                 # Δraw, 索引 i 对应 offsets[i+1]
        di_lo, di_hi = lo, hi - 1                             # 内区一阶差分范围
        ign_off = offsets[di_lo + int(np.argmax(d[di_lo:di_hi + 1])) + 1]
        return raw_mean, shape, peak_off, trough_off, ign_off

    for reg in ["all", "trend", "mid", "choppy"]:
        sub = ev if reg == "all" else ev[ev["regime"] == reg]
        if len(sub) < 50:
            out["regimes"][reg] = {"n": int(len(sub)), "note": "样本不足<50"}
            continue
        raw_mean, shape, peak_off, trough_off, ign_off = curve_stats(sub)
        out["regimes"][reg] = {
            "n": int(len(sub)),
            "raw_ratio_curve": [round(float(x), 4) for x in raw_mean],
            "log_shape_curve": [round(float(x), 4) for x in shape],
            "peak_offset": int(peak_off),
            "trough_offset": int(trough_off),
            "ignition_offset": int(ign_off),
            "ratio_at_-1": round(float(raw_mean[offsets.index(-1)]), 4),
            "ratio_at_0": round(float(raw_mean[offsets.index(0)]), 4),
            "ratio_at_+1": round(float(raw_mean[offsets.index(1)]), 4),
        }

    # status 裁决: 以最稳健的地标 = ignition (最大单日抬升) 相对 τ=0 的偏移。
    #   ignition < 0 -> lead (拐点前已起飞), == 0 -> sync (同步起飞), > 0 -> lag。
    #   peak_offset 作为辅助 (信号顶相对拐点的滞后)。
    ign = out["regimes"]["all"]["ignition_offset"]
    pk = out["regimes"]["all"]["peak_offset"]
    if ign < 0:
        status = "lead"
    elif ign == 0:
        status = "sync"
    else:
        status = "lag"
    out["suggested_status"] = status
    out["all_ignition_offset"] = int(ign)
    out["all_peak_offset"] = int(pk)
    # regime 不变性: 各 regime 的 ignition/peak 是否一致
    igns = {r: out["regimes"][r].get("ignition_offset") for r in ["trend", "mid", "choppy"]
            if "ignition_offset" in out["regimes"][r]}
    pks = {r: out["regimes"][r].get("peak_offset") for r in ["trend", "mid", "choppy"]
           if "peak_offset" in out["regimes"][r]}
    out["regime_invariant_ignition"] = len(set(igns.values())) == 1
    out["regime_invariant_peak"] = len(set(pks.values())) == 1
    out["regime_ignitions"] = igns
    out["regime_peaks"] = pks
    return out


def main():
    t0 = time.time()
    print("\n=== T-001 相位对齐事件研究 (ratio vs MA5 上拐点) ===\n", flush=True)
    ratio = compute_ratio_series()
    ma = compute_ma_regime()
    res = event_study(ratio, ma)

    print("\n=== 相位平均曲线 (raw ratio 均值, ±10 日) ===", flush=True)
    offs = res["offsets"]
    hdr = "regime    n      " + " ".join(f"{k:>6d}" for k in offs)
    print(hdr, flush=True)
    for reg in ["all", "trend", "mid", "choppy"]:
        r = res["regimes"].get(reg, {})
        if "raw_ratio_curve" not in r:
            print(f"{reg:8s} {r.get('n', 0):>6d}  (样本不足)", flush=True)
            continue
        cells = " ".join(f"{v:>6.2f}" for v in r["raw_ratio_curve"])
        print(f"{reg:8s} {r['n']:>6d}  {cells}", flush=True)

    print("\n=== 相位地标 (单位:日, <0=领先拐点 τ0) ===", flush=True)
    for reg in ["all", "trend", "mid", "choppy"]:
        r = res["regimes"].get(reg, {})
        if "peak_offset" not in r:
            continue
        print(f"  {reg:8s} trough={r['trough_offset']:+d}d  ignition={r['ignition_offset']:+d}d  "
              f"peak={r['peak_offset']:+d}d", flush=True)
    print(f"\n建议 status = {res['suggested_status']}  "
          f"(all ignition={res['all_ignition_offset']:+d}d, peak={res['all_peak_offset']:+d}d; "
          f"regime 不变 ignition={res['regime_invariant_ignition']} peak={res['regime_invariant_peak']})",
          flush=True)

    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图: ±10 日相位曲线 (raw ratio) 分 regime
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for reg, c in [("all", "k"), ("trend", "tab:green"),
                       ("mid", "tab:orange"), ("choppy", "tab:red")]:
            r = res["regimes"].get(reg, {})
            if "raw_ratio_curve" not in r:
                continue
            ax.plot(offs, r["raw_ratio_curve"], marker="o", ms=3, color=c,
                    label=f"{reg} (n={r['n']:,})")
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_xlabel("offset to MA5 upturn (days)  [tau=0]")
        ax.set_ylabel("mean pump ratio = P_up/(P_down+eps)")
        ax.set_title("T-001 phase-aligned ratio around MA5 upturn (lagged-confirmed)")
        ax.legend()
        ax.grid(alpha=0.3)
        figp = CACHE / "figs" / "t001_phase_curve.png"
        fig.tight_layout(); fig.savefig(figp, dpi=110)
        print(f"[fig] {figp.relative_to(ROOT)}", flush=True)
    except Exception as e:
        print(f"[fig] 跳过绘图: {e}", flush=True)

    print(f"\n[done] 结果 -> {RESULTS.relative_to(ROOT)}  耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
