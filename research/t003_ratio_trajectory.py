# -*- coding: utf-8 -*-
"""T-003 ratio 轨迹特征 — 5 类全因果 (causal) 轨迹特征 + 描述性 IC.

研究动机: 生产 v3c 逐日独立推理, 模型"看不到" ratio 的*时序轨迹* (它每天只吃当日特征,
          不知道 ratio 在涨/跌、加速/减速、处在自身历史的高位/低位、距上一个峰多久)。
          T-001/T-002 已证: 非平稳来自相位/时序而非窗口配比 → ratio 的*轨迹*是真正正交的增量。
          本任务把这条轨迹显式特征化, 留给 T-004 门控 / T-005 walk-forward 验证。

零前视纪律 (SIGN-R04):
  - 所有特征只用 t 及之前的数据 (diff / shift(+k) / 过去窗口 rolling)。绝不 shift(-k)。
  - 输出 parquet 列名全部避开 forward-field 黑名单 (r5/r10/.../dd*/fwd_/future_/next_/...)。
  - 用于描述性 IC 的 5 日前向收益只在内存计算, *绝不*写进 features parquet (否则触发 guard)。
  - verify.py 的 leakage guard 会扫 research/features/*.parquet 列名, 本脚本产物须通过。

5 类轨迹特征 (全 causal):
  1. Δratio  (速度):     ratio_vel_1 / ratio_vel_3 / ratio_vel_5     一阶差分, ratio 在涨还是跌、多快
  2. Δ²ratio (加速度):   ratio_acc_1 / ratio_acc_3                   二阶差分, 涨势在加速还是衰竭
  3. 背离象限:           ratio_div_quad (-2/-1/1/2) + ratio_div_strength   ratio 方向 vs 价格方向
  4. 自分位:             ratio_selfpct_20 / ratio_selfpct_60         ratio 在自身过去 20/60 日的分位
  5. 距峰天数:           ratio_days_since_hi_60 / ratio_days_since_lo_60   距上一个 60 日高/低多少根 bar

checkpoint (SIGN-R08): 产物 research/features/ratio_traj.parquet 存在即跳过重算。

产出:
  - research/features/ratio_traj.parquet   (ts_code, trade_date, ratio, + 11 轨迹特征)
  - research/cache/t003_results.json       (单因子 RankIC + regime 分层 IC, 描述性)
  - 控制台打印 IC 摘要
裁决 research/verdicts/T-003.json 由本脚本依真实数字写入 (status=built)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache"
FEATURES = ROOT / "research" / "features"
FEATURES.mkdir(parents=True, exist_ok=True)
RATIO_CACHE = CACHE / "ratio_series.parquet"
OUT_PARQUET = FEATURES / "ratio_traj.parquet"
RESULTS = CACHE / "t003_results.json"
VERDICT = ROOT / "research" / "verdicts" / "T-003.json"

WIN_START, WIN_END = "20240601", "20260101"
FWD_H = 5            # 描述性 IC 用 5 日前向收益 (与 r5 pump 模型对齐); *不入库*
SELF_W1, SELF_W2 = 20, 60
PEAK_W = 60

# 产物列名 (全 causal); 显式列出便于审计列名不撞 forward-field 黑名单
FEATURE_COLS = [
    "ratio_vel_1", "ratio_vel_3", "ratio_vel_5",          # 1 速度
    "ratio_acc_1", "ratio_acc_3",                          # 2 加速度
    "ratio_div_quad", "ratio_div_strength",               # 3 背离象限
    "ratio_selfpct_20", "ratio_selfpct_60",               # 4 自分位
    "ratio_days_since_hi_60", "ratio_days_since_lo_60",    # 5 距峰天数
]


def load_close() -> pd.DataFrame:
    """daily close (causal), 供背离象限的价格动量 + 描述性前向收益。"""
    print("[close] 加载 daily close 全历史 ...", flush=True)
    ddir = ROOT / "output" / "tushare_cache" / "daily"
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(ddir.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    return big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def compute_regime() -> pd.DataFrame:
    """T-001/T-002 同口径 regime: 过去 60 日 MA5/MA60 穿越数, 全 causal。"""
    print("[regime] 算 MA5/MA60 穿越数 (regime 分层用) ...", flush=True)
    ddir = ROOT / "output" / "tushare_cache" / "daily"
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(ddir.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = big.groupby("ts_code")["close"]
    big["ma5"] = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    big["ma60"] = g.transform(lambda s: s.rolling(60, min_periods=60).mean())
    big["ma5_above"] = (big["ma5"] > big["ma60"]).astype("float")
    big["cross"] = big.groupby("ts_code")["ma5_above"].diff().abs()
    big["ma5_60_cross_60d"] = (big.groupby("ts_code")["cross"]
                               .transform(lambda s: s.rolling(60, min_periods=30).sum()))
    return big[["ts_code", "trade_date", "ma5_60_cross_60d"]]


def _days_since_extreme(s: pd.Series, w: int, hi: bool) -> pd.Series:
    """距过去 w 日窗口内极值多少根 bar (causal, raw rolling apply)。0=今天就是极值。"""
    fn = (lambda a: a.shape[0] - 1 - int(np.argmax(a))) if hi else \
         (lambda a: a.shape[0] - 1 - int(np.argmin(a)))
    return s.rolling(w, min_periods=max(10, w // 3)).apply(fn, raw=True)


def build_features(ratio: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """构建 5 类 causal 轨迹特征。"""
    df = ratio.merge(close[["ts_code", "trade_date", "close"]],
                     on=["ts_code", "trade_date"], how="left")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = df.groupby("ts_code")["ratio"]

    # 1) 速度 Δratio (一阶差分, 多尺度)
    print("[feat] 1/5 速度 Δratio ...", flush=True)
    df["ratio_vel_1"] = g.diff(1)
    df["ratio_vel_3"] = df["ratio"] - g.shift(3)
    df["ratio_vel_5"] = df["ratio"] - g.shift(5)

    # 2) 加速度 Δ²ratio (二阶差分)
    print("[feat] 2/5 加速度 Δ²ratio ...", flush=True)
    df["ratio_acc_1"] = df.groupby("ts_code")["ratio_vel_1"].diff(1)
    df["ratio_acc_3"] = (df["ratio_vel_3"]
                         - df.groupby("ts_code")["ratio_vel_3"].shift(3))

    # 3) 背离象限: ratio 方向 vs 价格动量方向 (causal)
    print("[feat] 3/5 背离象限 ...", flush=True)
    gc = df.groupby("ts_code")["close"]
    price_mom_5 = df["close"] / gc.shift(5) - 1.0
    rv5 = df["ratio_vel_5"]
    # 象限编码: 2=量价齐升 / 1=多头背离(量升价滞) / -1=空头背离(价升量滞) / -2=量价齐跌
    quad = np.where((rv5 > 0) & (price_mom_5 > 0), 2,
            np.where((rv5 > 0) & (price_mom_5 <= 0), 1,
             np.where((rv5 <= 0) & (price_mom_5 > 0), -1, -2)))
    df["ratio_div_quad"] = quad.astype("float32")
    # 连续背离强度: 各自用过去 60 日滚动 std 归一后相减 (ratio 升得比价格快 => 正)
    rv5_sd = df.groupby("ts_code")["ratio_vel_5"].transform(
        lambda s: s.rolling(60, min_periods=20).std())
    pm5 = price_mom_5.copy()
    df["_pm5"] = pm5
    pm5_sd = df.groupby("ts_code")["_pm5"].transform(
        lambda s: s.rolling(60, min_periods=20).std())
    df["ratio_div_strength"] = (rv5 / (rv5_sd + 1e-9)) - (pm5 / (pm5_sd + 1e-9))
    df.drop(columns=["_pm5"], inplace=True)

    # 4) 自分位: ratio 在自身过去 20/60 日的分位 (causal rolling rank)
    print("[feat] 4/5 自分位 (rolling rank) ...", flush=True)
    df["ratio_selfpct_20"] = g.transform(
        lambda s: s.rolling(SELF_W1, min_periods=10).rank(pct=True))
    df["ratio_selfpct_60"] = g.transform(
        lambda s: s.rolling(SELF_W2, min_periods=20).rank(pct=True))

    # 5) 距峰天数: 距过去 60 日高/低多少根 bar (causal rolling argmax/argmin)
    print("[feat] 5/5 距峰天数 (rolling argmax/min, 较慢) ...", flush=True)
    df["ratio_days_since_hi_60"] = g.transform(
        lambda s: _days_since_extreme(s, PEAK_W, hi=True))
    df["ratio_days_since_lo_60"] = g.transform(
        lambda s: _days_since_extreme(s, PEAK_W, hi=False))

    for c in FEATURE_COLS:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).astype("float32")
    return df


def descriptive_ic(df: pd.DataFrame, close: pd.DataFrame, regime: pd.DataFrame) -> dict:
    """描述性单因子 RankIC + regime 分层 IC (SIGN-R03: 仅描述, 不作 gate)。
    前向收益 fwd5 只在内存计算, 不写入 features parquet。"""
    print(f"[ic] 计算描述性 RankIC (fwd{FWD_H}, 仅描述非 gate) ...", flush=True)
    c = close.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    # *前向*收益: 仅供 IC 评估, 绝不入库
    c["_fwd"] = c.groupby("ts_code")["close"].shift(-FWD_H) / c["close"] - 1.0
    ev = df.merge(c[["ts_code", "trade_date", "_fwd"]], on=["ts_code", "trade_date"], how="left")
    ev = ev.merge(regime, on=["ts_code", "trade_date"], how="left")
    ev = ev[(ev["trade_date"] >= WIN_START) & (ev["trade_date"] <= WIN_END)]
    ev = ev.dropna(subset=["_fwd"])

    # regime 分层 (T-001/T-002 同口径)
    p20 = ev["ma5_60_cross_60d"].quantile(0.20)
    p80 = ev["ma5_60_cross_60d"].quantile(0.80)
    ev["regime"] = "mid"
    ev.loc[ev["ma5_60_cross_60d"] <= p20, "regime"] = "trend"
    ev.loc[ev["ma5_60_cross_60d"] >= p80, "regime"] = "choppy"

    def daily_rankic(sub: pd.DataFrame, col: str) -> float:
        # 每日截面 Spearman(feature, fwd) 再跨日均值 = RankIC
        d = sub[[col, "_fwd", "trade_date"]].dropna()
        if len(d) < 500:
            return float("nan")
        ics = (d.groupby("trade_date")
                 .apply(lambda x: x[col].corr(x["_fwd"], method="spearman")
                        if x[col].nunique() > 2 and len(x) >= 20 else np.nan))
        return float(np.nanmean(ics.values))

    ic = {"overall": {}, "by_regime": {}}
    for col in FEATURE_COLS:
        ic["overall"][col] = round(daily_rankic(ev, col), 4)
    for reg in ["trend", "mid", "choppy"]:
        sub = ev[ev["regime"] == reg]
        ic["by_regime"][reg] = {col: round(daily_rankic(sub, col), 4) for col in FEATURE_COLS}
    ic["fwd_horizon"] = FWD_H
    ic["regime_cut"] = {"cross_p20": float(p20), "cross_p80": float(p80)}
    ic["n_eval_rows"] = int(len(ev))
    ic["note"] = "RankIC 描述性, 非 gate (SIGN-R03); 前向收益仅内存计算未入库"
    return ic


def main():
    t0 = time.time()
    print("\n=== T-003 ratio 轨迹特征 (5 类全因果) ===\n", flush=True)
    if not RATIO_CACHE.exists():
        sys.exit(f"[err] 缺 {RATIO_CACHE} — 先跑 T-001 生成 ratio_series.parquet")
    ratio = pd.read_parquet(RATIO_CACHE)
    ratio["trade_date"] = ratio["trade_date"].astype(str)
    close = load_close()

    if OUT_PARQUET.exists():
        print(f"[feat] checkpoint 命中 {OUT_PARQUET.name}, 跳过构建", flush=True)
        feat = pd.read_parquet(OUT_PARQUET)
    else:
        feat = build_features(ratio, close)
        keep = ["ts_code", "trade_date", "ratio"] + FEATURE_COLS
        out = feat[keep].copy()
        # 入库前再次确认无 forward-field 列名 (自检, 与 verify.py 同步)
        forbidden = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                    {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                    {"max_gain", "maxgain", "future_ret", "fwd_ret", "label", "target"}
        bad = [c for c in out.columns if c.lower() in forbidden
               or c.lower().startswith(("fwd_", "future_", "next_"))
               or c.lower().endswith("_forward")]
        assert not bad, f"列名撞 forward-field 黑名单: {bad}"
        out.to_parquet(OUT_PARQUET, index=False)
        print(f"[feat] 落盘 {OUT_PARQUET.relative_to(ROOT)} "
              f"({len(out):,} 行 x {len(out.columns)} 列)", flush=True)

    regime = compute_regime()
    ic = descriptive_ic(feat, close, regime)

    # 覆盖率/缺失诊断
    cov = {c: round(float(feat[c].notna().mean()), 4) for c in FEATURE_COLS}
    res = {
        "window": [WIN_START, WIN_END],
        "n_rows": int(len(feat)),
        "n_stocks": int(feat["ts_code"].nunique()),
        "feature_cols": FEATURE_COLS,
        "feature_categories": {
            "1_velocity_dratio": ["ratio_vel_1", "ratio_vel_3", "ratio_vel_5"],
            "2_accel_d2ratio": ["ratio_acc_1", "ratio_acc_3"],
            "3_divergence_quadrant": ["ratio_div_quad", "ratio_div_strength"],
            "4_self_percentile": ["ratio_selfpct_20", "ratio_selfpct_60"],
            "5_days_since_peak": ["ratio_days_since_hi_60", "ratio_days_since_lo_60"],
        },
        "coverage_notna": cov,
        "descriptive_ic": ic,
        "causal_note": "全部 diff/shift(+k)/过去窗口 rolling, 零 shift(-k); 前向收益仅内存评估未入库",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印 IC 摘要
    print("\n=== 单因子 RankIC (描述性, fwd5; 非 gate) ===", flush=True)
    print(f"{'feature':24s} {'overall':>8s} {'trend':>8s} {'mid':>8s} {'choppy':>8s}", flush=True)
    for col in FEATURE_COLS:
        o = ic["overall"][col]
        tr = ic["by_regime"]["trend"][col]
        md = ic["by_regime"]["mid"][col]
        ch = ic["by_regime"]["choppy"][col]
        print(f"{col:24s} {o:>8.4f} {tr:>8.4f} {md:>8.4f} {ch:>8.4f}", flush=True)

    # best |IC| 摘要供 verdict 一句话
    best = max(FEATURE_COLS, key=lambda c: abs(ic["overall"][c]) if ic["overall"][c] == ic["overall"][c] else 0)
    print(f"\n最强单因子 (|overall RankIC|): {best} = {ic['overall'][best]:+.4f}", flush=True)
    print(f"\n[done] features -> {OUT_PARQUET.relative_to(ROOT)} | results -> {RESULTS.relative_to(ROOT)}"
          f"  耗时 {time.time()-t0:.0f}s", flush=True)
    return res, best


if __name__ == "__main__":
    main()
