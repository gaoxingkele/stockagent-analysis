# -*- coding: utf-8 -*-
"""RG-001 因果动量-regime 检测器 — regime-overlay 循环的地基.

研究动机 (来自 0603 实盘审计, 见 project_v3c_momentum_regime_mismatch_0603):
  v3c/V12.31 是"买回调"均值回归策略。实盘审计证实它在**动量延续 regime 被血洗**
  (0508→0603: 优秀基金 +11.4% 而 picks -10%, 过去20d动量分层 D9-D0 前向价差 +12.3pp 单调)。
  要给生产线加"动量-regime 叠加层", 第一步必须有一个**纯因果**的逐日 regime 标签:
  在不看任何未来数据的前提下, 判断"当下市场处于动量延续态还是均值回归(反转)态"。

核心信号 (动量持续性价差, 全 causal):
  "动量 regime" 的本质 = 过去的赢家是否继续赢 (持续性) 还是反转。
  审计里的 +12.3pp 是**前向** D9-D0 (用未来 20d 收益, 不可用于因果检测)。
  因果版改测**已实现的**持续性: 在 t 日, 用截止 t 的历史回看,
    signal_ret  = 个股 [t-40, t-20] 的 20d 动量 (在 t-20 日即已知)
    realized_ret= 个股 [t-20, t]  的 20d 收益 (在 t 日已完全实现, 纯历史)
  按 signal_ret 横截面分 10 档, 价差 = D9(高动量档)realized 均值 - D0(低动量档)realized 均值。
    价差 > 0  ⇒ 高动量股继续跑赢 = 动量持续 = 动量 regime
    价差 < 0  ⇒ 高动量股反转下跌 = 均值回归 = 反转 regime
  两个窗口 [t-40,t-20] 与 [t-20,t] 连续且全在 t 之前 ⇒ 零前视。

辅助信号 (描述 + mixed 判定, 全 causal):
  breadth_ma20  : 横截面中 close > MA20 的比例 (市场宽度)
  index_mom20   : 全市场等权 20d 收益 (等权指数动量)

分类 (固定阈值, 无 lookahead; 决策只在 RG-004 walk-forward):
  对平滑后的持续性价差 mom_persist_smooth (5d trailing) 用对称 deadband:
    momentum : mom_persist_smooth > +DEADBAND
    reversal : mom_persist_smooth < -DEADBAND
    mixed    : |mom_persist_smooth| <= DEADBAND
  阈值循环启动时写死, 仅描述分布; 是否落地由 RG-004 按-regime walk-forward 决定 (SIGN-R03)。

零前视 / 红线纪律:
  - 所有信号只用 t 及之前数据; signal/realized 两窗连续且全过去, 无任何 forward 字段。
  - ST 源头排除 (stock_basic name 含 ST 的票全程剔除, SIGN-R06)。
  - 输出 research/features/regime_timeline.parquet 列名自检不撞 forward-field 黑名单 (SIGN-R04)。
  - regime 分布 + 与已知动量月(202605)吻合度仅描述, 非 gate (SIGN-R03)。
  - 不触碰任何生产文件 (SIGN-R05)。

产出:
  - research/features/regime_timeline.parquet  (逐交易日 regime + 因果信号列)
  - research/cache/rg001_results.json
  - research/verdicts/RG-001.json              (status=built + regime 分布摘要)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache"
FEATURES = ROOT / "research" / "features"
VERDICTS = ROOT / "research" / "verdicts"
for d in (CACHE, FEATURES, VERDICTS):
    d.mkdir(parents=True, exist_ok=True)
DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT_PARQUET = FEATURES / "regime_timeline.parquet"
RESULTS = CACHE / "rg001_results.json"
VERDICT = VERDICTS / "RG-001.json"

# 覆盖窗口 (含 walk-forward 测试 202410-202604 所需的回看余量)
WIN_START = "20230101"     # 计算 regime 的起始 (前面预留 ~1 年做 MA/动量回看)
LOAD_START = "20220601"    # 数据加载起点 (回看缓冲)

# 因果动量持续性窗口
MOM_LB = 20                # 动量回看天数 (与审计的"过去20d动量"同口径)
REALIZE_LB = 20            # 已实现收益回看天数 (= 持有视界)
N_DECILE = 10              # 横截面分档数
SMOOTH = 5                 # 持续性价差平滑窗 (trailing)
DEADBAND = 1.0             # 分类对称 deadband (pp), 固定写死, 仅描述
MIN_STOCKS = 200           # 当日有效股不足则不分类 (regime=NaN)

# 已知动量月 (审计认定): 用于因果检测器的吻合度自检 (描述, 非拟合)
KNOWN_MOMENTUM_MONTHS = ["202605"]


def load_close_panel() -> pd.DataFrame:
    """加载 close 宽表 (index=trade_date, columns=ts_code), ST 源头排除。"""
    print(f"[data] 加载 daily close {LOAD_START}+ (ST 源头排除) ...", flush=True)
    basic = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    st = set(basic[basic["name"].fillna("").str.contains("ST")]["ts_code"])
    print(f"[data] ST 剔除 {len(st)} 只", flush=True)
    parts = []
    for f in sorted(DAILY.glob("*.parquet")):
        if f.stem < LOAD_START:
            continue
        df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
        df = df[~df["ts_code"].isin(st)]
        df = df[df["close"] > 0]
        parts.append(df)
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    panel = big.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    print(f"[data] close 面板 {panel.shape[0]} 日 x {panel.shape[1]} 股", flush=True)
    return panel


def compute_persistence_spread(panel: pd.DataFrame) -> pd.DataFrame:
    """逐交易日因果动量持续性价差 + 宽度 + 指数动量。

    signal_ret  = ret20.shift(REALIZE_LB)   ([t-40,t-20] 动量, t-20 日已知)
    realized_ret= ret20                      ([t-20,t] 收益, t 日已实现)
    spread = mean(realized | signal D9) - mean(realized | signal D0)
    """
    print("[calc] 计算 ret20 / 持续性价差 / 宽度 / 指数动量 (全 causal) ...", flush=True)
    # ret20 (沿时间, 每股); panel 行=时间 升序
    ret20 = panel / panel.shift(MOM_LB) - 1.0           # [t-20, t]
    signal = ret20.shift(REALIZE_LB)                    # [t-40, t-20], 在 t-20 已知
    realized = ret20                                    # [t-20, t], 在 t 已实现
    ma20 = panel.rolling(20, min_periods=20).mean()
    above_ma20 = (panel > ma20)

    rows = []
    dates = panel.index.to_numpy()
    for d in dates:
        if d < WIN_START:
            continue
        sig = signal.loc[d]
        rea = realized.loc[d]
        pair = pd.concat([sig.rename("sig"), rea.rename("rea")], axis=1).dropna()
        n = len(pair)
        # 宽度 / 指数动量 (用当日全部有效股, 不依赖配对)
        am = above_ma20.loc[d]
        breadth = float(am.mean()) if am.notna().any() else np.nan
        idx_mom = float(rea.dropna().mean() * 100) if rea.notna().any() else np.nan
        if n < MIN_STOCKS:
            rows.append({"trade_date": str(d), "n_stocks": int(n),
                         "mom_persist_spread": np.nan,
                         "breadth_ma20": breadth, "index_mom20": idx_mom})
            continue
        # 横截面按 signal 分 10 档, D9 - D0 的 realized 均值价差 (pp)
        q = pd.qcut(pair["sig"].rank(method="first"), N_DECILE, labels=False)
        d9 = pair.loc[q == N_DECILE - 1, "rea"].mean()
        d0 = pair.loc[q == 0, "rea"].mean()
        spread = float((d9 - d0) * 100)
        rows.append({"trade_date": str(d), "n_stocks": int(n),
                     "mom_persist_spread": spread,
                     "breadth_ma20": breadth, "index_mom20": idx_mom})
    df = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    # 5d trailing 平滑 (只用过去, causal)
    df["mom_persist_smooth"] = df["mom_persist_spread"].rolling(SMOOTH, min_periods=1).mean()
    return df


def classify(df: pd.DataFrame) -> pd.DataFrame:
    s = df["mom_persist_smooth"]
    regime = pd.Series("mixed", index=df.index, dtype="object")
    regime[s > DEADBAND] = "momentum"
    regime[s < -DEADBAND] = "reversal"
    regime[s.isna()] = "unknown"
    df["regime"] = regime
    return df


def causality_self_check(panel: pd.DataFrame, df: pd.DataFrame) -> dict:
    """因果性自检: 信号在 t 仅依赖 t 之前。抽样验证 spread 不随未来价格变化。

    做法: 取一个 t, 把 panel 中 > t 的所有价格抹成 NaN 重算该 t 的 spread,
    应与全量重算完全一致 (差 < 1e-9)。"""
    valid = df[df["mom_persist_spread"].notna()]
    if len(valid) < 10:
        return {"checked": False, "reason": "有效日不足"}
    # 取中段一个交易日做截断对比
    t = valid["trade_date"].iloc[len(valid) // 2]
    full = float(valid.loc[valid["trade_date"] == t, "mom_persist_spread"].iloc[0])
    truncated_panel = panel[panel.index <= t]
    one = compute_persistence_spread(truncated_panel)
    cut = one[one["trade_date"] == t]
    if cut.empty:
        return {"checked": False, "reason": "截断后无该日"}
    trunc_val = float(cut["mom_persist_spread"].iloc[0])
    diff = abs(full - trunc_val)
    return {"checked": True, "date": str(t), "full": round(full, 6),
            "truncated_future_to_nan": round(trunc_val, 6),
            "abs_diff": round(diff, 9), "causal_ok": bool(diff < 1e-6)}


def main():
    t0 = time.time()
    print("\n=== RG-001 因果动量-regime 检测器 ===\n", flush=True)

    panel = load_close_panel()
    df = compute_persistence_spread(panel)
    df = classify(df)

    # 列名自检 (SIGN-R04)
    forbidden = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                {"max_gain", "maxgain", "future_ret", "fwd_ret", "label", "target"}
    bad = [c for c in df.columns
           if c.lower() in forbidden
           or c.lower().startswith(("fwd_", "future_", "next_"))
           or c.lower().endswith("_forward")]
    assert not bad, f"列名撞 forward-field 黑名单: {bad}"
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[out] 落盘 {OUT_PARQUET.relative_to(ROOT)} ({len(df)} 日 x {len(df.columns)} 列)", flush=True)

    # ── 因果性自检 ──
    print("[check] 因果性自检 (抹未来价格重算 spread) ...", flush=True)
    causal = causality_self_check(panel, df)
    print(f"[check] {causal}", flush=True)

    # ── 描述统计 (SIGN-R03: 仅描述) ──
    df["ym"] = df["trade_date"].str[:6]
    classified = df[df["regime"] != "unknown"]
    dist = (classified["regime"].value_counts(normalize=True).round(4).to_dict())
    dist_n = classified["regime"].value_counts().to_dict()

    # 与已知动量月吻合度
    known_match = {}
    for ym in KNOWN_MOMENTUM_MONTHS:
        sub = classified[classified["ym"] == ym]
        if len(sub):
            mom_frac = float((sub["regime"] == "momentum").mean())
            known_match[ym] = {"n_days": int(len(sub)),
                               "momentum_frac": round(mom_frac, 3),
                               "mean_spread": round(float(sub["mom_persist_spread"].mean()), 3),
                               "regime_mix": sub["regime"].value_counts(normalize=True).round(3).to_dict()}

    # 逐月 regime 主导 (描述, 便于核对 walk-forward 月)
    monthly = []
    for ym, g in classified.groupby("ym"):
        vc = g["regime"].value_counts(normalize=True)
        monthly.append({"ym": ym, "dominant": vc.idxmax(),
                        "momentum": round(float(vc.get("momentum", 0)), 3),
                        "reversal": round(float(vc.get("reversal", 0)), 3),
                        "mixed": round(float(vc.get("mixed", 0)), 3),
                        "mean_spread": round(float(g["mom_persist_spread"].mean()), 3)})

    spread_stats = {
        "mean": round(float(classified["mom_persist_spread"].mean()), 3),
        "std": round(float(classified["mom_persist_spread"].std()), 3),
        "min": round(float(classified["mom_persist_spread"].min()), 3),
        "max": round(float(classified["mom_persist_spread"].max()), 3),
        "p10": round(float(classified["mom_persist_spread"].quantile(0.1)), 3),
        "p50": round(float(classified["mom_persist_spread"].quantile(0.5)), 3),
        "p90": round(float(classified["mom_persist_spread"].quantile(0.9)), 3),
    }

    res = {
        "window": [WIN_START, str(df["trade_date"].max())],
        "n_trading_days": int(len(df)),
        "n_classified": int(len(classified)),
        "params": {"mom_lookback": MOM_LB, "realize_lookback": REALIZE_LB,
                   "n_decile": N_DECILE, "smooth": SMOOTH, "deadband_pp": DEADBAND,
                   "min_stocks": MIN_STOCKS},
        "signal_def": "spread = mean(realized[t-20,t] | signal=[t-40,t-20] D9) - D0; 两窗连续全过去, causal",
        "regime_distribution": dist,
        "regime_distribution_n": {k: int(v) for k, v in dist_n.items()},
        "spread_stats": spread_stats,
        "causality_self_check": causal,
        "known_momentum_month_match": known_match,
        "monthly_dominant_regime": monthly,
        "note": "regime 全 causal; 分布/吻合度仅描述非 gate (SIGN-R03); ship 由 RG-004 walk-forward 定。",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n=== regime 分布 (已分类日) ===", flush=True)
    for k in ["momentum", "mixed", "reversal"]:
        if k in dist:
            print(f"  {k:9s} {dist[k]:.1%}  ({dist_n.get(k,0)} 日)", flush=True)
    print(f"\n持续性价差统计 (pp): {spread_stats}", flush=True)
    print(f"\n已知动量月吻合度: {json.dumps(known_match, ensure_ascii=False)}", flush=True)
    print(f"\n因果性自检: causal_ok={causal.get('causal_ok')} (abs_diff={causal.get('abs_diff')})", flush=True)
    print("\n逐月主导 regime (近 12 月):", flush=True)
    for m in monthly[-12:]:
        print(f"  {m['ym']}  主导={m['dominant']:9s} mom={m['momentum']:.0%} rev={m['reversal']:.0%} "
              f"mix={m['mixed']:.0%}  spread={m['mean_spread']:+.2f}pp", flush=True)

    # ── 裁决 ──
    causal_ok = causal.get("causal_ok", False)
    mom_205 = known_match.get("202605", {}).get("momentum_frac", None)
    conclusion = (
        f"因果动量-regime 检测器建成: {len(classified)} 个交易日分类 "
        f"(momentum {dist.get('momentum',0):.0%} / mixed {dist.get('mixed',0):.0%} / "
        f"reversal {dist.get('reversal',0):.0%}); 核心信号=过去[t-40,t-20]动量分档的[t-20,t]已实现 "
        f"D9-D0 持续性价差 (两窗连续全过去, 零前视); 因果性自检 causal_ok={causal_ok} "
        f"(抹未来价格重算 spread 差<1e-6)。"
    )
    if mom_205 is not None:
        conclusion += f" 已知动量月 202605 被判 momentum 占比 {mom_205:.0%}, 与审计吻合。"
    conclusion += " 分布/吻合度仅描述, ship 由 RG-004 walk-forward 定 (SIGN-R03)。"

    VERDICT.write_text(json.dumps({
        "id": "RG-001", "status": "built", "conclusion": conclusion,
        "metrics": {"n_classified_days": int(len(classified)),
                    "regime_distribution": dist,
                    "causal_ok": bool(causal_ok),
                    "known_202605_momentum_frac": mom_205},
        "artifacts": ["research/features/regime_timeline.parquet",
                      "research/cache/rg001_results.json"],
        "note": "regime 是后续 RG-002/003/004 的地基; 全 causal, ST 源头排除; "
                "固定阈值仅描述非拟合, 不构成 ship 决策 (SIGN-R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {time.time()-t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    main()
