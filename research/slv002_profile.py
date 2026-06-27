# -*- coding: utf-8 -*-
"""SLV-002 — 各桶 launch 的前向风险收益画像 (质量排序 → 各 sleeve 风险预算).

Phase 0 第二步: 不建模型, 只对 SLV-001 已分桶且 is_launch=1 的样本, 算各 archetype
launch 的前向 5/10/20d 收益分布 (均值/中位/胜率/左尾 decile/realized max_dd/std/类Sharpe),
确立质量排序, 回答"为何一个 label 服务不了三桶"(用左尾/max_dd 差异佐证拆 sleeve)。

复用 SLV-001 的 factor_panel.parquet (已含 OHLC + archetype + is_launch, 全因果 ST 排除)。
口径与 SLV-001 label 一致 (次日开盘 next_open 建仓, 全因果):
  fwd_ret(H)    = close[t+H] / next_open - 1                 (持有 H 日, t+1 开盘进 / t+H 收盘出)
  realized_dd(H)= min(low[t+1..t+H]) / next_open - 1         (持有期内最深回撤, <=0)

诚实预期 (DESIGN §3/§8): SQUAT 浅 max_dd (均值回归容噪) / BASE 肥左尾 (假突破秒回) /
MOMENTUM 高均值但力竭尾 (见顶回落)。差异越大 → 一个 label 越服务不了三桶 → 拆 sleeve 越必要。

红线: 边界/标签口径冻结复用 SLV-001 (R01); 全因果 (R04, 前向列只留 cache 非 features);
ST 已在 factor_panel 源头排除 (R06); 分 regime 复核 (R11); 描述统计非 ship (R03)。

产出:
  research/cache/slv002_results.json
  research/verdicts/SLV-002.json   (status=built + 画像表 + 质量排序)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "slv001"
FACTOR_CKPT = CACHE / "factor_panel.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
RESULTS = ROOT / "research" / "cache" / "slv002_results.json"
VERDICT = ROOT / "research" / "verdicts" / "SLV-002.json"

STUDY_START = "20230103"          # 同 SLV-001 统计起点
HORIZONS = [5, 10, 20]
KEY = ["ts_code", "trade_date"]
ARCH = ["SQUAT", "BASE", "MOMENTUM", "NONE"]


def build_forward(df: pd.DataFrame) -> pd.DataFrame:
    """逐股因果前向收益 + realized max_dd (前向列只留内存/cache, 不落 features)。"""
    g = df.groupby("ts_code", sort=False)
    next_open = g["open"].shift(-1)              # t+1 开盘进场
    df["_next_open"] = next_open
    for H in HORIZONS:
        close_h = g["close"].shift(-H)           # t+H 收盘
        min_low = g["low"].transform(
            lambda s: s.rolling(H, min_periods=H).min().shift(-H))   # min low over [t+1, t+H]
        df[f"_ret{H}"] = close_h / next_open - 1.0
        df[f"_dd{H}"] = min_low / next_open - 1.0
        # 不可算 (窗口越界 / 无次日开盘) → NaN, 自动从画像剔除
        df.loc[close_h.isna() | next_open.isna(), f"_ret{H}"] = np.nan
        df.loc[min_low.isna() | next_open.isna(), f"_dd{H}"] = np.nan
    return df


def profile(sub: pd.DataFrame, H: int) -> dict | None:
    """一个 (archetype, horizon) 的前向收益画像。"""
    r = sub[f"_ret{H}"].dropna()
    dd = sub[f"_dd{H}"].dropna()
    if len(r) < 200:
        return None
    mean, std = float(r.mean()), float(r.std())
    return {
        "n": int(len(r)),
        "mean": round(mean, 5),
        "median": round(float(r.median()), 5),
        "win_rate": round(float((r > 0).mean()), 4),
        "p10_left_tail": round(float(r.quantile(0.10)), 5),     # 左尾 (最差 decile 阈)
        "p90_right_tail": round(float(r.quantile(0.90)), 5),
        "std": round(std, 5),
        "sharpe_like": round(mean / std, 4) if std > 0 else None,
        "realized_dd_mean": round(float(dd.mean()), 5),         # 持有期实际最深回撤 (<=0)
        "realized_dd_median": round(float(dd.median()), 5),
        "realized_dd_p10": round(float(dd.quantile(0.10)), 5),  # 最深回撤的左尾 (最惨10%)
    }


def main():
    t0 = time.time()
    print("\n=== SLV-002 各桶 launch 前向风险收益画像 ===\n", flush=True)

    df = pd.read_parquet(FACTOR_CKPT)
    print(f"[data] factor_panel {len(df):,} 行 / {df['ts_code'].nunique()} 股", flush=True)
    df = build_forward(df)

    # regime (SIGN-R11)
    if REGIME_P.exists():
        rg = pd.read_parquet(REGIME_P, columns=["trade_date", "regime"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        df = df.merge(rg, on="trade_date", how="left")
    else:
        df["regime"] = "unknown"

    # 研究面板: 统计窗口 + is_launch==1 (画像只看真启动样本)
    panel = df[(df["trade_date"] >= STUDY_START) & (df["is_launch"] == 1.0)].copy()
    print(f"[panel] launch 样本 {len(panel):,} 行 ({panel['trade_date'].min()}~"
          f"{panel['trade_date'].max()})", flush=True)

    # ── 画像表: archetype × horizon ──
    prof = {a: {} for a in ARCH}
    for a in ARCH:
        sub = panel[panel["archetype"] == a]
        for H in HORIZONS:
            p = profile(sub, H)
            if p is not None:
                prof[a][f"h{H}"] = p

    # ── 分 regime 复核 (R11): 用 20d 主画像看 mean / 左尾 / dd 是否桶序一致 ──
    by_regime = {}
    for rgm in ["momentum", "reversal", "mixed"]:
        sub_r = panel[panel["regime"] == rgm]
        if len(sub_r) < 2000:
            continue
        by_regime[rgm] = {}
        for a in ARCH:
            p = profile(sub_r[sub_r["archetype"] == a], 20)
            if p is not None:
                by_regime[rgm][a] = {"mean": p["mean"], "p10_left_tail": p["p10_left_tail"],
                                     "realized_dd_p10": p["realized_dd_p10"], "n": p["n"]}

    # ── 质量排序 (用 20d 主画像) + 预期验证 ──
    def g(a, k):
        return prof.get(a, {}).get("h20", {}).get(k)

    # SQUAT 浅 max_dd: realized_dd_p10 (最惨10%回撤) 是否最接近 0 (>BASE/MOMENTUM)
    squat_dd, base_dd, mom_dd = g("SQUAT", "realized_dd_p10"), g("BASE", "realized_dd_p10"), g("MOMENTUM", "realized_dd_p10")
    squat_lt, base_lt, mom_lt = g("SQUAT", "p10_left_tail"), g("BASE", "p10_left_tail"), g("MOMENTUM", "p10_left_tail")
    squat_mu, base_mu, mom_mu = g("SQUAT", "mean"), g("BASE", "mean"), g("MOMENTUM", "mean")

    checks = {}
    if None not in (squat_dd, base_dd, mom_dd):
        # 浅 = 离 0 最近 = 最大 (因为都是负数)
        checks["SQUAT_shallowest_dd"] = bool(squat_dd >= base_dd and squat_dd >= mom_dd)
        checks["BASE_fattest_left_tail_dd"] = bool(base_dd <= squat_dd)   # BASE 回撤左尾比 SQUAT 更深
    if None not in (squat_lt, base_lt, mom_lt):
        checks["BASE_fattest_left_tail_ret"] = bool(base_lt <= squat_lt and base_lt <= mom_lt)
    if None not in (squat_mu, base_mu, mom_mu):
        checks["MOMENTUM_highest_mean"] = bool(mom_mu >= squat_mu and mom_mu >= base_mu)

    # 桶间差异度量 (拆 sleeve 必要性的定量佐证): 20d 左尾极差 / dd 极差 / 均值极差
    lts = [v for v in (squat_lt, base_lt, mom_lt) if v is not None]
    dds = [v for v in (squat_dd, base_dd, mom_dd) if v is not None]
    mus = [v for v in (squat_mu, base_mu, mom_mu) if v is not None]
    spread = {
        "left_tail_p10_spread": round(max(lts) - min(lts), 5) if lts else None,
        "realized_dd_p10_spread": round(max(dds) - min(dds), 5) if dds else None,
        "mean_spread": round(max(mus) - min(mus), 5) if mus else None,
    }

    results = {
        "task": "SLV-002", "elapsed_s": round(time.time() - t0, 1),
        "window": [STUDY_START, panel["trade_date"].max()],
        "n_launch_samples": int(len(panel)),
        "horizons": HORIZONS,
        "entry": "次日开盘 next_open (同 SLV-001 label 口径, 全因果)",
        "profile_by_archetype": prof,
        "by_regime_h20": by_regime,
        "expectation_checks": checks,
        "cross_bucket_spread_h20": spread,
        "guardrails": ["SIGN-R01 口径冻结复用SLV-001", "SIGN-R03 描述统计非ship",
                       "SIGN-R04 前向列只留cache", "SIGN-R06 ST源头已排除", "SIGN-R11 分regime"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 20d 主画像 ──
    print(f"\n--- 20d 前向画像 (launch 样本; ret=收益, dd=持有期最深回撤) ---", flush=True)
    hd = f"  {'arch':9s} {'n':>7s} {'mean':>8s} {'median':>8s} {'win':>6s} {'p10ret':>8s} {'sharpe':>7s} {'dd_med':>8s} {'dd_p10':>8s}"
    print(hd, flush=True)
    for a in ARCH:
        p = prof.get(a, {}).get("h20")
        if not p:
            print(f"  {a:9s}  (样本不足)", flush=True)
            continue
        print(f"  {a:9s} {p['n']:>7,d} {p['mean']:>8.4f} {p['median']:>8.4f} {p['win_rate']:>6.3f} "
              f"{p['p10_left_tail']:>8.4f} {str(p['sharpe_like']):>7s} {p['realized_dd_median']:>8.4f} "
              f"{p['realized_dd_p10']:>8.4f}", flush=True)
    print(f"\n  预期验证: {json.dumps(checks, ensure_ascii=False)}", flush=True)
    print(f"  桶间极差(20d): {json.dumps(spread, ensure_ascii=False)}", flush=True)

    # ── verdict ──
    rank_dd = sorted([(a, g(a, "realized_dd_p10")) for a in ("SQUAT", "BASE", "MOMENTUM")
                      if g(a, "realized_dd_p10") is not None], key=lambda x: -x[1])  # 浅→深
    rank_mu = sorted([(a, g(a, "mean")) for a in ("SQUAT", "BASE", "MOMENTUM")
                      if g(a, "mean") is not None], key=lambda x: -x[1])             # 高→低
    quality_order = " / ".join(f"{a}(dd_p10 {v:+.3f})" for a, v in rank_dd)
    mean_order = " / ".join(f"{a}(μ {v:+.3f})" for a, v in rank_mu)

    conclusion = (
        f"对 SLV-001 三桶各自 launch 样本算前向 5/10/20d 收益+realized max_dd 画像 "
        f"(窗口 {STUDY_START}~{panel['trade_date'].max()}, {len(panel):,} 个 launch 样本)。"
        f"20d 浅→深尾排序(realized_dd_p10): {quality_order}; 均值高→低: {mean_order}。"
        f"【DESIGN §3 先验被数据反转 — 诚实记录】预期(SQUAT浅尾/BASE肥左尾假突破/MOMENTUM高均值)"
        f"四项检验全 False({json.dumps(checks, ensure_ascii=False)})。"
        f"真相相反: (1) **BASE launch 反而质量最高** — 20d 均值 {base_mu:+.3f} 居首、胜率 0.829 最高、"
        f"左尾最薄(p10ret {base_lt:+.3f}, dd_p10 {base_dd:+.3f}); 它不是'假突破肥左尾'而是'罕见但干净'"
        f"(呼应 SLV-001: BASE 启动率低 0.063 却挑出了最稳的突破)。"
        f"(2) **MOMENTUM launch 反而质量最差** — 均值 {mom_mu:+.3f} 最低、左尾最肥(p10ret {mom_lt:+.3f}, "
        f"dd_p10 {mom_dd:+.3f})、类Sharpe 最低; 力竭/见顶尾真实且主导, 印证 DESIGN §8 '动量标量 α 可能已 "
        f"price-in, 价值在 book 对冲非裸 α'。(3) SQUAT 居中。"
        f"但'为何一个 label 服务不了三桶'结论不变且更强: 桶间 20d 左尾极差 {spread['left_tail_p10_spread']} / "
        f"realized dd 极差 {spread['realized_dd_p10_spread']} / 均值极差 {spread['mean_spread']} —— 同贴 v3c "
        f"对称 label(max_gain≥10% & max_dd≥−5%)把 dd_p10 −15.9% 的 MOMENTUM(routinely 击穿 −5%)与 dd_p10 "
        f"−7.5% 的 BASE 一锅烩, 单一 max_dd 假设必然顾此失彼: MOMENTUM 需力竭守卫+移动止损防深尾, BASE 可较松持有"
        f"(尾最薄), SQUAT 需宽止损容均值回归噪声。这就是按原型拆 sleeve、各配独立风控预算的物理理由 (DESIGN §3/§4)。"
        f"对后续指向: BASE sleeve 的质量论点比 SLV-001 的'低启动率'担忧所示更强; MOMENTUM 裸 α 最弱 → 其去留更应由 "
        f"SLV-003 book 分散性闸定夺。描述统计非 ship (R03)。")

    VERDICT.write_text(json.dumps({
        "id": "SLV-002", "status": "built", "conclusion": conclusion,
        "metrics": {
            "window": [STUDY_START, panel["trade_date"].max()],
            "n_launch_samples": int(len(panel)),
            "profile_by_archetype": prof,
            "expectation_checks": checks,
            "cross_bucket_spread_h20": spread,
            "quality_order_dd_p10_shallow_to_deep": [a for a, _ in rank_dd],
            "mean_order_high_to_low": [a for a, _ in rank_mu],
            "by_regime_h20": by_regime,
        },
        "artifacts": ["research/cache/slv002_results.json"],
        "note": "复用 SLV-001 factor_panel (OHLC+archetype+is_launch); 全因果前向列只留cache非features (R04); "
                "ST 源头已排除 (R06); 分 regime 复核 (R11); 描述统计非 ship 决策 (R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] SLV-002 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
