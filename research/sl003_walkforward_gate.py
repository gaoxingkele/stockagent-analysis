# -*- coding: utf-8 -*-
"""SL-003 — r5 龙虎榜席位事件 sleeve walk-forward 决策 gate (净 α + 席位技能净增量 + 消融).

承接 SL-001 (引擎 + 三臂) / SL-002 (成本×持有期×容量)。前两 task 只产描述统计 (SIGN-R03
中间指标≠落地)。本 task 是**裁决闸 (isGate)**: 在 ≥30 月 walk-forward 上算真正的净 α, 隔离
"席位技能" vs "龙虎榜效应", 跑事前注册的 5 分叉 playbook, 写 PASS/真小/REJECT_* 裁决。

席位战绩 (SEAT-001 sf_edge_r5) 本身是 expanding 因果信号 (avail<=D, D+1 生效) — 因此逐月
α 序列**天然 walk-forward** (每月的信号只用该月之前的数据, 无需逐月重训)。本 gate = 把
事件对齐的逐月净 α 序列建出来, 算 mean/t/Sharpe/regime/outlier。

核心量 (主测 H=5 / 30bps round-trip, 复用 SL-001 picks):
  - 市场基准 (等权全市场 r5): 对每个 entry_date e, market[e] = 全市场所有股
    open[e+H]/open[e]-1 的等权均值 (同 D+1 开盘建 / D+1+H 开盘平 口径, ST 已源头排除)。
  - 净 α = Arm_A net − market (扣市场)。   gross α = Arm_A gross − market。
  - 席位技能净增量 = Arm_A − Arm_B (同集动量排序对照; cost 抵消 → = gross_A − gross_B)。
    Arm_A vs Arm_B 唯一差异 = 排序键 (席位战绩 vs 动量) ⇒ 净增量 = 纯席位技能, 扣[动量]。
  - 逐月: 当月所有 entry-day 的事件等权均值 → 月度序列; t = mean/(std/√n); Sharpe = mean/std×√12。
  - regime 分层 (SIGN-R11): event_date 的 regime (动量/反转), 报告各 regime 净 α, 检查不集中。
  - 单月 outlier (SIGN): 剔除贡献最大的单月后净 α 仍 > 0 才算稳健。

事前注册 gate (冻结 SIGN-R01, 不可改阈值):
  净 α(Arm_A−等权市场) >= +0.50pp 且 |t|>=3 (≥30月) 且 Arm_A−Arm_B > 0 显著(|t|>=2) 且
  Sharpe >= 1.0 且 单月 outlier 剔后净 α>0 且 分 regime 不集中于单一 regime。

5 分叉 (responsePlaybook, REJECT 合法完成 SIGN-R02):
  PASS              — 全部 gate 条件满足
  真小              — 净 α>0 但 <+0.50pp 或 Sharpe<1 (promising, 不强上)
  REJECT_成本吃掉   — gross α>0 但 net α<=0 (信号真但成本不经济)
  REJECT_龙虎榜效应 — Arm_A ≈ Arm_B (不是席位技能, 是买龙虎榜本身)
  REJECT_负         — 净 α<0 且 gross α 也<0 (干净否)

输出:
  research/cache/sl003_monthly.parquet   (逐月 net_A/market/alpha/skill + regime)
  research/verdicts/SL-003.json          (status ∈ 5 分叉 + 全部 gate 指标)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import sl001_sleeve_engine as eng  # noqa: E402  复用引擎/常数 (载体/因果一致, SIGN-R01)

CACHE = ROOT / "research" / "cache"
SL_CACHE = CACHE / "sl001"
PICKS_CACHE = SL_CACHE / "picks.parquet"           # 主测 H=5
MKT_CACHE = SL_CACHE / "market_bench_H5.parquet"   # 等权全市场基准 (checkpoint)
MONTHLY_OUT = CACHE / "sl003_monthly.parquet"
VERDICT = ROOT / "research" / "verdicts" / "SL-003.json"

# ── 主测口径 (冻结 SIGN-R01, 复用 SL-001) ──
H = eng.H_HOLD            # 5
TOP_N = eng.TOP_N         # 10
COST_RT = eng.COST_RT     # 30bps round-trip
RET_CLIP = eng.RET_CLIP

# ── 事前注册 gate 阈值 (冻结, 绝不改 SIGN-R01) ──
GATE_ALPHA_PP = 0.50      # 净月度 α 下限 (pp)
GATE_T = 3.0              # 净 α |t| 下限
GATE_SKILL_T = 2.0        # 席位技能净增量 |t| 显著门槛 (Arm_A−Arm_B)
GATE_SHARPE = 1.0         # 净 α 月序列年化 Sharpe 下限
MIN_MONTHS = 30           # walk-forward 最少月数


def build_market_bench(st: set) -> pd.Series:
    """等权全市场 H 日 open→open 前向收益, 按 trade_date(=entry_date) 取均值 (ST 已源头排除)。"""
    if MKT_CACHE.exists():
        print(f"[mkt] checkpoint 命中 {MKT_CACHE.name}", flush=True)
        s = pd.read_parquet(MKT_CACHE)
        return s.set_index("entry_date")["market_ret"]
    px = eng.load_prices(st)  # 引擎已做 ST 源头排除 + checkpoint
    px = px[["ts_code", "trade_date", "open"]].sort_values(["ts_code", "trade_date"])
    # 每股 H 日后开盘 (同股序列内 shift; 跨股 NaN 自然丢弃)
    px["open_fwd"] = px.groupby("ts_code")["open"].shift(-H)
    px = px.dropna(subset=["open_fwd"])
    px = px[(px["open"] > 0) & (px["open_fwd"] > 0)]
    px["fwd_ret"] = (px["open_fwd"] / px["open"] - 1.0).clip(-RET_CLIP, RET_CLIP)
    mkt = px.groupby("trade_date")["fwd_ret"].mean().rename("market_ret")
    mkt = mkt.reset_index().rename(columns={"trade_date": "entry_date"})
    SL_CACHE.mkdir(parents=True, exist_ok=True)
    mkt.to_parquet(MKT_CACHE, index=False)
    print(f"[mkt] 落盘 {MKT_CACHE.name} ({len(mkt):,} 日, 等权全市场 H={H} open→open)", flush=True)
    return mkt.set_index("entry_date")["market_ret"]


def per_event_metrics(picks: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """逐 entry_date: Arm_A/B net + market + 净α + 席位技能净增量 + regime。"""
    pnl = eng.run_arms(picks, TOP_N, COST_RT)
    wide = pnl.pivot_table(index="entry_date", columns="arm", values="gross")
    wide = wide.rename(columns={"A": "gross_A", "B": "gross_B", "M": "gross_M"})
    wide = wide.reset_index()
    wide["entry_date"] = wide["entry_date"].astype(str)
    # 仅保留有席位臂 (Arm_A) 的事件日
    wide = wide.dropna(subset=["gross_A"]).copy()
    wide["net_A"] = wide["gross_A"] - COST_RT
    wide["net_B"] = wide["gross_B"] - COST_RT
    wide["market"] = wide["entry_date"].map(market)
    wide = wide.dropna(subset=["market"]).copy()
    wide["alpha"] = wide["net_A"] - wide["market"]            # 净 α (扣市场)
    wide["gross_alpha"] = wide["gross_A"] - wide["market"]    # 毛 α
    wide["skill"] = wide["gross_A"] - wide["gross_B"]         # 席位技能净增量 (cost 抵消)
    wide["month"] = wide["entry_date"].str[:6]
    return wide


def attach_regime(wide: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """按 event_date 的 regime 标记 (信号日 regime; entry=next(event))。"""
    if not eng.REGIME_P.exists():
        wide["regime"] = "unknown"
        return wide
    reg = pd.read_parquet(eng.REGIME_P)[["trade_date", "regime"]].copy()
    reg["trade_date"] = reg["trade_date"].astype(str)
    # entry_date → event_date 映射 (picks 已有两列)
    e2e = picks[["entry_date", "event_date"]].drop_duplicates()
    e2e["entry_date"] = e2e["entry_date"].astype(str)
    e2e["event_date"] = e2e["event_date"].astype(str)
    wide = wide.merge(e2e, on="entry_date", how="left")
    wide = wide.merge(reg.rename(columns={"trade_date": "event_date"}), on="event_date", how="left")
    wide["regime"] = wide["regime"].fillna("unknown")
    return wide


def monthly_series(wide: pd.DataFrame, col: str) -> pd.Series:
    """事件等权逐月均值序列。"""
    return wide.groupby("month")[col].mean()


def tstat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / (sd / np.sqrt(len(x))))


def sharpe_ann(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(12.0))


def stat_block(m: pd.Series) -> dict:
    x = m.to_numpy()
    return {
        "n_months": int(len(x)),
        "monthly_mean_pp": round(float(np.nanmean(x)) * 100, 4),
        "monthly_std_pp": round(float(np.nanstd(x, ddof=1)) * 100, 4) if len(x) > 1 else None,
        "t": round(tstat(x), 3),
        "sharpe_ann": round(sharpe_ann(x), 3),
        "pos_month_ratio": round(float((x > 0).mean()), 4),
        "worst_month_pp": round(float(np.nanmin(x)) * 100, 4),
        "best_month_pp": round(float(np.nanmax(x)) * 100, 4),
    }


def outlier_drop(m: pd.Series) -> dict:
    """剔除贡献最大的单月 (|偏离均值| 最大) 后重算均值/t — 检查不被单月驱动。"""
    x = m.to_numpy()
    mean_full = float(np.nanmean(x))
    dev = np.abs(x - mean_full)
    drop_i = int(np.nanargmax(dev))
    drop_month = str(m.index[drop_i])
    kept = np.delete(x, drop_i)
    return {
        "dropped_month": drop_month,
        "dropped_value_pp": round(float(x[drop_i]) * 100, 4),
        "mean_after_drop_pp": round(float(np.nanmean(kept)) * 100, 4),
        "t_after_drop": round(tstat(kept), 3),
        "still_positive": bool(np.nanmean(kept) > 0),
    }


def regime_block(wide: pd.DataFrame, col: str) -> dict:
    """各 regime 的逐月均值 (事件等权), 检查 α 不集中于单一 regime。"""
    out = {}
    for rg, g in wide.groupby("regime"):
        m = g.groupby("month")[col].mean()
        out[str(rg)] = {
            "n_months": int(len(m)),
            "monthly_mean_pp": round(float(m.mean()) * 100, 4),
            "t": round(tstat(m.to_numpy()), 3),
            "pos_month_ratio": round(float((m.to_numpy() > 0).mean()), 4),
        }
    return out


def decide(alpha_s: dict, skill_s: dict, gross_alpha_s: dict,
           outl: dict, regime_alpha: dict) -> tuple[str, dict, list]:
    """事前注册 5 分叉 (冻结 SIGN-R01/R02)。"""
    a_mean = alpha_s["monthly_mean_pp"]
    a_t = alpha_s["t"]
    a_sharpe = alpha_s["sharpe_ann"]
    g_mean = gross_alpha_s["monthly_mean_pp"]
    sk_mean = skill_s["monthly_mean_pp"]
    sk_t = skill_s["t"]

    skill_sig = (sk_mean > 0) and (abs(sk_t) >= GATE_SKILL_T)
    # regime 集中度: 净 α 是否在两个主 regime (动量/反转) 都为正 (排除 unknown)
    main_rg = {k: v for k, v in regime_alpha.items() if k in ("momentum", "reversal")}
    rg_both_pos = len(main_rg) >= 2 and all(v["monthly_mean_pp"] > 0 for v in main_rg.values())
    rg_concentrated = not rg_both_pos

    gate_checks = {
        "alpha>=+0.50pp": a_mean >= GATE_ALPHA_PP,
        "|t_alpha|>=3": abs(a_t) >= GATE_T,
        "skill(A-B)>0 & |t|>=2": skill_sig,
        "sharpe>=1.0": (a_sharpe is not None) and (a_sharpe >= GATE_SHARPE),
        "outlier_drop_still_pos": outl["still_positive"],
        "regime_not_concentrated": rg_both_pos,
    }

    # ── 5 分叉决策树 ──
    if a_mean <= 0:
        status = "REJECT_成本吃掉" if g_mean > 0 else "REJECT_负"
    elif not skill_sig:
        status = "REJECT_龙虎榜效应"   # 净 α>0 但不是席位技能 (Arm_A≈Arm_B)
    elif all(gate_checks.values()):
        status = "PASS"
    else:
        status = "真小"               # 净 α>0 且是席位技能, 但未达 magnitude/sharpe/稳健
    return status, gate_checks, list(gate_checks.keys())


def main() -> int:
    t0 = time.time()
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== SL-003 walk-forward 决策 gate (净 α + 席位技能净增量 + 消融) ===\n", flush=True)

    st = eng.load_st_set()
    print(f"[st] ST 源头排除 {len(st)} 只", flush=True)
    market = build_market_bench(st)
    print(f"[mkt] 等权全市场基准 {len(market)} 日", flush=True)

    picks = pd.read_parquet(PICKS_CACHE)
    picks["entry_date"] = picks["entry_date"].astype(str)
    picks["event_date"] = picks["event_date"].astype(str)
    print(f"[picks] {len(picks):,} 候选 / {picks['entry_date'].nunique()} entry 日 "
          f"(席位可信 {picks['has_seat'].mean():.1%})", flush=True)

    wide = per_event_metrics(picks, market)
    wide = attach_regime(wide, picks)
    n_months = wide["month"].nunique()
    print(f"[events] {len(wide)} entry 日 / {n_months} 月 "
          f"({wide['entry_date'].min()}~{wide['entry_date'].max()})", flush=True)
    if n_months < MIN_MONTHS:
        print(f"[WARN] 月数 {n_months} < {MIN_MONTHS} (walk-forward 样本不足)", flush=True)

    # ── 逐月序列 ──
    m_alpha = monthly_series(wide, "alpha")
    m_gross_alpha = monthly_series(wide, "gross_alpha")
    m_skill = monthly_series(wide, "skill")
    m_netA = monthly_series(wide, "net_A")
    m_market = monthly_series(wide, "market")

    monthly = pd.DataFrame({
        "month": m_alpha.index,
        "net_A": m_netA.values, "market": m_market.values,
        "alpha": m_alpha.values, "gross_alpha": m_gross_alpha.values,
        "skill_A_minus_B": m_skill.values,
    })
    monthly.to_parquet(MONTHLY_OUT, index=False)
    print(f"[monthly] -> {MONTHLY_OUT.relative_to(ROOT)} ({len(monthly)} 月)", flush=True)

    alpha_s = stat_block(m_alpha)
    gross_alpha_s = stat_block(m_gross_alpha)
    skill_s = stat_block(m_skill)
    outl = outlier_drop(m_alpha)
    regime_alpha = regime_block(wide, "alpha")
    regime_skill = regime_block(wide, "skill")

    # ── 打印 ──
    print("\n=== 净 α (Arm_A net − 等权全市场), 逐月事件等权 ===", flush=True)
    print(f"  月数={alpha_s['n_months']}  月均={alpha_s['monthly_mean_pp']:+.3f}pp  "
          f"t={alpha_s['t']}  Sharpe(年)={alpha_s['sharpe_ann']}  "
          f"正月%={alpha_s['pos_month_ratio']:.1%}  "
          f"最差月={alpha_s['worst_month_pp']:+.2f}pp", flush=True)
    print(f"  毛 α 月均={gross_alpha_s['monthly_mean_pp']:+.3f}pp (t={gross_alpha_s['t']})", flush=True)
    print("\n=== 席位技能净增量 (Arm_A − Arm_B, 扣动量) ===", flush=True)
    print(f"  月均={skill_s['monthly_mean_pp']:+.3f}pp  t={skill_s['t']}  "
          f"正月%={skill_s['pos_month_ratio']:.1%}", flush=True)
    print("\n=== 单月 outlier 剔除 (净 α) ===", flush=True)
    print(f"  剔 {outl['dropped_month']} ({outl['dropped_value_pp']:+.2f}pp) → "
          f"月均 {outl['mean_after_drop_pp']:+.3f}pp (t={outl['t_after_drop']}), "
          f"仍正={outl['still_positive']}", flush=True)
    print("\n=== 分 regime 净 α (SIGN-R11) ===", flush=True)
    for rg, d in regime_alpha.items():
        print(f"  {rg:10s} 月数={d['n_months']:>3d} 月均={d['monthly_mean_pp']:+.3f}pp "
              f"t={d['t']} 正月%={d['pos_month_ratio']:.1%}", flush=True)

    status, gate_checks, gate_order = decide(
        alpha_s, skill_s, gross_alpha_s, outl, regime_alpha)

    print("\n=== 事前注册 gate 断言 (冻结 SIGN-R01) ===", flush=True)
    for k in gate_order:
        print(f"  [{'✓' if gate_checks[k] else '✗'}] {k}", flush=True)
    print(f"\n=== 裁决: {status} ===", flush=True)

    conclusion_map = {
        "PASS": "净 α 扛过成本+对照+outlier+regime → 进 SL-004 独立 sleeve 落地。",
        "真小": "净 α>0 且是席位技能, 但未达 +0.50pp/Sharpe1/稳健门槛 → 记 promising 待优化, 不强上。",
        "REJECT_成本吃掉": "毛 α>0 但净 α<=0 → 信号真但成本不经济 (短线高换手吃掉), 文档化容量/换手。",
        "REJECT_龙虎榜效应": "Arm_A≈Arm_B → 不是席位技能, 是'买龙虎榜'本身 → 文档化。",
        "REJECT_负": "净 α<0 且毛 α 也<0 → 干净 REJECT。",
    }

    verdict = {
        "id": "SL-003", "status": status,
        "conclusion": (
            f"r5 龙虎榜席位事件 sleeve walk-forward 裁决={status}。"
            f"主测 H={H}/{int(COST_RT*10000)}bps往返, {alpha_s['n_months']} 月事件对齐 walk-forward。"
            f"净 α(Arm_A net−等权全市场)={alpha_s['monthly_mean_pp']:+.3f}pp/月 t={alpha_s['t']} "
            f"Sharpe(年)={alpha_s['sharpe_ann']} 正月%={alpha_s['pos_month_ratio']:.0%} "
            f"最差月={alpha_s['worst_month_pp']:+.2f}pp; "
            f"毛 α={gross_alpha_s['monthly_mean_pp']:+.3f}pp; "
            f"席位技能净增量(Arm_A−Arm_B, 扣动量)={skill_s['monthly_mean_pp']:+.3f}pp t={skill_s['t']}; "
            f"单月outlier剔({outl['dropped_month']})后 {outl['mean_after_drop_pp']:+.3f}pp 仍正={outl['still_positive']}。"
            f" {conclusion_map[status]} (REJECT 合法完成 SIGN-R02, 中间指标≠落地仅净α walk-forward 定调 SIGN-R03)"),
        "artifact": str(MONTHLY_OUT.relative_to(ROOT)),
        "preRegisteredGate": {
            "alpha_pp": GATE_ALPHA_PP, "t": GATE_T, "skill_t": GATE_SKILL_T,
            "sharpe": GATE_SHARPE, "min_months": MIN_MONTHS,
            "frozen": "启动写死, 本 task 未改任何阈值 (SIGN-R01)",
        },
        "gate_checks": gate_checks,
        # verify.py isGate PASS 需要的 4 项指标 (无论裁决都给, 方便审计)
        "metrics": {
            "alpha_delta_pp": alpha_s["monthly_mean_pp"],
            "sharpe": alpha_s["sharpe_ann"],
            "worst_month": alpha_s["worst_month_pp"],
            "pos_alpha_month_ratio": alpha_s["pos_month_ratio"],
            "alpha_t": alpha_s["t"],
            "net_alpha_full": alpha_s,
            "gross_alpha_full": gross_alpha_s,
            "skill_A_minus_B_full": skill_s,
            "outlier_drop": outl,
            "regime_alpha": regime_alpha,
            "regime_skill": regime_skill,
            "n_months": alpha_s["n_months"],
        },
        "design": {
            "walk_forward": "席位战绩 expanding 因果 (avail<=D, D+1 生效) → 逐月 α 序列天然 walk-forward",
            "market_benchmark": f"等权全市场 H={H} open→open 前向收益 (ST 源头排除), 按 entry_date",
            "alpha": "Arm_A net − market (扣市场)", "skill": "Arm_A − Arm_B (扣动量, cost 抵消)",
            "monthly": "事件等权: 当月所有 entry-day 均值; t=mean/(std/√n); Sharpe=mean/std×√12",
            "main_cell": f"H={H}, N={TOP_N}, cost={int(COST_RT*10000)}bps round-trip (复用 SL-001, SIGN-R01)",
        },
        "artifacts": [str(MONTHLY_OUT.relative_to(ROOT)), str(MKT_CACHE.relative_to(ROOT))],
        "guardrails": ["SIGN-R01 gate 阈值冻结未改", "SIGN-R02 REJECT 合法完成",
                       "SIGN-R03 仅净α walk-forward 定调", "SIGN-R06 ST 源头排除",
                       "SIGN-R11 分 regime", "单月 outlier 剔除检验"],
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status} -> {VERDICT.relative_to(ROOT)}", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
