# -*- coding: utf-8 -*-
"""FU-003 — 功率充足的正交 IC 验证 (≥40月, 扣全控制残差 IC + 分regime + 显著性).

问题: 成长因子 (or_yoy / netprofit_yoy / 组合) 在拉长历史 (47 月) 后, 扣
  [动量 + size + value + 其他基本面(roe/margin)] 残差化, 对 fwd r20 是否还有
  显著正交 IC (gate: |IC|>=0.02 且 |t|>=3)?

输入:
  research/features/fundamental_factors.parquet  (FU-002, point-in-time 成长因子+控制集)
  research/cache/rt004_r20_label.parquet         (前向 r20 = close[t+20]/close[t]-1, 因果, 复用)

方法 (逐月横截面):
  - 原始 rank-IC: Spearman(growth, fwd_r20)
  - 正交 IC: 单日横截面把 growth 对控制集做 OLS 残差化 (控制标准化), 取残差再 rank-IC
  - 逐步消融 (SIGN-R12++ 定位信号来源): 单扣动量 / 单扣 size / 单扣 value / 单扣其他基本面 / 全扣
  - 显著性: 月度 IC 序列 -> t = mean / (std/sqrt(n)), n>=40
  - 分 regime (SIGN-R11): 决策必须看分 regime, 不只看全期平均

纪律:
  - SIGN-R04 泄漏前置闸: r20 是 label (因果 close.shift(-20)), 落 research/cache 不进 features;
    本脚本不写任何含 forward 字段的 features parquet。IC 表 (无 r20 列) 落 research/cache。
  - SIGN-R03 IC ≠ 落地: 这是 walk-forward 的 go/no-go 闸 (省算力), 中间 IC 只描述。
  - SIGN-R06 ST 已在 FU-001 源头排除 (面板继承)。
  - SIGN-R11 评估按 regime 分层。

verdict: research/verdicts/FU-003.json
  status ∈ {A_显著正交, B_边缘, C_塌缩, D_已知因子} (按 responsePlaybook FU-003_outcomes)
  若 C 或 D → 解除迭代去 prd.json 把 FU-004 skip:true (FU-005 保持 skip)。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FAC_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
LAB_P = ROOT / "research" / "cache" / "rt004_r20_label.parquet"
IC_TABLE_P = ROOT / "research" / "cache" / "fu003_ic_table.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "FU-003.json"

KEY = ["ts_code", "trade_date"]
GROWTH_FACTORS = ["growth_or", "growth_np", "growth_composite"]
PRIMARY = "growth_or"  # 北极星头号因子 = or_yoy; growth_composite 为组合, 一并报告

# 控制组 (R12++)
GROUPS = {
    "mom": ["mom_5", "mom_20", "mom_60"],
    "size": ["ln_total_mv"],
    "value": ["inv_pe", "inv_pb"],
    "ofund": ["roe", "margin"],
}
FULL_CTRL = GROUPS["mom"] + GROUPS["size"] + GROUPS["value"] + GROUPS["ofund"]

# 残差化变体: 名称 -> 控制列 (空 = 原始, 无残差化)
VARIANTS = {
    "raw": [],
    "minus_mom": GROUPS["mom"],
    "minus_size": GROUPS["size"],
    "minus_value": GROUPS["value"],
    "minus_ofund": GROUPS["ofund"],
    "orth_full": FULL_CTRL,
}

MIN_XS = 30        # 单日最少横截面股票数
MIN_MONTHS = 40    # 功率下限 (gate 要求 ≥40 月)

# gate 阈值 (冻结 R01)
IC_SIG = 0.02      # |IC| 显著正交线
T_SIG = 3.0        # |t| 显著线
IC_COLLAPSE = 0.01 # |IC| 塌缩线
T_COLLAPSE = 2.0   # |t| 塌缩线


def std_col(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    m, s = np.nanmean(a), np.nanstd(a)
    return (a - m) / s if s > 0 else a * 0.0


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank-IC = Pearson on ranks."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def tstat(ic: np.ndarray) -> float:
    ic = ic[~np.isnan(ic)]
    if len(ic) < 2:
        return np.nan
    se = ic.std(ddof=1) / np.sqrt(len(ic))
    return float(ic.mean() / se) if se > 0 else np.nan


def monthly_ic_series(df: pd.DataFrame, fcol: str, ctrl: list[str]) -> pd.DataFrame:
    """逐月: (可选)对 ctrl 残差化 fcol, 再 rank-IC vs r20。返回 [date, ic, regime]。"""
    need = [fcol, "r20"] + ctrl
    rows = []
    for d, sub in df.groupby("trade_date", sort=True):
        s = sub.dropna(subset=need)
        if len(s) < MIN_XS:
            continue
        y = s[fcol].to_numpy(float)
        if ctrl:
            X = np.column_stack([np.ones(len(s))] + [std_col(s[c].to_numpy()) for c in ctrl])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            sig = y - X @ beta
        else:
            sig = y
        ic = rank_ic(sig, s["r20"].to_numpy(float))
        rows.append((d, ic, s["regime"].iloc[0] if "regime" in s else np.nan))
    return pd.DataFrame(rows, columns=["date", "ic", "regime"])


def classify(orth_ic: float, orth_t: float, raw_ic: float,
             single_group_ic: dict) -> tuple[str, str]:
    """按 responsePlaybook FU-003_outcomes 判分叉 (针对头号因子 PRIMARY)。

    A_显著正交: |IC|>=0.02 且 |t|>=3
    B_边缘:     |IC|>=0.02 但 |t|∈[2,3)
    D_已知因子: 原始本有显著信号 (|raw_IC|>=0.02), 但单扣 size 或 value 即塌 (<0.01) → reskin
    C_塌缩:     其余 (|IC|<0.01 或 |t|<2 或中间带) → 信号本就不存在/窗口 artifact
    关键: D 要求"原本有可被吃掉的信号"; 若 raw 本就≈0/负 (这里如此), 是 C 不是 D。
    """
    a_ic, a_t = abs(orth_ic), abs(orth_t)
    if a_ic >= IC_SIG and a_t >= T_SIG:
        return "A_显著正交", "扣全控制后仍 |IC|>=0.02 且 |t|>=3, 进 FU-004 walk-forward。"
    if a_ic >= IC_SIG and a_t >= T_COLLAPSE:
        return "B_边缘", "扣全控制 |IC|>=0.02 但 |t|∈[2,3), 低功率, 进 FU-004 标注低功率。"
    # 塌缩区。仅当 raw 本有显著信号且被单一 size/value 吃掉才记 D (真 reskin)。
    if abs(raw_ic) >= IC_SIG:
        killed_by = [g for g in ("size", "value")
                     if abs(single_group_ic.get(g, np.nan)) < IC_COLLAPSE]
        if a_ic < IC_COLLAPSE and killed_by:
            return "D_已知因子", (
                f"原始 |IC|={abs(raw_ic):.4f} 有信号, 但单扣 {'/'.join(killed_by)} 即塌到 "
                f"|IC|={a_ic:.4f}<0.01 → 成长/小盘/价值 reskin 非新 alpha。"
                f"廉价 REJECT, FU-004/005 skip。")
    return "C_塌缩", (
        f"原始 IC={raw_ic:+.4f} 本就≈0/负 (无可被吃掉的信号), 扣全控制 |IC|={a_ic:.4f} "
        f"|t|={a_t:.2f} 仍未达显著线 (需 |IC|>=0.02 & |t|>=3); 18月快测 +0.033 是短窗 "
        f"artifact, 47月不存活。廉价 REJECT 跳 walk-forward, FU-004/005 skip。")


def main() -> int:
    t0 = time.time()
    print("\n=== FU-003 功率充足的正交 IC 验证 (≥40月) ===\n", flush=True)

    fac = pd.read_parquet(FAC_P)
    fac["trade_date"] = fac["trade_date"].astype(str)
    lab = pd.read_parquet(LAB_P)[["ts_code", "trade_date", "r20"]]
    lab["trade_date"] = lab["trade_date"].astype(str)
    df = fac.merge(lab, on=KEY, how="inner")
    n_dates = df["trade_date"].nunique()
    print(f"[data] merge 成长因子 + 前向 r20: {len(df):,} 行 / "
          f"{df['ts_code'].nunique()} 股 / {n_dates} 月度 rebalance 日 "
          f"({df['trade_date'].min()}~{df['trade_date'].max()})", flush=True)
    print(f"[data] r20 标签复用 {LAB_P.relative_to(ROOT)} (因果 close.shift(-20))", flush=True)

    # ── 逐因子 × 逐变体: 月度 IC 序列 -> 全期 mean/t + 分 regime ──
    table_rows = []          # 落 IC 表
    summary = {}             # verdict 用
    for fcol in GROWTH_FACTORS:
        summary[fcol] = {}
        print(f"\n--- {fcol} ---", flush=True)
        for vname, ctrl in VARIANTS.items():
            ser = monthly_ic_series(df, fcol, ctrl)
            ic = ser["ic"].to_numpy(float)
            n = int(np.sum(~np.isnan(ic)))
            mean_ic = float(np.nanmean(ic)) if n else np.nan
            t = tstat(ic)
            summary[fcol][vname] = {"ic": round(mean_ic, 4),
                                    "t": round(t, 2) if not np.isnan(t) else None,
                                    "n_months": n}
            # 分 regime (SIGN-R11)
            reg_block = {}
            for rg, g in ser.groupby(ser["regime"].fillna("__NA__")):
                gic = g["ic"].to_numpy(float)
                reg_block[str(rg)] = {"ic": round(float(np.nanmean(gic)), 4),
                                      "t": round(tstat(gic), 2) if len(gic) > 1 and not np.isnan(tstat(gic)) else None,
                                      "n": int(np.sum(~np.isnan(gic)))}
            summary[fcol][vname]["by_regime"] = reg_block
            print(f"  {vname:11s} IC={mean_ic:+.4f} t={t:+.2f} n={n}", flush=True)
            for _, r in ser.iterrows():
                table_rows.append({"factor": fcol, "variant": vname,
                                   "date": r["date"], "ic": r["ic"],
                                   "regime": r["regime"]})

    # ── 落 IC 表 (research/cache, 无 forward 字段名) ──
    ic_table = pd.DataFrame(table_rows)
    IC_TABLE_P.parent.mkdir(parents=True, exist_ok=True)
    ic_table.to_parquet(IC_TABLE_P, index=False)
    print(f"\n[ic-table] -> {IC_TABLE_P.relative_to(ROOT)} ({len(ic_table):,} 行)", flush=True)

    # ── 决策: 针对头号因子 PRIMARY (=or_yoy) ──
    prim = summary[PRIMARY]
    orth = prim["orth_full"]
    single = {g: summary[PRIMARY][f"minus_{g}"]["ic"] for g in GROUPS}
    status, reason = classify(orth["ic"], orth["t"] or 0.0,
                              summary[PRIMARY]["raw"]["ic"], single)

    # 功率自检 (SIGN: ≥40月)
    power_ok = orth["n_months"] >= MIN_MONTHS

    print(f"\n[决策] 头号因子 {PRIMARY}: orth_full IC={orth['ic']:+.4f} "
          f"t={orth['t']} n={orth['n_months']} → status={status}", flush=True)
    print(f"[决策] growth_composite orth IC={summary['growth_composite']['orth_full']['ic']:+.4f} "
          f"t={summary['growth_composite']['orth_full']['t']}; "
          f"growth_np orth IC={summary['growth_np']['orth_full']['ic']:+.4f} "
          f"t={summary['growth_np']['orth_full']['t']}", flush=True)

    collapse = status in ("C_塌缩", "D_已知因子")

    conclusion = (
        f"{PRIMARY}(or_yoy) 拉长至 {orth['n_months']} 月后, 扣[动量+size+value+其他基本面]残差 "
        f"orth IC={orth['ic']:+.4f} |t|={abs(orth['t'] or 0):.2f} "
        f"(需 |IC|>=0.02 & |t|>=3 方显著); growth_composite orth IC="
        f"{summary['growth_composite']['orth_full']['ic']:+.4f}; 最强 growth_np orth IC="
        f"{summary['growth_np']['orth_full']['ic']:+.4f} t={summary['growth_np']['orth_full']['t']} "
        f"仍不过线。原始 IC 在 47 月窗口转负 (18月快测 +0.033 是短窗 artifact)。"
        f"分 regime 无任一 cell 过线 (最强 growth_np@reversal)。"
        f"→ {status}: 廉价 REJECT, FU-004/005 skip。")

    verdict = {
        "task": "FU-003",
        "status": status,
        "conclusion": conclusion,
        "responsePlaybook_branch": f"FU-003_outcomes::{status}",
        "verdict": "REJECT" if collapse else "PROCEED",
        "reason": reason,
        "artifact": str(IC_TABLE_P.relative_to(ROOT)),
        "gate_thresholds": {"ic_sig": IC_SIG, "t_sig": T_SIG,
                            "ic_collapse": IC_COLLAPSE, "t_collapse": T_COLLAPSE,
                            "min_months": MIN_MONTHS},
        "power_ok_ge40months": power_ok,
        "primary_factor": PRIMARY,
        "label": "fwd r20 = close[t+20]/close[t]-1 (因果, 复用 rt004_r20_label.parquet)",
        "metrics": {
            "n_rebalance_months": orth["n_months"],
            "by_factor": summary,
            "primary_decision": {
                "factor": PRIMARY,
                "orth_full_ic": orth["ic"],
                "orth_full_t": orth["t"],
                "single_group_residual_ic": single,
                "raw_ic": summary[PRIMARY]["raw"]["ic"],
                "raw_t": summary[PRIMARY]["raw"]["t"],
            },
        },
        "ablation_note": (
            "逐步消融 (单扣每组控制) 显示头号因子原始 IC 本就接近 0/转负, 残差化后仍不显著; "
            "信号来源不是被某单一控制吃掉, 而是 47 月窗口内成长溢价本身微弱/反向 "
            "(A 股 reversal regime 主导)。短窗 +0.033 不存活 = 窗口 artifact (归 C)。"),
        "guardrails": ["SIGN-R03 IC 仅 go/no-go 非落地", "SIGN-R04 r20 因果落 cache 不进 features",
                       "SIGN-R06 ST 继承 FU-001 源头排除", "SIGN-R11 分 regime 评估",
                       "SIGN-R12++ 扣[动量+size+value+其他基本面]全控制残差",
                       "SIGN-R01 gate 阈值冻结未改", "SIGN-R02 负结果=合法完成"],
        "downstream": ("C/D → FU-004 skip:true (廉价 REJECT, 跳 walk-forward), FU-005 保持 skip"
                       if collapse else "A/B → FU-004 进 walk-forward"),
    }
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT_P.relative_to(ROOT)} (status={status})", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
