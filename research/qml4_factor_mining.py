# -*- coding: utf-8 -*-
"""QML-4 — 自动因子挖掘 (GP/随机符号搜索) + 正交IC廉价闸 + DSR 多重检验.

QuantML 合集: GP/RL-AlphaGen/LLM 自动生成 Alpha 因子。gplearn 不可用 → 自建紧凑随机符号搜索:
价量原语 {ret/overnight/intraday/vol/amount} × 算子 {tsmean/tsstd/tsdelta/tssum/tszscore/tsmax/
tsmin + div/mul/sub} × 窗口 {5,10,20,60} 组合生成 N 个候选因子, 全部过 oa_screen 正交 IC 廉价闸
(扣 FULL_CTRL 残差, NW-t), 统计存活数; 最强者报 DSR (n_trials=N, 多重检验惩罚极狠)。

AI Scientist 教训 (gate>吞吐): 枯竭价量空间自动搜索量产假阳性, 强 gate 一轮抓。预期: 多数因子
C_塌缩(残差被 FULL_CTRL 吃), 即便少数过廉价闸 DSR(N 试验)也杀。本任务=证明该教训 + 完成 Tier3。

全 backward(价量原语 t 只用 ≤t); 复用 oa_daily_panel; 确定性(seed). 生产冻结(R05)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import oa_screen
import oa_vol_factors as ov
import research_env as renv

CACHE = ROOT / "research" / "cache"
CTRL_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
VERDICT = ROOT / "research" / "verdicts" / "QML-4.json"
KEY = ["ts_code", "trade_date"]
N_FACTORS = 80
SEED = 20260619
WINDOWS = [5, 10, 20, 60]
BASES = ["ret", "overnight", "intraday", "vol", "amount"]


def g_roll(panel, col, w, op):
    g = panel.groupby("ts_code", sort=False)[col]
    if op == "tsmean": return g.transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean())
    if op == "tsstd": return g.transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).std())
    if op == "tssum": return g.transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).sum())
    if op == "tsdelta": return panel[col] - g.transform(lambda s: s.shift(w))
    if op == "tsmax": return g.transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).max())
    if op == "tsmin": return g.transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).min())
    if op == "tszscore":
        m = g.transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).mean())
        sd = g.transform(lambda s: s.rolling(w, min_periods=max(3, w // 2)).std())
        return (panel[col] - m) / (sd + 1e-9)
    raise ValueError(op)


UNARY = ["tsmean", "tsstd", "tssum", "tsdelta", "tsmax", "tsmin", "tszscore"]
BINARY = ["div", "mul", "sub"]


def gen_and_compute(panel, rng):
    """随机生成 N 个深度<=2 价量符号因子, 计算 → {name: series} (只在生成的列上)。"""
    facts, names = {}, []
    tries = 0
    while len(names) < N_FACTORS and tries < N_FACTORS * 4:
        tries += 1
        if rng.random() < 0.5:  # 一元
            b = rng.choice(BASES); op = rng.choice(UNARY); w = int(rng.choice(WINDOWS))
            name = f"{op}_{b}_{w}"
            if name in facts: continue
            s = g_roll(panel, b, w, op)
        else:  # 二元: op(unary(b1,w1), unary(b2,w2))
            b1, b2 = rng.choice(BASES), rng.choice(BASES)
            o1, o2 = rng.choice(UNARY), rng.choice(UNARY)
            w1, w2 = int(rng.choice(WINDOWS)), int(rng.choice(WINDOWS))
            bop = rng.choice(BINARY)
            name = f"{bop}__{o1}_{b1}_{w1}__{o2}_{b2}_{w2}"
            if name in facts: continue
            s1 = g_roll(panel, b1, w1, o1); s2 = g_roll(panel, b2, w2, o2)
            if bop == "div": s = s1 / (s2.abs() + 1e-9)
            elif bop == "mul": s = s1 * s2
            else: s = s1 - s2
        s = s.replace([np.inf, -np.inf], np.nan)
        facts[name] = s.to_numpy()
        names.append(name)
    return facts, names


def main():
    t0 = time.time()
    print(f"\n=== QML-4 自动因子挖掘 (随机符号搜索 N={N_FACTORS}) + 正交闸 + DSR ===\n", flush=True)
    panel = ov.build_panel().sort_values(KEY).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    print("[gen] 生成+计算价量符号因子 ...", flush=True)
    facts, names = gen_and_compute(panel, rng)
    print(f"[gen] {len(names)} 因子算完 ({time.time()-t0:.0f}s)", flush=True)

    ctrl_dates = set(pd.read_parquet(CTRL_P, columns=["trade_date"])["trade_date"].astype(str).unique())
    base_df = panel[KEY].copy()
    for nm in names:
        base_df[nm] = facts[nm]
    mask = base_df["trade_date"].isin(ctrl_dates)
    fdf = base_df[mask].reset_index(drop=True)
    del facts, base_df
    print(f"[screen] 54 月度日切片 {len(fdf):,} 行, 过 oa_screen 正交闸 ...", flush=True)

    verdict = oa_screen.screen_factors(fdf, names, "QML-4-MINING", primary=names[0], write=False)
    cl = verdict["classifications"]
    survivors = verdict["survivors"]
    n_A = sum(1 for c in cl.values() if c["status"] == "A_显著正交")
    n_B = sum(1 for c in cl.values() if c["status"] == "B_边缘")
    n_C = sum(1 for c in cl.values() if c["status"] == "C_塌缩")
    n_D = sum(1 for c in cl.values() if c["status"] == "D_已知因子")
    best = verdict["best_factor"]; best_ic = verdict["best_orth_ic"]; best_t = verdict["best_orth_t"]

    # DSR: 最强候选当作"搜索出的", n_trials=N (多重检验). 用 orth IC 的 t 近似 → 这里直接报 expected_max
    # 把 orth IC 月度序列的 per-obs Sharpe 类比: 用 |best_t| 与 N 算 deflated 显著性
    # 简化诚实版: N 次搜索下, 期望最大 |t| 基准 (Bonferroni 视角)
    from scipy import stats as _st
    # best 因子月度 IC t 已是 NW-t; N 次独立搜索的期望最大 |z| ≈ Phi^-1(1 - 1/(2N))
    z_crit_bonf = float(_st.norm.ppf(1 - 0.05 / (2 * len(names))))   # Bonferroni 校正临界 z
    dsr_pass = bool(abs(best_t or 0) >= z_crit_bonf)

    status = "PROCEED_phase2" if (survivors and dsr_pass) else "REJECT_cheap"
    conclusion = (
        f"随机符号搜索 {len(names)} 价量因子, 正交闸(扣FULL_CTRL残差,NW-t): "
        f"A_显著={n_A} B_边缘={n_B} C_塌缩={n_C} D_reskin={n_D}, 存活(A/B)={len(survivors)}. "
        f"最强 {best} orth_full IC={best_ic:+.4f} NW-t={best_t}. "
        f"多重检验: Bonferroni 临界 |z|={z_crit_bonf:.2f}(N={len(names)}), 最强|t|={'%.2f'%abs(best_t or 0)} "
        f"{'>=临界 → 过多重检验' if dsr_pass else '<临界 → 多重检验后不显著'}. → {status}. "
        + ("少数存活且过多重检验 → Phase-2 walk-forward。" if status == "PROCEED_phase2"
           else "自动搜索的价量因子残差被 FULL_CTRL 吃 + 多重检验惩罚 → 无真增量。印证 AI Scientist 教训: "
                "枯竭价量空间高吞吐搜索量产假阳性, 强 gate 一轮抓。与 QML-1/2/3 合: 模型/搜索轴均非杠杆。"))
    out = {"id": "QML-4", "status": status, "n_generated": len(names),
           "screen_breakdown": {"A": n_A, "B": n_B, "C": n_C, "D": n_D},
           "survivors": survivors[:20], "n_survivors": len(survivors),
           "best_factor": best, "best_orth_ic": best_ic, "best_orth_nw_t": best_t,
           "multiple_testing": {"n_trials": len(names), "bonferroni_z_crit": round(z_crit_bonf, 3),
                                "best_abs_t": round(abs(best_t or 0), 3), "passes": dsr_pass},
           "conclusion": conclusion,
           "guardrails": ["R01 gate冻结", "R03 IC仅go/no-go", "R04 价量原语backward",
                          "R12++ 扣FULL_CTRL残差", "R15 多重检验Bonferroni(n_trials=N)",
                          "R16 廉价闸先杀", "R05 生产冻结", "AI Scientist: gate>吞吐"]}
    VERDICT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== QML-4 汇总 ===", flush=True)
    print(f"  生成 {len(names)} | A={n_A} B={n_B} C={n_C} D={n_D} | 存活={len(survivors)}", flush=True)
    print(f"  最强 {best} orth IC={best_ic:+.4f} NW-t={best_t} | Bonferroni临界|z|={z_crit_bonf:.2f}", flush=True)
    print(f"  [决策] {status}", flush=True)
    print(f"  {conclusion}", flush=True)
    print(f"[done] {(time.time()-t0)/60:.1f} min -> {VERDICT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
