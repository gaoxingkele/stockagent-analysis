# -*- coding: utf-8 -*-
"""OA-000 — 共享正交筛查工具 (Track C2 campaign 的 Phase-1 廉价闸 + 多重检验总账).

所有正交因子族 (OA-VOL / OA-VP / OA-COM / OA-RC / OA-NEWS) 复用本模块跑 Phase-1:
  把候选因子对 FULL_CTRL=[动量+size+value+roe/margin] 逐月横截面残差化, 残差 rank-IC vs
  前向 r20, 逐步消融定位来源, 月度 IC 序列出 t, 分 regime, 逐因子 A/B/C/D 分类。

设计承袭 fu003_orthogonal_ic.py (同 gate 阈值 |IC|>=0.02 & |t|>=3, ≥40月, 冻结 R01),
泛化为任意因子族可调用。控制面板/标签全复用既有工件 (零额外 IO 成本):
  - FULL_CTRL + regime + industry: research/features/fundamental_factors.parquet (54 月度调仓日)
  - 前向 r20 (因果 close.shift(-20)): research/cache/rt004_r20_label.parquet

★ 多重检验总账 (SIGN-R15): 每次 screen 把测过的 (family, factor, variant) 写入
  research/cache/oa_manifest.json → n_trials, 供下游 walk-forward 的 DSR (deflated Sharpe)
  惩罚"搜了 N 个因子, best-of-N 会蒙到假阳性"。

纪律: SIGN-R03 IC 仅 go/no-go (walk-forward 才 ship); R04 候选只含 backward 列, 不写 forward;
  R06 ST 已在面板/标签源头排除; R11 分 regime; R12++ 扣全控制残差 + 逐步消融; R16 廉价闸先杀。
  本模块不读/不写生产文件 (R05)。`python research/oa_screen.py` 跑合成自检 (无需缓存)。
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CTRL_P = ROOT / "research" / "features" / "fundamental_factors.parquet"
LAB_P = ROOT / "research" / "cache" / "rt004_r20_label.parquet"
CACHE = ROOT / "research" / "cache"
VERDICT_DIR = ROOT / "research" / "verdicts"
MANIFEST_P = CACHE / "oa_manifest.json"

KEY = ["ts_code", "trade_date"]

# 控制组 (R12++) — 复用 fundamental_factors 已有列
GROUPS = {
    "mom": ["mom_5", "mom_20", "mom_60"],
    "size": ["ln_total_mv"],
    "value": ["inv_pe", "inv_pb"],
    "ofund": ["roe", "margin"],
}
FULL_CTRL = GROUPS["mom"] + GROUPS["size"] + GROUPS["value"] + GROUPS["ofund"]
VARIANTS = {
    "raw": [],
    "minus_mom": GROUPS["mom"],
    "minus_size": GROUPS["size"],
    "minus_value": GROUPS["value"],
    "minus_ofund": GROUPS["ofund"],
    "orth_full": FULL_CTRL,
}

MIN_XS = 30        # 单日最少横截面股票数
MIN_MONTHS = 40    # 功率下限 (≥40 月)

# gate 阈值 (冻结 R01, 同 fu003)
IC_SIG = 0.02
T_SIG = 3.0
IC_COLLAPSE = 0.01
T_COLLAPSE = 2.0

# forward 黑名单 (与 verify.py 一致): 候选因子列绝不命中
FORWARD_TOKENS = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                 {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                 {"max_gain", "maxgain", "future_ret", "fwd_ret", "label", "target"}
import re as _re
_FWD_PAT = [r"^r\d+_next", r"^fwd_", r"^future_", r"^next_", r"_forward$",
            r"^label$", r"^target$", r"^fr\d+$"]


def assert_backward(cols) -> None:
    """泄漏前置闸 (R04): 候选因子列不得命中 forward 黑名单。"""
    bad = [c for c in cols if c.lower() in FORWARD_TOKENS
           or any(_re.search(p, c.lower()) for p in _FWD_PAT)]
    if bad:
        raise AssertionError(f"候选含 forward 字段 (泄漏): {bad}")


def std_col(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    m, s = np.nanmean(a), np.nanstd(a)
    return (a - m) / s if s > 0 else a * 0.0


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def tstat(ic: np.ndarray) -> float:
    """朴素 t (IID 假设)。仅供参考; gate 用 tstat_nw (codex 0618: 重叠 r20 label 序列自相关)。"""
    ic = ic[~np.isnan(ic)]
    if len(ic) < 2:
        return np.nan
    se = ic.std(ddof=1) / np.sqrt(len(ic))
    return float(ic.mean() / se) if se > 0 else np.nan


def tstat_nw(ic: np.ndarray, lag: int = 2) -> float:
    """Newey-West t (HAC) of the IC-series mean — 校正重叠 r20 label 的序列自相关 (codex 0618 #1/#4)。

    月度调仓 + r20(20交易日)horizon 重叠极小, NW t≈朴素 t, 但这是诚实判据。
    长程方差 Ω = γ0 + 2·Σ_{l=1}^{L}(1 - l/(L+1))·γ_l;  se = sqrt(Ω/n);  t = mean/se。
    """
    ic = ic[~np.isnan(ic)]
    n = len(ic)
    if n < lag + 2:
        return tstat(ic)
    x = ic - ic.mean()
    g0 = float(np.dot(x, x) / n)
    omega = g0
    for l in range(1, lag + 1):
        gl = float(np.dot(x[l:], x[:-l]) / n)
        omega += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    if omega <= 0:
        return tstat(ic)
    se = np.sqrt(omega / n)
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
    """A_显著正交 / B_边缘 / D_已知因子(reskin) / C_塌缩 (同 fu003 playbook)。"""
    a_ic, a_t = abs(orth_ic), abs(orth_t)
    if a_ic >= IC_SIG and a_t >= T_SIG:
        return "A_显著正交", "扣全控制后仍 |IC|>=0.02 且 |t|>=3 → 进 walk-forward GATE。"
    if a_ic >= IC_SIG and a_t >= T_COLLAPSE:
        return "B_边缘", "扣全控制 |IC|>=0.02 但 |t|∈[2,3), 低功率 → 进 walk-forward 标低功率。"
    if abs(raw_ic) >= IC_SIG:
        killed_by = [g for g in ("mom", "size", "value")
                     if abs(single_group_ic.get(g, np.nan)) < IC_COLLAPSE]
        if a_ic < IC_COLLAPSE and killed_by:
            return "D_已知因子", (
                f"原始 |IC|={abs(raw_ic):.4f} 有信号, 但单扣 {'/'.join(killed_by)} 即塌到 "
                f"|IC|={a_ic:.4f}<0.01 → 动量/小盘/价值 reskin 非新 alpha。廉价 REJECT。")
    return "C_塌缩", (
        f"原始 IC={raw_ic:+.4f} 本就≈0/弱, 扣全控制 |IC|={a_ic:.4f} |t|={a_t:.2f} "
        f"未达显著线 (需 |IC|>=0.02 & |t|>=3) → 信号不存在/窗口 artifact。廉价 REJECT。")


# ─────────────────────── 多重检验总账 (SIGN-R15) ───────────────────────
def update_manifest(family: str, trials: list[str]) -> dict:
    """把本族测过的因子变体名累加进 campaign manifest → n_trials 供 DSR。"""
    man = {"families": {}, "n_trials_total": 0, "trial_names": []}
    if MANIFEST_P.exists():
        man = json.loads(MANIFEST_P.read_text(encoding="utf-8"))
    man.setdefault("families", {})
    man["families"][family] = sorted(set(trials))
    all_trials = []
    for fam, ts in man["families"].items():
        all_trials += [f"{fam}::{t}" for t in ts]
    man["trial_names"] = sorted(set(all_trials))
    man["n_trials_total"] = len(man["trial_names"])
    MANIFEST_P.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_P.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return man


# ─────────────────────── 一站式 screen ───────────────────────
def _load_controls() -> pd.DataFrame:
    ctrl = pd.read_parquet(CTRL_P)
    ctrl["trade_date"] = ctrl["trade_date"].astype(str)
    lab = pd.read_parquet(LAB_P)[["ts_code", "trade_date", "r20"]]
    lab["trade_date"] = lab["trade_date"].astype(str)
    keep = KEY + ["industry", "regime"] + FULL_CTRL
    keep = [c for c in keep if c in ctrl.columns]
    return ctrl[keep].merge(lab, on=KEY, how="inner")


def screen_factors(factor_df: pd.DataFrame, factor_cols: list[str],
                   family: str, primary: str | None = None,
                   ctrl_panel: pd.DataFrame | None = None,
                   write: bool = True) -> dict:
    """对一族候选因子跑 Phase-1。返回 verdict dict; write=True 落 ic_table+verdict+manifest。

    factor_df: 含 KEY + factor_cols (backward); 与控制面板 inner join (自动对齐到 54 月度日)。
    primary: 北极星因子名 (族级裁决依据); 默认取 factor_cols[0]。
    """
    assert_backward(factor_cols)
    primary = primary or factor_cols[0]
    fdf = factor_df.copy()
    fdf["trade_date"] = fdf["trade_date"].astype(str)
    panel = ctrl_panel if ctrl_panel is not None else _load_controls()
    df = panel.merge(fdf[KEY + factor_cols], on=KEY, how="inner")
    n_dates = df["trade_date"].nunique()

    table_rows, summary = [], {}
    for fcol in factor_cols:
        summary[fcol] = {}
        for vname, ctrl in VARIANTS.items():
            ser = monthly_ic_series(df, fcol, ctrl)
            ic = ser["ic"].to_numpy(float)
            n = int(np.sum(~np.isnan(ic)))
            mean_ic = float(np.nanmean(ic)) if n else np.nan
            t = tstat_nw(ic)          # gate 判据 = Newey-West HAC t (codex 0618)
            t_naive = tstat(ic)       # 仅参考
            reg_block = {}
            for rg, g in ser.groupby(ser["regime"].fillna("__NA__")):
                gic = g["ic"].to_numpy(float)
                gt = tstat_nw(gic)
                reg_block[str(rg)] = {"ic": round(float(np.nanmean(gic)), 4),
                                      "t": round(gt, 2) if len(gic) > 1 and not np.isnan(gt) else None,
                                      "n": int(np.sum(~np.isnan(gic)))}
            summary[fcol][vname] = {"ic": round(mean_ic, 4),
                                    "t": round(t, 2) if not np.isnan(t) else None,
                                    "t_naive": round(t_naive, 2) if not np.isnan(t_naive) else None,
                                    "n_months": n, "by_regime": reg_block}
            for _, r in ser.iterrows():
                table_rows.append({"factor": fcol, "variant": vname,
                                   "date": r["date"], "ic": r["ic"], "regime": r["regime"]})

    # 逐因子分类
    classifications = {}
    for fcol in factor_cols:
        s = summary[fcol]
        orth = s["orth_full"]
        single = {g: s[f"minus_{g}"]["ic"] for g in GROUPS}
        status, reason = classify(orth["ic"], orth["t"] or 0.0, s["raw"]["ic"], single)
        classifications[fcol] = {"status": status, "reason": reason,
                                 "orth_ic": orth["ic"], "orth_t": orth["t"],
                                 "raw_ic": s["raw"]["ic"], "n_months": orth["n_months"]}

    survivors = [f for f, c in classifications.items()
                 if c["status"] in ("A_显著正交", "B_边缘")]
    # 族级: 最强存活因子 (按 |orth_ic|); 若无存活, 全族 REJECT
    best = max(classifications.items(),
              key=lambda kv: abs(kv[1]["orth_ic"]) if kv[1]["orth_ic"] is not None else -1)
    family_verdict = "PROCEED" if survivors else "REJECT"

    man = update_manifest(family, factor_cols) if write else {"n_trials_total": len(factor_cols)}

    verdict = {
        "task": family, "status": "built",
        "verdict": family_verdict,
        "n_rebalance_months": n_dates,
        "power_ok_ge40months": bool(n_dates >= MIN_MONTHS),
        "primary_factor": primary,
        "survivors": survivors,
        "best_factor": best[0], "best_orth_ic": best[1]["orth_ic"], "best_orth_t": best[1]["orth_t"],
        "classifications": classifications,
        "by_factor_full": summary,
        "label": "fwd r20 (因果, rt004_r20_label)",
        "control_set": FULL_CTRL,
        "gate_thresholds": {"ic_sig": IC_SIG, "t_sig": T_SIG,
                            "ic_collapse": IC_COLLAPSE, "min_months": MIN_MONTHS},
        "n_trials_campaign_so_far": man["n_trials_total"],
        "conclusion": (
            f"{family}: {len(factor_cols)} 候选因子 Phase-1 筛查 ({n_dates}月). "
            f"存活(A/B): {survivors if survivors else '无'}; "
            f"最强 {best[0]} orth_full IC={best[1]['orth_ic']:+.4f} t={best[1]['orth_t']} "
            f"→ 族裁决 {family_verdict}。" +
            ("解除对应 walk-forward GATE。" if survivors else "全族廉价 REJECT, GATE 保持 skip。")),
        "guardrails": ["SIGN-R03 IC 仅 go/no-go", "SIGN-R04 候选 backward 自查通过",
                       "SIGN-R06 ST 源头排除 (面板继承)", "SIGN-R11 分 regime",
                       "SIGN-R12++ 扣全控制残差+逐步消融", "SIGN-R15 manifest 记 n_trials",
                       "SIGN-R16 廉价闸先杀", "SIGN-R01 gate 阈值冻结"],
    }
    if write:
        ic_table = pd.DataFrame(table_rows)
        ic_p = CACHE / f"oa_{family.lower().replace('oa-','').replace('-','_')}_ic_table.parquet"
        ic_table.to_parquet(ic_p, index=False)
        verdict["artifact"] = str(ic_p.relative_to(ROOT))
        VERDICT_DIR.mkdir(parents=True, exist_ok=True)
        (VERDICT_DIR / f"{family}.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return verdict


# ─────────────────────── 合成自检 ───────────────────────
def _selftest() -> None:
    print("=== oa_screen.py self-test (合成) ===", flush=True)
    rng = np.random.default_rng(0)
    dates = [f"2022{m:02d}28" for m in range(1, 13)] + [f"2023{m:02d}28" for m in range(1, 13)] + \
            [f"2024{m:02d}28" for m in range(1, 13)] + [f"2025{m:02d}28" for m in range(1, 11)]
    rows = []
    for d in dates:
        for i in range(120):
            mom = rng.normal(); sz = rng.normal(); r20 = 0.3 * mom + rng.normal(0, 1)
            noise = rng.normal()
            ortho = rng.normal(); r20 += 0.5 * ortho  # 注入与控制正交的真信号
            rows.append({"ts_code": f"{i:06d}.SZ", "trade_date": d, "regime": "mixed",
                         "industry": "x", "mom_5": mom, "mom_20": mom, "mom_60": mom,
                         "ln_total_mv": sz, "inv_pe": rng.normal(), "inv_pb": rng.normal(),
                         "roe": rng.normal(), "margin": rng.normal(), "r20": r20,
                         "f_noise": noise, "f_ortho": ortho, "f_mom_reskin": mom})
    panel = pd.DataFrame(rows)
    fac = panel[KEY + ["f_noise", "f_ortho", "f_mom_reskin"]].copy()
    v = screen_factors(fac, ["f_noise", "f_ortho", "f_mom_reskin"], "OA-SELFTEST",
                       ctrl_panel=panel.drop(columns=["f_noise", "f_ortho", "f_mom_reskin"]),
                       write=False)
    cl = v["classifications"]
    print(f"  f_noise   → {cl['f_noise']['status']} (orth IC={cl['f_noise']['orth_ic']:+.4f})", flush=True)
    print(f"  f_ortho   → {cl['f_ortho']['status']} (orth IC={cl['f_ortho']['orth_ic']:+.4f})", flush=True)
    print(f"  f_mom_reskin → {cl['f_mom_reskin']['status']} (orth IC={cl['f_mom_reskin']['orth_ic']:+.4f})", flush=True)
    assert cl["f_noise"]["status"] == "C_塌缩", cl["f_noise"]
    assert cl["f_ortho"]["status"] == "A_显著正交", cl["f_ortho"]
    assert "f_ortho" in v["survivors"], v["survivors"]
    print("\n[selftest] PASSED — 噪声→C, 正交真信号→A, 存活识别正确", flush=True)


if __name__ == "__main__":
    _selftest()
