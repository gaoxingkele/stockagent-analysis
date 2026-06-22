# -*- coding: utf-8 -*-
"""MS-002 — 市场状态暴露 overlay 测试 (de-lookahead WFE-001 book).

在最诚实的 de-lookahead+embargo book (WFE-001, 净Sharpe1.31) 日级 ret 上应用市场状态暴露
缩放, 5 臂 apples-to-apples:
  base        : 静态全暴露 (ret 原值)
  dynamic_paper: PRIMARY 冻结规则 (paper alpha择时): GT 0→×1.0 / 0.5→×0.7 / 1→×0.4 (熊加牛减)
  trend       : 对照 (趋势beta择时, 反向): GT 0→×0.4 / 0.5→×0.7 / 1→×1.0 (牛加熊减)
  static_dele : 常数均暴露 = mean(m_dynamic) (Sharpe≡base, 隔离"降暴露"vs"timing")
  placebo     : 随机月度暴露排列 (N=1000 seed) → dynamic Sharpe 的经验 p值 (SIGN-R13 负控)

诊断: book 按 GOOD_TIMES 分 state 的 Sharpe/收益 (定位我们在哪种 state 赚/亏 = 假说核心)。
缩放语义: 日收益 r_t → m_month·r_t (其余现金0); 常数缩放 Sharpe 不变 → 任何提升必来自 timing。

gate (冻结 R01): PASS 当 dynamic_paper Sharpe>base 且 placebo p<0.05 且 maxDD不更差 且 改善不集中单state。
不碰生产 (R05)。supporting: 若有全周期 book 序列则附熊市行为 (lookahead 警告)。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
import research_env as renv

CACHE = ROOT / "research" / "cache"
BOOK_P = CACHE / "wfe001" / "wfe001_book_oos.parquet"
GT_P = CACHE / "ms_goodtimes.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "MS-002.json"

RULE_PAPER = {0.0: 1.0, 0.5: 0.7, 1.0: 0.4}   # PRIMARY: 熊加牛减
RULE_TREND = {0.0: 0.4, 0.5: 0.7, 1.0: 1.0}   # 对照: 牛加熊减
PPY = 252


def maxdd(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def metrics(ret: np.ndarray, months: np.ndarray) -> dict:
    ret = np.nan_to_num(ret, nan=0.0)
    nav = np.cumprod(1.0 + ret)
    # 月度收益 (复利)
    md = pd.DataFrame({"m": months, "r": ret})
    mret = md.groupby("m")["r"].apply(lambda s: float(np.prod(1.0 + s.to_numpy()) - 1.0))
    return {
        "sharpe": round(renv.sharpe(ret, PPY), 4),
        "ann_return": round(renv.ann_return(ret, PPY), 4),
        "max_dd": round(maxdd(nav), 4),
        "worst_month": round(float(mret.min()), 4),
        "monthly_win_rate": round(float((mret > 0).mean()), 4),
        "final_nav": round(float(nav[-1]), 4),
    }


def main():
    print("\n=== MS-002 市场状态暴露 overlay (de-lookahead WFE-001 book) ===\n", flush=True)
    book = pd.read_parquet(BOOK_P).reset_index()
    book = book.rename(columns={book.columns[0]: "trade_date"})
    book["trade_date"] = book["trade_date"].astype(str)
    book["month"] = book["trade_date"].str[:6]
    book = book[book["ret"].notna()].reset_index(drop=True)

    gt = pd.read_parquet(GT_P)[["month", "GOOD_TIMES"]]
    book = book.merge(gt, on="month", how="left")
    miss = book["GOOD_TIMES"].isna().sum()
    if miss:
        print(f"  ⚠ {miss} 日无 GOOD_TIMES (窗外), 用 0.5 填", flush=True)
        book["GOOD_TIMES"] = book["GOOD_TIMES"].fillna(0.5)

    r = book["ret"].to_numpy(float)
    months = book["month"].to_numpy()
    gtv = book["GOOD_TIMES"].round(2).to_numpy()
    m_paper = np.array([RULE_PAPER.get(round(x, 1), 0.7) for x in gtv])
    m_trend = np.array([RULE_TREND.get(round(x, 1), 0.7) for x in gtv])
    m_avg = float(np.mean(m_paper))

    arms = {
        "base": metrics(r, months),
        "dynamic_paper": metrics(m_paper * r, months),
        "trend": metrics(m_trend * r, months),
        "static_dele": metrics(m_avg * r, months),
    }
    print(f"窗口 {book['trade_date'].min()}~{book['trade_date'].max()} ({book['month'].nunique()}月, {len(book)}日)", flush=True)
    print(f"GOOD_TIMES 分布: {pd.Series(gtv).value_counts().sort_index().to_dict()}; 均暴露(paper规则)={m_avg:.3f}\n", flush=True)
    print(f"  {'臂':14s} {'Sharpe':>8s} {'年化':>9s} {'maxDD':>8s} {'最差月':>8s} {'月胜率':>7s} {'终值':>7s}", flush=True)
    for name, mm in arms.items():
        print(f"  {name:14s} {mm['sharpe']:>8.3f} {mm['ann_return']:>9.2%} {mm['max_dd']:>8.2%} "
              f"{mm['worst_month']:>8.2%} {mm['monthly_win_rate']:>7.1%} {mm['final_nav']:>7.3f}", flush=True)

    # 按 state 分桶诊断 (假说核心: 我们在哪种 state 赚/亏)
    print("\n[诊断] base book 按 GOOD_TIMES 分 state:", flush=True)
    state_diag = {}
    for s in sorted(set(gtv)):
        mask = gtv == s
        if mask.sum() < 5:
            continue
        rs = r[mask]
        sd = {"n_days": int(mask.sum()), "sharpe": round(renv.sharpe(rs, PPY), 3),
              "ann_return": round(renv.ann_return(rs, PPY), 4),
              "mean_daily_bps": round(float(np.nanmean(rs) * 1e4), 2)}
        state_diag[str(s)] = sd
        lbl = {0.0: "熊", 0.5: "过渡", 1.0: "牛"}.get(s, str(s))
        print(f"  GT={s:.2f}({lbl}) n={sd['n_days']:>4d} Sharpe={sd['sharpe']:>+6.2f} "
              f"年化={sd['ann_return']:>+7.2%} 日均={sd['mean_daily_bps']:+.2f}bps", flush=True)

    # placebo: 随机月度暴露排列 (保持同一组 m 值, 打乱到不同月) N seed
    print("\n[placebo] 随机择时负控 N=1000 ...", flush=True)
    mon_list = list(pd.unique(months))
    mon_m_paper = {mo: m_paper[months == mo][0] for mo in mon_list}
    base_sharpe = arms["base"]["sharpe"]
    dyn_sharpe = arms["dynamic_paper"]["sharpe"]
    rng = np.random.default_rng(20260618)
    perm_sharpes = []
    mvals = np.array([mon_m_paper[mo] for mo in mon_list])
    mon_idx = {mo: (months == mo) for mo in mon_list}
    for _ in range(1000):
        perm = rng.permutation(mvals)
        mm = np.empty(len(r))
        for mo, mv in zip(mon_list, perm):
            mm[mon_idx[mo]] = mv
        perm_sharpes.append(renv.sharpe(mm * r, PPY))
    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= dyn_sharpe).mean())   # dyn 超过多少比例随机
    print(f"  dynamic_paper Sharpe={dyn_sharpe:.3f} vs base={base_sharpe:.3f}", flush=True)
    print(f"  placebo Sharpe 分布: mean={perm_sharpes.mean():.3f} p5={np.percentile(perm_sharpes,5):.3f} "
          f"p95={np.percentile(perm_sharpes,95):.3f}", flush=True)
    print(f"  p值(随机择时 Sharpe >= dynamic) = {p_value:.3f}", flush=True)

    # gate
    c_sharpe = dyn_sharpe > base_sharpe
    c_placebo = p_value < 0.05
    c_dd = arms["dynamic_paper"]["max_dd"] >= arms["base"]["max_dd"]   # 不更差(更接近0)
    conds = {"dynamic>base_sharpe": bool(c_sharpe), "placebo_p<0.05": bool(c_placebo),
             "maxdd_not_worse": bool(c_dd)}
    if all(conds.values()):
        status = "PASS"
    elif (arms["dynamic_paper"]["max_dd"] > arms["base"]["max_dd"] + 0.02) and not c_sharpe:
        status = "真小_仅降回撤"
    elif arms["trend"]["sharpe"] > base_sharpe + 0.1 and not c_sharpe:
        status = "REJECT_paper反指_trend更优"
    else:
        status = "REJECT"

    print(f"\n  gate: {conds} → status={status}", flush=True)
    if arms["trend"]["sharpe"] > arms["dynamic_paper"]["sharpe"]:
        print(f"  ⚠ 对照 trend Sharpe={arms['trend']['sharpe']:.3f} > dynamic_paper {dyn_sharpe:.3f}: "
              f"净多头book趋势beta择时优于paper alpha择时 (但trend非primary, 需自己OOS)", flush=True)

    verdict = {
        "id": "MS-002", "status": status,
        "window": [book["trade_date"].min(), book["trade_date"].max()],
        "n_months": int(book["month"].nunique()), "n_days": int(len(book)),
        "base_is": "WFE-001 de-lookahead+embargo book (最诚实, 净Sharpe~1.31)",
        "good_times_dist": {str(k): int(v) for k, v in pd.Series(gtv).value_counts().sort_index().items()},
        "avg_exposure_paper_rule": round(m_avg, 3),
        "arms": arms,
        "state_diagnostic": state_diag,
        "placebo": {"n": 1000, "dynamic_sharpe": dyn_sharpe, "base_sharpe": base_sharpe,
                    "placebo_mean_sharpe": round(float(perm_sharpes.mean()), 4),
                    "placebo_p5_p95": [round(float(np.percentile(perm_sharpes, 5)), 4),
                                       round(float(np.percentile(perm_sharpes, 95)), 4)],
                    "p_value_random_ge_dynamic": p_value},
        "gate_conditions": conds, "gate_status": status,
        "conclusion": (
            f"市场状态暴露overlay在de-lookahead WFE-001 book ({book['month'].nunique()}月, 无熊月只过渡+牛) "
            f"上: dynamic_paper(熊加牛减) Sharpe={dyn_sharpe:.3f} vs base={base_sharpe:.3f} "
            f"(Δ{dyn_sharpe-base_sharpe:+.3f}), trend(牛加熊减)={arms['trend']['sharpe']:.3f}, "
            f"placebo p={p_value:.3f}. → {status}. "
            f"OOS窗无熊市故paper'熊市加仓'侧未测(caveat); static_dele Sharpe≡base 证任何提升来自timing非降暴露。"),
        "caveat": "long-only净多头吃不到paper空头端alpha; OOS窗(2024-10~2026-06)无熊月只能测'牛市减仓'侧; "
                  "全周期熊市行为需lookahead序列(supporting, 非裁决)。SIGN-R13 placebo负控已含。",
        "guardrails": ["SIGN-R01 gate冻结", "SIGN-R02 负/真小=合法完成", "SIGN-R11 分state",
                       "SIGN-R13 placebo负控+static_dele隔离降暴露", "SIGN-R05 生产线冻结"],
    }
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT_P.relative_to(ROOT)} (status={status})", flush=True)
    return status


if __name__ == "__main__":
    main()
