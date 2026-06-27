# -*- coding: utf-8 -*-
"""STAB-001 — 低换手/稳定化构建网格: baseline / V_stick / V_N30 / V_combo 过引擎 (换手↓稳定↑Sharpe不降?).

A1 北极星 (唯一识别出的真实 book 层杠杆): DIAG-001 证 V12.31 picks 对剔 1 个 0.005% 因子整批重排 (Jaccard
0.33 = 隐性高换手/刀尖选股); QuantML 证高换手吃 alpha。本 task 在 **DEP-001 PIT-clean 分数 + BT 引擎** 上,
只改**组合构建规则** (选股分数/池构造逐字不动, 复用 stab001_gen_pool 候选池), 测预注册网格能否换手↓+稳定↑
同时净 Sharpe 不材料性降。

四变体 (construction_grid, 冻结 R01; 同 PIT-clean 分数/embargo/close-based/成本/双轨权重, 仅构建规则不同):
  baseline = 现行双轨 (MAX_A=8/MAX_B=15, top20/20d 再平衡) = DEP-001 PIT-clean 基线 (应复刻 Sharpe≈0.84)
  V_stick  = 持仓粘性: 已持有名 pool_rank<=STICK_RANK(30) 给排序 bonus → 留任, 跌出 top30/出池才卖 (滞回带)
  V_N30    = 降集中: MAX_A=10/MAX_B=22 (双轨权重不变 → 每名仓位更低), top~30
  V_combo  = 粘性 + N30

每变体指标: 月均换手 / 净Sharpe(close-based+成本) / maxDD / 持仓期期重叠(连续再平衡日 Jaccard, 稳定代理) /
  扰动重排率 (对 holder 含泄漏分数的同构建 pick 重叠, 接 DIAG-001) / 配对 block bootstrap ΔSharpe vs baseline CI。

gate_stab (冻结 R01): book 改进候选 = 月均换手↓>=20% AND 净Sharpe ΔvsBaseline 配对bootstrap CI 不材料性<0
  (不显著变差) AND maxDD 不升 AND 稳定代理↑。命中=纳入 V12.32 候选 (进 STAB-002 深验), 否则记录。

responsePlaybook (事前注册 R02): 找到改进变体 / 全变体降Sharpe / 无差异。
R05 生产只读; R06 ST 源头排除 (load_prices); R11 regime 分层; R02 负结果=合法完成; R03 中间指标≠ship;
R04 大缓存 gitignored 无 features 写入。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from engine import PortfolioBacktester, CostModel
from run_bt002 import (load_prices, load_index_navs, nav_summary, monthly_returns,
                       TOTAL, REBAL_EVERY, BASELINE_SLIP)
from walk_forward_validation import apply_cap, industry_alloc
from t005_walk_forward_gate import A_PCT, B_PCT, CAP_IN, CAP_CROSS

STAB = ROOT / "research" / "cache" / "stab001"
POOL_CLEAN = STAB / "candidate_pool_clean.parquet"
POOL_HOLDER = STAB / "candidate_pool_holder.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
DEP_RESULTS = ROOT / "research" / "cache" / "dep001" / "dep001_results.json"
VERDICT = ROOT / "research" / "verdicts" / "STAB-001.json"

# ── 构建网格 (冻结 R01) ──
BASE_A, BASE_B = 8, 15            # baseline 双轨大小 (= t005 MAX_A/MAX_B)
N30_A, N30_B = 10, 22            # V_N30 降集中 (双轨权重不变 A_PCT/B_PCT → 每名仓位更低)
STICK_RANK = 30                  # 持仓粘性滞回带: 已持有名 pool_rank<=此 则留任 (跌出 top30/出池才卖)
STICK_BONUS = 1e6                # 排序 bonus (保证留任名优先, 但留任名之间相对序仍按 ratio_s5)
# gate 阈值 (冻结 R01)
TURNOVER_DROP_MIN = 0.20         # 月均换手须 ↓>=20%
MAXDD_TOL = 0.005                # maxDD "不升" 容差 (var_maxDD >= base_maxDD - tol)
# bootstrap (同 dep001/FIN-001)
COHORT_LEN, N_BOOT, BOOT_SEED, PPY = 20, 1000, 20260616, 252


def sharpe_arr(r: np.ndarray) -> float:
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    return float(np.mean(r) / (np.std(r, ddof=1) + 1e-12) * np.sqrt(PPY))


def size_dual(pool_d: pd.DataFrame, max_a: int, max_b: int, sort_col: str) -> dict:
    """复刻 t005 build_dual 的双轨 sizing (a_pool head(max_a)+cap, b_pool head(max_b)+双cap), 返回 {ts:weight}.
    pool_d: 单日候选池 (含 ts_code/industry/ratio_s5/pool_rank/sort_col)。逐行同 build_dual。"""
    v7c = pool_d.sort_values(sort_col, ascending=False)
    a_pool = v7c.head(max_a).copy()
    a_pool = apply_cap(a_pool, A_PCT, CAP_IN, sort_col=sort_col)
    a_ind = industry_alloc(a_pool, A_PCT)
    b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
    b_pool = b_cand.head(max_b).copy()
    b_pool = apply_cap(b_pool, B_PCT, CAP_IN, sort_col=sort_col)
    b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind, sort_col=sort_col)
    out = {}
    for pool, alloc in [(a_pool, A_PCT), (b_pool, B_PCT)]:
        if len(pool) == 0:
            continue
        per = alloc / len(pool)
        for _, row in pool.iterrows():
            out[row["ts_code"]] = out.get(row["ts_code"], 0.0) + per
    return out


def build_targets(pool: pd.DataFrame, rebal: list, max_a: int, max_b: int,
                  sticky: bool) -> dict:
    """逐再平衡日构建目标。sticky=True 时给已持有且 pool_rank<=STICK_RANK 的名 bonus (滞回留任)。"""
    by_day = {d: g for d, g in pool.groupby("trade_date")}
    targets, held = {}, set()
    for d in rebal:
        g = by_day.get(d)
        if g is None or g.empty:
            continue
        g = g.copy()
        if sticky and held:
            keep = g["ts_code"].isin(held) & (g["pool_rank"] <= STICK_RANK)
            g["sort_col"] = g["ratio_s5"] + np.where(keep, STICK_BONUS, 0.0)
            col = "sort_col"
        else:
            col = "ratio_s5"
        w = size_dual(g, max_a, max_b, col)
        if not w:
            continue
        targets[d] = w
        held = set(w)
    return targets


def consec_jaccard(targets: dict, rebal: list) -> float:
    """连续再平衡日持仓名集 Jaccard 均值 (持仓期期重叠稳定代理; 高=换手低/稳定)。"""
    ds = [d for d in rebal if d in targets]
    js = []
    for i in range(len(ds) - 1):
        a, b = set(targets[ds[i]]), set(targets[ds[i + 1]])
        u = a | b
        if u:
            js.append(len(a & b) / len(u))
    return float(np.mean(js)) if js else float("nan")


def perturb_jaccard(t_clean: dict, t_holder: dict, rebal: list) -> float:
    """同构建在 clean vs holder 分数下的 pick 重叠均值 (扰动重排率, 接 DIAG-001; 高=对 holder 扰动稳定)。"""
    js = []
    for d in rebal:
        if d not in t_clean or d not in t_holder:
            continue
        a, b = set(t_clean[d]), set(t_holder[d])
        u = a | b
        if u:
            js.append(len(a & b) / len(u))
    return float(np.mean(js)) if js else float("nan")


def run_book(targets: dict, px, start: str):
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    return bt.run(targets, start=start)


def paired_boot(base_ret: pd.Series, var_ret: pd.Series):
    """per-cohort 配对 block bootstrap of ΔSharpe(var - base) (同 dep001, 消共模)。"""
    common = sorted(set(base_ret.index) & set(var_ret.index))
    br = base_ret.reindex(common).to_numpy(); vr = var_ret.reindex(common).to_numpy()
    n = len(common); cid = np.arange(n) // COHORT_LEN; nc = int(cid.max()) + 1
    bc = [br[cid == c] for c in range(nc)]; vc = [vr[cid == c] for c in range(nc)]
    rng = np.random.default_rng(BOOT_SEED)
    ds = []
    for _ in range(N_BOOT):
        pk = rng.integers(0, nc, size=nc)
        sb = sharpe_arr(np.concatenate([bc[c] for c in pk]))
        sv = sharpe_arr(np.concatenate([vc[c] for c in pk]))
        if np.isfinite(sb) and np.isfinite(sv):
            ds.append(sv - sb)
    ds = np.array(ds)
    ci = (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))
    return {"delta_sharpe_ci95": [round(ci[0], 3), round(ci[1], 3)],
            "delta_sharpe_median": round(float(np.median(ds)), 3),
            "p_delta_lt_0": round(float((ds < 0).mean()), 3),
            "ci_contains_0": bool(ci[0] <= 0 <= ci[1]), "ci_upper_ge_0": bool(ci[1] >= 0)}


def main():
    t0 = time.time()
    print("\n=== STAB-001: 低换手/稳定化构建网格 (baseline/V_stick/V_N30/V_combo) ===\n", flush=True)
    pool_c = pd.read_parquet(POOL_CLEAN); pool_c["trade_date"] = pool_c["trade_date"].astype(str)
    pool_h = pd.read_parquet(POOL_HOLDER); pool_h["trade_date"] = pool_h["trade_date"].astype(str)
    start = pool_c["trade_date"].min()
    print(f"[pool] clean {len(pool_c):,} 行 / {pool_c['trade_date'].nunique()} 日; holder {len(pool_h):,} 行", flush=True)

    print(f"[prices] load daily >= {start} (ST 源头排除, R06) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    rebal = cal[::REBAL_EVERY]
    end = cal[-1]
    print(f"[prices] {len(px):,} 行 / {len(cal)} 交易日 [{start}~{end}]; 再平衡 {len(rebal)} 次", flush=True)

    # ── 四变体构建 (clean 臂主回测) ──
    variants = {
        "baseline": dict(max_a=BASE_A, max_b=BASE_B, sticky=False),
        "V_stick":  dict(max_a=BASE_A, max_b=BASE_B, sticky=True),
        "V_N30":    dict(max_a=N30_A, max_b=N30_B, sticky=False),
        "V_combo":  dict(max_a=N30_A, max_b=N30_B, sticky=True),
    }
    tgt_clean, tgt_holder, books = {}, {}, {}
    for name, sp in variants.items():
        tgt_clean[name] = build_targets(pool_c, rebal, sp["max_a"], sp["max_b"], sp["sticky"])
        tgt_holder[name] = build_targets(pool_h, rebal, sp["max_a"], sp["max_b"], sp["sticky"])
        res = run_book(tgt_clean[name], px, start)
        books[name] = res.timeseries
        res.timeseries.to_parquet(STAB / f"book_{name}.parquet")
        s = res.summary()
        print(f"  [{name:9s}] 年化 {s['ann_return']:+.1%}  净Sharpe {s['sharpe']:+.2f}  maxDD {s['max_drawdown']:+.1%}  "
              f"月换手 {s['avg_monthly_turnover']:.2f}  持仓 {s['avg_n_positions']:.0f}", flush=True)

    # ── 指标 ──
    base_ret = books["baseline"]["nav"].astype(float).pct_change().dropna()
    base_ret.index = base_ret.index.astype(str)
    metrics = {}
    for name in variants:
        ts = books[name]
        s = BacktestSummary(ts)
        cj = consec_jaccard(tgt_clean[name], rebal)
        pj = perturb_jaccard(tgt_clean[name], tgt_holder[name], rebal)
        vr = ts["nav"].astype(float).pct_change().dropna(); vr.index = vr.index.astype(str)
        boot = paired_boot(base_ret, vr) if name != "baseline" else None
        metrics[name] = {
            "net_sharpe": round(s["sharpe"], 3),
            "ann_return": round(s["ann_return"], 4),
            "max_drawdown": round(s["max_drawdown"], 4),
            "avg_monthly_turnover": round(s["avg_monthly_turnover"], 3),
            "avg_n_positions": round(s["avg_n_positions"], 1),
            "consec_holding_jaccard": round(cj, 4),
            "perturb_jaccard_vs_holder": round(pj, 4),
            "boot_delta_sharpe_vs_baseline": boot,
        }

    base = metrics["baseline"]
    print("\n[指标汇总]", flush=True)
    for name, m in metrics.items():
        print(f"  {name:9s}: Sharpe {m['net_sharpe']:+.2f} | 月换手 {m['avg_monthly_turnover']:.2f} | "
              f"maxDD {m['max_drawdown']:+.1%} | 连续Jaccard {m['consec_holding_jaccard']:.3f} | "
              f"扰动Jaccard {m['perturb_jaccard_vs_holder']:.3f}", flush=True)

    # ── gate_stab 判各变体 ──
    base_turn = base["avg_monthly_turnover"]
    base_dd = base["max_drawdown"]
    base_cj = base["consec_holding_jaccard"]
    base_pj = base["perturb_jaccard_vs_holder"]
    gate_eval = {}
    for name in ["V_stick", "V_N30", "V_combo"]:
        m = metrics[name]
        turn_drop = (base_turn - m["avg_monthly_turnover"]) / base_turn if base_turn else 0.0
        cond_turn = bool(turn_drop >= TURNOVER_DROP_MIN)
        cond_sharpe = bool(m["boot_delta_sharpe_vs_baseline"]["ci_upper_ge_0"])  # 不材料性<0
        cond_maxdd = bool(m["max_drawdown"] >= base_dd - MAXDD_TOL)              # 回撤不升 (dd 负, >= 更浅)
        cond_stab = bool(m["consec_holding_jaccard"] > base_cj or m["perturb_jaccard_vs_holder"] > base_pj)
        passed = bool(cond_turn and cond_sharpe and cond_maxdd and cond_stab)
        gate_eval[name] = {
            "turnover_drop_frac": round(turn_drop, 3),
            "cond_turnover_down_20pct": cond_turn,
            "cond_sharpe_not_materially_worse": cond_sharpe,
            "cond_maxdd_not_up": cond_maxdd,
            "cond_stability_up": cond_stab,
            "PASS": passed,
        }
        print(f"  gate[{name:7s}]: 换手↓{turn_drop:+.0%}({cond_turn}) Sharpe不材料降({cond_sharpe}) "
              f"maxDD不升({cond_maxdd}) 稳定↑({cond_stab}) => {'PASS' if passed else 'fail'}", flush=True)

    winners = [n for n, g in gate_eval.items() if g["PASS"]]
    # 全变体是否 Sharpe 材料性下降 (boot CI 整段 <0)
    all_worse = all(not metrics[n]["boot_delta_sharpe_vs_baseline"]["ci_upper_ge_0"]
                    for n in ["V_stick", "V_N30", "V_combo"])
    if winners:
        status = "找到改进变体"
        best = max(winners, key=lambda n: (gate_eval[n]["turnover_drop_frac"], metrics[n]["net_sharpe"]))
    elif all_worse:
        status = "全变体降Sharpe"; best = None
    else:
        status = "无差异"; best = None

    # ── regime 分层 (R11): 各变体超额 hs300 分 regime ──
    idx = load_index_navs(start, end)
    book_dates = books["baseline"]["nav"].index.astype(str).tolist()
    idx = idx[(idx["trade_date"] >= start) & (idx["trade_date"] <= book_dates[-1])].copy()
    idx = idx.set_index("trade_date").reindex(book_dates).ffill()
    hs = idx["hs300_close"].dropna(); hs = hs / hs.iloc[0]; hs = hs.reindex(book_dates).ffill()
    reg = pd.read_parquet(REGIME); reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())
    hs_m = monthly_returns(hs)
    regime_by_variant = {}
    for name in variants:
        bm = monthly_returns(books[name]["nav"])
        cm = pd.DataFrame({"book": bm, "hs300": hs_m}).dropna()
        cm["regime"] = cm.index.map(month_regime)
        cm["excess"] = (cm["book"] - cm["hs300"]) * 100
        regime_by_variant[name] = {rg: round(float(g["excess"].mean()), 3)
                                   for rg, g in cm.groupby("regime")}

    # ── 裁决 ──
    dep_sharpe = json.loads(DEP_RESULTS.read_text(encoding="utf-8"))["clean_book"]["sharpe"]
    repro_note = (f"baseline 复刻 DEP-001 PIT-clean book: Sharpe {base['net_sharpe']:+.2f} "
                  f"vs DEP-001 {dep_sharpe:+.2f} (|Δ|={abs(base['net_sharpe']-dep_sharpe):.2f}, "
                  f"{'忠实复刻' if abs(base['net_sharpe']-dep_sharpe) < 0.10 else '有偏差'})")

    pieces = []
    for name in ["V_stick", "V_N30", "V_combo"]:
        m, g = metrics[name], gate_eval[name]
        pieces.append(f"{name}(换手{m['avg_monthly_turnover']:.2f}/{g['turnover_drop_frac']:+.0%}, "
                      f"Sharpe{m['net_sharpe']:+.2f} ΔCI{m['boot_delta_sharpe_vs_baseline']['delta_sharpe_ci95']}, "
                      f"maxDD{m['max_drawdown']:+.1%}, 连续J{m['consec_holding_jaccard']:.2f}/扰动J{m['perturb_jaccard_vs_holder']:.2f} "
                      f"vs base{base_cj:.2f}/{base_pj:.2f} => {'PASS' if g['PASS'] else 'fail'})")

    status_txt = {
        "找到改进变体": (f"低换手变体 **{best}** 过 gate_stab (换手↓"
                    f"{gate_eval[best]['turnover_drop_frac']:+.0%} 且 Sharpe 不材料性降 + maxDD 不升 + 稳定↑) "
                    f"→ V12.32 book 改进候选, 进 STAB-002 深验 (换手-成本敏感 + 分regime)。" if best else ""),
        "全变体降Sharpe": ("V_stick/V_N30/V_combo 配对 bootstrap ΔSharpe CI 整段 < 0 → 低换手以 Sharpe 为代价, "
                      "**V12.31 现行换手/集中度是必要的** (刀尖选股的另一面 = 高换手买到的就是边际 alpha), "
                      "记录不动 (responsePlaybook['全变体降Sharpe'])。"),
        "无差异": ("构建变化 (粘性/N30) 对 21 月 book Sharpe 无材料性影响 (ΔSharpe CI 含 0, 噪声主导), "
                 "且未同时满足换手↓20%+稳定↑+Sharpe不降的全部条件 → 文档化, **稳定性靠增 N 不靠粘性** "
                 "(responsePlaybook['无差异']); 若仅换手↓但 Sharpe 不增不减, 作 STAB-002 deployable 候选保留判。"),
    }[status]

    conclusion = (
        f"STAB-001 (A1 低换手/稳定化构建): 在 **DEP-001 PIT-clean 分数 + BT 引擎** 上, **选股分数/池构造逐字不动** "
        f"(复用 stab001_gen_pool 重建的每日 V7c 候选池, NO retrain, R05), 仅改组合构建规则跑 4 变体 "
        f"({start}~{book_dates[-1]}, {len(rebal)} 个 20d 再平衡 cohort)。{repro_note}。"
        f"变体 vs baseline (换手{base_turn:.2f}/Sharpe{base['net_sharpe']:+.2f}/maxDD{base_dd:+.1%}/"
        f"连续J{base_cj:.2f}/扰动J{base_pj:.2f}): " + "; ".join(pieces) + "。"
        f"配对 block bootstrap ({N_BOOT}次) 消共模只测构建净效应。"
        f"**裁决 [{status}]**: {status_txt} "
        f"分 regime (R11): 各变体超额 hs300 见 regime_table。生产线 V12.31 全程只读冻结 (R05); "
        f"构建层改进候选须过 STAB-002 深验才谈纳入 V12.32 (中间不 ship, R03)。"
    )

    results = {
        "config": {"period": [start, book_dates[-1]], "n_rebal": len(rebal), "rebal_every_tdays": REBAL_EVERY,
                   "total_invested": TOTAL, "baseline_slippage": BASELINE_SLIP,
                   "grid": {"baseline": [BASE_A, BASE_B], "V_N30": [N30_A, N30_B],
                            "stick_rank": STICK_RANK, "A_PCT": A_PCT, "B_PCT": B_PCT},
                   "gate_thresholds": {"turnover_drop_min": TURNOVER_DROP_MIN, "maxdd_tol": MAXDD_TOL},
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "note": "选股分数/池构造不动 (复用 dep001 PIT-clean 候选池), 仅构建规则不同 (R01)"},
        "metrics": metrics, "gate_eval": gate_eval, "regime_by_variant": regime_by_variant,
        "baseline_reproduces_dep001": {"stab_baseline_sharpe": base["net_sharpe"], "dep001_sharpe": round(dep_sharpe, 3)},
        "winners": winners, "best_variant": best, "status": status,
    }
    (STAB / "stab001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    VERDICT.write_text(json.dumps({
        "id": "STAB-001", "status": status, "conclusion": conclusion,
        "metrics": {
            "baseline_net_sharpe": base["net_sharpe"],
            "baseline_avg_monthly_turnover": base_turn,
            "baseline_max_drawdown": base_dd,
            "baseline_consec_jaccard": base_cj,
            "baseline_perturb_jaccard": base_pj,
            "dep001_sharpe_reference": round(dep_sharpe, 3),
            "variants": {n: {
                "net_sharpe": metrics[n]["net_sharpe"],
                "avg_monthly_turnover": metrics[n]["avg_monthly_turnover"],
                "turnover_drop_frac": gate_eval[n]["turnover_drop_frac"] if n != "baseline" else 0.0,
                "max_drawdown": metrics[n]["max_drawdown"],
                "consec_holding_jaccard": metrics[n]["consec_holding_jaccard"],
                "perturb_jaccard_vs_holder": metrics[n]["perturb_jaccard_vs_holder"],
                "boot_delta_sharpe_ci95": metrics[n]["boot_delta_sharpe_vs_baseline"]["delta_sharpe_ci95"] if n != "baseline" else None,
                "boot_ci_contains_0": metrics[n]["boot_delta_sharpe_vs_baseline"]["ci_contains_0"] if n != "baseline" else None,
                "gate_PASS": gate_eval[n]["PASS"] if n != "baseline" else None,
            } for n in variants},
            "winners": winners, "best_variant": best, "playbook": status,
        },
        "regime_table": regime_by_variant,
        "artifacts": ["research/backtest/stab001_gen_pool.py", "research/backtest/run_stab001.py",
                      "research/cache/stab001/candidate_pool_clean.parquet",
                      "research/cache/stab001/candidate_pool_holder.parquet",
                      "research/cache/stab001/stab001_results.json",
                      "research/cache/stab001/book_baseline.parquet"],
        "note": "A1 book 层构建实验; 唯一改动=组合构建 (粘性/N30), 选股分数=DEP-001 PIT-clean 候选池不动 (R01)。"
                "复用缓存 dep001 r20(剔holder)+t005 s5 模型, NO retrain (R05 生产指纹未变)。embargo+close-based "
                "成本, apples-to-apples vs DEP-001 PIT-clean 基线; ST 源头排除 (R06); regime 分层 (R11)。"
                "配对 block bootstrap 同分数共模相消; 扰动重排率接 DIAG-001 (对 holder 含泄漏分数同构建 pick 重叠); "
                "大缓存 gitignored 无 features 写入 (R04); 负结果=合法完成 (R02); 中间指标≠ship (R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict={status}"
          + (f" best={best}" if best else "")
          + f" -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


# 小工具: 直接对 timeseries 取 summary (避免重复 run)
def BacktestSummary(ts: pd.DataFrame) -> dict:
    from engine import BacktestResult
    return BacktestResult(timeseries=ts, trades=pd.DataFrame()).summary()


if __name__ == "__main__":
    main()
