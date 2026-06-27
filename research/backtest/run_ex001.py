# -*- coding: utf-8 -*-
"""EX-001 — 出场策略测 (分批止盈 TP + 回撤止损 SL), 同 V12.31 picks 受控 Δ.

承用户 triple-barrier 直觉。19× 价格/形态选股信号穷尽 → 增量不在新信号, 在**出场策略**。
对**同一批 V12.31 picks** (与 BT-002 完全一致的入场) 套不同出场规则, 比 book 净 Sharpe/年化/maxDD/换手,
分 regime。**入场逐位一致 → 各 arm 的 Δ 是受控差分, 共模 (r20 池 lookahead/集中) 相消** (SIGN-R03)。

预注册出场网格 (冻结于 prd.preRegisteredGate.exit_grid, **不搜不调** SIGN-R01):
  baseline : 现行固定 ~r20(=20 交易日)持有, 到期收盘平仓。
  TP-only  : 三档分批止盈 +10%/+20%/+30% 各卖原仓 1/3; 触发用日内 high 达标价 (缺口越过→开盘价;
             涨停不可卖→顺延); 残仓 40d backstop 收盘平。
  SL-only  : 自入场峰值回撤 {-8%, -12%} 触发, 卖全部残仓; 用日内 low (缺口跌破→开盘价; 跌停不可卖→顺延);
             无触发则 40d backstop 收盘平。两档分别报。
  TP+SL    : 组合 (TP 分批 + SL 兜底), 两档 SL 分别报。
  时间 backstop 40d (去固定 20d 后兜底)。close-based 作保守界参考另算。

受控 Δ 设计: 所有 arm 共享**同一组 cohort 入场** (每 20 交易日按 V12.31 picks 的 alloc 加权建仓,
日 D 收盘决定 → D+1 开盘成交, T+1)。各 arm 仅**出场逻辑**不同。book = cohort 子账户的等权聚合
(constant-investment 叠加; 某 cohort 全平后转现金, 期间无活 cohort 则当日 book 收益=0 = 止盈后钱闲置的真实拖累)。

成本模型复用 engine.CostModel (冻结); ST 源头排除 (R06); regime 分层必带 (R11);
描述性受控对照 = 合法完成 (R02); 中间指标≠ship (R03); 绝对量级含 r20 池 lookahead 非可交易声明。
大缓存写 research/cache/ex001/ (gitignored)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from engine import CostModel

CACHE001 = ROOT / "research" / "cache" / "bt001"
CACHE = ROOT / "research" / "cache" / "ex001"
CACHE.mkdir(parents=True, exist_ok=True)
PICKS = CACHE001 / "picks_v1231_daily.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "EX-001.json"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"

# ── 冻结参数 (R01, = prd.preRegisteredGate.exit_grid) ──
TOTAL = 0.90              # 投资比例 (双轨 A+B), 余 0.10 现金 (= BT-002)
REBAL_EVERY = 20         # cohort 入场周期 ≈ r20 持有
BASELINE_HOLD = 20       # baseline 固定持有交易日
BACKSTOP = 40            # 出场 arm 时间兜底 (交易日)
LIMIT = 9.8              # |pct_chg|>=LIMIT 当日不可成交
TP_LEVELS = [0.10, 0.20, 0.30]   # 三档止盈, 各卖原仓 1/3
SL_LEVELS = [0.08, 0.12]         # 两档回撤止损 (自入场峰值)
SLIP = 0.0010            # 基线滑点 0.10%/边 (= BT-002 基线)


def load_prices(start: str) -> pd.DataFrame:
    parts = []
    for f in sorted(DAILY_DIR.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high",
                                        "low", "close", "pct_chg"])
        d["trade_date"] = d["trade_date"].astype(str)
        d = d[d["trade_date"] >= start]
        if not d.empty:
            parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    basic = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    st = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    px = px[~px["ts_code"].isin(st)].reset_index(drop=True)   # ST 源头排除 (R06)
    return px


def build_lookup(px: pd.DataFrame):
    """ts_code -> {date: (open,high,low,close,pct)}; 全局日历。"""
    cal = sorted(px["trade_date"].unique())
    lut: dict[str, dict[str, tuple]] = {}
    for ts, g in px.groupby("ts_code"):
        lut[ts] = {d: (o, h, l, c, p) for d, o, h, l, c, p in
                   zip(g["trade_date"], g["open"], g["high"], g["low"], g["close"], g["pct_chg"])}
    return cal, lut


# ───────────── 单仓出场模拟 (受控: 仅此处随 arm 变) ─────────────
def simulate_position(rows: list, entry_price: float, arm: dict,
                      buy_rate: float, sell_rate: float, close_based: bool = False):
    """rows: [(o,h,l,c,pct) or None] 长 = 窗口, index0=入场日(当日 T+1 不可卖).
    返回逐日 position_value 序列 (= 残股市值 + 已实现现金, 含买入成本扣减), 长 = len(rows)。
    arm: {'tp':bool, 'sl':float|None, 'hold':int(baseline 时设, 否则 None)}。
    close_based=True: TP/SL 用收盘价判定与成交 (保守界)。"""
    shares0 = 1.0 / entry_price          # 单位名义 1.0 建仓 (权重在 cohort 层缩放)
    shares = shares0
    realized = -buy_rate                  # 买入成本 (名义 1.0 * buy_rate)
    peak = entry_price
    tp_done = [False] * len(TP_LEVELS)
    n = len(rows)
    hold = arm.get("hold")               # baseline 固定持有
    maxi = (hold if hold is not None else BACKSTOP)
    vals = [np.nan] * n
    last_close = entry_price

    def sell(frac_shares, price):
        nonlocal shares, realized
        sv = frac_shares * price
        realized += sv * (1.0 - sell_rate)
        shares -= frac_shares

    done = False
    for i in range(n):
        row = rows[i]
        if row is None:                  # 停牌: 沿用上一收盘估值, 不可交易
            vals[i] = shares * last_close + realized
            continue
        o, h, l, c, pct = row
        last_close = c

        if (not done) and i >= 1 and i <= maxi:
            limit_up = pct >= LIMIT
            limit_down = pct <= -LIMIT
            if arm.get("tp") and shares > 1e-15:
                for k, thr in enumerate(TP_LEVELS):
                    if tp_done[k]:
                        continue
                    lvl = entry_price * (1.0 + thr)
                    hit = (c >= lvl) if close_based else (h >= lvl)
                    if hit and not limit_up:
                        if close_based:
                            fill = c
                        else:
                            fill = o if o > lvl else lvl   # 缺口越过→开盘价
                        sell(shares0 / 3.0, fill)
                        tp_done[k] = True
            if arm.get("sl") is not None and shares > 1e-15:
                peak = max(peak, c if close_based else h)
                sl_lvl = peak * (1.0 - arm["sl"])
                hit = (c <= sl_lvl) if close_based else (l <= sl_lvl)
                if hit and not limit_down:
                    if close_based:
                        fill = c
                    else:
                        fill = o if o < sl_lvl else sl_lvl   # 缺口跌破→开盘价
                    sell(shares, fill)
                    done = True
            # baseline / backstop: 到期收盘平残仓
            if i >= maxi and shares > 1e-15 and not done:
                if not (limit_up or limit_down):     # 涨跌停不可平→顺延次日
                    sell(shares, c)
                    done = True

        vals[i] = shares * last_close + realized

    # 窗口末仍有残仓 (末日涨跌停/停牌顺延未成) → 回溯到最后一个可交易日收盘平, 之后日值=已实现现金
    if shares > 1e-15:
        for i in range(n - 1, -1, -1):
            if rows[i] is not None and abs(rows[i][4]) < LIMIT:
                sell(shares, rows[i][3])      # 收盘平残仓
                for j in range(i, n):
                    vals[j] = realized
                break
    return vals


# ───────────── cohort 聚合 → book NAV ─────────────
def run_arm(rebal_days, picks, cal, lut, arm, cm: CostModel, close_based=False):
    """各 cohort: 入场日 = rebal_day 的次一交易日; 仓位权重 = alloc_pct 归一到 TOTAL。
    cohort NAV(t) = 静态现金(1-TOTAL) + Σ_i w_i * position_value_i(t)。
    book: 逐日 = 当日活跃 cohort 的等权平均日收益 (无活跃 cohort → 0)。返回 nav Series(index=date)。"""
    buy_rate, sell_rate = cm.buy_rate(), cm.sell_rate()
    cal_idx = {d: i for i, d in enumerate(cal)}
    cohort_ret = {}   # date -> list of cohort daily returns

    for rd in rebal_days:
        ri = cal_idx.get(rd)
        if ri is None or ri + 1 >= len(cal):
            continue
        entry_i = ri + 1
        entry_date = cal[entry_i]
        g = picks[picks["trade_date"] == rd]
        if g.empty:
            continue
        tot_alloc = g["alloc_pct"].sum()
        if tot_alloc <= 0:
            continue
        win_len = (BASELINE_HOLD if arm.get("hold") is not None else BACKSTOP) + 6
        win_idx = list(range(entry_i, min(entry_i + win_len, len(cal))))
        win_dates = [cal[j] for j in win_idx]
        if len(win_dates) < 2:
            continue

        # cohort 逐日市值: 静态现金 + Σ w_i * pos_val_i
        cash_static = 1.0 - TOTAL
        port = np.zeros(len(win_dates))
        deployed = 0.0
        for _, r in g.iterrows():
            ts = r["ts_code"]
            w = r["alloc_pct"] / tot_alloc * TOTAL
            sd = lut.get(ts)
            if sd is None:
                continue
            er = sd.get(entry_date)
            if er is None or not np.isfinite(er[0]) or er[0] <= 0 or abs(er[4]) >= LIMIT:
                continue   # 入场日涨跌停/停牌不可建仓 → 跳过 (各 arm 一致, 共模)
            entry_price = er[0]
            rows = [sd.get(d) for d in win_dates]
            vals = simulate_position(rows, entry_price, arm, buy_rate, sell_rate, close_based)
            port += w * np.array([v if np.isfinite(v) else 0.0 for v in vals])
            deployed += w
        nav = cash_static + port      # cohort NAV 路径
        if nav[0] <= 0:
            continue
        rets = nav[1:] / nav[:-1] - 1.0
        for k, d in enumerate(win_dates[1:]):
            cohort_ret.setdefault(d, []).append(float(rets[k]))

    # 聚合: 逐日等权平均 cohort 收益
    all_dates = sorted(cohort_ret.keys())
    book_ret = pd.Series({d: np.mean(cohort_ret[d]) for d in all_dates})
    nav = (1.0 + book_ret).cumprod()
    return nav


def nav_summary(nav: pd.Series, ppy: int = 252) -> dict:
    nav = nav.dropna().astype(float)
    n = len(nav)
    if n < 2:
        return {"n_days": n, "error": "insufficient"}
    arr = nav.to_numpy()
    ret = pd.Series(arr).pct_change().dropna().to_numpy()
    ann = (arr[-1] / arr[0]) ** (ppy / max(n - 1, 1)) - 1.0
    vol = float(np.nanstd(ret, ddof=1))
    sharpe = (float(np.nanmean(ret)) / (vol + 1e-12)) * np.sqrt(ppy)
    peak = np.maximum.accumulate(arr)
    maxdd = float((arr / peak - 1.0).min())
    return {"n_days": int(n), "total_return": float(arr[-1] / arr[0] - 1.0),
            "ann_return": float(ann), "sharpe": float(sharpe), "max_drawdown": maxdd}


def monthly_returns(nav: pd.Series) -> pd.Series:
    s = nav.dropna().astype(float)
    ym = s.index.astype(str).str[:6]
    last = s.groupby(ym).last()
    return last.pct_change().dropna()


def main():
    t0 = time.time()
    print("\n=== EX-001: 出场策略测 (分批止盈 TP + 回撤止损 SL), 同 picks 受控 Δ ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / {picks['month'].nunique()} 月", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal, lut = build_lookup(px)
    print(f"[prices] {len(px):,} 行 / {px['ts_code'].nunique()} 股 / {len(cal)} 交易日", flush=True)

    rebal_days = [d for d in cal[::REBAL_EVERY] if d in set(picks["trade_date"].unique())]
    print(f"[cohorts] {len(rebal_days)} 个 (每 {REBAL_EVERY} 交易日入场)", flush=True)

    cm = CostModel(slippage=SLIP, enabled=True)

    # ── arm 定义 (冻结网格 R01) ──
    arms = {
        "baseline":   {"hold": BASELINE_HOLD},
        "TP_only":    {"tp": True},
        "SL_8":       {"sl": 0.08},
        "SL_12":      {"sl": 0.12},
        "TP_SL_8":    {"tp": True, "sl": 0.08},
        "TP_SL_12":   {"tp": True, "sl": 0.12},
    }

    print("\n[intraday 主口径] 各出场 arm book ...", flush=True)
    navs, summ = {}, {}
    for name, arm in arms.items():
        nav = run_arm(rebal_days, picks, cal, lut, arm, cm, close_based=False)
        navs[name] = nav
        summ[name] = nav_summary(nav)
        s = summ[name]
        print(f"  {name:10s}: 年化 {s['ann_return']:+.1%}  净Sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s['max_drawdown']:+.1%}  天数 {s['n_days']}", flush=True)

    # ── close-based 保守界 (TP_only / SL_8 / TP_SL_8) ──
    print("\n[close-based 保守界参考] ...", flush=True)
    cb_summ = {}
    for name in ["TP_only", "SL_8", "TP_SL_8"]:
        nav = run_arm(rebal_days, picks, cal, lut, arms[name], cm, close_based=True)
        cb_summ[name] = nav_summary(nav)
        s = cb_summ[name]
        print(f"  {name:10s}(close): 年化 {s['ann_return']:+.1%}  净Sharpe {s['sharpe']:+.2f}  "
              f"maxDD {s['max_drawdown']:+.1%}", flush=True)

    # ── 受控 Δ vs baseline ──
    base = summ["baseline"]
    deltas = {}
    print("\n[受控 Δ vs baseline] (Sharpe / 年化 / maxDD)", flush=True)
    for name in arms:
        if name == "baseline":
            continue
        d = {
            "d_sharpe": round(summ[name]["sharpe"] - base["sharpe"], 3),
            "d_ann_return_pp": round((summ[name]["ann_return"] - base["ann_return"]) * 100, 2),
            "d_maxdd_pp": round((summ[name]["max_drawdown"] - base["max_drawdown"]) * 100, 2),
        }
        deltas[name] = d
        print(f"  {name:10s}: ΔSharpe {d['d_sharpe']:+.3f}  Δ年化 {d['d_ann_return_pp']:+.2f}pp  "
              f"ΔmaxDD {d['d_maxdd_pp']:+.2f}pp", flush=True)

    # ── regime 分解 (R11) ──
    print("\n[regime 分解] 各 arm 月度收益按月 regime (R11) ...", flush=True)
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())

    base_m = monthly_returns(navs["baseline"])
    regime_delta = {}
    for name in arms:
        if name == "baseline":
            continue
        m = monthly_returns(navs[name])
        df = pd.DataFrame({"arm": m, "base": base_m}).dropna()
        df["regime"] = df.index.map(month_regime)
        df["d_pp"] = (df["arm"] - df["base"]) * 100
        regime_delta[name] = {rg: round(float(g["d_pp"].mean()), 3)
                              for rg, g in df.groupby("regime")}
    for name, rd in regime_delta.items():
        seg = "  ".join(f"{rg}:{v:+.2f}pp" for rg, v in rd.items())
        print(f"  {name:10s}: {seg}", flush=True)

    # ── 先验验证 ──
    tp_helps = bool(deltas["TP_only"]["d_sharpe"] > 0)
    sl_hurts = bool(deltas["SL_8"]["d_sharpe"] < 0 and deltas["SL_12"]["d_sharpe"] < 0)
    best_arm = max(deltas, key=lambda k: deltas[k]["d_sharpe"])

    results = {
        "config": {"total": TOTAL, "rebal_every": REBAL_EVERY, "baseline_hold": BASELINE_HOLD,
                   "backstop": BACKSTOP, "tp_levels": TP_LEVELS, "sl_levels": SL_LEVELS,
                   "slippage": SLIP, "n_cohorts": len(rebal_days),
                   "period": [start, cal[-1]]},
        "summary_intraday": summ, "summary_close_based": cb_summ,
        "deltas_vs_baseline": deltas, "regime_delta_vs_baseline": regime_delta,
        "prior_check": {"tp_helps_sharpe": tp_helps, "sl_hurts_sharpe": sl_hurts,
                        "best_arm": best_arm},
        "caveat": "受控 Δ (入场逐位一致) 共模相消可信; 绝对量级含 r20 池 lookahead + 集中 + cohort 等权聚合, "
                  "非可交易声明 (同 BT-001/002)。出场层无 walk-forward, 仅描述性受控对照 (EX-003 才是 ship gate)。",
    }
    (CACHE / "ex001_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for name, nav in navs.items():
        nav.to_frame("nav").to_parquet(CACHE / f"ex001_nav_{name}.parquet")

    # ── 裁决句 ──
    d_tp = deltas["TP_only"]; d_sl8 = deltas["SL_8"]; d_sl12 = deltas["SL_12"]
    d_c8 = deltas["TP_SL_8"]; d_c12 = deltas["TP_SL_12"]
    if tp_helps and sl_hurts:
        verdict_dir = "符合先验: 分批止盈帮 (顺均值回归 edge), 回撤止损伤 (砍在反弹前) → TP 进出场层候选, SL 弃; 指向 EX-003 用上屏障为主"
    elif (not tp_helps) and (not sl_hurts):
        verdict_dir = "意外: 止盈未帮且止损未伤, 需 walk-forward 去 lookahead 确认"
    elif tp_helps and (not sl_hurts):
        verdict_dir = "止盈帮, 止损未明显伤 → TP 候选, SL 中性"
    else:
        verdict_dir = "出场策略对 V12.31 book 无 Sharpe 增益 → book 瓶颈不在出场, 文档化 (指向 EX-003 重标/风格)"

    conclusion = (
        f"对同一批 V12.31 picks ({len(rebal_days)} cohort, {start}~{cal[-1]}) 套预注册出场网格 (受控 Δ, 入场逐位一致)。"
        f"baseline (固定 {BASELINE_HOLD}d 持有): 年化 {base['ann_return']:+.1%} 净Sharpe {base['sharpe']:+.2f} maxDD {base['max_drawdown']:+.1%}。"
        f"受控 Δ vs baseline: TP-only ΔSharpe {d_tp['d_sharpe']:+.3f} (Δ年化 {d_tp['d_ann_return_pp']:+.2f}pp, ΔmaxDD {d_tp['d_maxdd_pp']:+.2f}pp); "
        f"SL-8 ΔSharpe {d_sl8['d_sharpe']:+.3f} / SL-12 ΔSharpe {d_sl12['d_sharpe']:+.3f}; "
        f"TP+SL-8 ΔSharpe {d_c8['d_sharpe']:+.3f} / TP+SL-12 ΔSharpe {d_c12['d_sharpe']:+.3f}。"
        f"先验验证: 分批止盈帮 Sharpe={tp_helps}, 回撤止损伤 Sharpe={sl_hurts} (两档 SL 均负)。最优 arm={best_arm}。"
        f"{verdict_dir}。【绝对量级非可交易声明】含 r20 池共模 lookahead + 集中 + cohort 等权聚合 (同 BT-001/002); "
        f"出场层无 walk-forward, 仅受控描述性对照, ship 由 EX-003 r20 triple-barrier 重标 walk-forward 定。"
    )

    VERDICT.write_text(json.dumps({
        "id": "EX-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "n_cohorts": len(rebal_days),
            "baseline_ann_return": round(base["ann_return"], 4),
            "baseline_net_sharpe": round(base["sharpe"], 3),
            "baseline_max_drawdown": round(base["max_drawdown"], 4),
            "deltas_vs_baseline": deltas,
            "summary_intraday": {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, (int, float))}
                                 for k, v in summ.items()},
            "summary_close_based": {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, (int, float))}
                                    for k, v in cb_summ.items()},
            "tp_helps_sharpe": tp_helps,
            "sl_hurts_sharpe": sl_hurts,
            "best_arm": best_arm,
        },
        "regime_delta_vs_baseline": regime_delta,
        "cost_model": cm.describe(),
        "artifacts": [
            "research/backtest/run_ex001.py",
            "research/cache/ex001/ex001_results.json",
            "research/cache/ex001/ex001_nav_*.parquet",
        ],
        "note": "出场网格冻结于 prd.preRegisteredGate.exit_grid (R01, 不搜不调); 入场=BT-002 同 picks 共模相消 (R03); "
                "ST 源头排除 (R06); regime 分层必带 (R11); 描述性受控对照=合法完成 (R02); 中间指标≠ship (R03); "
                "大缓存 gitignored; 生产线只读未碰 (R05)。intraday 主口径 + close-based 保守界。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
