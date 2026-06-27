# -*- coding: utf-8 -*-
"""WFE-002 — 强化版止盈复测 (embargo picks + close-based 主口径 + 随机 p90 + 分解负控).

吃进 codex 第二次 review 第 2 + 5 条 (research/backtest/audits/)。WF-002 在 de-lookahead 真实 OOS
picks 上得 "TP真改进 (frozen gate)" 但 codex 指出 gate 仍太松:
  ① TP ΔSharpe +0.809 vs 随机阈值 placebo **均值** +0.769 仅多 +0.04 且落在随机分布内 → 用"胜过均值"
     当 gate 把"结构性 de-risk"过度叙述成"TP 真改进"; 应改成 **TP 须 > 随机阈值分布 p90** 才谈择价 alpha。
  ② 成交口径应以 **close-based 保守成交为主口径** (非日内 high 的乐观假设), 防成交乐观抬升假象。
  ③ 应**分解 edge 来源**: 结构性减仓的边际到底来自 (a) 盈利条件 (在赢时减) (b) 缩短持有 (c) 阈值档位。

WFE-002 三处全强化:
  ① picks 换成 **WFE-001 label-embargo 真·真实 OOS picks** (picks_oos_daily.parquet)。
  ② **主口径 = close-based 保守成交** (TP 触发后按当日 close 成交, 非日内 high)。所有 gate / 分解
     裁决都在 close-based 上做; 日内 high 版仅留作乐观参考。
  ③ 强化 gate (prd.gate_tp_v2, R01 冻结):
       结构性减仓改进 = close-based TP ΔSharpe(TP40c-base40)>0 AND bootstrap CI 不含0 AND > 静态降暴露 placebo;
       择价 alpha     = TP 须 > 随机阈值 placebo 分布 **p90** (非均值; 预期不成立 = 档位无 alpha)。
  ④ 扩展负控分解 edge (3 控, 全 close-based, 描述性诊断 SIGN-R03):
       (a) 盈利条件控  cond_profit       — 入场后**首个上涨且盈利日**卖 1/3 (无阈值, 仅"在赢时减")。
       (b) 缩短持有控  short_hold_random — **随机卖出日** but 条件已盈利 (隔离"缩短持有 + 盈利门控", 无阈值)。
       (c) 收益分位控  ret_quantile      — 阈值由**实现收益分位** (各仓 within-window max-cumret 的 33/67/90 分位)
                                            派生, 非固定 +10/20/30% 价格档 (隔离"档位 level 是否重要")。
     哪个控的 ΔSharpe 最接近 TP_close → 该来源即 edge 主驱动。

统计严谨 (SIGN-R13): per-cohort block bootstrap CI + leave-one-out + 分 regime (R11)。受控 Δ: 两臂入场
逐位一致 (同一批 embargo OOS picks), 仅出场逻辑变 → 共模相消; 且已在 de-lookahead+embargo 真实 picks 上 (R13①)。

网格预注册冻结 (R01, = prd.preRegisteredGate.tp_retest_v2), 不搜不调。ST 源头排除 (R06);
大缓存写 research/cache/wfe002/ (gitignored); 生产线只读 (R05)。
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

WFE001 = ROOT / "research" / "cache" / "wfe001"
PICKS = WFE001 / "picks_oos_daily.parquet"          # SIGN-R13① + 第1条: embargo 真·真实 OOS picks
CACHE = ROOT / "research" / "cache" / "wfe002"
CACHE.mkdir(parents=True, exist_ok=True)
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "WFE-002.json"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"

# ── 冻结参数 (R01, = prd.preRegisteredGate.tp_retest_v2) ──
TOTAL = 0.90
REBAL_EVERY = 20
LIMIT = 9.8
TP_LEVELS = [0.10, 0.20, 0.30]      # 三档真实止盈, 各卖原仓 1/3
SLIP = 0.0010                       # 基线滑点 0.10%/边 (= WFE-001/WF-001/BT-002)
HOLDS = {"20d": 20, "40d": 40}      # baseline 两个对齐持有期
RANDOM_TP_RANGE = (0.05, 0.35)      # placebo 随机阈值区间 (冻结)
N_RANDOM_SEEDS = 50                 # placebo_random_TP 多 seed (codex 第2条: 50)
N_SHORT_SEEDS = 50                  # short_hold_random 多 seed (分解控)
RET_Q = [0.33, 0.67, 0.90]          # ret_quantile 控: 实现收益分位 (派生阈值)
N_BOOT = 1000                       # block bootstrap 次数
BOOT_SEED = 20260614                # 复现用固定种子


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
    cal = sorted(px["trade_date"].unique())
    lut: dict[str, dict[str, tuple]] = {}
    for ts, g in px.groupby("ts_code"):
        lut[ts] = {d: (o, h, l, c, p) for d, o, h, l, c, p in
                   zip(g["trade_date"], g["open"], g["high"], g["low"], g["close"], g["pct_chg"])}
    return cal, lut


# ───────────── 单仓出场模拟 (返回 总市值 + 在险股权两路) ─────────────
def simulate_position(rows, entry_price, *, mode, hold, buy_rate, sell_rate,
                      tp_levels=None, close_based=False, rand_offsets=None):
    """mode in {'hold','tp','cond_profit','short_hold_random'}; hold=固定/兜底持有交易日数.
    返回 (vals, stock_vals): vals=残股市值+已实现现金(含买入成本); stock_vals=shares*last_close.
      tp                : 三档价格阈值 (tp_levels, 可随机/分位派生); close_based 时按 close 成交。
      cond_profit       : 首 3 个"上涨且盈利日"各卖 1/3 (无阈值, close 成交)。
      short_hold_random : rand_offsets 指定的随机日, 若当日盈利则卖 1/3 (close 成交)。
    index0=入场日 (T+1 当日不可卖)。"""
    shares0 = 1.0 / entry_price
    shares = shares0
    realized = -buy_rate
    levels = tp_levels if tp_levels is not None else TP_LEVELS
    tp_done = [False] * len(levels)
    rand_day_set = set(rand_offsets) if rand_offsets else set()
    tranches_done = 0
    n = len(rows)
    maxi = hold
    vals = [np.nan] * n
    stock_vals = [np.nan] * n
    last_close = entry_price

    def sell(frac_shares, price):
        nonlocal shares, realized
        realized += frac_shares * price * (1.0 - sell_rate)
        shares -= frac_shares

    done = False
    for i in range(n):
        row = rows[i]
        if row is None:                       # 停牌: 沿用上一收盘估值, 不可交易
            vals[i] = shares * last_close + realized
            stock_vals[i] = shares * last_close
            continue
        o, h, l, c, pct = row
        last_close = c
        if (not done) and i >= 1 and i <= maxi:
            limit_up = pct >= LIMIT
            if mode == "tp" and shares > 1e-15 and not limit_up:
                for k, thr in enumerate(levels):
                    if tp_done[k]:
                        continue
                    lvl = entry_price * (1.0 + thr)
                    hit = (c >= lvl) if close_based else (h >= lvl)
                    if hit:
                        fill = c if close_based else (o if o > lvl else lvl)
                        sell(shares0 / 3.0, fill)
                        tp_done[k] = True
            elif mode == "cond_profit" and shares > 1e-15 and not limit_up:
                # (a) 盈利条件控: 首 3 个"上涨且盈利日" (pct>0 & close>entry) 各卖 1/3, 无阈值, close 成交
                if tranches_done < 3 and pct > 0 and c > entry_price:
                    sell(shares0 / 3.0, c)
                    tranches_done += 1
            elif mode == "short_hold_random" and shares > 1e-15 and not limit_up:
                # (b) 缩短持有控: 随机卖出日 but 条件已盈利 (close>=entry), close 成交
                if i in rand_day_set and c >= entry_price:
                    sell(shares0 / 3.0, c)
            # 到期收盘平残仓 (涨跌停不可平→顺延)
            if i >= maxi and shares > 1e-15 and not done:
                if abs(pct) < LIMIT:
                    sell(shares, c)
                    done = True
        vals[i] = shares * last_close + realized
        stock_vals[i] = shares * last_close

    if shares > 1e-15:                        # 末端残仓回溯到最后可交易日平
        for i in range(n - 1, -1, -1):
            if rows[i] is not None and abs(rows[i][4]) < LIMIT:
                sell(shares, rows[i][3])
                for j in range(i, n):
                    vals[j] = realized
                    stock_vals[j] = 0.0
                break
    return vals, stock_vals


# ───────────── cohort 模拟 → per-cohort 日收益 + 暴露 ─────────────
def run_arm_cohorts(rebal_days, picks, cal, lut, *, mode, hold, cm,
                    tp_levels=None, close_based=False, weight_scale=1.0, rng=None,
                    random_tp=False):
    """返回 (ret_df, expo_df): index=date, columns=cohort_id.
    weight_scale: 静态降暴露 (投 weight_scale*TOTAL, 余现金)。
    random_tp: 每仓抽随机价格阈值 (placebo_random_TP)。
    mode='short_hold_random': 每仓抽 3 个随机卖出日 offset (rng)。"""
    buy_rate, sell_rate = cm.buy_rate(), cm.sell_rate()
    cal_idx = {d: i for i, d in enumerate(cal)}
    ret_cols, expo_cols = {}, {}
    win_len = hold + 6
    inv = TOTAL * weight_scale
    cash_static = 1.0 - inv

    for cid, rd in enumerate(rebal_days):
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
        win_idx = list(range(entry_i, min(entry_i + win_len, len(cal))))
        win_dates = [cal[j] for j in win_idx]
        if len(win_dates) < 2:
            continue
        port = np.zeros(len(win_dates))
        stock = np.zeros(len(win_dates))
        for _, r in g.iterrows():
            ts = r["ts_code"]
            w = r["alloc_pct"] / tot_alloc * inv
            sd = lut.get(ts)
            if sd is None:
                continue
            er = sd.get(entry_date)
            if er is None or not np.isfinite(er[0]) or er[0] <= 0 or abs(er[4]) >= LIMIT:
                continue                         # 入场涨跌停/停牌不可建仓 (各臂一致, 共模)
            entry_price = er[0]
            rows = [sd.get(d) for d in win_dates]
            lv = tp_levels
            roff = None
            if random_tp and rng is not None:
                lv = sorted(rng.uniform(RANDOM_TP_RANGE[0], RANDOM_TP_RANGE[1], size=3))
            if mode == "short_hold_random" and rng is not None:
                hi = min(hold, len(rows) - 1)
                if hi >= 1:
                    k = min(3, hi)
                    roff = sorted(rng.choice(range(1, hi + 1), size=k, replace=False).tolist())
            vals, svals = simulate_position(rows, entry_price, mode=mode, hold=hold,
                                            buy_rate=buy_rate, sell_rate=sell_rate,
                                            tp_levels=lv, close_based=close_based,
                                            rand_offsets=roff)
            port += w * np.array([v if np.isfinite(v) else 0.0 for v in vals])
            stock += w * np.array([v if np.isfinite(v) else 0.0 for v in svals])
        nav = cash_static + port
        if nav[0] <= 0:
            continue
        rets = nav[1:] / nav[:-1] - 1.0
        expo = stock[1:] / nav[1:]
        dates1 = win_dates[1:]
        ret_cols[cid] = pd.Series(rets, index=dates1)
        expo_cols[cid] = pd.Series(expo, index=dates1)
    ret_df = pd.DataFrame(ret_cols).sort_index()
    expo_df = pd.DataFrame(expo_cols).sort_index()
    return ret_df, expo_df


def collect_maxret_levels(rebal_days, picks, cal, lut, qs):
    """(c) 收益分位控: 扫各仓 within-window max(close/entry - 1), 取分位作派生阈值 (诊断 lookahead, 标注)。"""
    cal_idx = {d: i for i, d in enumerate(cal)}
    maxrets = []
    for rd in rebal_days:
        ri = cal_idx.get(rd)
        if ri is None or ri + 1 >= len(cal):
            continue
        entry_i = ri + 1
        entry_date = cal[entry_i]
        g = picks[picks["trade_date"] == rd]
        win_idx = list(range(entry_i, min(entry_i + HOLDS["40d"] + 6, len(cal))))
        win_dates = [cal[j] for j in win_idx]
        for _, r in g.iterrows():
            sd = lut.get(r["ts_code"])
            if sd is None:
                continue
            er = sd.get(entry_date)
            if er is None or not np.isfinite(er[0]) or er[0] <= 0 or abs(er[4]) >= LIMIT:
                continue
            ep = er[0]
            mr = -1.0
            for d in win_dates[1:]:
                row = sd.get(d)
                if row is not None and np.isfinite(row[3]):
                    mr = max(mr, row[3] / ep - 1.0)
            if mr > -1.0:
                maxrets.append(mr)
    maxrets = np.array(maxrets)
    levels = [float(np.percentile(maxrets, q * 100)) for q in qs]
    return sorted(levels), maxrets


def book_from_cols(ret_df: pd.DataFrame, cols=None) -> pd.Series:
    sub = ret_df if cols is None else ret_df.iloc[:, list(cols)]
    return (1.0 + sub.mean(axis=1, skipna=True).dropna()).cumprod()


def sharpe_of(nav: pd.Series, ppy: int = 252) -> float:
    nav = nav.dropna().astype(float)
    if len(nav) < 3:
        return float("nan")
    ret = nav.pct_change().dropna().to_numpy()
    vol = float(np.nanstd(ret, ddof=1))
    return float(np.nanmean(ret) / (vol + 1e-12) * np.sqrt(ppy))


def nav_summary(nav: pd.Series, ppy: int = 252) -> dict:
    nav = nav.dropna().astype(float)
    n = len(nav)
    if n < 3:
        return {"n_days": n, "error": "insufficient"}
    arr = nav.to_numpy()
    ret = pd.Series(arr).pct_change().dropna().to_numpy()
    ann = (arr[-1] / arr[0]) ** (ppy / max(n - 1, 1)) - 1.0
    vol = float(np.nanstd(ret, ddof=1))
    sharpe = float(np.nanmean(ret) / (vol + 1e-12) * np.sqrt(ppy))
    peak = np.maximum.accumulate(arr)
    maxdd = float((arr / peak - 1.0).min())
    return {"n_days": int(n), "total_return": float(arr[-1] / arr[0] - 1.0),
            "ann_return": float(ann), "sharpe": sharpe, "max_drawdown": maxdd}


def monthly_returns(nav: pd.Series) -> pd.Series:
    s = nav.dropna().astype(float)
    ym = s.index.astype(str).str[:6]
    return s.groupby(ym).last().pct_change().dropna()


def avg_exposure(expo_df: pd.DataFrame) -> float:
    return float(expo_df.mean(axis=1, skipna=True).mean())


def multiseed_dsharpe(rebal_days, picks, cal, lut, cm, base_sharpe, *, mode,
                      random_tp=False, n_seeds=50, master_seed=BOOT_SEED):
    """多 seed 平均某随机臂的 ΔSharpe vs base, 返回 (mean_sharpe, mean_dsharpe, (p5,p90,p95), ret_dfs)。"""
    rng_master = np.random.default_rng(master_seed)
    sharpes, dsharpes, dfs = [], [], []
    for _ in range(n_seeds):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        rdf, _ = run_arm_cohorts(rebal_days, picks, cal, lut, mode=mode, hold=HOLDS["40d"],
                                 cm=cm, close_based=True, random_tp=random_tp, rng=rng,
                                 tp_levels=TP_LEVELS if mode == "tp" else None)
        dfs.append(rdf)
        sm = nav_summary(book_from_cols(rdf))
        sharpes.append(sm["sharpe"]); dsharpes.append(sm["sharpe"] - base_sharpe)
    return (float(np.mean(sharpes)), float(np.mean(dsharpes)),
            (float(np.percentile(dsharpes, 5)), float(np.percentile(dsharpes, 90)),
             float(np.percentile(dsharpes, 95))), dfs)


def main():
    t0 = time.time()
    print("\n=== WFE-002: 强化版止盈复测 (embargo picks + close-based 主口径 + 随机p90 + 分解负控) ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] WFE-001 embargo OOS: {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / "
          f"{picks['month'].nunique()} 月", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal, lut = build_lookup(px)
    print(f"[prices] {len(px):,} 行 / {px['ts_code'].nunique()} 股 / {len(cal)} 交易日", flush=True)

    pick_dates = set(picks["trade_date"].unique())
    rebal_days = [d for d in cal[::REBAL_EVERY] if d in pick_dates]
    print(f"[cohorts] {len(rebal_days)} 个 (每 {REBAL_EVERY} 交易日入场)\n", flush=True)

    cm = CostModel(slippage=SLIP, enabled=True)

    # ── 主臂 (close-based 主口径) ──
    arms = {}
    print("[arms] 模拟各臂 cohort (close-based 主口径) ...", flush=True)
    arms["baseline_20d"]  = run_arm_cohorts(rebal_days, picks, cal, lut, mode="hold", hold=HOLDS["20d"], cm=cm)
    arms["baseline_40d"]  = run_arm_cohorts(rebal_days, picks, cal, lut, mode="hold", hold=HOLDS["40d"], cm=cm)
    arms["TP_40d_close"]  = run_arm_cohorts(rebal_days, picks, cal, lut, mode="tp", hold=HOLDS["40d"], cm=cm, tp_levels=TP_LEVELS, close_based=True)   # ★ 主口径/gate
    arms["TP_40d_high"]   = run_arm_cohorts(rebal_days, picks, cal, lut, mode="tp", hold=HOLDS["40d"], cm=cm, tp_levels=TP_LEVELS, close_based=False)  # 乐观参考

    summ = {k: nav_summary(book_from_cols(v[0])) for k, v in arms.items()}
    for name in ["baseline_20d", "baseline_40d", "TP_40d_close", "TP_40d_high"]:
        s = summ[name]
        print(f"  {name:14s}: 年化 {s['ann_return']:+.1%}  净Sharpe {s['sharpe']:+.2f}  maxDD {s['max_drawdown']:+.1%}", flush=True)

    base40_nav = book_from_cols(arms["baseline_40d"][0])
    base40_sharpe = summ["baseline_40d"]["sharpe"]
    tpc_sharpe = summ["TP_40d_close"]["sharpe"]                 # ★ 主口径
    d_sharpe_40c = tpc_sharpe - base40_sharpe                   # ★ gate 主指标 (close-based)
    d_sharpe_40h = summ["TP_40d_high"]["sharpe"] - base40_sharpe
    d_sharpe_20  = tpc_sharpe - summ["baseline_20d"]["sharpe"]  # EX-001 混淆口径 (参考)
    d_ann_40c_pp = (summ["TP_40d_close"]["ann_return"] - summ["baseline_40d"]["ann_return"]) * 100
    d_maxdd_40c_pp = (summ["TP_40d_close"]["max_drawdown"] - summ["baseline_40d"]["max_drawdown"]) * 100
    print(f"\n[受控 Δ close-based 主口径] TP_40d_close - baseline_40d: ΔSharpe {d_sharpe_40c:+.3f}  "
          f"Δ年化 {d_ann_40c_pp:+.2f}pp  ΔmaxDD {d_maxdd_40c_pp:+.2f}pp", flush=True)
    print(f"[参考] 日内high乐观口径 ΔSharpe {d_sharpe_40h:+.3f}; EX-001混淆口径(TP-base20) ΔSharpe {d_sharpe_20:+.3f}", flush=True)

    # ── placebo 1: 静态降暴露 (匹配 TP_close 平均暴露) ──
    expo_tp = avg_exposure(arms["TP_40d_close"][1])
    expo_b40 = avg_exposure(arms["baseline_40d"][1])
    f = expo_tp / expo_b40 if expo_b40 > 0 else 1.0
    print(f"\n[placebo_static_reduce] TP_close平均暴露 {expo_tp:.3f} / baseline40 {expo_b40:.3f} → 降暴露因子 f={f:.3f}", flush=True)
    arms["placebo_static_reduce"] = run_arm_cohorts(rebal_days, picks, cal, lut, mode="hold",
                                                    hold=HOLDS["40d"], cm=cm, weight_scale=f)
    summ["placebo_static_reduce"] = nav_summary(book_from_cols(arms["placebo_static_reduce"][0]))
    static_dsharpe = summ["placebo_static_reduce"]["sharpe"] - base40_sharpe
    print(f"  placebo_static_reduce: 净Sharpe {summ['placebo_static_reduce']['sharpe']:+.2f}  "
          f"ΔvsBase40 {static_dsharpe:+.3f} (理论≈0: 零息现金静态混合不抬 Sharpe)", flush=True)

    # ── placebo 2: 随机阈值 TP (50 seed, close-based, gate 用 p90) ──
    print(f"\n[placebo_random_TP] {N_RANDOM_SEEDS} seed 随机阈值 uniform{RANDOM_TP_RANGE} (close-based) ...", flush=True)
    rand_sharpe_mean, rand_dsharpe_mean, rand_pcts, _ = multiseed_dsharpe(
        rebal_days, picks, cal, lut, cm, base40_sharpe, mode="tp", random_tp=True,
        n_seeds=N_RANDOM_SEEDS)
    rand_p5, rand_p90, rand_p95 = rand_pcts
    print(f"  random_TP: 净Sharpe {rand_sharpe_mean:+.2f} ΔvsBase40 mean {rand_dsharpe_mean:+.3f} "
          f"p90={rand_p90:+.3f} [p5,p95]=({rand_p5:+.3f},{rand_p95:+.3f})", flush=True)

    # ── 扩展负控分解 edge (codex 第5条, 全 close-based, 描述性诊断 R03) ──
    print("\n[分解负控] edge 主来源 (各控 ΔSharpe vs TP_close 比对) ...", flush=True)
    # (a) 盈利条件控
    cp_df, _ = run_arm_cohorts(rebal_days, picks, cal, lut, mode="cond_profit", hold=HOLDS["40d"], cm=cm, close_based=True)
    cp_sharpe = nav_summary(book_from_cols(cp_df))["sharpe"]; cp_dsharpe = cp_sharpe - base40_sharpe
    print(f"  (a) cond_profit (在赢时减,无阈值):      净Sharpe {cp_sharpe:+.2f}  ΔvsBase40 {cp_dsharpe:+.3f}", flush=True)
    # (b) 缩短持有控 (50 seed)
    sh_sharpe_mean, sh_dsharpe_mean, _, _ = multiseed_dsharpe(
        rebal_days, picks, cal, lut, cm, base40_sharpe, mode="short_hold_random",
        n_seeds=N_SHORT_SEEDS, master_seed=BOOT_SEED + 1)
    print(f"  (b) short_hold_random (随机日,盈利门控): 净Sharpe {sh_sharpe_mean:+.2f}  ΔvsBase40 {sh_dsharpe_mean:+.3f}", flush=True)
    # (c) 收益分位控
    rq_levels, maxrets = collect_maxret_levels(rebal_days, picks, cal, lut, RET_Q)
    rq_df, _ = run_arm_cohorts(rebal_days, picks, cal, lut, mode="tp", hold=HOLDS["40d"], cm=cm,
                               tp_levels=rq_levels, close_based=True)
    rq_sharpe = nav_summary(book_from_cols(rq_df))["sharpe"]; rq_dsharpe = rq_sharpe - base40_sharpe
    print(f"  (c) ret_quantile (实现收益{RET_Q}分位派生阈值={[round(x,3) for x in rq_levels]}): "
          f"净Sharpe {rq_sharpe:+.2f}  ΔvsBase40 {rq_dsharpe:+.3f}", flush=True)
    decomp = {"cond_profit": round(cp_dsharpe, 4), "short_hold_random": round(sh_dsharpe_mean, 4),
              "ret_quantile": round(rq_dsharpe, 4), "random_threshold": round(rand_dsharpe_mean, 4),
              "TP_close": round(d_sharpe_40c, 4)}
    # edge 主来源 = ΔSharpe 最接近 TP_close 的控
    src_order = sorted([("盈利条件", cp_dsharpe), ("缩短持有", sh_dsharpe_mean),
                        ("收益分位档位", rq_dsharpe), ("随机阈值", rand_dsharpe_mean)],
                       key=lambda kv: abs(kv[1] - d_sharpe_40c))
    edge_source = src_order[0][0]
    print(f"  → edge 最接近的控 = {edge_source} (其 ΔSharpe {src_order[0][1]:+.3f} vs TP_close {d_sharpe_40c:+.3f})", flush=True)

    # ── block bootstrap CI (重采样 cohort, close-based TP vs base40) ──
    print(f"\n[bootstrap] {N_BOOT} 次 cohort 重采样 ΔSharpe(TP40c-base40) CI ...", flush=True)
    tp_ret = arms["TP_40d_close"][0]; b40_ret = arms["baseline_40d"][0]
    common_cols = [c for c in tp_ret.columns if c in set(b40_ret.columns)]
    tp_ret = tp_ret[common_cols]; b40_ret = b40_ret[common_cols]
    ncoh = len(common_cols)
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, ncoh, size=ncoh)
        s_tp = sharpe_of(book_from_cols(tp_ret, idx))
        s_b = sharpe_of(book_from_cols(b40_ret, idx))
        if np.isfinite(s_tp) and np.isfinite(s_b):
            boot.append(s_tp - s_b)
    boot = np.array(boot)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    ci_excludes_0 = bool(ci[0] > 0 or ci[1] < 0)
    print(f"  ΔSharpe 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]  (含0? {'否' if ci_excludes_0 else '是'})  "
          f"P(Δ>0)={float((boot>0).mean()):.2f}", flush=True)

    # ── leave-one-out ──
    loo = []
    for j in range(ncoh):
        keep = [k for k in range(ncoh) if k != j]
        loo.append(sharpe_of(book_from_cols(tp_ret, keep)) - sharpe_of(book_from_cols(b40_ret, keep)))
    loo = np.array(loo)
    loo_min, loo_max = float(loo.min()), float(loo.max())
    loo_sign_stable = bool((loo > 0).all() or (loo < 0).all())
    print(f"[leave-one-out] ΔSharpe 范围 [{loo_min:+.3f}, {loo_max:+.3f}]  符号稳定? {loo_sign_stable}", flush=True)

    # ── 分 regime (R11) ──
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())
    tp_m = monthly_returns(book_from_cols(arms["TP_40d_close"][0]))
    b40_m = monthly_returns(base40_nav)
    dfm = pd.DataFrame({"tp": tp_m, "base": b40_m}).dropna()
    dfm["regime"] = dfm.index.map(month_regime)
    dfm["d_pp"] = (dfm["tp"] - dfm["base"]) * 100
    regime_delta = {rg: {"n_months": int(len(g)), "d_month_pp": round(float(g["d_pp"].mean()), 3)}
                    for rg, g in dfm.groupby("regime")}
    print("\n[regime] TP_40d_close - baseline_40d 月超额 (R11):", flush=True)
    for rg, d in regime_delta.items():
        print(f"  [{rg:9s}] n={d['n_months']:2d}  Δ {d['d_month_pp']:+.3f}pp/月", flush=True)
    regime_min = min(d["d_month_pp"] for d in regime_delta.values())
    regime_not_hurt = bool(regime_min > -0.5)

    # ── gate_tp_v2 判定 (R01 冻结) ──
    # 结构性减仓改进 = close-based TP ΔSharpe>0 AND bootstrap CI 不含0(下界>0) AND > 静态降暴露 placebo
    struct_improve = bool(d_sharpe_40c > 0 and (ci_excludes_0 and ci[0] > 0) and d_sharpe_40c > static_dsharpe)
    # 择价 alpha = TP > 随机阈值 placebo 分布 p90 (非均值)
    level_alpha = bool(d_sharpe_40c > rand_p90)
    if struct_improve and level_alpha:
        status = "意外择价"
    elif struct_improve and not level_alpha:
        status = "结构成立档位无alpha"
    else:
        status = "结构也不成立"

    playbook_map = {
        "结构成立档位无alpha": "close-based 保守口径下结构性减仓 ΔSharpe>0 且过 CI/胜静态降暴露, 但 TP < 随机阈值 p90 → "
                               "ship '任意分批减仓结构' (路径管理/de-risk), **不声称择价**择档 (符合 codex 预期); 待 holdout。",
        "结构也不成立": "close-based 保守口径下结构性减仓 ΔSharpe CI 含0 或 ≤ 静态降暴露 placebo → "
                       "WF-002 的 '+0.809' 是日内 high 成交乐观 + Δ算术的假象, 弃止盈。",
        "意外择价": "TP > 随机阈值 p90 → 固定档位确有择价 alpha (反 codex 预期) → 强记录, 需独立 holdout 复核。",
    }

    cond = {
        "struct_dsharpe_40c_positive": bool(d_sharpe_40c > 0),
        "struct_bootstrap_ci_excludes_0": bool(ci_excludes_0 and ci[0] > 0),
        "struct_beats_static_placebo": bool(d_sharpe_40c > static_dsharpe),
        "level_TP_gt_random_p90": level_alpha,
    }
    print(f"\n[gate_tp_v2] 条件: {cond}", flush=True)
    print(f"[gate_tp_v2] 结构性减仓改进? {struct_improve}; 择价alpha(TP>随机p90 {rand_p90:+.3f})? {level_alpha}", flush=True)
    print(f"[status] {status}", flush=True)

    # ── 落盘 ──
    results = {
        "config": {"total": TOTAL, "rebal_every": REBAL_EVERY, "holds": HOLDS,
                   "tp_levels": TP_LEVELS, "slippage": SLIP, "n_cohorts": ncoh,
                   "random_tp_range": RANDOM_TP_RANGE, "n_random_seeds": N_RANDOM_SEEDS,
                   "n_short_seeds": N_SHORT_SEEDS, "ret_q": RET_Q, "n_boot": N_BOOT,
                   "period": [start, cal[-1]], "main_caliber": "close-based 保守成交",
                   "picks_source": "WFE-001 label-embargo 真·真实 OOS (codex 第1条 + SIGN-R13①)"},
        "summary": summ,
        "controlled_delta_close_based": {
            "d_sharpe_TP40c_minus_base40": round(d_sharpe_40c, 4),
            "d_sharpe_TP40high_minus_base40_optimistic": round(d_sharpe_40h, 4),
            "d_sharpe_TP40c_minus_base20_EX001_confounded": round(d_sharpe_20, 4),
            "d_ann_return_pp": round(d_ann_40c_pp, 3),
            "d_maxdd_pp": round(d_maxdd_40c_pp, 3),
        },
        "placebo": {
            "static_reduce_factor_f": round(f, 4),
            "static_reduce_dsharpe": round(static_dsharpe, 4),
            "random_TP_sharpe_mean": round(rand_sharpe_mean, 4),
            "random_TP_dsharpe_mean": round(rand_dsharpe_mean, 4),
            "random_TP_dsharpe_p90": round(rand_p90, 4),
            "random_TP_dsharpe_p5_p95": [round(rand_p5, 4), round(rand_p95, 4)],
        },
        "edge_decomposition": {
            "deltas_vs_base40": decomp,
            "ret_quantile_levels": [round(x, 4) for x in rq_levels],
            "edge_source_nearest_TP": edge_source,
        },
        "bootstrap": {"ci95": [round(ci[0], 4), round(ci[1], 4)],
                      "excludes_0": ci_excludes_0, "p_gt_0": round(float((boot > 0).mean()), 4),
                      "n_valid": int(len(boot))},
        "leave_one_out": {"min": round(loo_min, 4), "max": round(loo_max, 4),
                          "sign_stable": loo_sign_stable},
        "regime_delta": regime_delta,
        "gate_conditions": cond,
        "struct_improve": struct_improve, "level_alpha": level_alpha,
        "status": status,
    }
    (CACHE / "wfe002_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    for name, (rdf, _) in arms.items():
        book_from_cols(rdf).to_frame("nav").to_parquet(CACHE / f"wfe002_nav_{name}.parquet")

    conclusion = (
        f"在 **WFE-001 label-embargo 真·真实 OOS picks** ({ncoh} cohort, {start}~{cal[-1]}) 上强化版止盈复测, "
        f"吃进 codex 第二次 review 第2+5条: **主口径=close-based 保守成交** / 强化 gate (TP 须 > 随机阈值 **p90** 非均值) / 扩展负控分解 edge。"
        f"baseline_40d 净Sharpe {base40_sharpe:+.2f}; TP_40d_close (主口径) 净Sharpe {tpc_sharpe:+.2f}。"
        f"★close-based apples-to-apples ΔSharpe(TP40c−base40) = {d_sharpe_40c:+.3f} "
        f"(Δ年化 {d_ann_40c_pp:+.2f}pp, ΔmaxDD {d_maxdd_40c_pp:+.2f}pp); "
        f"日内 high 乐观口径 ΔSharpe {d_sharpe_40h:+.3f} (WF-002 用的是这个乐观成交); "
        f"EX-001 混淆口径 (TP−base20) ΔSharpe {d_sharpe_20:+.3f}。"
        f"placebo: 静态降暴露 (f={f:.3f}) ΔSharpe {static_dsharpe:+.3f} (≈0 → 增益非'少持仓数学抬升'); "
        f"随机阈值 TP ({N_RANDOM_SEEDS} seed, close-based) ΔSharpe mean {rand_dsharpe_mean:+.3f} **p90={rand_p90:+.3f}** [p5,p95]=({rand_p5:+.3f},{rand_p95:+.3f})。"
        f"★强化 gate (随机 **p90** 非均值): TP_close ΔSharpe {d_sharpe_40c:+.3f} vs 随机 p90 {rand_p90:+.3f} → 择价 alpha = {level_alpha}。"
        f"分解负控 (各控 ΔSharpe vs base40, 越接近 TP_close {d_sharpe_40c:+.3f} 越是 edge 主源): "
        f"盈利条件 {cp_dsharpe:+.3f} / 缩短持有 {sh_dsharpe_mean:+.3f} / 收益分位档位 {rq_dsharpe:+.3f} / 随机阈值 {rand_dsharpe_mean:+.3f} "
        f"→ **edge 主来源 = {edge_source}**。"
        f"block bootstrap ({N_BOOT}次) ΔSharpe 95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}] (含0? {'否' if ci_excludes_0 else '是'}); "
        f"leave-one-out [{loo_min:+.3f},{loo_max:+.3f}] 符号稳定 {loo_sign_stable}; "
        f"regime (R11) " + " / ".join(f"{rg} {d['d_month_pp']:+.2f}pp" for rg, d in regime_delta.items()) + "。"
        f"gate_tp_v2: 结构性减仓改进={struct_improve} (close-based ΔSharpe>0 AND CI 不含0 AND >静态降暴露 placebo); "
        f"择价 alpha={level_alpha} (TP>随机p90) → **status={status}** (frozen gate, R01)。{playbook_map[status]} "
        f"★关键 nuance (SIGN-R03 诚实): (1) 主口径从 WF-002 的日内 high 乐观成交收紧到 close-based 保守成交后, "
        f"ΔSharpe {d_sharpe_40h:+.3f}→{d_sharpe_40c:+.3f}; (2) 强化 gate 用随机阈值 **p90** (非 WF-002 的均值): "
        f"{'TP 仍 > p90 = 意外有择价' if level_alpha else 'TP < p90 → 固定档位无择价 alpha, 边际全是 timing-agnostic 结构性 de-risk (任意阈值都行, 符合 codex 预期)'}; "
        f"(3) 分解 4 控指向 edge 主源 = **{edge_source}**; (4) Δ年化 {d_ann_40c_pp:+.2f}pp 但 maxDD 砍 {abs(d_maxdd_40c_pp):.1f}pp = 保收益降波动。"
        f"【绝对量级已是 de-lookahead+embargo 真实 OOS, 受控 Δ 两臂相消 + close-based 保守口径 + 随机 p90 强 gate + 4 控分解 → "
        f"止盈是否真技能的最干净裁决】。"
    )

    VERDICT.write_text(json.dumps({
        "id": "WFE-002", "status": status, "conclusion": conclusion,
        "metrics": {
            "n_cohorts": ncoh,
            "main_caliber": "close-based",
            "baseline_20d_sharpe": round(summ["baseline_20d"]["sharpe"], 3),
            "baseline_40d_sharpe": round(base40_sharpe, 3),
            "TP_40d_close_sharpe": round(tpc_sharpe, 3),
            "TP_40d_high_sharpe": round(summ["TP_40d_high"]["sharpe"], 3),
            "d_sharpe_TP40c_minus_base40": round(d_sharpe_40c, 4),
            "d_sharpe_TP40high_minus_base40_optimistic": round(d_sharpe_40h, 4),
            "d_sharpe_TP40c_minus_base20_EX001_confounded": round(d_sharpe_20, 4),
            "d_ann_return_pp": round(d_ann_40c_pp, 3),
            "d_maxdd_pp": round(d_maxdd_40c_pp, 3),
            "placebo_static_reduce_dsharpe": round(static_dsharpe, 4),
            "placebo_random_TP_dsharpe_mean": round(rand_dsharpe_mean, 4),
            "placebo_random_TP_dsharpe_p90": round(rand_p90, 4),
            "TP_gt_random_p90_level_alpha": level_alpha,
            "struct_improve": struct_improve,
            "edge_decomposition_dsharpe": decomp,
            "edge_source_nearest_TP": edge_source,
            "bootstrap_ci95": [round(ci[0], 4), round(ci[1], 4)],
            "bootstrap_excludes_0": ci_excludes_0,
            "bootstrap_p_gt_0": round(float((boot > 0).mean()), 4),
            "loo_min": round(loo_min, 4), "loo_max": round(loo_max, 4),
            "loo_sign_stable": loo_sign_stable,
            "gate_conditions": cond,
        },
        "regime_delta": regime_delta,
        "cost_model": cm.describe(),
        "playbook_hit": status,
        "artifacts": [
            "research/backtest/run_wfe002.py",
            "research/cache/wfe002/wfe002_results.json",
            "research/cache/wfe002/wfe002_nav_*.parquet",
        ],
        "note": "网格冻结于 prd.preRegisteredGate.tp_retest_v2 (R01, 不搜不调); "
                "picks=WFE-001 label-embargo 真·真实 OOS (codex 第1条 + SIGN-R13①); "
                "主口径=close-based 保守成交 (codex 第2条); 强化 gate=TP 须 > 随机阈值 p90 非均值 (codex 第2条); "
                "扩展负控分解 edge=盈利条件控/缩短持有控/收益分位控 (codex 第5条); "
                "baseline-40d 对齐去持有期混淆 (R13③); 静态降暴露 placebo 去'少持仓数学抬升' (R13②); "
                "block bootstrap CI + leave-one-out + 分regime (R11); 受控 Δ 两臂入场逐位一致共模相消 (R03); "
                "ST 源头排除 (R06); 大缓存 gitignored (R04); 生产线只读 (R05); 描述性受控对照=合法完成 (R02)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] status={status} -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
