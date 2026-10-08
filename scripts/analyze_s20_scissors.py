#!/usr/bin/env python
"""Scissors on the frozen S20 offensive list: how big a pool, how hard a cut, which factor.

The frozen list (config/s20_pure_v1.json) is already a scissors rule: take the
stage1 top 100, cut the 40% with the highest NATR(14), keep the stage1 top 20
of what is left. This script maps the neighbourhood of that rule and then
tries a second cut on top of it.

Descriptive only. Both windows (dev 2025-03-03..2026-01-26, a 50% stock
sample; confirm 2026-01-27..2026-08-05, the full market) have been used many
times for this family of questions, so nothing here is a confirmation; the
reserved window from 2026-08-06 is not read (metaRSI guard). Nothing is
written to the contract or the published list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "C:/aicoding/mylib/skills/meta-rsi/scripts")
from analyze_s20_pure_r11_safe_up import load  # noqa: E402
from experiment_cross_filter_v2 import (  # noqa: E402
    DEV_CACHE, EXT_CACHE, FOLDS, RESERVED_FROM, STALE_IN_EXT, TRAIN_ATR_PCT, fit_head, load_paths, nw_se, stock_feats,
)
from experiment_r5_s5_cross_filter import add_events, auc  # noqa: E402
from metarsi import guard_read  # noqa: E402

OUT = ROOT / "output/experiments/s20_scissors_20261004"
V2 = ROOT / "output/experiments/cross_filter_v2_20261004"
FROZEN = (100, 0.4, 20)


def dn7_panel() -> pd.DataFrame:
    """y_dn7 for every stock-day: the low touches -7% of the next open within 5 sessions of entry."""
    daily = ROOT / "output/tushare_cache/daily"
    last = sorted(d.stem for d in daily.glob("*.parquet") if d.stem < RESERVED_FROM)[-1]
    files = [d for d in sorted(daily.glob("*.parquet")) if "20240101" <= d.stem]
    files = files[: [d.stem for d in files].index(last) + 6]      # five sessions of outcome after the last signal day
    px = pd.concat([pd.read_parquet(d, columns=["ts_code", "trade_date", "open", "low"]) for d in files])
    px["trade_date"] = px["trade_date"].astype(str)
    px["ts_code"] = px["ts_code"].astype(str)
    op = px.pivot(index="trade_date", columns="ts_code", values="open").sort_index()
    lo = px.pivot(index="trade_date", columns="ts_code", values="low").sort_index()
    fwd_min = lo[::-1].rolling(5, min_periods=4).min()[::-1].shift(-1)      # lowest low over sessions d+1 .. d+5
    entry = op.shift(-1)
    dd = (fwd_min / entry.where(entry > 0) - 1.0) * 100.0          # worst low against the entry open, percent
    out = dd.stack().rename("dd5").reset_index()
    out["y_dn7"] = (out.dd5 <= -7.0).astype(float)
    return out[out.trade_date < RESERVED_FROM]


def dn7_scores(keys: pd.DataFrame, y7: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward head for the -7% label, same folds, features and training slice as the -5% head."""
    cache = OUT / "p_dn7.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    feats = [f for f in stock_feats() if f not in STALE_IN_EXT]
    dev = pd.read_parquet(DEV_CACHE, columns=["ts_code", "trade_date", *feats])
    dev["trade_date"] = dev["trade_date"].astype(str)
    dev[feats] = dev[feats].astype(np.float32)
    ext = pd.read_parquet(EXT_CACHE, columns=["ts_code", "trade_date", *feats])
    ext = ext[ext.trade_date < RESERVED_FROM]
    cand = keys.merge(pd.concat([dev, ext], ignore_index=True), on=["ts_code", "trade_date"], how="inner")
    train = dev.merge(y7, on=["ts_code", "trade_date"], how="inner").dropna(subset=["y_dn7"])
    del dev, ext
    train["y_dn7"] = train["y_dn7"].astype(np.int8)
    train = train[train.groupby("trade_date").atr_pct.rank(pct=True) >= TRAIN_ATR_PCT]
    cand["p_dn7"] = np.nan
    for name, fit_end, tune_lo, tune_hi, test_lo, test_hi in FOLDS:
        fit = train[train.trade_date <= fit_end]
        tune = train[(train.trade_date >= tune_lo) & (train.trade_date <= tune_hi)]
        sel = (cand.trade_date >= test_lo) & (cand.trade_date <= test_hi)
        model = fit_head(fit, tune, feats, "y_dn7")
        cand.loc[sel, "p_dn7"] = model.predict_proba(cand.loc[sel, feats])[:, 1]
        print(f"{name} y_dn7: fit {len(fit):,} base {fit.y_dn7.mean():.3f} trees {model.best_iteration_} "
              f"tune AUC {auc(tune.y_dn7, model.predict_proba(tune[feats])[:, 1]):.3f} atr-only {auc(tune.y_dn7, tune.atr_pct):.3f}",
              flush=True)
    out = cand[["ts_code", "trade_date", "p_dn7"]].dropna()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def build() -> pd.DataFrame:
    p = load()[["ts_code", "trade_date", "stage1_probability", "natr14", "period"]]
    guard_read(ROOT, p.trade_date.min(), p.trade_date.max())
    f = add_events(p.merge(load_paths(), on=["ts_code", "trade_date"], how="inner"))
    f["s_rank"] = f.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    f = f[f.s_rank <= 200].reset_index(drop=True)
    y7 = dn7_panel()
    f = f.merge(y7, on=["ts_code", "trade_date"], how="left")
    f = f.merge(dn7_scores(f[["ts_code", "trade_date"]], y7), on=["ts_code", "trade_date"], how="left")
    days = sorted(f.trade_date.unique())
    all_days = sorted(path.stem for path in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    start = all_days[max(all_days.index(days[0]) - 25, 0)]
    px = pd.concat([
        pd.read_parquet(ROOT / "output/tushare_cache/daily" / f"{d}.parquet", columns=["ts_code", "trade_date", "pct_chg"])
        for d in all_days if start <= d <= days[-1]
    ])
    px["trade_date"] = px["trade_date"].astype(str)
    px["ts_code"] = px["ts_code"].astype(str)
    px = px.sort_values(["ts_code", "trade_date"])
    px["lr"] = np.log1p(px.pct_chg.fillna(0.0) / 100.0)
    roll = px.groupby("ts_code").lr
    px["r5"] = (np.exp(roll.transform(lambda x: x.rolling(5, min_periods=5).sum())) - 1.0) * 100.0     # last 5 sessions, signal day included
    px["r20"] = (np.exp(roll.transform(lambda x: x.rolling(20, min_periods=15).sum())) - 1.0) * 100.0
    f = f.merge(px[["ts_code", "trade_date", "pct_chg", "r5", "r20"]], on=["ts_code", "trade_date"], how="left")
    heads = pd.read_parquet(V2 / "scored_candidates.parquet", columns=["ts_code", "trade_date", "p_dn5"])
    f = f.merge(heads, on=["ts_code", "trade_date"], how="left")

    # Factors off the price-volatility axis: money flow, turnover, size. All known at the signal-day close.
    cache = ROOT / "output/tushare_cache"
    cal = sorted(path.stem for path in (cache / "daily").glob("*.parquet"))
    first = cal[max(cal.index(days[0]) - 6, 0)]
    cols = ["buy_sm_amount", "buy_md_amount", "buy_lg_amount", "buy_elg_amount", "sell_lg_amount", "sell_elg_amount"]
    mf = pd.concat([pd.read_parquet(cache / "moneyflow" / f"{d}.parquet", columns=["ts_code", "trade_date", *cols])
                    for d in cal if first <= d <= days[-1] and (cache / "moneyflow" / f"{d}.parquet").exists()])
    mf["trade_date"] = mf["trade_date"].astype(str)
    mf["ts_code"] = mf["ts_code"].astype(str)
    mf = mf.sort_values(["ts_code", "trade_date"])
    mf["net"] = mf.buy_lg_amount + mf.buy_elg_amount - mf.sell_lg_amount - mf.sell_elg_amount
    mf["gross"] = mf.buy_sm_amount + mf.buy_md_amount + mf.buy_lg_amount + mf.buy_elg_amount
    roll = mf.groupby("ts_code")[["net", "gross"]].rolling(5, min_periods=3).sum().reset_index(level=0, drop=True)
    mf["main5"] = roll["net"] / roll["gross"].where(roll["gross"] > 0)
    f = f.merge(mf[["ts_code", "trade_date", "main5"]], on=["ts_code", "trade_date"], how="left")
    db = pd.concat([pd.read_parquet(cache / "daily_basic" / f"{d}.parquet", columns=["ts_code", "trade_date", "turnover_rate_f", "circ_mv"])
                    for d in days if (cache / "daily_basic" / f"{d}.parquet").exists()])
    db["trade_date"] = db["trade_date"].astype(str)
    db["ts_code"] = db["ts_code"].astype(str)
    return f.merge(db, on=["ts_code", "trade_date"], how="left")


def funnel(f: pd.DataFrame, pool: int, cap: float) -> pd.DataFrame:
    """stage1 top `pool` -> drop the `cap` share with the highest NATR -> rank what is left by stage1."""
    q = f[f.s_rank <= pool].copy()
    pct = q.groupby("trade_date").natr14.rank(pct=True)
    q = q[pct.isna() | (pct <= 1 - cap)].copy()
    q["list_rank"] = q.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    return q


def stat(q: pd.DataFrame, base_day: pd.Series | None = None) -> str:
    day = q.groupby("trade_date").ret_v1.mean()
    text = (f"ret {day.mean():+5.2f} pure_up {100 * q.pure_up.mean():4.1f} pure_down {100 * q.pure_down.mean():4.1f} "
            f"dn5 {100 * q.y_dn5.mean():4.1f} dn7 {100 * q.y_dn7.mean():4.1f} len {len(q) / q.trade_date.nunique():4.1f}")
    if base_day is not None:
        d = (day - base_day.reindex(day.index)).dropna()
        se = nw_se(d.to_numpy())
        month = d.groupby(d.index.str[:6]).mean()
        text += f" | vs frozen {d.mean():+5.2f} (t {d.mean() / se:+5.2f}, months up {int((month > 0).sum())}/{len(month)})"
    return text


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = build()
    market = dn7_panel()
    lines = [
        "S20 scissors map. ret = +15%/-10% exit return per listed name (list basis, 0.3% cost), day mean.",
        "dn5 / dn7 = share that touch -5% / -7% within 5 sessions of entry; pure_down = touch -10% before +15% within 20.",
        "Descriptive: both windows are used data. The frozen rule is pool 100, cut 40% by NATR, keep 20.",
        "",
    ]
    n_rules = 0
    for period in ("dev", "confirm"):
        part = f[f.period == period]
        frozen = funnel(part, FROZEN[0], FROZEN[1])
        base20 = frozen[frozen.list_rank <= 20]
        base_day = base20.groupby("trade_date").ret_v1.mean()
        lines.append(f"===== {period}: {part.trade_date.nunique()} days =====")
        lines.append(f"raw stage1 top 20            : {stat(part[part.s_rank <= 20], base_day)}")
        lines.append(f"frozen list (100, 40%, 20)   : {stat(base20)}")
        lines.append(f"frozen list, places 1-10     : {stat(base20[base20.list_rank <= 10], base_day)}")
        lines.append(f"frozen list, places 11-20    : {stat(base20[base20.list_rank > 10], base_day)}")
        lines.append("")
        lines.append("-- A. pool size x cut share, same NATR scissors --")
        for keep in (20, 10):
            for pool in (30, 50, 100, 200):
                for cap in (0.0, 0.2, 0.4, 0.6):
                    if pool * (1 - cap) < keep:
                        continue
                    q = funnel(part, pool, cap)
                    n_rules += 1
                    lines.append(f"keep {keep:2d} | pool {pool:3d} cut {int(cap * 100):2d}% : {stat(q[q.list_rank <= keep], base_day)}")
            lines.append("")
        top50 = frozen[frozen.list_rank <= 50]
        lines.append("-- can the -7% drop be told apart inside the frozen list's first 50 places? AUC for y_dn7 --")
        lines.append(f"model P(-7%) {auc(top50.y_dn7, top50.p_dn7):.3f} | model P(-5%) {auc(top50.y_dn7, top50.p_dn5):.3f} | "
                     f"NATR {auc(top50.y_dn7, top50.natr14):.3f} | float turnover {auc(top50.y_dn7, top50.turnover_rate_f):.3f} | "
                     f"stage1 score {auc(top50.y_dn7, top50.stage1_probability):.3f} | base rate {top50.y_dn7.mean():.3f}")
        lines.append("")
        lines.append("-- B. a second cut on the frozen list: take its first M places, keep K by another factor --")
        keys = {
            "stage1 order (no second cut)": lambda g: g.list_rank,
            "lowest NATR": lambda g: g.natr14,
            "lowest model P(-5% in 5d)": lambda g: g.p_dn5,
            "lowest model P(-7% in 5d)": lambda g: g.p_dn7,
            "drop the day's biggest fallers": lambda g: -g.pct_chg,
            "drop the day's biggest risers": lambda g: g.pct_chg,
            "drop biggest 5d main-money outflow": lambda g: -g.main5,
            "drop highest float turnover": lambda g: g.turnover_rate_f,
            "drop smallest float cap": lambda g: -g.circ_mv,
        }
        for keep in (20, 10):
            for m in (20, 30, 50):
                if m <= keep:
                    continue
                cand = frozen[frozen.list_rank <= m]
                for name, key in keys.items():
                    picked = pd.concat([g.assign(_o=key(g)).sort_values(["_o", "list_rank"]).head(keep)
                                        for _, g in cand.groupby("trade_date")])
                    n_rules += name != "stage1 order (no second cut)"
                    lines.append(f"keep {keep:2d} of first {m:2d} | {name:34s}: {stat(picked, base_day)}")
                lines.append("")
        lines.append("-- D. how deep the first five sessions go: share touching each line, and the average worst drop --")
        first50 = frozen[frozen.list_rank <= 50]

        def keep_by(col: str, n: int, sign: float = 1.0) -> pd.DataFrame:
            return pd.concat([g.assign(_o=sign * g[col]).sort_values(["_o", "list_rank"]).head(n)
                              for _, g in first50.groupby("trade_date")])

        sweep = {
            "raw stage1 top 20": part[part.s_rank <= 20],
            "frozen list (20)": base20,
            "frozen first 10 places": base20[base20.list_rank <= 10],
            "first 50 -> 20 lowest NATR": keep_by("natr14", 20),
            "first 50 -> 20 lowest turnover": keep_by("turnover_rate_f", 20),
            "first 50 -> 20 lowest P(-7%)": keep_by("p_dn7", 20),
            "first 50 -> 10 lowest NATR": keep_by("natr14", 10),
            "whole market that day": None,
        }
        for name, q in sweep.items():
            if q is None:
                q = market[market.trade_date.isin(set(part.trade_date))]
            shares = " ".join(f"-{k}%: {100 * (q.dd5 <= -k).mean():4.1f}" for k in (5, 6, 7, 8, 10))
            ret = f"ret {q.groupby('trade_date').ret_v1.mean().mean():+5.2f}" if "ret_v1" in q else "ret   n/a"
            lines.append(f"{name:32s}: {shares} | mean worst {q.dd5.mean():+5.2f} median {q.dd5.median():+5.2f} | {ret}")
        lines.append("")
        lines.append("-- E. what the NATR cut throws away: stage1 top 100 by NATR band, and by how the name had been moving --")
        pool = part[part.s_rank <= 100].copy()
        pool["npct"] = pool.groupby("trade_date").natr14.rank(pct=True)
        pool["band"] = np.select([pool.npct <= 0.4, pool.npct <= 0.6], ["kept by both (lowest 40%)", "kept at 40%, cut at 60%"],
                                 "cut by both (highest 40%)")
        prior15 = (1 + pool.r20 / 100) / (1 + pool.r5 / 100) * 100 - 100
        pool["move"] = np.select(
            [(pool.r5 >= 10) & (pool.r5 > prior15), pool.r5 >= 10, pool.r5 <= -5],
            ["rising faster (5d >= +10%, above prior 15d)", "rising fast, not faster", "falling (5d <= -5%)"], "in between")

        def out(q: pd.DataFrame) -> str:
            return (f"n/day {len(q) / part.trade_date.nunique():5.1f} | +10% in 5d {100 * q.y_up.mean():4.1f} | +15% first {100 * q.pure_up.mean():4.1f} | "
                    f"hit +15% in 20d {100 * q.hit15.mean():4.1f} | -7% in 5d {100 * q.y_dn7.mean():4.1f} | -10% first {100 * q.pure_down.mean():4.1f} | "
                    f"ret {q.ret_v1.mean():+5.2f}")

        for band in ("kept by both (lowest 40%)", "kept at 40%, cut at 60%", "cut by both (highest 40%)"):
            lines.append(f"{band:28s}: {out(pool[pool.band == band])}")
        lines.append("")
        for move in ("rising faster (5d >= +10%, above prior 15d)", "rising fast, not faster", "in between", "falling (5d <= -5%)"):
            m = pool[pool.move == move]
            lines.append(f"[{move}] share of pool {100 * len(m) / len(pool):4.1f}% ; of these, cut at 40%: "
                         f"{100 * (m.band == 'cut by both (highest 40%)').mean():4.1f}%, cut at 60%: {100 * (m.band != 'kept by both (lowest 40%)').mean():4.1f}%")
            for band in ("kept by both (lowest 40%)", "kept at 40%, cut at 60%", "cut by both (highest 40%)"):
                q = m[m.band == band]
                if len(q) >= 30:
                    lines.append(f"    {band:28s}: {out(q)}")
        lines.append("")
        list40 = funnel(part, 100, 0.4)
        list40 = list40[list40.list_rank <= 20]
        list60 = funnel(part, 100, 0.6)
        list60 = list60[list60.list_rank <= 20]
        k40 = set(zip(list40.ts_code, list40.trade_date))
        k60 = set(zip(list60.ts_code, list60.trade_date))
        lost = pool[[key in k40 and key not in k60 for key in zip(pool.ts_code, pool.trade_date)]]
        gained = pool[[key in k60 and key not in k40 for key in zip(pool.ts_code, pool.trade_date)]]
        lines.append(f"names the 60% rule drops from the frozen list : {out(lost)}")
        lines.append(f"names it puts in their place               : {out(gained)}")
        lines.append(f"  of the dropped names, rising faster: {100 * (lost.move.str.startswith('rising faster')).mean():4.1f}% ; "
                     f"of the names put in: {100 * (gained.move.str.startswith('rising faster')).mean():4.1f}%")
        lines.append("")
        if period == "confirm":   # the dev scores are a row sample, so yesterday's list is half missing there
            lines.append("-- C. frozen list: names also on yesterday's frozen list against names new today --")
            cal = sorted(path.stem for path in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
            prev = {b: a for a, b in zip(cal[:-1], cal[1:])}
            members = base20.groupby("trade_date").ts_code.agg(set)
            flag = [code in members.get(prev.get(day), set()) for code, day in zip(base20.ts_code, base20.trade_date)]
            had_prev = base20.trade_date.map(lambda d: prev.get(d) in members.index)
            stay, new = base20[np.array(flag) & had_prev], base20[~np.array(flag) & had_prev]
            lines.append(f"on yesterday's list too : n={len(stay):4d} {stat(stay)}")
            lines.append(f"new today               : n={len(new):4d} {stat(new)}")
            d = (stay.groupby("trade_date").ret_v1.mean() - new.groupby("trade_date").ret_v1.mean()).dropna()
            lines.append(f"same-day difference stayers minus new: {d.mean():+.2f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)})")
            lines.append("")
    lines.append(f"rules looked at in this run, per window: {n_rules // 2}")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
