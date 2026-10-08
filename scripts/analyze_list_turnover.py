#!/usr/bin/env python
"""What to do when tomorrow's list differs from today's.

Descriptive study on the development (2025-03-03 .. 2026-01-26, 50% stock
sample) and confirmation (2026-01-27 .. 2026-08-05, full market) windows.
Signal days from 2026-08-06 on are reserved and never read. Nothing here
changes a published list or a frozen contract.

Part 1 (addendum to experiment_cross_filter_v2.py): choose 20 names out of
the top 50 instead of the top 40, and how the order inside an unchanged top
20 lines up with outcomes.

Part 2: on each day d with a list on d-1, split names into
  stayers   in the top 20 on d-1 and on d
  entrants  in the top 20 on d, not on d-1
  dropouts  in the top 20 on d-1, not on d (by where they landed)
and read every group's outcome from d forward (entry at the next open), which
is the choice a holder or a buyer faces on d. Then two simple books are
compared at equal size: buy the day's top 20, versus keep yesterday's names
while they stay inside the top 50 and fill the gaps from today's ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import (  # noqa: E402
    CONF_PRED, DEV_PRED, EXP, RESERVED_FROM, load_paths, nw_se, order_key,
)
from experiment_r5_s5_cross_filter import add_events  # noqa: E402

V2 = ROOT / "output/experiments/cross_filter_v2_20261004"
OUT = ROOT / "output/experiments/list_turnover_20261004"
K, HOLD_ZONE = 20, 50


def load_frame() -> pd.DataFrame:
    dev = pd.read_parquet(DEV_PRED, columns=["ts_code", "trade_date", "s20_20r_platt", "r20_nested_anchor"])
    dev = dev.rename(columns={"s20_20r_platt": "s_score", "r20_nested_anchor": "r_score"})
    conf = pd.read_parquet(CONF_PRED, columns=["ts_code", "trade_date", "stage1_probability", "r20_p20_reference"])
    conf = conf.rename(columns={"stage1_probability": "s_score", "r20_p20_reference": "r_score"})
    rank = pd.concat([dev.assign(period="dev"), conf.assign(period="confirm")], ignore_index=True)
    rank["trade_date"] = rank["trade_date"].astype(str)
    rank["ts_code"] = rank["ts_code"].astype(str)
    rank = rank[rank.trade_date < RESERVED_FROM]
    for tag in ("s", "r"):
        rank[f"{tag}_rank"] = rank.groupby("trade_date")[f"{tag}_score"].rank(ascending=False, method="first")
    frame = add_events(rank.merge(load_paths(), on=["ts_code", "trade_date"], how="inner"))
    band = pd.read_parquet(EXP / "band_panel.parquet", columns=["ts_code", "trade_date", "cls_a5_d10", "ret_a5_b15_d10"])
    band["trade_date"] = band["trade_date"].astype(str)
    band["ts_code"] = band["ts_code"].astype(str)
    frame = frame.merge(band, on=["ts_code", "trade_date"], how="inner")
    frame["band_ok"] = frame.cls_a5_d10.isin([1, 3]).astype(np.int8)
    return frame


def day_pairs(days: list[str]) -> list[tuple[str, str]]:
    cal = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    nxt = {a: b for a, b in zip(cal[:-1], cal[1:])}
    have = set(days)
    return [(d, nxt[d]) for d in days if nxt.get(d) in have]


def line(q: pd.DataFrame) -> str:
    if q.empty:
        return "n=0"
    return (f"n={len(q):5d} ret {q.ret_v1.mean():+.2f} pure_up {100 * q.pure_up.mean():4.1f} "
            f"pure_down {100 * q.pure_down.mean():4.1f} hit15 {100 * q.hit15.mean():4.1f} band_ok {100 * q.band_ok.mean():4.1f}")


def diff_line(a: pd.DataFrame, b: pd.DataFrame, label: str) -> str:
    x = a.groupby("trade_date").ret_v1.mean()
    y = b.groupby("trade_date").ret_v1.mean()
    d = (x - y).dropna()
    se = nw_se(d.to_numpy())
    month = d.groupby(d.index.str[:6]).mean()
    return (f"    {label}: {d.mean():+.2f} per trade (se {se:.2f}, t {d.mean() / se:+.2f}, days {len(d)}, "
            f"months up {int((month > 0).sum())}/{len(month)})")


def part1(lines: list[str]) -> pd.DataFrame:
    cand = pd.read_parquet(V2 / "scored_candidates.parquet")
    frame = add_events(cand.merge(load_paths(), on=["ts_code", "trade_date"], how="inner"))
    lines.append("== Part 1a: 20 names chosen out of the top 50 (same exposure as the plain top 20) ==")
    for period in ("dev", "confirm"):
        part = frame[frame.period == period]
        for pool, rank_col in (("S", "s_rank"), ("R", "r_rank")):
            base = part[part[rank_col] <= K]
            base_day = base.groupby("trade_date").ret_v1.mean()
            lines.append(f"{period} {pool} plain top 20: ret {base_day.mean():+.2f} pure_up {100 * base.pure_up.mean():.1f} "
                         f"pure_down {100 * base.pure_down.mean():.1f}")
            for how in ("atr", "rank+atr", "dn", "up-dn"):
                picked = pd.concat([
                    g.assign(_o=order_key(g, rank_col, how)).sort_values(["_o", rank_col]).head(K)
                    for _, g in part[part[rank_col] <= HOLD_ZONE].groupby("trade_date")
                ])
                d = (picked.groupby("trade_date").ret_v1.mean() - base_day).dropna()
                se = nw_se(d.to_numpy())
                lines.append(f"  top50 -> 20 by {how:8s}: ret {picked.groupby('trade_date').ret_v1.mean().mean():+.2f} "
                             f"diff {d.mean():+.2f} (t {d.mean() / se:+.2f}) pure_up {100 * picked.pure_up.mean():.1f} "
                             f"pure_down {100 * picked.pure_down.mean():.1f} share from 21-50 "
                             f"{100 * (picked[rank_col] > K).mean():.0f}%")
    lines.append("")
    lines.append("== Part 1b: order inside an unchanged top 20 (first 10 against last 10) ==")
    for period in ("dev", "confirm"):
        part = frame[frame.period == period]
        for pool, rank_col in (("S", "s_rank"), ("R", "r_rank")):
            top = part[part[rank_col] <= K]
            for how in ("rank", "atr", "rank+atr"):
                o = pd.concat([g.assign(_p=order_key(g, rank_col, how).rank(method="first"))
                               for _, g in top.groupby("trade_date")])
                first, last = o[o._p <= 10], o[o._p > 10]
                d = (first.groupby("trade_date").ret_v1.mean() - last.groupby("trade_date").ret_v1.mean()).dropna()
                se = nw_se(d.to_numpy())
                lines.append(f"{period} {pool} order by {how:8s}: first 10 ret {first.ret_v1.mean():+.2f} pure_down "
                             f"{100 * first.pure_down.mean():.1f} | last 10 ret {last.ret_v1.mean():+.2f} pure_down "
                             f"{100 * last.pure_down.mean():.1f} | first-last {d.mean():+.2f} (t {d.mean() / se:+.2f})")
    lines.append("")
    return frame


def part2(frame: pd.DataFrame, lines: list[str]) -> None:
    lines.append("== Part 2: tomorrow's list differs from today's ==")
    for period in ("dev", "confirm"):
        part = frame[frame.period == period]
        pairs = day_pairs(sorted(part.trade_date.unique()))
        by_day = {d: g for d, g in part.groupby("trade_date")}
        for pool, rank_col in (("S", "s_rank"), ("R", "r_rank")):
            groups = {k: [] for k in ("stay", "enter", "drop_21_50", "drop_51_plus", "drop_unranked")}
            overlap, age_rows = [], []
            streak: dict[str, int] = {}
            last_day = None
            for prev, day in pairs:
                g0, g1 = by_day[prev], by_day[day]
                top0 = set(g0.loc[g0[rank_col] <= K, "ts_code"])
                top1 = g1[g1[rank_col] <= K]
                overlap.append(len(top0 & set(top1.ts_code)) / K)
                groups["stay"].append(top1[top1.ts_code.isin(top0)])
                groups["enter"].append(top1[~top1.ts_code.isin(top0)])
                out = g1[g1.ts_code.isin(top0) & (g1[rank_col] > K)]
                groups["drop_21_50"].append(out[out[rank_col] <= HOLD_ZONE])
                groups["drop_51_plus"].append(out[out[rank_col] > HOLD_ZONE])
                gone = top0 - set(g1.ts_code)
                groups["drop_unranked"].append(pd.DataFrame({"ts_code": sorted(gone), "trade_date": day}))
                if last_day != prev:      # a gap in the calendar restarts the count
                    streak = {c: 1 for c in top0}
                streak = {c: streak.get(c, 0) + 1 for c in top1.ts_code}
                age_rows.append(top1.assign(age=top1.ts_code.map(streak)))
                last_day = day
            g = {k: pd.concat(v, ignore_index=True) for k, v in groups.items()}
            lines.append(f"{period} {pool}: day pairs {len(pairs)}, mean share of yesterday's 20 still in today's 20 "
                         f"{100 * np.mean(overlap):.0f}% (p10 {100 * np.quantile(overlap, .1):.0f}%, "
                         f"p90 {100 * np.quantile(overlap, .9):.0f}%)")
            lines.append(f"  stayers          {line(g['stay'])}")
            lines.append(f"  entrants         {line(g['enter'])}")
            lines.append(f"  dropped to 21-50 {line(g['drop_21_50'])}")
            lines.append(f"  dropped past 50  {line(g['drop_51_plus'])}")
            lines.append(f"  dropped, no score on d: n={len(g['drop_unranked'])}")
            lines.append(diff_line(g["enter"], g["stay"], "entrants minus stayers"))
            lines.append(diff_line(g["drop_21_50"], g["stay"], "dropped to 21-50 minus stayers"))
            lines.append(diff_line(g["drop_51_plus"], g["stay"], "dropped past 50 minus stayers"))
            lines.append(diff_line(g["drop_21_50"], g["enter"], "dropped to 21-50 minus entrants"))
            lines.append(diff_line(g["drop_51_plus"], g["enter"], "dropped past 50 minus entrants"))
            age = pd.concat(age_rows, ignore_index=True)
            age["bucket"] = pd.cut(age.age, [0, 1, 2, 4, 9, 999], labels=["1", "2", "3-4", "5-9", "10+"])
            for b, q in age.groupby("bucket", observed=True):
                lines.append(f"  days in a row in the top 20 = {b:>4s}: {line(q)}")

            # Two books of the same size, scored as one new 20-day trade per name per day.
            fresh, sticky, held = [], [], set()
            last_day = None
            turnover = []
            for day in sorted(by_day):
                gd = by_day[day].sort_values(rank_col)
                top = gd[gd[rank_col] <= K]
                if last_day is None or (last_day, day) not in set(pairs):
                    held = set()
                keep = gd[gd.ts_code.isin(held) & (gd[rank_col] <= HOLD_ZONE)].head(K)
                fill = gd[~gd.ts_code.isin(set(keep.ts_code))].head(K - len(keep))
                book = pd.concat([keep, fill])
                turnover.append(len(fill) / K if held else np.nan)
                fresh.append(top)
                sticky.append(book)
                held, last_day = set(book.ts_code), day
            fresh_df, sticky_df = pd.concat(fresh), pd.concat(sticky)
            lines.append(f"  book A, the day's top 20 : {line(fresh_df)}")
            lines.append(f"  book B, keep while in top {HOLD_ZONE}, fill from today's ranking: {line(sticky_df)} "
                         f"| new names per day {100 * np.nanmean(turnover):.0f}% against {100 * (1 - np.mean(overlap)):.0f}% for A")
            lines.append(diff_line(sticky_df, fresh_df, "book B minus book A"))
            lines.append("")


def exit_offset(q: pd.DataFrame) -> np.ndarray:
    """Session on which the +15%/-10% rule closes the trade; 20 when neither is touched."""
    up = q.up15_day.to_numpy(dtype=int)
    dn = q.dn10_day.to_numpy(dtype=int)
    up = np.where((up > 0) & (up <= 20), up, 0)
    dn = np.where((dn > 0) & (dn <= 20), dn, 0)
    stop = (dn > 0) & ((up == 0) | (dn <= up))
    return np.where(stop, dn, np.where(up > 0, up, 20))


def part3(frame: pd.DataFrame, lines: list[str]) -> None:
    """A 20-slot account: a slot frees when its trade closes, and the day's list fills the gaps in order."""
    lines.append("== Part 3: a 20-slot account that only buys when a slot is free ==")
    cal = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    idx = {d: i for i, d in enumerate(cal)}
    rng = np.random.default_rng(20261004)
    for period in ("dev", "confirm"):
        part = frame[frame.period == period]
        for pool, rank_col in (("S", "s_rank"), ("R", "r_rank")):
            top = part[part[rank_col] <= K].copy()
            top["off"] = exit_offset(top)
            days = {d: g for d, g in top.groupby("trade_date")}

            def run(order: str) -> pd.DataFrame:
                book: list[tuple[int, str]] = []
                taken = []
                for day in sorted(days):
                    i = idx[day]
                    book = [(f, c) for f, c in book if f > i]
                    held = {c for _, c in book}
                    g = days[day]
                    if order == "random":
                        g = g.iloc[rng.permutation(len(g))]
                    else:
                        key = order_key(g, rank_col, order.removeprefix("reverse "))
                        g = g.assign(_o=-key if order.startswith("reverse") else key).sort_values(["_o", rank_col])
                    g = g[~g.ts_code.isin(held)].head(K - len(book))
                    book += [(i + int(o), c) for o, c in zip(g.off, g.ts_code)]
                    taken.append(g)
                return pd.concat(taken, ignore_index=True)

            n_days = len(days)
            lines.append(f"{period} {pool}: {n_days} signal days, {len(top)} names listed")
            for order in ("rank", "rank+atr", "atr", "reverse rank+atr"):
                t = run(order)
                lines.append(f"  fill by {order:16s}: bought {len(t):4d} ({100 * len(t) / len(top):.0f}% of listed, "
                             f"{len(t) / n_days:.1f}/day) ret {t.ret_v1.mean():+.2f} pure_up {100 * t.pure_up.mean():.1f} "
                             f"pure_down {100 * t.pure_down.mean():.1f} account {t.ret_v1.sum() / K:+.1f}%")
            runs = [run("random") for _ in range(30)]
            lines.append(f"  fill by random (30 runs)  : bought {np.mean([len(t) for t in runs]):.0f} "
                         f"ret {np.mean([t.ret_v1.mean() for t in runs]):+.2f} "
                         f"(run sd {np.std([t.ret_v1.mean() for t in runs]):.2f}) pure_down "
                         f"{100 * np.mean([t.pure_down.mean() for t in runs]):.1f} "
                         f"account {np.mean([t.ret_v1.sum() / K for t in runs]):+.1f}% "
                         f"(run sd {np.std([t.ret_v1.sum() / K for t in runs]):.1f})")
            lines.append("")


def part4(frame: pd.DataFrame, lines: list[str]) -> None:
    """Do names that leave the top 20 go on to fall? Close-to-close returns from day d, S list, confirm window."""
    lines.append("== Part 4: after a name leaves the S top 20 (confirm window, close-to-close from the next open) ==")
    f = frame[frame.period == "confirm"]
    extra = pd.read_parquet(EXP / "path_panel.parquet", columns=["ts_code", "trade_date", "ret5", "ret10"])
    extra["trade_date"] = extra["trade_date"].astype(str)
    extra["ts_code"] = extra["ts_code"].astype(str)
    days = sorted(f.trade_date.unique())
    px = pd.concat([
        pd.read_parquet(path, columns=["ts_code", "trade_date", "pct_chg"])
        for path in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet"))
        if days[0] <= path.stem <= days[-1]
    ])
    px["trade_date"] = px["trade_date"].astype(str)
    px["ts_code"] = px["ts_code"].astype(str)
    f = f.merge(extra, on=["ts_code", "trade_date"], how="left").merge(px, on=["ts_code", "trade_date"], how="left")
    by_day = {d: g for d, g in f.groupby("trade_date")}
    groups = {k: [] for k in ("stayers", "entrants", "dropped to 21-50", "dropped past 50", "all dropouts", "whole market")}
    for prev, day in day_pairs(days):
        g0, g1 = by_day[prev], by_day[day]
        top0 = set(g0.loc[g0.s_rank <= K, "ts_code"])
        top1 = g1[g1.s_rank <= K]
        out = g1[g1.ts_code.isin(top0) & (g1.s_rank > K)]
        groups["stayers"].append(top1[top1.ts_code.isin(top0)])
        groups["entrants"].append(top1[~top1.ts_code.isin(top0)])
        groups["dropped to 21-50"].append(out[out.s_rank <= HOLD_ZONE])
        groups["dropped past 50"].append(out[out.s_rank > HOLD_ZONE])
        groups["all dropouts"].append(out)
        groups["whole market"].append(g1)
    g = {k: pd.concat(v, ignore_index=True) for k, v in groups.items()}

    def row(q: pd.DataFrame) -> str:
        return (f"n={len(q):6d} day-d change {q.pct_chg.mean():+5.2f}% | ret5 {q.ret5.mean():+5.2f} ({100 * (q.ret5 < 0).mean():4.1f}% down) | "
                f"ret10 {q.ret10.mean():+5.2f} ({100 * (q.ret10 < 0).mean():4.1f}% down) | ret20 {q.ret20.mean():+5.2f} "
                f"({100 * (q.ret20 < 0).mean():4.1f}% down, median {q.ret20.median():+5.2f}) | -10% first {100 * q.pure_down.mean():4.1f}% | "
                f"+15% first {100 * q.pure_up.mean():4.1f}% | rule ret {q.ret_v1.mean():+5.2f}")

    for name, q in g.items():
        lines.append(f"  {name:17s} {row(q)}")
    for col in ("ret5", "ret20"):
        for other in ("whole market", "stayers", "entrants"):
            x = g["all dropouts"].groupby("trade_date")[col].mean()
            y = g[other].groupby("trade_date")[col].mean()
            d = (x - y).dropna()
            se = nw_se(d.to_numpy())
            lines.append(f"    all dropouts minus {other}, same day, {col}: {d.mean():+.2f} (t {d.mean() / se:+.2f}, days {len(d)})")
    d = g["all dropouts"]
    for name, q in (("fell on day d", d[d.pct_chg < 0]), ("rose on day d", d[d.pct_chg >= 0]),
                    ("fell 5% or more on day d", d[d.pct_chg <= -5]), ("rose 5% or more on day d", d[d.pct_chg >= 5])):
        lines.append(f"  dropouts that {name:25s} {row(q)}")
    lines.append("")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "List turnover study. ret = +15%/-10% exit return per name, entry at the next open after day d.",
        "t values use Newey-West with 19 lags. Dev is a 50% stock sample, confirm is the full market.",
        f"Signal days from {RESERVED_FROM} are not read.",
        "",
    ]
    scored = part1(lines)
    full = load_frame()
    part2(full, lines)
    part3(scored, lines)
    part4(full, lines)
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
