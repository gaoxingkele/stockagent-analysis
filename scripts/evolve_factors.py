#!/usr/bin/env python
"""Evolve / synthesise new screening factors inside the S20 and R20 top-100 pools (idea I0014).

  python scripts/evolve_factors.py

User request 2026-11-11: "在 R20 系列和 S20 系列筛选机制上，通过演化和合成新的因子，来提高筛选中的成功率" (both pools).
Genetic programming over the 282 TA-Lib / Qlib-Alpha158 factors already built in output/experiments/factor_zoo_20261015/factors.parquet.
Everything is on USED windows (signal days 2025-03-03..2026-08-05; nothing from 2026-08-06 on is read), evidence tier reused_holdout at best,
and the evolved factors are NOT a validated rule.  The pools are the same top-100 pools as T0107/T0108/N0036 (not the final 20-name lists).
Success = +15% before -10% within 20 sessions from the next open (list basis U15D10), stop-first = -10% first.
PRE-REGISTERED:
 Three disjoint segments per pool (search / select / test, in time order):
   S20: search 2025-03-03..2025-08-29 | select 2026-01-27..2026-04-30 | test 2026-05-01..2026-08-05
   R20: search 2026-01-27..2026-04-13 | select 2026-04-14..2026-06-12 | test 2026-06-15..2026-08-05
 Terminals: within-day percentile of each factor inside the pool.  Operators add, sub, mul, max, min, neg(1-x), absdiff; depth <= 3.
 Fitness on search = min over the two time-halves of the same-day paired (top tercile minus bottom tercile) success difference t-statistic,
   -inf unless the per-trade return difference is > 0 in both halves, and unless |rank corr with NATR| <= 0.6 (lesson of I0009/I0010).
 GP: population 200, 20 generations, tournament 4, crossover + subtree/point mutation, elitism 10, 3 seeds per pool; hall of fame = top 20 distinct
   (rank corr < 0.9) by fitness.  Evaluated candidate count is reported.
 Select: hall-of-fame members re-scored on the select segment; the Bonferroni bar counts the 20 candidates x 2 pools (|t| >= 2.5 for 20, 2.73 for 40
   when both pools are read together; the one-pool bar 2.50 is used per pool).  Passing = same sign, bar, per-trade difference > 0.
 Test: only the single best select-passing factor per pool is scored once on test (success diff, t, stop-first diff, per trade, and the effect of dropping
   the worst third of the pool); baselines: the best single zoo factor chosen the same way (search -> select) and low NATR.
 Placebo: the same GP run on within-day shuffled success labels (3 seeds per pool): how large the search fitness gets with no signal, and how many
   placebo hall-of-fame members pass select (that count is what chance looks like).
Falsified (I0014) if, in a pool, no evolved factor passes select, or the one that does has test success diff t < 1.96 / opposite sign / per-trade difference <= 0,
or the placebo passes select as often as the real labels.
Output: output/experiments/evolve_factors_20261111/report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_bottom_patterns import pools  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402

OUT = ROOT / "output/experiments/evolve_factors_20261111"
ZOO = ROOT / "output/experiments/factor_zoo_20261015/factors.parquet"
SEG = {"S20": {"search": ("20250303", "20250829"), "select": ("20260127", "20260430"), "test": ("20260501", "20260805")},
       "R20": {"search": ("20260127", "20260413"), "select": ("20260414", "20260612"), "test": ("20260615", "20260805")}}
POP, GENS, TOURN, ELITE, SEEDS, HOF = 200, 20, 4, 10, 3, 20
OPS = ("add", "sub", "mul", "max", "min", "neg", "absdiff")
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------ expression trees: ("t", i) | (op, a[, b])
def rand_tree(rng, nt: int, depth: int):
    if depth == 0 or rng.random() < 0.25:
        return ("t", int(rng.integers(nt)))
    op = OPS[int(rng.integers(len(OPS)))]
    if op == "neg":
        return (op, rand_tree(rng, nt, depth - 1))
    return (op, rand_tree(rng, nt, depth - 1), rand_tree(rng, nt, depth - 1))


def evaluate(tree, X: np.ndarray) -> np.ndarray:
    k = tree[0]
    if k == "t":
        return X[:, tree[1]]
    a = evaluate(tree[1], X)
    if k == "neg":
        return 1.0 - a
    b = evaluate(tree[2], X)
    return {"add": a + b, "sub": a - b, "mul": a * b, "max": np.maximum(a, b), "min": np.minimum(a, b), "absdiff": np.abs(a - b)}[k]


def size(tree) -> int:
    return 1 if tree[0] == "t" else 1 + sum(size(c) for c in tree[1:])


def depth(tree) -> int:
    return 0 if tree[0] == "t" else 1 + max(depth(c) for c in tree[1:])


def subtrees(tree, path=()):
    yield path, tree
    if tree[0] != "t":
        for i, c in enumerate(tree[1:], 1):
            yield from subtrees(c, path + (i,))


def replace(tree, path, new):
    if not path:
        return new
    lst = list(tree)
    lst[path[0]] = replace(tree[path[0]], path[1:], new)
    return tuple(lst)


def show(tree, names) -> str:
    k = tree[0]
    if k == "t":
        return names[tree[1]]
    if k == "neg":
        return f"(1-{show(tree[1], names)})"
    return f"{k}({show(tree[1], names)},{show(tree[2], names)})"


def vary(rng, a, b, nt):
    c = a
    if rng.random() < 0.6:
        pa = list(subtrees(a))
        pb = list(subtrees(b))
        p, _ = pa[int(rng.integers(len(pa)))]
        _, sub = pb[int(rng.integers(len(pb)))]
        c = replace(a, p, sub)
    if rng.random() < 0.5:
        pc = list(subtrees(c))
        p, _ = pc[int(rng.integers(len(pc)))]
        c = replace(c, p, rand_tree(rng, nt, 2))
    if depth(c) > 3:
        return a
    return c


# ------------------------------------------------------------------ data and scoring
class Seg:
    """One segment of one pool: percentile terminals, day index, outcomes."""

    def __init__(self, df: pd.DataFrame, names: list[str], natr_col: str):
        df = df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
        self.day = df.trade_date.to_numpy()
        self.X = df.groupby("trade_date")[names].rank(pct=True).fillna(0.5).to_numpy(np.float32)
        self.success = df.success.to_numpy(float)
        self.stop = df.stop.to_numpy(float)
        self.ret = df.ret_base.to_numpy(float)
        self.natr = df.groupby("trade_date")[natr_col].rank(pct=True).fillna(0.5).to_numpy()
        self.bounds = np.flatnonzero(np.r_[True, self.day[1:] != self.day[:-1], True])
        self.days = self.day[self.bounds[:-1]]

    def perday(self):
        for i in range(len(self.bounds) - 1):
            yield i, slice(self.bounds[i], self.bounds[i + 1])

    def half_mask(self):
        n = len(self.days)
        return np.arange(n) < n // 2

    def diffs(self, score: np.ndarray, y_list):
        """Per-day (top tercile minus bottom tercile) means of each y in y_list."""
        out = [[] for _ in y_list]
        for i, s in self.perday():
            sc = score[s]
            if len(sc) < 12:
                for o in out:
                    o.append(np.nan)
                continue
            r = sc.argsort(kind="stable").argsort() / (len(sc) - 1)
            top, bot = r > 2 / 3, r <= 1 / 3
            for o, y in zip(out, y_list):
                yy = y[s]
                o.append(yy[top].mean() - yy[bot].mean())
        return [np.array(o) for o in out]


def fitness(seg: Seg, tree, X=None) -> tuple[float, np.ndarray]:
    sc = evaluate(tree, seg.X)
    if np.nanstd(sc) < 1e-6:
        return -np.inf, sc
    if abs(pd.Series(sc).corr(pd.Series(seg.natr), method="spearman")) > 0.6:
        return -np.inf, sc
    ds, dr = seg.diffs(sc, [seg.success, seg.ret])
    h = seg.half_mask()
    ts = [tstat(pd.Series(ds[m]).dropna()) if np.isfinite(ds[m]).sum() > 5 else np.nan for m in (h, ~h)]
    if not np.all(np.isfinite(ts)) or np.nanmean(dr[h]) <= 0 or np.nanmean(dr[~h]) <= 0:
        return -np.inf, sc
    return float(min(ts)) - 0.02 * size(tree), sc


def evolve(seg: Seg, names: list[str], seed: int, counter: list[int]):
    rng = np.random.default_rng(seed)
    nt = len(names)
    cache: dict = {}

    def fit(t):
        if t not in cache:
            cache[t] = fitness(seg, t)[0]
            counter[0] += 1
        return cache[t]
    pop = [rand_tree(rng, nt, 3) for _ in range(POP)]
    for _ in range(GENS):
        fs = np.array([fit(t) for t in pop])
        order = np.argsort(-fs)
        new = [pop[i] for i in order[:ELITE]]
        while len(new) < POP:
            pa = max((int(rng.integers(POP)) for _ in range(TOURN)), key=lambda i: fs[i])
            pb = max((int(rng.integers(POP)) for _ in range(TOURN)), key=lambda i: fs[i])
            new.append(vary(rng, pop[pa], pop[pb], nt))
        pop = new
    return {t: v for t, v in cache.items() if np.isfinite(v)}


def hall_of_fame(seg: Seg, cands: dict, k: int = HOF):
    out, scores = [], []
    for t, v in sorted(cands.items(), key=lambda kv: -kv[1])[:400]:
        sc = evaluate(t, seg.X)
        if any(abs(pd.Series(sc).corr(pd.Series(o), method="spearman")) > 0.9 for o in scores):
            continue
        out.append((t, v))
        scores.append(sc)
        if len(out) >= k:
            break
    return out


def score_on(seg: Seg, score: np.ndarray) -> dict:
    ds, dr, dstop = seg.diffs(score, [seg.success, seg.ret, seg.stop])
    ok = np.isfinite(ds)
    return dict(sd=100 * np.nanmean(ds), t=tstat(pd.Series(ds[ok])) if ok.sum() > 5 else np.nan, pt=np.nanmean(dr), stop=100 * np.nanmean(dstop), days=int(ok.sum()))


def drop_worst_third(seg: Seg, score: np.ndarray) -> tuple[float, float, float]:
    """Effect of dropping the worst third of the pool: kept-name success / per-trade return minus the whole pool, same day."""
    ds, dp, dst = [], [], []
    for i, s in seg.perday():
        sc = score[s]
        if len(sc) < 12:
            continue
        r = sc.argsort(kind="stable").argsort() / (len(sc) - 1)
        keep = r > 1 / 3
        ds.append(seg.success[s][keep].mean() - seg.success[s].mean())
        dp.append(seg.ret[s][keep].mean() - seg.ret[s].mean())
        dst.append(seg.stop[s][keep].mean() - seg.stop[s].mean())
    return 100 * np.mean(ds), np.mean(dp), 100 * np.mean(dst)


def build_segs(placebo_seed: int | None = None):
    P = pools()
    zoo = pd.read_parquet(ZOO)
    names = [c for c in zoo.columns if c not in ("ts_code", "trade_date")]
    zoo[names] = zoo[names].replace([np.inf, -np.inf], np.nan)
    names = [c for c in names if zoo[c].notna().mean() > 0.5 and zoo[c].nunique() > 5]
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet", columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["success"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    o["stop"] = ((o.dn10_day > 0) & ((o.up15_day == 0) | (o.dn10_day <= o.up15_day))).astype(float)
    return P, zoo, names, o


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    P, zoo, names, o = build_segs()
    say(f"terminals {len(names)}; GP population {POP}, generations {GENS}, seeds {SEEDS}; hall of fame {HOF}; NATR column t_NATR")
    bar_sel = norm.isf(0.025 / HOF)
    for pname, segs in SEG.items():
        base = P[pname][P[pname].trade_date.between("20250303", "20260805")].merge(o, on=["ts_code", "trade_date"]).merge(zoo, on=["ts_code", "trade_date"], how="left")
        S = {k: Seg(base[base.trade_date.between(a, b)], names, "t_NATR") for k, (a, b) in segs.items()}
        say(f"\n{'=' * 90}\n{pname} top-100 pool: " + "; ".join(f"{k} {a}..{b} {len(S[k].day):,} rows/{len(S[k].days)} days" for k, (a, b) in segs.items())
            + f"; test mean success {100 * S['test'].success.mean():.1f}%, stop-first {100 * S['test'].stop.mean():.1f}%, per trade {S['test'].ret.mean():+.2f}")
        for label, shuffle in (("REAL", False), ("PLACEBO (within-day shuffled labels on search)", True)):
            seg = S["search"]
            if shuffle:
                rng = np.random.default_rng(99)
                idx = np.arange(len(seg.day))
                for i, s in seg.perday():
                    idx[s] = rng.permutation(idx[s])
                seg = Seg(base[base.trade_date.between(*segs["search"])].sort_values(["trade_date", "ts_code"]).reset_index(drop=True).assign(
                    success=lambda d: d.success.to_numpy()[idx], stop=lambda d: d.stop.to_numpy()[idx], ret_base=lambda d: d.ret_base.to_numpy()[idx]), names, "t_NATR")
            counter = [0]
            cands: dict = {}
            for sd in range(SEEDS):
                cands.update(evolve(seg, names, 1000 * (pname == "R20") + sd + (500 if shuffle else 0), counter))
            hof = hall_of_fame(seg, cands)
            say(f"\n-- {label}: evaluated {counter[0]:,} candidates, {len(cands):,} passed the constraints; best search fitness (min half t) {hof[0][1] if hof else float('nan'):+.2f}; "
                f"hall of fame {len(hof)}")
            passed = []
            for t, v in hof:
                r = score_on(S["select"], evaluate(t, S["select"].X))
                ok = r["t"] >= bar_sel and r["pt"] > 0
                if ok:
                    passed.append((t, v, r))
            say(f"   select ({segs['select'][0]}..{segs['select'][1]}): bar {bar_sel:.2f}; passed {len(passed)} of {len(hof)}")
            for t, v in hof[:5]:
                r = score_on(S["select"], evaluate(t, S["select"].X))
                say(f"   search {v:+.2f} | select success diff {r['sd']:+.1f}pp (t {r['t']:+.1f}), per trade {r['pt']:+.2f}, stop diff {r['stop']:+.1f}pp | {show(t, names)}")
            if shuffle:
                continue
            if not passed:
                say("   no evolved factor passed select -> not scored on test")
            else:
                t, v, r = max(passed, key=lambda x: x[2]["t"])
                rt = score_on(S["test"], evaluate(t, S["test"].X))
                dw = drop_worst_third(S["test"], evaluate(t, S["test"].X))
                say(f"   BEST select-passing: {show(t, names)}  (size {size(t)})")
                say(f"   select: success diff {r['sd']:+.1f}pp (t {r['t']:+.1f}), per trade {r['pt']:+.2f}")
                say(f"   TEST ({segs['test'][0]}..{segs['test'][1]}): top-minus-bottom tercile success {rt['sd']:+.1f}pp (t {rt['t']:+.1f}), stop-first {rt['stop']:+.1f}pp, per trade {rt['pt']:+.2f}, {rt['days']} days")
                say(f"   TEST drop-worst-third vs whole pool: success {dw[0]:+.2f}pp, per trade {dw[1]:+.2f}, stop-first {dw[2]:+.2f}pp")
            # baselines on the same protocol: best single zoo factor on search, then test
            sing = []
            for j, nm in enumerate(names):
                f, _ = fitness(S["search"], ("t", j))
                if np.isfinite(f):
                    sing.append((f, j))
            sing.sort(reverse=True)
            if sing:
                j = sing[0][1]
                r = score_on(S["select"], S["select"].X[:, j])
                rt = score_on(S["test"], S["test"].X[:, j])
                say(f"   baseline best single factor on search: {names[j]} (search {sing[0][0]:+.2f}); select {r['sd']:+.1f}pp (t {r['t']:+.1f}); test {rt['sd']:+.1f}pp (t {rt['t']:+.1f}), per trade {rt['pt']:+.2f}")
            nn = S["test"]
            rt = score_on(nn, -nn.natr)
            say(f"   baseline low NATR on test: success {rt['sd']:+.1f}pp (t {rt['t']:+.1f}), stop-first {rt['stop']:+.1f}pp, per trade {rt['pt']:+.2f}")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
