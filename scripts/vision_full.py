#!/usr/bin/env python
"""Full-market picture CNN (idea I0013, follow-up of I0012): train on the whole market instead of the S20 funnel only.

  python scripts/vision_full.py [--stage build|train|all]

User request 2026-10-31: "用全市场大样本再训练一遍".  Same chart renderer as vision_smoke.py (candles + MA5/10/20/60, volume, MACD, RSI14 panels).
Samples: every signal day 2024-09..2026-08-05 (used windows only; nothing from 2026-08-06 on is a signal day), 200 random stocks per day from the
whole market (>=160 bars of history).  Label = U15D10 first-passage per-trade return from the next open (+15% / -10% / day-20 close, cost 0.3%;
a day touching both counts as the stop).  Tier reused_holdout.
PRE-REGISTERED:
 Targets: (A) within-day percentile of the per-trade return; (B) the same percentile taken WITHIN day x NATR tercile (volatility stripped; lesson of I0010).
 Train signal days <= 2025-11-28, test >= 2026-02-02 (20-session purge).  CNN as in I0012 (fixed, 6 epochs); L=100 with 2 seeds, L=50 and 150 one seed.
 Baseline: gradient boosting on the 12 hand features with the same targets; placebo CNN (L=100) with within-day(-tercile) shuffled targets.
 Metrics on the full-market test set: daily rank IC with the per-trade return (NW t), top-decile minus bottom-decile same-day per trade, and the same inside
 each NATR tercile.  Then the CNN is applied to the I0012 S20-funnel test rows (pool S top 20 + 20 other survivors, signal days >= 2026-02-02) for the daily IC
 with their per-trade return.
Falsified (I0013) if the target-B CNN daily IC has NW t < 1.96, or does not beat the hand-feature boosting IC, or the placebo reaches the same IC, or the
IC inside the S20 funnel is not > 0.
Output: output/experiments/vision_full_20261031/report.txt; images cached outside the repo (env VISION_FULL_CACHE).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import vision_smoke as V  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402
from price_panel import adjusted_daily  # noqa: E402

OUT = ROOT / "output/experiments/vision_full_20261031"
CACHE = Path(os.environ.get("VISION_FULL_CACHE", Path(tempfile.gettempdir()) / "vision_full_20261031"))
OLD = V.CACHE
FIRST, LAST = "20240902", "20260805"
PER_DAY = 200
WINDOWS = V.WINDOWS
LINES: list[str] = []
TARGETS = ("yB",)                       # target A was started first (CNN L=100 seed 7 only, then stopped: ~15 min per CNN on this CPU)
PLAN = ((100, (7, 8)), (150, (7,)))     # L=50 dropped for time


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def label(st: dict, e: int) -> tuple[float, float] | None:
    n = len(st["c"])
    if e + 20 >= n:
        return None
    entry = st["o"][e + 1]
    if not entry > 0:
        return None
    h, l, c = st["h"][e + 1:e + 21], st["l"][e + 1:e + 21], st["c"][e + 20]
    stop, up = np.where(l <= 0.9 * entry)[0], np.where(h >= 1.15 * entry)[0]
    s0, u0 = (stop[0] if len(stop) else 99), (up[0] if len(up) else 99)
    if s0 <= u0 and s0 < 99:
        r = -10.0
    elif u0 < 99:
        r = 15.0
    else:
        r = (c / entry - 1) * 100
    return r - 0.3, float(st["natr"][e])


def _work(args):
    frame, want, tag = args
    imgs = {L: [] for L in WINDOWS}
    meta, feats = [], []
    for code, g in frame.groupby("ts_code", sort=False):
        if code not in want:
            continue
        g = g.reset_index(drop=True)
        st = V.indicators(g)
        pos = {d: i for i, d in enumerate(st["date"])}
        for d in want[code]:
            e = pos.get(d)
            if e is None or e < 160:
                continue
            lab = label(st, e)
            if lab is None:
                continue
            ims = [V.render(st, e, L) for L in WINDOWS]
            if any(im is None for im in ims):
                continue
            hf = V.hand_features(st, e)
            if not np.isfinite(hf).all():
                continue
            for L, im in zip(WINDOWS, ims):
                imgs[L].append(im)
            meta.append((code, d, lab[0], lab[1]))
            feats.append(hf)
    for L in WINDOWS:
        np.save(CACHE / f"chunk{tag}_{L}.npy", np.stack(imgs[L]) if imgs[L] else np.zeros((0, 3, V.H, L), np.uint8))
    return meta, np.array(feats)


def build() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    px = adjusted_daily("20240102", "20261029", None)
    days = sorted(d for d in px.trade_date.unique() if FIRST <= d <= LAST)
    ok = px.groupby("ts_code").size()
    codes = np.array(sorted(ok[ok >= 170].index))
    rng = np.random.default_rng(20261031)
    rows = []
    for d in days:
        act = px[px.trade_date == d].ts_code.to_numpy()
        act = np.intersect1d(act, codes)
        pick = rng.choice(act, min(PER_DAY, len(act)), replace=False)
        rows.append(pd.DataFrame({"ts_code": pick, "trade_date": d}))
    smp = pd.concat(rows, ignore_index=True)
    say(f"samples {len(smp)} over {len(days)} days ({days[0]}..{days[-1]}), {smp.ts_code.nunique()} stocks")
    uc = sorted(set(smp.ts_code))
    parts = [set(uc[i::8]) for i in range(8)]
    chunks = []
    for i, p in enumerate(parts):
        want = {c: sorted(g.trade_date) for c, g in smp[smp.ts_code.isin(p)].groupby("ts_code")}
        chunks.append((px[px.ts_code.isin(p)], want, i))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_work, chunks))
    meta = pd.DataFrame([m for r in res for m in r[0]], columns=["ts_code", "trade_date", "ret", "natr"])
    np.save(CACHE / "hand.npy", np.concatenate([r[1] for r in res if len(r[1])]))
    n = len(meta)
    for L in WINDOWS:
        mm = np.lib.format.open_memmap(CACHE / f"img_{L}.npy", mode="w+", dtype=np.uint8, shape=(n, 3, V.H, L))
        o = 0
        for i in range(8):
            a = np.load(CACHE / f"chunk{i}_{L}.npy", mmap_mode="r")
            mm[o:o + len(a)] = a
            o += len(a)
            del a
            os.remove(CACHE / f"chunk{i}_{L}.npy")
        mm.flush()
        del mm
    meta.to_parquet(CACHE / "meta.parquet", index=False)
    say(f"rendered {n} images per window; cache {CACHE}")


def targets(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["tier"] = d.groupby("trade_date").natr.rank(pct=True).mul(3).astype(int).clip(upper=2)
    d["yA"] = d.groupby("trade_date").ret.rank(pct=True)
    d["yB"] = d.groupby(["trade_date", "tier"]).ret.rank(pct=True)
    return d


def evaluate(te: pd.DataFrame, col: str, tag: str) -> None:
    ic = V.daily_ic(te, col, "ret")
    q = te.groupby("trade_date")[col].rank(pct=True, method="first")
    gap = (te[q > 0.9].groupby("trade_date").ret.mean() - te[q <= 0.1].groupby("trade_date").ret.mean()).dropna()
    tier_gaps = []
    for k in range(3):
        t2 = te[te.tier == k]
        q2 = t2.groupby("trade_date")[col].rank(pct=True, method="first")
        g2 = (t2[q2 > 0.8].groupby("trade_date").ret.mean() - t2[q2 <= 0.2].groupby("trade_date").ret.mean()).dropna()
        tier_gaps.append(f"{g2.mean():+.2f}")
    say(f"   {tag:28s} daily IC {ic.mean():+.4f} (NW t {tstat(ic):+.1f}); top decile minus bottom {gap.mean():+.2f} (t {tstat(gap):+.1f}); "
        f"within NATR tercile low/mid/high quintile gap {'/'.join(tier_gaps)}; corr with NATR {te[col].corr(te.natr):+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["build", "train", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("build", "all") and not (CACHE / "meta.parquet").exists():
        build()
    if a.stage == "build":
        (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
        return 0
    from sklearn.ensemble import HistGradientBoostingRegressor
    meta = pd.read_parquet(CACHE / "meta.parquet")
    meta["row"] = np.arange(len(meta))
    hand = np.load(CACHE / "hand.npy")
    imgs = {L: np.load(CACHE / f"img_{L}.npy", mmap_mode="r") for L in WINDOWS}
    d = targets(meta)
    tr, te = d[d.trade_date <= V.OUT_TR_END].copy(), d[d.trade_date >= V.TE_START].copy()
    say(f"train rows {len(tr)} ({tr.trade_date.nunique()} days), test rows {len(te)} ({te.trade_date.nunique()} days); test mean per trade {te.ret.mean():+.2f}")

    # I0012 funnel rows (pool S + other survivors, test period) for the transfer check
    osmp = pd.read_parquet(OLD / "samples.parquet")
    ometa = pd.read_parquet(OLD / "meta.parquet")
    ometa["orow"] = np.arange(len(ometa))
    fun = osmp[osmp.grp.isin(["S", "F"])].drop_duplicates(["ts_code", "trade_date"]).merge(ometa, on=["ts_code", "trade_date"])
    fun = fun[fun.ret_v1.notna() & (fun.trade_date >= V.TE_START)].copy()
    oimgs = {L: np.load(OLD / f"img_{L}.npy", mmap_mode="r") for L in WINDOWS}
    ohand = np.load(OLD / "hand.npy")
    say(f"S20-funnel transfer rows {len(fun)} ({fun.trade_date.nunique()} days)")

    def funnel(pred, tag):
        fun["p"] = pred
        ic = V.daily_ic(fun, "p", "ret_v1")
        q = fun.groupby("trade_date").p.rank(pct=True, method="first")
        gap = (fun[q > 2 / 3].groupby("trade_date").ret_v1.mean() - fun[q <= 1 / 3].groupby("trade_date").ret_v1.mean()).dropna()
        say(f"   -> S20 funnel {tag:22s} daily IC {ic.mean():+.4f} (NW t {tstat(ic):+.1f}); top third minus bottom third {gap.mean():+.2f} (t {tstat(gap):+.1f})")

    te["natr_neg"] = -te.natr
    for tgt in TARGETS:
        say(f"\n== target {'A: within-day percentile' if tgt == 'yA' else 'B: within day x NATR tercile percentile (volatility stripped)'} ==")
        evaluate(te.assign(nat=te.natr_neg).rename(columns={"nat": "x"}), "x", "baseline low NATR")
        hr = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=1).fit(hand[tr.row], tr[tgt])
        te["hand"] = hr.predict(hand[te.row])
        evaluate(te, "hand", "hand-feature boosting")
        funnel(hr.predict(ohand[fun.orow]), "hand boosting")
        for L, seeds in PLAN:
            Xtr = imgs[L]
            for sd in seeds:
                pr = V.train_cnn(_Idx(Xtr, tr.row.to_numpy()), tr[tgt].to_numpy(), "reg", seed=sd)
                te["cnn"] = pr(_Idx(Xtr, te.row.to_numpy()))
                evaluate(te, "cnn", f"CNN L={L} seed {sd}")
                funnel(pr(_Idx(oimgs[L], fun.orow.to_numpy())), f"CNN L={L} seed {sd}")
        if tgt == "yB":
            grp = ["trade_date", "tier"]
        else:
            grp = ["trade_date"]
        sh = tr[tgt].groupby([tr[g] for g in grp]).transform(lambda v: np.random.default_rng(5).permutation(v.to_numpy()))
        pr = V.train_cnn(_Idx(imgs[100], tr.row.to_numpy()), sh.to_numpy(), "reg", seed=7)
        te["plc"] = pr(_Idx(imgs[100], te.row.to_numpy()))
        evaluate(te, "plc", "placebo CNN L=100")
        funnel(pr(_Idx(oimgs[100], fun.orow.to_numpy())), "placebo CNN L=100")
        (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


class _Idx:
    """Lazy row selection of a memory-mapped image array so that 80k images are never held in RAM at once."""

    def __init__(self, arr, rows):
        self.arr, self.rows = arr, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        ix = np.arange(len(self.rows))[ix] if isinstance(ix, slice) else np.asarray(ix)
        r = self.rows[ix]
        o = np.argsort(r)
        out = np.empty((len(r),) + self.arr.shape[1:], np.uint8)
        out[o] = self.arr[r[o]]
        return out


if __name__ == "__main__":
    raise SystemExit(main())
