#!/usr/bin/env python
"""Vision smoke test: do the recommended lists share chart shapes, and can a small CNN on the chart picture say anything about outcomes?

  python scripts/vision_smoke.py [--stage build|train|all]

User request 2026-10-31 (idea I0012): take 50 / 100 / 150 daily bars before the signal day for the names on the pool S (S20 contract top 20),
pool A and V12.31 Top20 lists; is the picture similar / clustered more than random names of the same day, and does a picture-only CNN carry decision
information?  The picture is a main panel (candles + MA5/10/20/60) plus volume, MACD (DIF/DEA/histogram) and RSI(14) sub-panels, drawn from adjusted
daily bars by numpy (3 channels: R candles, G moving averages, B sub-panels); one image per (stock, signal day, window length).
PRE-REGISTERED, descriptive, used windows only (signal days 2025-03-03..2026-08-05; nothing from 2026-08-06 on is read), tier reused_holdout.

 Part 1 similarity (no learning): mean pairwise correlation of the z-scored log-close path of the list members vs 300 random 20-subsets of
        300 random stocks of the same day; z over the random subsets; reported by list and window length.
 Part 2 CNN list-vs-market: S vs M (random market names, 25 per day): train signal days <= 2025-12-31, test >= 2026-02-02.  R20 family (pool A
        + V12.31 Top20) vs M: train 2026-04-14..06-12, test 2026-07-06..08-05 (the R20 lists exist from 04-14).  Fixed small CNN, 6 epochs, no tuning.
        AUC with day-block bootstrap; baseline = gradient boosting on 12 hand-made features (momentum, NATR, range position, MA distances, RSI, volume ratio).
 Part 3 CNN outcome: inside the S20 funnel (top 20 + 20 random of the other survivors per day, 2025-03..2026-08) predict the within-day percentile
        of the U15D10 per-trade return; train <= 2025-11-28 (20-session purge), test >= 2026-02-02; daily rank IC, NW(19) t, top-third minus
        bottom-third same-day; baselines NATR and the hand-feature boosting; placebo = same CNN trained on within-day shuffled targets.
Falsified (I0012) if (a) CNN AUC is not >= tabular baseline + 0.02 or the similarity z < 2, and (b) outcome IC NW t < 1.96, or not above the NATR baseline,
or the placebo reaches the same IC.
Output: output/experiments/vision_smoke_20261031/{report.txt, examples.png}; images are cached outside the repo (env VISION_CACHE).
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
from analyze_r20_s20_nested import tstat  # noqa: E402
from price_panel import adjusted_daily  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402

OUT = ROOT / "output/experiments/vision_smoke_20261031"
CACHE = Path(os.environ.get("VISION_CACHE", Path(tempfile.gettempdir()) / "vision_smoke_20261031"))
START, END = "20250303", "20260805"
WINDOWS = (50, 100, 150)
H, PH, VH, MH, RH = 128, 64, 20, 24, 20
TR_END, TE_START = "20251231", "20260202"
OUT_TR_END = "20251128"
R_TR_END, R_TE_START = "20260612", "20260706"
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------ indicators and rendering
def indicators(g: pd.DataFrame) -> dict:
    o, h, l, c, v = (g[k].to_numpy(float) for k in ("open", "high", "low", "close", "vol"))
    cs = pd.Series(c)
    ma = {k: cs.rolling(k, min_periods=k).mean().to_numpy() for k in (5, 10, 20, 60)}
    e12, e26 = cs.ewm(span=12, adjust=False).mean(), cs.ewm(span=26, adjust=False).mean()
    dif = (e12 - e26)
    dea = dif.ewm(span=9, adjust=False).mean()
    d = cs.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    natr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy() / c
    return dict(date=g.trade_date.to_numpy(), o=o, h=h, l=l, c=c, v=v, ma5=ma[5], ma10=ma[10], ma20=ma[20], ma60=ma[60],
                dif=dif.to_numpy(), dea=dea.to_numpy(), hist=(dif - dea).to_numpy(), rsi=rsi, natr=natr)


def line(img: np.ndarray, ch: int, r: np.ndarray, val: int, base: int) -> None:
    for i in range(len(r)):
        a = r[i - 1] if i else r[i]
        lo, hi = (a, r[i]) if a <= r[i] else (r[i], a)
        img[ch, base + lo:base + hi + 1, i] = val


def render(st: dict, e: int, L: int) -> np.ndarray | None:
    s = e - L + 1
    if s < 100:
        return None
    sl = slice(s, e + 1)
    o, h, l, c, v = (st[k][sl] for k in ("o", "h", "l", "c", "v"))
    mas = [st[k][sl] for k in ("ma5", "ma10", "ma20", "ma60")]
    if any(np.isnan(m).any() for m in mas) or np.isnan(st["rsi"][sl]).any() or (c <= 0).any():
        return None
    lo = min(l.min(), *(m.min() for m in mas))
    hi = max(h.max(), *(m.max() for m in mas))
    sc = (PH - 1) / (hi - lo + 1e-12)
    ry = lambda x: np.clip(((hi - x) * sc).round().astype(int), 0, PH - 1)  # noqa: E731
    rh, rl, ro, rc = ry(h), ry(l), ry(o), ry(c)
    img = np.zeros((3, H, L), np.uint8)
    top, bot, upd = np.minimum(ro, rc), np.maximum(ro, rc), c >= o
    for i in range(L):
        img[0, rh[i]:rl[i] + 1, i] = 150
        img[0, top[i]:bot[i] + 1, i] = 255 if upd[i] else 90
    for m, val in zip(mas, (255, 200, 150, 100)):
        line(img, 1, ry(m), val, 0)
    vh = np.round(v / (v.max() + 1e-9) * (VH - 1)).astype(int)
    for i in range(L):
        img[2, PH + VH - 1 - vh[i]:PH + VH, i] = 255 if upd[i] else 120
    m0 = PH + VH
    hist, dif, dea = st["hist"][sl], st["dif"][sl], st["dea"][sl]
    mx = max(np.abs(hist).max(), np.abs(dif).max(), np.abs(dea).max()) + 1e-12
    mid = MH // 2
    rows = lambda x: np.clip(mid - np.round(x / mx * (mid - 1)).astype(int), 0, MH - 1)  # noqa: E731
    rhs = rows(hist)
    for i in range(L):
        a, b = sorted((mid, rhs[i]))
        img[2, m0 + a:m0 + b + 1, i] = 255 if hist[i] >= 0 else 120
    for x, val in ((dif, 220), (dea, 170)):
        r = rows(x)
        img[2, m0 + r, np.arange(L)] = val
    r0 = m0 + MH
    for lvl in (30, 70):
        img[2, r0 + int(round((100 - lvl) / 100 * (RH - 1))), :] = np.maximum(img[2, r0 + int(round((100 - lvl) / 100 * (RH - 1))), :], 40)
    rr = np.clip(np.round((100 - st["rsi"][sl]) / 100 * (RH - 1)).astype(int), 0, RH - 1)
    line(img, 2, rr, 255, r0)
    return img


def hand_features(st: dict, e: int) -> list[float]:
    c = st["c"]
    f = lambda k: c[e] / c[e - k] - 1  # noqa: E731
    w = c[e - 99:e + 1]
    return [f(20), f(60), f(120), st["natr"][e], (c[e] - w.min()) / (w.max() - w.min() + 1e-12), c[e] / st["ma20"][e] - 1, c[e] / st["ma60"][e] - 1,
            st["ma20"][e] / st["ma20"][e - 5] - 1, st["rsi"][e], st["hist"][e] / c[e], np.mean(st["v"][e - 4:e + 1]) / (np.mean(st["v"][e - 19:e + 1]) + 1e-9),
            c[e] / c[e - 149:e + 1].max() - 1]


def _work(args):
    frame, want = args                                     # want: ts_code -> list of trade_date
    res = {L: [] for L in WINDOWS}
    meta, feats, paths = [], [], []
    for code, g in frame.groupby("ts_code", sort=False):
        if code not in want:
            continue
        g = g.reset_index(drop=True)
        st = indicators(g)
        pos = {d: i for i, d in enumerate(st["date"])}
        for d in want[code]:
            e = pos.get(d)
            if e is None or e < 160:
                continue
            ims = [render(st, e, L) for L in WINDOWS]
            if any(im is None for im in ims):
                continue
            hf = hand_features(st, e)
            if not np.isfinite(hf).all():
                continue
            for L, im in zip(WINDOWS, ims):
                res[L].append(im)
            meta.append((code, d))
            feats.append(hf)
            paths.append(np.log(st["c"][e - 149:e + 1]).astype(np.float32))
    return ({L: (np.stack(v) if v else np.zeros((0, 3, H, L), np.uint8)) for L, v in res.items()}, meta, np.array(feats), np.array(paths))


# ------------------------------------------------------------------ samples
def samples() -> pd.DataFrame:
    rng = np.random.default_rng(20261031)
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    fr = fr[(fr.trade_date >= START) & (fr.trade_date <= END)]
    surv = select(fr, PureConfig(top_k=100), "U15D10")
    out = outcomes()
    out = out[(out.trade_date >= START) & (out.trade_date <= END)][["ts_code", "trade_date", "ret_v1", "state"]]
    rows = []
    for d, g in surv.groupby("trade_date"):
        s = g[g.list_rank <= 20]
        o = g[g.list_rank > 20]
        rows.append(pd.DataFrame({"ts_code": s.ts_code, "trade_date": d, "grp": "S", "rank": s.list_rank, "natr_rank": s.natr14}))
        if len(o):
            o = o.sample(min(20, len(o)), random_state=int(d))
            rows.append(pd.DataFrame({"ts_code": o.ts_code, "trade_date": d, "grp": "F", "rank": o.list_rank, "natr_rank": o.natr14}))
    for d, g in fr.groupby("trade_date"):
        m = g.sample(min(25, len(g)), random_state=int(d) + 1)
        rows.append(pd.DataFrame({"ts_code": m.ts_code, "trade_date": d, "grp": "M", "rank": np.nan, "natr_rank": m.natr14}))
    import glob
    r20 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "output/r20_history/r20_lists_*.parquet")))])
    r20 = r20[(r20.trade_date >= "20260414") & (r20.trade_date <= END)]
    for lst, tag in (("pool_a", "A"), ("v12_31_top20", "V")):
        x = r20[r20.list == lst].sort_values(["trade_date", "list_rank"]).groupby("trade_date").head(20)
        rows.append(pd.DataFrame({"ts_code": x.ts_code, "trade_date": x.trade_date, "grp": tag, "rank": x.list_rank, "natr_rank": np.nan}))
    s = pd.concat(rows, ignore_index=True)
    s["trade_date"] = s.trade_date.astype(str)
    # a name drawn into M that is also in S that day is dropped from M
    sk = set(zip(s[s.grp == "S"].ts_code, s[s.grp == "S"].trade_date))
    s = s[~((s.grp == "M").to_numpy() & np.array([(a, b) in sk for a, b in zip(s.ts_code, s.trade_date)]))]
    return s.merge(out, on=["ts_code", "trade_date"], how="left")


def build() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    smp = samples()
    say(f"samples: {smp.groupby('grp').size().to_dict()}, days {smp.trade_date.nunique()} ({smp.trade_date.min()}..{smp.trade_date.max()})")
    codes = sorted(set(smp.ts_code))
    px = adjusted_daily("20240102", END, set(codes))
    key = smp.drop_duplicates(["ts_code", "trade_date"])
    want = {c: sorted(g.trade_date) for c, g in key.groupby("ts_code")}
    chunks = [(px[px.ts_code.isin(set(codes[i::8]))], {k: want[k] for k in codes[i::8] if k in want}) for i in range(8)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_work, chunks))
    meta = [m for r in res for m in r[1]]
    for L in WINDOWS:
        np.save(CACHE / f"img_{L}.npy", np.concatenate([r[0][L] for r in res]))
    np.save(CACHE / "hand.npy", np.concatenate([r[2] for r in res if len(r[2])]))
    np.save(CACHE / "paths.npy", np.concatenate([r[3] for r in res if len(r[3])]))
    pd.DataFrame(meta, columns=["ts_code", "trade_date"]).to_parquet(CACHE / "meta.parquet", index=False)
    smp.to_parquet(CACHE / "samples.parquet", index=False)
    say(f"rendered images {len(meta)} (of {len(key)} unique stock-days); cache {CACHE}")


# ------------------------------------------------------------------ models
def train_cnn(X: np.ndarray, y: np.ndarray, kind: str, seed: int = 7, epochs: int = 6):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(8)

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    net = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 48), blk(48, 64), nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss() if kind == "cls" else nn.MSELoss()
    n = len(X)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        for g in opt.param_groups:
            g["lr"] = 2e-3 * (0.5 if ep >= epochs - 2 else 1.0)
        perm = rng.permutation(n)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            xb = torch.from_numpy(X[b]).float() / 255.0
            yb = torch.from_numpy(y[b]).float()
            opt.zero_grad()
            loss = loss_fn(net(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
    net.eval()

    def predict(Z: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 256):
                out.append(net(torch.from_numpy(Z[i:i + 256]).float() / 255.0).squeeze(1).numpy())
        return np.concatenate(out)
    return predict


def auc_boot(y: np.ndarray, p: np.ndarray, day: np.ndarray, reps: int = 200) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, p)
    days = np.unique(day)
    idx = {d: np.where(day == d)[0] for d in days}
    rng = np.random.default_rng(3)
    bs = []
    for _ in range(reps):
        pick = rng.choice(days, len(days))
        ii = np.concatenate([idx[d] for d in pick])
        if 0 < y[ii].mean() < 1:
            bs.append(roc_auc_score(y[ii], p[ii]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def daily_ic(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    return df.groupby("trade_date").apply(lambda g: g[a].rank().corr(g[b].rank()) if len(g) >= 10 else np.nan).dropna()


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
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    meta = pd.read_parquet(CACHE / "meta.parquet")
    smp = pd.read_parquet(CACHE / "samples.parquet")
    hand = np.load(CACHE / "hand.npy")
    paths = np.load(CACHE / "paths.npy")
    imgs = {L: np.load(CACHE / f"img_{L}.npy", mmap_mode="r") for L in WINDOWS}
    meta["row"] = np.arange(len(meta))
    say(f"cache: {len(meta)} images; samples by group {smp.groupby('grp').size().to_dict()}")

    # example picture
    try:
        from PIL import Image
        ex = []
        for grp in ("S", "M"):
            rows_ = smp[smp.grp == grp].merge(meta, on=["ts_code", "trade_date"]).head(1).row.tolist() + smp[smp.grp == grp].merge(meta, on=["ts_code", "trade_date"]).iloc[500:501].row.tolist()
            for r in rows_:
                ex.append(np.pad(np.asarray(imgs[100][r]).transpose(1, 2, 0), ((0, 0), (0, 10), (0, 0))))
        Image.fromarray(np.concatenate(ex, axis=1)).resize((len(ex) * 110 * 3, H * 3), Image.NEAREST).save(OUT / "examples.png")
    except Exception as exc:  # noqa: BLE001
        say(f"example image skipped: {exc}")

    # ---------------- Part 1 similarity
    say("\n== Part 1: similarity of the 150-bar paths (no learning) ==")
    full = pd.DataFrame({"ts_code": meta.ts_code, "trade_date": meta.trade_date, "row": meta.row})
    rng = np.random.default_rng(11)
    for grp, label in (("S", "pool S"), ("A", "pool A"), ("V", "V12.31 Top20")):
        for L in WINDOWS:
            zs, raw, base = [], [], []
            days = sorted(smp[smp.grp == grp].trade_date.unique())[::3]
            for d in days:
                g = smp[(smp.grp == grp) & (smp.trade_date == d)].merge(full, on=["ts_code", "trade_date"])
                m = smp[(smp.grp == "M") & (smp.trade_date == d)].merge(full, on=["ts_code", "trade_date"])
                if len(g) < 10 or len(m) < 15:
                    continue
                pool = pd.concat([g, m])
                P = paths[pool.row.to_numpy(), -L:]
                P = (P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9)
                C = (P @ P.T) / L
                ng, nm = len(g), len(m)

                def mean_pair(ix):
                    sub = C[np.ix_(ix, ix)]
                    return (sub.sum() - np.trace(sub)) / (len(ix) * (len(ix) - 1))
                obs = mean_pair(np.arange(ng))
                k = min(ng, nm)
                draws = [mean_pair(ng + rng.choice(nm, k, replace=False)) for _ in range(100)]
                zs.append((obs - np.mean(draws)) / (np.std(draws) + 1e-9))
                raw.append(obs)
                base.append(np.mean(draws))
            if zs:
                say(f"{label:14s} L={L:3d}: mean pairwise corr {np.mean(raw):+.3f} vs random same-day subsets {np.mean(base):+.3f}; mean z {np.mean(zs):+.2f} over {len(zs)} days; days with z>2 {np.mean(np.array(zs) > 2):.0%}")

    # ---------------- Part 2 CNN list vs market
    say("\n== Part 2: picture-only CNN, list vs same-day random market names ==")
    from sklearn.metrics import roc_auc_score  # noqa: F401
    d2 = smp.drop_duplicates(["ts_code", "trade_date", "grp"]).merge(meta, on=["ts_code", "trade_date"])
    tasks = [("pool S vs M", d2[d2.grp.isin(["S", "M"])].assign(y=lambda x: (x.grp == "S").astype(float)), TR_END, TE_START),
             ("R20 family (A+V12.31) vs M", d2[d2.grp.isin(["A", "V", "M"]) & (d2.trade_date >= "20260414")].assign(y=lambda x: x.grp.isin(["A", "V"]).astype(float)), R_TR_END, R_TE_START)]
    res2 = {}
    for name, d, tr_end, te_start in tasks:
        tr = d[d.trade_date <= tr_end]
        te = d[d.trade_date >= te_start]
        say(f"{name}: train rows {len(tr)} (pos {tr.y.mean():.2f}), test rows {len(te)} (pos {te.y.mean():.2f}), test days {te.trade_date.nunique()}")
        hb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=1).fit(hand[tr.row], tr.y)
        ab = auc_boot(te.y.to_numpy(), hb.predict_proba(hand[te.row])[:, 1], te.trade_date.to_numpy())
        say(f"   tabular baseline (12 hand features) AUC {ab[0]:.3f} [{ab[1]:.3f}, {ab[2]:.3f}]")
        for L in WINDOWS:
            Xtr, Xte = np.asarray(imgs[L][tr.row.to_numpy()]), np.asarray(imgs[L][te.row.to_numpy()])
            pr = train_cnn(Xtr, tr.y.to_numpy(), "cls")
            ac = auc_boot(te.y.to_numpy(), pr(Xte), te.trade_date.to_numpy())
            res2[(name, L)] = ac[0]
            say(f"   CNN L={L:3d} AUC {ac[0]:.3f} [{ac[1]:.3f}, {ac[2]:.3f}]")

    # ---------------- Part 3 CNN outcome inside the funnel
    say("\n== Part 3: picture-only CNN vs outcome inside the S20 funnel (top 20 + 20 random other survivors per day) ==")
    d3 = smp[smp.grp.isin(["S", "F"])].drop_duplicates(["ts_code", "trade_date"]).merge(meta, on=["ts_code", "trade_date"])
    d3 = d3[d3.ret_v1.notna()].copy()
    d3["y"] = d3.groupby("trade_date").ret_v1.rank(pct=True)
    tr, te = d3[d3.trade_date <= OUT_TR_END], d3[d3.trade_date >= TE_START].copy()
    say(f"train rows {len(tr)} ({tr.trade_date.nunique()} days), test rows {len(te)} ({te.trade_date.nunique()} days); mean per trade test {te.ret_v1.mean():+.2f}")
    te["natr_neg"] = -hand[te.row.to_numpy(), 3]
    ic = daily_ic(te, "natr_neg", "ret_v1")
    say(f"   baseline low-NATR: daily IC {ic.mean():+.4f} (NW t {tstat(ic):+.1f})")
    hr = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=1).fit(hand[tr.row], tr.y)
    te["hand_pred"] = hr.predict(hand[te.row])
    ic = daily_ic(te, "hand_pred", "ret_v1")
    say(f"   baseline hand-feature boosting: daily IC {ic.mean():+.4f} (NW t {tstat(ic):+.1f})")
    for L in WINDOWS:
        Xtr, Xte = np.asarray(imgs[L][tr.row.to_numpy()]), np.asarray(imgs[L][te.row.to_numpy()])
        pr = train_cnn(Xtr, tr.y.to_numpy(), "reg")
        te[f"cnn{L}"] = pr(Xte)
        ic = daily_ic(te, f"cnn{L}", "ret_v1")
        q = te.groupby("trade_date")[f"cnn{L}"].rank(pct=True, method="first")
        gap = (te[q > 2 / 3].groupby("trade_date").ret_v1.mean() - te[q <= 1 / 3].groupby("trade_date").ret_v1.mean()).dropna()
        say(f"   CNN L={L:3d}: daily IC {ic.mean():+.4f} (NW t {tstat(ic):+.1f}); top third minus bottom third per trade {gap.mean():+.2f} (t {tstat(gap):+.1f}); corr with NATR {te[f'cnn{L}'].corr(-te.natr_neg):+.2f}")
    sh = tr.assign(y=tr.groupby("trade_date").y.transform(lambda v: np.random.default_rng(5).permutation(v.to_numpy())))
    Xtr = np.asarray(imgs[100][sh.row.to_numpy()])
    pr = train_cnn(Xtr, sh.y.to_numpy(), "reg")
    te["plc"] = pr(np.asarray(imgs[100][te.row.to_numpy()]))
    ic = daily_ic(te, "plc", "ret_v1")
    say(f"   placebo (L=100, within-day shuffled targets): daily IC {ic.mean():+.4f} (NW t {tstat(ic):+.1f})")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
