#!/usr/bin/env python
"""Start-up v6: read the phase of a move from the path already walked after entry (hold / add / reduce).

Follows wiki/2026-10-07_pump-v5-clean-start.md: whether the next five sessions will be a clean move was
hardly visible on the signal day (v5, v5b). v6 asks the question again at checkpoints k = 1, 2, 3
sessions after the entry (entry = next open after the signal day, checkpoint = close of session k), with
the path so far as extra information. The decisive comparison is v6 against the same model given only
the signal-day context ("k0"): if the walked path does not raise the AUC, v6 adds nothing.

Rows: every stock, every signal day, k in {1,2,3}. Split-adjusted prices (scripts/price_panel.py).
Features (all known at the checkpoint close):
  path   ret_k and ret_k/ATR, max rise and max dip since entry in ATR units, path efficiency since entry,
         last session's move in ATR units, up sessions so far, opening gap in ATR units,
         volume since entry / 20-session average before the signal day, k
  context (signal day) NATR, close/MA20, close/MA120, MA120 slope (20 sessions), RSI14, 5- and 20-session return
Targets: method C clean start-up / start-down over the 5 sessions after the checkpoint close.
Windows: train checkpoints whose label horizon ends by 2025-09-30 (sample 25% of stock-days);
         valid 2025-10..2026-01 (reported only; trees fixed at 300, 15 leaves, 2000 rows/leaf);
         test signal days 2026-01-27..2026-07-24 (labels end by 2026-08-05).
S20 use: frozen offensive lists in the test window. For a name still held at checkpoint k (no +15% or -10%
touch on sessions 1..k), hold value = final S20 exit return - return from exiting at the checkpoint close.
Rank the held names of a day by v6 (P(down) - P(up)) and compare with the S0001-style baseline (ret_k).
Output: output/experiments/pump_v6_20261007/{up.txt, down.txt, k0_up.txt, k0_down.txt, report.txt, s20_checkpoints.parquet}
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from price_panel import adjusted_daily, wilder_rsi  # noqa: E402
from stockagent_analysis.pump_labels import clean_start_labels  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/pump_v6_20261007"
START, END = "20240102", "20260805"
TRAIN_END, VALID_START, VALID_END = "20250930", "20251001", "20260126"
TEST_START, TEST_LAST_SIGNAL = "20260127", "20260724"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbosity=-1, seed=20261007, num_threads=8)
ROUNDS = 300
CONTEXT = ["natr", "dist20", "dist120", "ma120_slope", "rsi", "past5", "past20"]
PATH = ["k", "ret_k", "ret_k_atr", "max_up_atr", "max_dn_atr", "eff_k", "last_atr", "up_sessions", "gap_atr", "vol_ratio"]


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = adjusted_daily(START, END)
    g = px.groupby("ts_code", sort=False)
    tr = np.maximum(px.high - px.low, np.maximum((px.high - px.pre_close).abs(), (px.low - px.pre_close).abs()))
    atr = tr.groupby(px.ts_code, sort=False).transform(lambda s: s.rolling(14, min_periods=10).mean())
    ma20 = g.close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    ma120 = g.close.transform(lambda s: s.rolling(120, min_periods=120).mean())
    ctx = pd.DataFrame({"natr": atr / px.close, "dist20": px.close / ma20 - 1, "dist120": px.close / ma120 - 1,
                        "ma120_slope": ma120 / ma120.groupby(px.ts_code, sort=False).shift(20) - 1,
                        "rsi": wilder_rsi(px.close, px.ts_code), "past5": g.close.pct_change(5), "past20": g.close.pct_change(20)})
    lab = clean_start_labels(px[["ts_code", "trade_date", "high", "low", "close", "pre_close"]])
    lab = lab.set_index(["ts_code", "trade_date"]).label
    entry = g.open.shift(-1)
    vol20 = g.vol.transform(lambda s: s.rolling(20, min_periods=10).mean())
    # S20 exit from the entry: first touch of +15% / -10% on sessions 1..20, else close of session 20; cost 0.3
    up_day = pd.Series(0, index=px.index)
    dn_day = pd.Series(0, index=px.index)
    for j in range(20, 0, -1):
        up_day = up_day.mask(g.high.shift(-j) >= entry * 1.15, j)
        dn_day = dn_day.mask(g.low.shift(-j) <= entry * 0.90, j)
    ret20 = (g.close.shift(-20) / entry - 1) * 100
    final = np.where((dn_day > 0) & ((up_day == 0) | (dn_day <= up_day)), -10.0,
                     np.where(up_day > 0, 15.0, ret20)) - 0.3
    final = pd.Series(final, index=px.index).where(g.close.shift(-20).notna())
    keep_rows = (px.trade_date >= "20240715").to_numpy()   # MA120 needs 120 sessions of history
    base = px[["ts_code", "trade_date"]].join(ctx)
    base["atr"] = atr
    rows = []
    closes = [g.close.shift(-j) for j in range(1, 4)]
    highs = [g.high.shift(-j) for j in range(1, 4)]
    lows = [g.low.shift(-j) for j in range(1, 4)]
    vols = [g.vol.shift(-j) for j in range(1, 4)]
    dates = [g.trade_date.shift(-j) for j in range(1, 4)]
    for k in (1, 2, 3):
        r = base.copy()
        r["k"] = k
        ck = closes[k - 1]
        r["ret_k"] = ck / entry - 1
        r["ret_k_atr"] = (ck - entry) / atr
        r["max_up_atr"] = (pd.concat(highs[:k], axis=1).max(axis=1) - entry) / atr
        r["max_dn_atr"] = (pd.concat(lows[:k], axis=1).min(axis=1) - entry) / atr
        path = (closes[0] - entry).abs() + sum((closes[j] - closes[j - 1]).abs() for j in range(1, k))
        r["eff_k"] = (ck - entry) / path.replace(0, np.nan)
        prev = entry if k == 1 else closes[k - 2]
        r["last_atr"] = (ck - prev) / atr
        r["up_sessions"] = sum((closes[j] > (entry if j == 0 else closes[j - 1])).astype(int) for j in range(k))
        r["gap_atr"] = (entry - px.close) / atr
        r["vol_ratio"] = pd.concat(vols[:k], axis=1).mean(axis=1) / vol20
        r["ck_date"] = dates[k - 1]
        r["alive"] = ((up_day == 0) | (up_day > k)) & ((dn_day == 0) | (dn_day > k))
        r["hold_value"] = final - ((ck / entry - 1) * 100 - 0.3)
        r["label"] = lab.reindex(pd.MultiIndex.from_arrays([px.ts_code, dates[k - 1]])).to_numpy()
        r = r[keep_rows]
        num = r.select_dtypes("float64").columns
        r[num] = r[num].astype(np.float32)
        rows.append(r)
    data = pd.concat(rows, ignore_index=True)
    data = data[data.label.notna() & data.dist120.notna() & data.ret_k.notna()].reset_index(drop=True)
    return data, px


def daily_auc(frame: pd.DataFrame, y: str, s: str) -> float:
    vals = [roc_auc_score(g[y], g[s]) for _, g in frame.groupby("trade_date") if g[y].nunique() == 2]
    return float(np.mean(vals))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = build()
    data["yu"], data["yd"] = (data.label == 2).astype(int), (data.label == 1).astype(int)
    rng = np.random.default_rng(20261007)
    keep = pd.Series(rng.random(data[["ts_code", "trade_date"]].drop_duplicates().shape[0]) < 0.25)
    sd = data[["ts_code", "trade_date"]].drop_duplicates().assign(s=keep.to_numpy())
    data = data.merge(sd, on=["ts_code", "trade_date"])
    train = data[(data.ck_date <= TRAIN_END) & data.s & (data.trade_date >= "20240715")]
    train = train[train.ck_date.notna()]
    train = train[train.trade_date <= "20250915"]      # label horizon (checkpoint + 5 sessions) ends before 2025-09-30
    valid = data[(data.trade_date >= VALID_START) & (data.trade_date <= "20260109") & data.s]
    test = data[(data.trade_date >= TEST_START) & (data.trade_date <= TEST_LAST_SIGNAL)].copy()
    valid = valid.copy()
    lines = [f"rows train {len(train):,} valid {len(valid):,} test {len(test):,}; test signal days {test.trade_date.nunique()}",
             f"test class shares: clean-up {test.yu.mean():.3f} clean-down {test.yd.mean():.3f}", ""]
    models = {}
    for name, feats in (("v6", CONTEXT + PATH), ("k0", CONTEXT)):
        for side, y in (("up", "yu"), ("down", "yd")):
            b = lgb.train(PARAMS, lgb.Dataset(train[feats], label=train[y]), num_boost_round=ROUNDS)
            b.save_model(str(OUT / f"{name}_{side}.txt"))
            models[(name, side)] = (b, feats)
    for name, side in models:
        b, feats = models[(name, side)]
        test.loc[:, f"{name}_{side}"] = b.predict(test[feats])
        valid.loc[:, f"{name}_{side}"] = b.predict(valid[feats])
    lines.append("== daily AUC by checkpoint (v6 = path + context, k0 = signal-day context only) ==")
    for part_name, part in (("valid", valid), ("test", test)):
        for k in (1, 2, 3):
            x = part[part.k == k]
            lines.append(f"[{part_name} k={k}] clean-up: v6 {daily_auc(x, 'yu', 'v6_up'):.3f} k0 {daily_auc(x, 'yu', 'k0_up'):.3f} ret_k {daily_auc(x, 'yu', 'ret_k'):.3f} "
                         f"| clean-down: v6 {daily_auc(x, 'yd', 'v6_down'):.3f} k0 {daily_auc(x, 'yd', 'k0_down'):.3f} -ret_k {daily_auc(x.assign(n=-x.ret_k), 'yd', 'n'):.3f}")
    test["month"] = test.trade_date.str[:6]
    lines.append("test by month (k=2): v6 down " + " ".join(f"{m}:{daily_auc(g, 'yd', 'v6_down'):.2f}" for m, g in test[test.k == 2].groupby("month"))
                 + " | v6 up " + " ".join(f"{m}:{daily_auc(g, 'yu', 'v6_up'):.2f}" for m, g in test[test.k == 2].groupby("month")))
    for side in ("up", "down"):
        b, feats = models[("v6", side)]
        gain = pd.Series(b.feature_importance("gain"), index=feats)
        lines.append(f"v6 {side} gain: " + ", ".join(f"{k} {100 * v / gain.sum():.1f}%" for k, v in gain.sort_values(ascending=False).head(8).items()))

    # ---- S20 frozen lists: held names at each checkpoint
    frozen = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    frozen = frozen[(frozen.trade_date >= TEST_START) & (frozen.trade_date <= TEST_LAST_SIGNAL)]
    lists = select(frozen, PureConfig(), "U15D10")[["ts_code", "trade_date"]]
    s = test.merge(lists, on=["ts_code", "trade_date"])
    s = s[s.alive & s.hold_value.notna()].copy()
    s["score"] = s.v6_down - s.v6_up
    s["seg"] = np.where(s.trade_date <= "20260522", "early", "late")
    s.to_parquet(OUT / "s20_checkpoints.parquet", index=False)
    lines += ["", f"== S20 frozen lists, names still held at the checkpoint: hold value = final exit - exit now (points) =="]
    for k in (1, 2, 3):
        x = s[s.k == k].copy()
        lines.append(f"k={k}: held {len(x)} names, {x.trade_date.nunique()} days; mean hold value {x.hold_value.mean():+.2f}")
        for col, name in (("score", "v6 P(down)-P(up)"), ("ret_k", "ret so far (S0001-style, low = worse)")):
            sign = 1 if col == "score" else -1
            x["t"] = x.groupby("trade_date")[col].transform(lambda v: pd.qcut((sign * v).rank(method="first"), 3, labels=False) if len(v) >= 6 else np.nan)
            terc = x.groupby("t").hold_value.agg(["size", "mean"])
            hi = x[x.t == 2].groupby("trade_date").hold_value.mean()
            lo = x[x.t == 0].groupby("trade_date").hold_value.mean()
            d = (hi - lo).dropna()
            lines.append(f"    {name:40s} worst third {terc['mean'].get(2, np.nan):+5.2f} | middle {terc['mean'].get(1, np.nan):+5.2f} | best third {terc['mean'].get(0, np.nan):+5.2f} "
                         f"| worst-best same day {d.mean():+.2f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)})")
            for seg in ("early", "late"):
                xs = x[x.seg == seg]
                d2 = (xs[xs.t == 2].groupby("trade_date").hold_value.mean() - xs[xs.t == 0].groupby("trade_date").hold_value.mean()).dropna()
                lines.append(f"        [{seg}] worst-best {d2.mean():+.2f} (t {d2.mean() / nw_se(d2.to_numpy()):+.2f}, days {len(d2)})")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
