#!/usr/bin/env python
"""Cross filter v2: fixed list length, equal exposure, confirm window scored once.

Redesign of the 2026-10-04 R5/S5 cross-filter study. Research only: nothing
here writes production weights or changes a published list.

What changed against experiment_r5_s5_cross_filter.py:
  * Features run past 2026-01-26. The stock-level factors for 2026-01-27 ..
    2026-09-30 come from factor_groups_extension (*_ext_rebuild) and are
    written once to feat_ext_20260127_20260930.parquet in the output folder.
  * The 5-day heads are trained walk-forward on the high-volatility part of
    the sampled universe since 2024 (daily atr_pct percentile >= 0.70, where
    the lists live), not on 79 days of the top-200 pool.
  * Every rule keeps a fixed number of names and the money is split equally
    among them, so holding fewer names is no longer scored as skill.
  * A no-model volatility veto is the baseline every model rule has to beat.
  * The rule is chosen on the development days (2025-03-03 .. 2026-01-26) by
    code, then the confirmation days (2026-01-27 .. 2026-08-05) are scored.
    Signal days from 2026-08-06 on stay reserved and are never read here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_r5_s5_cross_filter import add_events, auc  # noqa: E402

EXP = ROOT / "output/experiments/s20_pure_20260928"
OUT = ROOT / "output/experiments/cross_filter_v2_20261004"
DEV_CACHE = EXP / "feat_cache_bps5000.parquet"
EXT_DIR = ROOT / "output/factor_lab_3y/factor_groups_extension"
EXT_CACHE = OUT / "feat_ext_20260127_20260930.parquet"
DEV_PRED = ROOT / "output/experiments/s20_20r_residual_portable/predictions.parquet"
CONF_PRED = ROOT / "output/experiments/s20_20r_confirmation/run_v1/predictions.parquet"
RESERVED_FROM = "20260806"
TRAIN_ATR_PCT = 0.70
POOL = 200
SEED = 20261004

# name, last fit day, early-stop window, scored window. Labels look 5 sessions
# ahead of the next open; every early-stop window's last label ends before its
# scored window starts.
FOLDS = [
    ("wf1", "20241220", "20250102", "20250221", "20250303", "20250430"),
    ("wf2", "20250430", "20250513", "20250620", "20250701", "20250829"),
    ("wf3", "20250829", "20250908", "20251024", "20251103", "20260126"),
    ("confirm", "20251128", "20251208", "20260116", "20260127", "20260805"),
]


def stock_feats() -> list[str]:
    import pyarrow.parquet as pq

    feats = list(pd.read_csv(EXP / "feat_cols.csv")["f"])
    ext_cols = set(pq.ParquetFile(next(iter(sorted(EXT_DIR.glob("*_ext_rebuild.parquet"))))).schema.names)
    return [f for f in feats if f in ext_cols]


# The extension store cannot recompute these from daily bars; it repeats each
# stock's 2026-01-26 value on every later day. The heads must not train on them.
STALE_IN_EXT = {"adx"}


def build_ext(feats: list[str]) -> None:
    if EXT_CACHE.exists():
        return
    parts = []
    for path in sorted(EXT_DIR.glob("*_ext_rebuild.parquet")):
        g = pd.read_parquet(path, columns=["ts_code", "trade_date", *feats])
        g["ts_code"] = g["ts_code"].astype(str)
        g["trade_date"] = g["trade_date"].astype(str)
        g = g[g.ts_code.str.endswith((".SH", ".SZ"))]
        g[feats] = g[feats].astype(np.float32)
        parts.append(g)
    ext = pd.concat(parts, ignore_index=True)
    if ext.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate stock-date keys in the extension store")
    ext.to_parquet(EXT_CACHE, index=False)
    print(f"wrote {EXT_CACHE.name}: rows {len(ext):,} days {ext.trade_date.nunique()} "
          f"{ext.trade_date.min()}..{ext.trade_date.max()} feats {len(feats)}", flush=True)


def load_paths() -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "ret20", "up10_day", "up15_day", "dn5_day", "dn10_day"]
    path = pd.read_parquet(EXP / "path_panel.parquet", columns=cols)
    path["trade_date"] = path["trade_date"].astype(str)
    path["ts_code"] = path["ts_code"].astype(str)
    path = path[path.trade_date < RESERVED_FROM]
    return path.dropna(subset=cols[2:])


def load_ranks() -> pd.DataFrame:
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
    return rank[(rank.s_rank <= POOL) | (rank.r_rank <= POOL)].reset_index(drop=True)


def fit_head(fit: pd.DataFrame, tune: pd.DataFrame, feats: list[str], label: str):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, n_jobs=8, verbose=-1,
    )
    model.fit(
        fit[feats], fit[label], eval_set=[(tune[feats], tune[label])], eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def score_candidates(feats: list[str], paths: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    cache = OUT / "scored_candidates.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    dev = pd.read_parquet(DEV_CACHE, columns=["ts_code", "trade_date", *feats])
    dev["trade_date"] = dev["trade_date"].astype(str)
    dev[feats] = dev[feats].astype(np.float32)
    ext = pd.read_parquet(EXT_CACHE)
    ext = ext[ext.trade_date < RESERVED_FROM]

    cand = ranks.merge(pd.concat([dev, ext], ignore_index=True), on=["ts_code", "trade_date"], how="inner")
    print(f"candidates with features {len(cand):,} of {len(ranks):,}", flush=True)

    lab = paths[["ts_code", "trade_date", "up10_day", "dn5_day"]]
    train = dev.merge(lab, on=["ts_code", "trade_date"], how="inner")
    del dev, ext
    train["y_dn5"] = ((train.dn5_day > 0) & (train.dn5_day <= 5)).astype(np.int8)
    train["y_up"] = ((train.up10_day > 0) & (train.up10_day <= 5)).astype(np.int8)
    train = train[train.groupby("trade_date").atr_pct.rank(pct=True) >= TRAIN_ATR_PCT]
    print(f"training rows in the high-volatility slice {len(train):,}", flush=True)

    cand["p_dn5"] = np.nan
    cand["p_up"] = np.nan
    feats = [f for f in feats if f not in STALE_IN_EXT]
    for name, fit_end, tune_lo, tune_hi, test_lo, test_hi in FOLDS:
        fit = train[train.trade_date <= fit_end]
        tune = train[(train.trade_date >= tune_lo) & (train.trade_date <= tune_hi)]
        sel = (cand.trade_date >= test_lo) & (cand.trade_date <= test_hi)
        for label, col in (("y_dn5", "p_dn5"), ("y_up", "p_up")):
            model = fit_head(fit, tune, feats, label)
            cand.loc[sel, col] = model.predict_proba(cand.loc[sel, feats])[:, 1]
            p_tune = model.predict_proba(tune[feats])[:, 1]
            print(f"{name} {label}: fit {len(fit):,} base {fit[label].mean():.3f} trees {model.best_iteration_} "
                  f"tune AUC {auc(tune[label], p_tune):.3f} atr-only {auc(tune[label], tune.atr_pct):.3f}", flush=True)
    keep = ["ts_code", "trade_date", "period", "s_rank", "r_rank", "atr_pct", "amplitude", "p_dn5", "p_up"]
    cand = cand[keep].dropna(subset=["p_dn5", "p_up"]).reset_index(drop=True)
    cand.to_parquet(cache, index=False)
    return cand


# rule name -> (candidates taken by rank, names kept, how they are ordered)
RULES = {
    "base20": (20, 20, "rank"),
    "rank10": (20, 10, "rank"),
    "atr10": (20, 10, "atr"),
    "dn10": (20, 10, "dn"),
    "up10": (20, 10, "up"),
    "cross_model10": (20, 10, "up-dn"),
    "cross_rank_atr10": (20, 10, "rank+atr"),
    "cross_rank_dn10": (20, 10, "rank+dn"),
    "wide_atr20": (40, 20, "atr"),
    "wide_dn20": (40, 20, "dn"),
    "wide_cross_model20": (40, 20, "up-dn"),
    "wide_rank_atr20": (40, 20, "rank+atr"),
}
PRIMARY = "atr10"  # written down from the first-round diagnostic before this script existed


def order_key(g: pd.DataFrame, rank_col: str, how: str) -> pd.Series:
    """Lower is better."""
    pct = lambda s: s.rank(pct=True)  # noqa: E731
    if how == "rank":
        return g[rank_col]
    if how == "atr":
        return g["atr_pct"]
    if how == "dn":
        return g["p_dn5"]
    if how == "up":
        return -g["p_up"]
    if how == "up-dn":
        return pct(g["p_dn5"]) - pct(g["p_up"])
    if how == "rank+atr":
        return pct(g[rank_col]) + pct(g["atr_pct"])
    if how == "rank+dn":
        return pct(g[rank_col]) + pct(g["p_dn5"])
    raise ValueError(how)


def pick(df: pd.DataFrame, rank_col: str, rule: str) -> pd.DataFrame:
    cap, k, how = RULES[rule]
    parts = []
    for _, g in df[df[rank_col] <= cap].groupby("trade_date", sort=True):
        parts.append(g.assign(_o=order_key(g, rank_col, how)).sort_values(["_o", rank_col]).head(k))
    return pd.concat(parts, ignore_index=True)


def nw_se(x: np.ndarray, lag: int = 19) -> float:
    """Newey-West standard error of the mean; holds overlap for 20 sessions."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = float(e @ e) / n
    for j in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - j / (lag + 1)) * float(e[j:] @ e[:-j]) / n
    return float(np.sqrt(max(var, 0.0) / n))


def stats(picked: pd.DataFrame, base_day: pd.Series | None) -> dict:
    day = picked.groupby("trade_date").ret_v1.mean()
    out = {
        "days": int(len(day)),
        "avg_len": round(len(picked) / len(day), 1),
        "ret_v1": round(float(day.mean()), 2),
        "pure_up": round(100 * float(picked.pure_up.mean()), 1),
        "pure_down": round(100 * float(picked.pure_down.mean()), 1),
        "hit15": round(100 * float(picked.hit15.mean()), 1),
        "band_ok": round(100 * float(picked.cls_a5_d10.isin([1, 3]).mean()), 1),
        "ret_band": round(float(picked.groupby("trade_date").ret_a5_b15_d10.mean().mean()), 2),
    }
    if base_day is not None:
        diff = (day - base_day.reindex(day.index)).dropna()
        se = nw_se(diff.to_numpy())
        month = diff.groupby(diff.index.str[:6]).mean()
        out.update(
            diff=round(float(diff.mean()), 2), nw_se=round(se, 2),
            t=round(float(diff.mean() / se), 2) if se > 0 else None,
            months_up=f"{int((month > 0).sum())}/{len(month)}",
        )
    return out


def evaluate(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    part = frame[frame.period == period]
    rows = []
    for pool, rank_col in (("S", "s_rank"), ("R", "r_rank")):
        base_day = pick(part, rank_col, "base20").groupby("trade_date").ret_v1.mean()
        for rule in RULES:
            st = stats(pick(part, rank_col, rule), None if rule == "base20" else base_day)
            rows.append({"period": period, "pool": pool, "rule": rule, **st})
    return pd.DataFrame(rows)


def in_list_auc(frame: pd.DataFrame, period: str) -> list[str]:
    part = frame[frame.period == period]
    lines = []
    for pool, rank_col in (("S", "s_rank"), ("R", "r_rank")):
        top = part[part[rank_col] <= 20]
        lines.append(
            f"{period} {pool} top20 n={len(top)}: "
            f"dn5 label AUC model {auc(top.y_dn5, top.p_dn5):.3f} atr {auc(top.y_dn5, top.atr_pct):.3f} | "
            f"+10% label AUC model {auc(top.y_up, top.p_up):.3f} atr {auc(top.y_up, top.atr_pct):.3f} | "
            f"pure-down AUC p_dn5 {auc(top.pure_down, top.p_dn5):.3f} atr {auc(top.pure_down, top.atr_pct):.3f} "
            f"rank {auc(top.pure_down, top[rank_col]):.3f} | "
            f"base dn5 {top.y_dn5.mean():.3f} up {top.y_up.mean():.3f}"
        )
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = stock_feats()
    build_ext(feats)
    paths = load_paths()
    ranks = load_ranks()
    cand = score_candidates(feats, paths, ranks)

    band = pd.read_parquet(EXP / "band_panel.parquet", columns=["ts_code", "trade_date", "cls_a5_d10", "ret_a5_b15_d10"])
    band["trade_date"] = band["trade_date"].astype(str)
    band["ts_code"] = band["ts_code"].astype(str)
    frame = add_events(cand.merge(paths, on=["ts_code", "trade_date"], how="inner"))
    frame = frame.merge(band, on=["ts_code", "trade_date"], how="inner")
    print(f"scored rows with outcomes {len(frame):,}; days dev {frame[frame.period == 'dev'].trade_date.nunique()} "
          f"confirm {frame[frame.period == 'confirm'].trade_date.nunique()}", flush=True)

    dev = evaluate(frame, "dev")
    lines = [
        "Cross filter v2. Every rule keeps a fixed number of names; ret_v1 is the day mean of the",
        "+15%/-10% exit return with the money split equally among kept names. diff = rule minus the",
        "unfiltered top 20 on the same day; nw_se allows 20 sessions of overlap.",
        f"Dev window is a 50% stock sample; confirm window is the full market. Reserved from {RESERVED_FROM}.",
        "",
        *in_list_auc(frame, "dev"),
        "",
        "== development 2025-03-03 .. 2026-01-26 ==",
        dev.drop(columns="period").to_string(index=False),
        "",
    ]
    # One challenger per pool, chosen on dev only, among same-exposure rules.
    chosen = {}
    for pool in ("S", "R"):
        d = dev[(dev.pool == pool) & (dev.rule != "base20")].sort_values(["ret_v1", "pure_down"], ascending=[False, True])
        chosen[pool] = d.iloc[0]["rule"]
        lines.append(f"dev choice for {pool}: {chosen[pool]} (primary written down in advance: {PRIMARY})")
    (OUT / "dev_choice.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    conf = evaluate(frame, "confirm")
    lines += [
        "",
        *in_list_auc(frame, "confirm"),
        "",
        "== confirmation 2026-01-27 .. 2026-08-05 (scored after the dev choice) ==",
        conf.drop(columns="period").to_string(index=False),
        "",
    ]
    for pool in ("S", "R"):
        for tag, rule in (("primary", PRIMARY), ("dev choice", chosen[pool])):
            r = conf[(conf.pool == pool) & (conf.rule == rule)].iloc[0]
            b = conf[(conf.pool == pool) & (conf.rule == "base20")].iloc[0]
            lines.append(
                f"confirm {pool} {tag} {rule}: ret {r.ret_v1} vs base {b.ret_v1} (diff {r['diff']} se {r.nw_se} t {r.t}, "
                f"months up {r.months_up}); pure_down {r.pure_down} vs {b.pure_down}; pure_up {r.pure_up} vs {b.pure_up}; "
                f"band_ok {r.band_ok} vs {b.band_ok}"
            )
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    pd.concat([dev, conf], ignore_index=True).to_csv(OUT / "grid.csv", index=False)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
