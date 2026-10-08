#!/usr/bin/env python
"""R5/S5 upside screen, then the S20 head, on the same clock as the break study.

The production R5 regresses a 5-day close return and its scores sit near +1
percentage point, so a literal "predicted rise > 10%" cut is empty. The models
fit here are research-only classifiers of a different event: the high versus
the next open touches +10% inside 5 sessions. Nothing is written to
output/production.

Clock, matching scripts/experiment_r5_s5_break_filter.py:
  train       signal dates < 2025-08-21
  early-stop  2025-08-21 .. 2025-11-24
  test        >= 2025-11-24, through the feature overlap end 2026-01-26

S5 is fit on the daily stage-1 top 200. R5 is fit on the daily top 200 by
the nested R20 score. S20's screen is the existing stage-1 rank (1 = best),
not a newly trained 20-day model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.s20_pure import PureConfig, exit_return, three_state  # noqa: E402

EXP = ROOT / "output/experiments/s20_pure_20260928"
OUT = ROOT / "output/experiments/r5_s5_up10_20261004"
FEAT = EXP / "feat_cache_bps5000.parquet"
KEEP_FEATS = [
    "adx", "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
    "ma5_ma20", "ma20_ma60", "macd_hist", "rsi_6", "rsi_14", "rsi_24",
    "kdj_k", "kdj_d", "boll_pct", "boll_width", "atr_pct", "bias_5",
    "bias_10", "bias_20", "roc_10", "roc_20", "cci_14", "wr_14", "mfi_14",
    "trix", "vol_ratio_5", "vol_ratio_20", "amplitude", "amount_ratio_20",
]


def auc(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def add_events(df: pd.DataFrame) -> pd.DataFrame:
    up10 = df["up10_day"].to_numpy(dtype=int)
    up15 = df["up15_day"].to_numpy(dtype=int)
    up20 = df["up20_day"].to_numpy(dtype=int)
    dn5 = df["dn5_day"].to_numpy(dtype=int)
    dn10 = df["dn10_day"].to_numpy(dtype=int)
    out = df.copy()
    out["y10"] = ((up10 > 0) & (up10 <= 5)).astype(np.int8)
    out["hit15"] = ((up15 > 0) & (up15 <= 20)).astype(np.int8)
    out["spike15"] = ((up15 > 0) & (up15 <= 5)).astype(np.int8)
    out["later15"] = ((up15 > 5) & (up15 <= 20)).astype(np.int8)
    g = np.zeros(len(out), dtype=np.float32)
    for level, day in ((10, up10), (15, up15), (20, up20)):
        g = np.where((day > 0) & (day <= 5), level, g)
    out["g5"] = g
    st = three_state(up15, dn10, dn5)
    out["pure_up"] = (st == "pure_up").astype(np.int8)
    out["pure_down"] = (st == "pure_down").astype(np.int8)
    rule = PureConfig().rule("U15D10")
    out["ret_v1"] = exit_return(up15, dn10, out["ret20"].to_numpy(dtype=float), rule)
    return out


def summarize(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"n": 0, "days": 0, "avg_len": 0.0}
    per = df.groupby("trade_date").size()
    y10 = df["y10"] == 1
    hit_given = df.loc[y10, "hit15"]
    pure_given = df.loc[y10, "pure_up"]
    later_given = df.loc[y10, "later15"]
    spike_given = df.loc[y10, "spike15"]

    def pct(s) -> float | None:
        return None if len(s) == 0 else round(100 * float(s.mean()), 1)

    return {
        "n": int(len(df)),
        "days": int(df.trade_date.nunique()),
        "avg_len": round(float(per.mean()), 2),
        "up10_5": round(100 * float(df.y10.mean()), 1),
        "hit15": round(100 * float(df.hit15.mean()), 1),
        "pure_up": round(100 * float(df.pure_up.mean()), 1),
        "pure_down": round(100 * float(df.pure_down.mean()), 1),
        "both": round(100 * float((df.y10 * df.hit15).mean()), 1),
        "later15": round(100 * float(df.later15.mean()), 1),
        "ret_v1": round(float(df.ret_v1.mean()), 2),
        "hit15_if_up10": pct(hit_given),
        "pure_if_up10": pct(pure_given),
        "later15_if_up10": pct(later_given),
        "spike15_if_up10": pct(spike_given),
        "n_up10": int(y10.sum()),
    }


def top_by(df: pd.DataFrame, col: str, k: int, ascending: bool) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.sort_values(["trade_date", col], ascending=[True, ascending])
        .groupby("trade_date", sort=False)
        .head(k)
    )


def fit_cls(train: pd.DataFrame, valid: pd.DataFrame, label: str):
    import lightgbm as lgb

    cols = [c for c in KEEP_FEATS if c in train.columns]
    y = train[label].to_numpy(dtype=float)
    ok = np.isfinite(y)
    pos = max(float(y[ok].sum()), 1.0)
    neg = max(float(ok.sum() - y[ok].sum()), 1.0)
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=neg / pos,
        random_state=20261004, n_jobs=4, verbose=-1,
    )
    model.fit(
        train.loc[ok, cols], y[ok].astype(int),
        eval_set=[(valid[cols], valid[label].astype(float))],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model, cols


def fit_reg(train: pd.DataFrame, valid: pd.DataFrame, label: str):
    import lightgbm as lgb

    cols = [c for c in KEEP_FEATS if c in train.columns]
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=20261004, n_jobs=4, verbose=-1,
    )
    model.fit(
        train[cols], train[label].to_numpy(dtype=float),
        eval_set=[(valid[cols], valid[label].to_numpy(dtype=float))],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model, cols


def load_model_frame() -> pd.DataFrame:
    rank = pd.read_parquet(
        EXP / "r04_preds_U15_D10.parquet",
        columns=[
            "ts_code", "trade_date", "stage1_score", "s1_rank", "ret20",
            "up10_day", "up15_day", "up20_day", "dn5_day", "dn10_day",
        ],
    )
    r20 = pd.read_parquet(
        ROOT / "output/experiments/s20_20r_residual_portable/predictions.parquet",
        columns=["ts_code", "trade_date", "r20_nested_anchor"],
    )
    for df in (rank, r20):
        df["trade_date"] = df["trade_date"].astype(str)
        df["ts_code"] = df["ts_code"].astype(str)
    m = rank.merge(r20, on=["ts_code", "trade_date"], how="inner")
    m["s_rank"] = m["s1_rank"].astype(int)
    m["r_rank"] = (
        m.groupby("trade_date")["r20_nested_anchor"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    use = m[(m.s_rank <= 200) | (m.r_rank <= 200)].copy()
    feats = pd.read_parquet(FEAT, columns=["ts_code", "trade_date", *KEEP_FEATS])
    feats["trade_date"] = feats["trade_date"].astype(str)
    feats["ts_code"] = feats["ts_code"].astype(str)
    use = use.merge(feats, on=["ts_code", "trade_date"], how="left")
    return add_events(use)


def wider_oracle() -> list[dict]:
    """Path conditional on the stage-1 book, including confirmation days.

    No model is applied. rank in the forward file is the stage-1 order.
    """
    scored = pd.read_parquet(
        ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
        columns=["ts_code", "trade_date", "period", "rank", "stage1_score"],
    )
    scored["trade_date"] = scored["trade_date"].astype(str)
    scored["ts_code"] = scored["ts_code"].astype(str)
    dates = set(scored.trade_date)
    path = pd.read_parquet(
        EXP / "path_panel.parquet",
        columns=["ts_code", "trade_date", "ret20", "up10_day", "up15_day", "up20_day", "dn5_day", "dn10_day"],
    )
    path["trade_date"] = path["trade_date"].astype(str)
    path["ts_code"] = path["ts_code"].astype(str)
    path = path[path.trade_date.isin(dates)]
    m = scored.merge(path, on=["ts_code", "trade_date"], how="inner")
    m = add_events(m)
    rows = []
    head = m[m["rank"] <= 20]
    for period, g in list(head.groupby("period")) + [("all", head)]:
        stat = summarize(g)
        stat.update(scope=f"stage1_top20:{period}", rule="unfiltered")
        rows.append(stat)
        kept = g[g.y10 == 1]
        stat2 = summarize(kept)
        stat2.update(scope=f"stage1_top20:{period}", rule="oracle_up10_in_5d")
        rows.append(stat2)
    print(
        f"wider oracle days {m.trade_date.min()}..{m.trade_date.max()} "
        f"rows {len(m):,} periods {sorted(m.period.unique())}",
        flush=True,
    )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_model_frame()
    days = sorted(frame.trade_date.unique())
    vcut = days[int(len(days) * 0.55)]
    cut = days[int(len(days) * 0.70)]
    print(
        f"model rows {len(frame):,} days {days[0]}..{days[-1]} n_days {len(days)} "
        f"train<{vcut} valid {vcut}..{cut} test>={cut}",
        flush=True,
    )
    print(
        f"pool y10 rate {frame.y10.mean():.3f}  hit15 {frame.hit15.mean():.3f}  "
        f"pure_up {frame.pure_up.mean():.3f}",
        flush=True,
    )

    test = frame[frame.trade_date >= cut].copy()
    scores = {}
    pred_notes = []
    for name, rank_col in (("S5", "s_rank"), ("R5", "r_rank")):
        pool = frame[frame[rank_col] <= 200]
        tr = pool[pool.trade_date < vcut]
        va = pool[(pool.trade_date >= vcut) & (pool.trade_date < cut)]
        te = test[test[rank_col] <= 200]
        print(f"fit {name} class train {len(tr)} pos {tr.y10.mean():.3f}", flush=True)
        cls, cols = fit_cls(tr, va, "y10")
        p_te = cls.predict_proba(te[cols])[:, 1]
        p_va = cls.predict_proba(va[cols])[:, 1]
        test.loc[te.index, f"p_{name}"] = p_te
        print(
            f"  {name} test AUC {auc(te.y10, p_te):.3f} valid AUC {auc(va.y10, p_va):.3f} "
            f"base {te.y10.mean():.3f} pred mean {p_te.mean():.3f} "
            f"p50 {np.median(p_te):.3f} p90 {np.quantile(p_te, 0.9):.3f}",
            flush=True,
        )
        reg, rcols = fit_reg(tr, va, "g5")
        g_te = reg.predict(te[rcols])
        test.loc[te.index, f"g_{name}"] = g_te
        pred_notes.append(
            f"{name} barrier-floor pred on test: "
            f"p50 {np.median(g_te):.2f} p90 {np.quantile(g_te, 0.90):.2f} "
            f"p99 {np.quantile(g_te, 0.99):.2f} max {g_te.max():.2f} "
            f"n>=10 {int((g_te >= 10).sum())} of {len(g_te)}"
        )
        print("  " + pred_notes[-1], flush=True)
        scores[name] = te

    # Oracle conditionals on every overlap day, before looking at model picks.
    oracle_rows = []
    for scope, mask in (
        ("overlap_s_top20", frame.s_rank <= 20),
        ("overlap_r_top20", frame.r_rank <= 20),
        ("test_s_top20", (frame.trade_date >= cut) & (frame.s_rank <= 20)),
        ("test_r_top20", (frame.trade_date >= cut) & (frame.r_rank <= 20)),
    ):
        g = frame[mask]
        st = summarize(g)
        st.update(scope=scope, rule="unfiltered")
        oracle_rows.append(st)
        st2 = summarize(g[g.y10 == 1])
        st2.update(scope=scope, rule="oracle_up10_in_5d")
        oracle_rows.append(st2)

    print("wider oracle, reading path panel", flush=True)
    oracle_rows.extend(wider_oracle())

    s_pool = test[test.s_rank <= 200].dropna(subset=["p_S5"])
    r_pool = test[test.r_rank <= 200].dropna(subset=["p_R5"])
    s40 = top_by(s_pool, "p_S5", 40, ascending=False)
    r40 = top_by(r_pool, "p_R5", 40, ascending=False)
    s20p = top_by(s_pool, "p_S5", 20, ascending=False)
    r20p = top_by(r_pool, "p_R5", 20, ascending=False)
    both40 = s40.merge(r40[["ts_code", "trade_date"]], on=["ts_code", "trade_date"], how="inner")

    def s20_head(df: pd.DataFrame) -> pd.DataFrame:
        return df[df.s_rank <= 20]

    rules = {
        "S20_top20": test[test.s_rank <= 20],
        "R20_top20": test[test.r_rank <= 20],
        "S5_top20_by_p": s20p,
        "R5_top20_by_p": r20p,
        "S5_top40_and_S20": s20_head(s40),
        "R5_top40_and_S20": s20_head(r40),
        "both_top40_and_S20": s20_head(both40),
        "S5_top40_rerank_S20_fill20": top_by(s40, "s_rank", 20, ascending=True),
        "R5_top40_rerank_S20_fill20": top_by(r40, "s_rank", 20, ascending=True),
        "both_top40_rerank_S20_fill20": top_by(both40, "s_rank", 20, ascending=True),
        "S5_g5_ge10": s_pool[s_pool.g_S5 >= 10],
        "R5_g5_ge10": r_pool[r_pool.g_R5 >= 10],
        "S5_g5_ge10_and_S20": s20_head(s_pool[s_pool.g_S5 >= 10]),
        "R5_g5_ge10_and_S20": s20_head(r_pool[r_pool.g_R5 >= 10]),
        "test_oracle_S20_and_up10": test[(test.s_rank <= 20) & (test.y10 == 1)],
        "test_oracle_R20_and_up10": test[(test.r_rank <= 20) & (test.y10 == 1)],
    }

    band = pd.read_parquet(
        EXP / "band_panel.parquet",
        columns=["ts_code", "trade_date", "cls_a5_d10", "ret_a5_b15_d10"],
    )
    band["trade_date"] = band.trade_date.astype(str)
    band["ts_code"] = band.ts_code.astype(str)
    band = band[band.trade_date >= cut]

    rows = []
    lines = [
        f"train < {vcut}  early-stop {vcut}..{cut}  test >= {cut}",
        "y10 = high vs next open touches +10% inside 5 sessions.",
        "hit15 = +15% touched inside 20 sessions. pure_up = +15% before -10% and before a -5% shake.",
        "S20 screen = existing stage-1 rank <= 20. Research models are not saved.",
        *pred_notes,
        "",
    ]
    for rule, picked in rules.items():
        stat = summarize(picked)
        if not picked.empty:
            m = picked.merge(band, on=["ts_code", "trade_date"], how="inner")
            if not m.empty:
                stat["band_ok"] = round(100 * float(m.cls_a5_d10.isin([1, 3]).mean()), 1)
                stat["band_bad"] = round(100 * float((m.cls_a5_d10 == 2).mean()), 1)
                stat["ret_band"] = round(float(m.ret_a5_b15_d10.mean()), 2)
        stat.update(scope="test_model", rule=rule)
        rows.append(stat)
        lines.append(f"{rule:32s} { {k: stat.get(k) for k in ('n','days','avg_len','up10_5','hit15','pure_up','pure_down','both','later15','ret_v1','hit15_if_up10','pure_if_up10','later15_if_up10','spike15_if_up10','band_ok')} }")

    lines.append("")
    for stat in oracle_rows:
        rows.append(stat)
        lines.append(
            f"{stat.get('scope','')} {stat.get('rule','')} "
            f"n={stat.get('n')} days={stat.get('days')} avg={stat.get('avg_len')} "
            f"up10={stat.get('up10_5')} hit15={stat.get('hit15')} pure={stat.get('pure_up')} "
            f"down={stat.get('pure_down')} later15={stat.get('later15')} ret={stat.get('ret_v1')} "
            f"hit15|up10={stat.get('hit15_if_up10')} pure|up10={stat.get('pure_if_up10')} "
            f"later|up10={stat.get('later15_if_up10')} spike|up10={stat.get('spike15_if_up10')}"
        )

    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
