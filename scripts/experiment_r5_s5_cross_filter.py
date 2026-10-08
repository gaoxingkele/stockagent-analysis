#!/usr/bin/env python
"""Cross the 5-day downside veto with the 5-day +10% screen, then the S20 rank.

Does not write production weights. Same clock as the 2026-10-04 break and
upside studies: train before 2025-08-21, early-stop through 2025-11-24, test
from 2025-11-24 through 2026-01-26.

The rule that gets called the cross-filter solution is chosen on the
early-stop window only, by account return: each day's kept names are summed
and divided by 20, and a day with no names counts as 0. Ties break toward
higher pure-up, then a longer book. The test window is scored after that
choice and is not used to pick the winner.
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
OUT = ROOT / "output/experiments/r5_s5_cross_20261004"
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
    dn5 = df["dn5_day"].to_numpy(dtype=int)
    dn10 = df["dn10_day"].to_numpy(dtype=int)
    out = df.copy()
    out["y_up"] = ((up10 > 0) & (up10 <= 5)).astype(np.int8)
    out["y_dn5"] = ((dn5 > 0) & (dn5 <= 5)).astype(np.int8)
    out["hit15"] = ((up15 > 0) & (up15 <= 20)).astype(np.int8)
    st = three_state(up15, dn10, dn5)
    out["pure_up"] = (st == "pure_up").astype(np.int8)
    out["pure_down"] = (st == "pure_down").astype(np.int8)
    rule = PureConfig().rule("U15D10")
    out["ret_v1"] = exit_return(up15, dn10, out["ret20"].to_numpy(dtype=float), rule)
    return out


def add_y7(frame: pd.DataFrame) -> pd.DataFrame:
    daily_dir = ROOT / "output/tushare_cache/daily"
    files = sorted(p for p in daily_dir.glob("*.parquet") if p.stem >= "20250201")
    px = pd.concat(
        [pd.read_parquet(p, columns=["ts_code", "trade_date", "open", "low"]) for p in files],
        ignore_index=True,
    )
    px["trade_date"] = px["trade_date"].astype(str)
    px["ts_code"] = px["ts_code"].astype(str)
    dates = np.array(sorted(px.trade_date.unique()))
    idx = {d: i for i, d in enumerate(dates)}
    px = px[px.ts_code.isin(set(frame.ts_code))]
    op = px.pivot(index="trade_date", columns="ts_code", values="open").reindex(dates)
    lo = px.pivot(index="trade_date", columns="ts_code", values="low").reindex(dates)
    codes = {c: op.columns.get_loc(c) for c in frame.ts_code.unique() if c in op.columns}
    y7 = np.full(len(frame), np.nan)
    for n, (code, day) in enumerate(zip(frame.ts_code.to_numpy(), frame.trade_date.to_numpy())):
        j = codes.get(code)
        i = idx.get(day)
        if j is None or i is None or i + 5 >= len(dates):
            continue
        entry = op.iat[i + 1, j]
        if not (entry > 0) or not np.isfinite(entry):
            continue
        window = lo.iloc[i + 1:i + 6, j].to_numpy(dtype=float)
        if np.isfinite(window).sum() < 4:
            continue
        y7[n] = np.nanmin(window) / entry - 1.0 <= -0.07
    frame = frame.copy()
    frame["y_dn7"] = y7
    print(f"y7 rate {np.nanmean(y7):.3f}", flush=True)
    return frame


def load_frame() -> pd.DataFrame:
    rank = pd.read_parquet(
        EXP / "r04_preds_U15_D10.parquet",
        columns=[
            "ts_code", "trade_date", "s1_rank", "ret20",
            "up10_day", "up15_day", "dn5_day", "dn10_day",
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
    return add_y7(add_events(use))


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
        eval_set=[(valid.loc[np.isfinite(valid[label].to_numpy(dtype=float)), cols],
                   valid.loc[np.isfinite(valid[label].to_numpy(dtype=float)), label].astype(float))],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model, cols


def pool_slice(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    if pool == "s":
        return df[df.s_rank <= 200]
    if pool == "r":
        return df[df.r_rank <= 200]
    return df[(df.s_rank <= 200) & (df.r_rank <= 200)]


def apply_gates(q: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if q.empty:
        return q
    mask = np.ones(len(q), dtype=bool)
    values = q.reset_index(drop=True)
    for col in (spec.get("dn"), spec.get("dn2")):
        if col:
            mask &= values[col].to_numpy(dtype=float) < spec["dn_at"]
    frac = spec.get("up_frac")
    for col in (spec.get("up"), spec.get("up2")):
        if col and frac:
            thr = np.nanquantile(values[col].to_numpy(dtype=float), 1.0 - frac)
            mask &= values[col].to_numpy(dtype=float) >= thr
    return q.iloc[np.flatnonzero(mask)]


def pick_rule(df: pd.DataFrame, days: list[str], spec: dict) -> pd.DataFrame:
    base = pool_slice(df, spec["pool"])
    groups = {d: g for d, g in base.groupby("trade_date", sort=False)}
    rank_col = {"s": "s_rank", "r": "r_rank", "both": "s_rank"}[spec["pool"]]
    fill_by = spec.get("fill_by", rank_col)
    parts = []
    mode = spec["mode"]
    for day in days:
        g = groups.get(day)
        if g is None or g.empty:
            continue
        if mode == "head":
            q = apply_gates(g[g[rank_col] <= spec["cap"]], spec)
            q = q.sort_values(fill_by, ascending=True)
            if spec.get("fill"):
                q = q.head(20)
            elif spec["cap"] > 20:
                q = q[q[rank_col] <= 20]
        elif mode == "top_score":
            q = g.sort_values(spec["score"], ascending=False).head(spec["k"])
        elif mode == "up_then_s20":
            q = g.sort_values(spec["up"], ascending=False).head(spec["k"])
            q = q[q.s_rank <= 20]
        elif mode == "soft":
            q = g[g[rank_col] <= spec["cap"]].copy()
            if q.empty:
                continue
            acc = None
            for up, dn in spec["pairs"]:
                leg = q[up].rank(pct=True) - q[dn].rank(pct=True)
                acc = leg if acc is None else acc + leg
            q = q.assign(soft=acc / len(spec["pairs"]))
            q = q.sort_values("soft", ascending=False).head(spec.get("k", 20))
        else:
            raise ValueError(mode)
        if not q.empty:
            parts.append(q)
    if not parts:
        return base.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


def score_picks(picked: pd.DataFrame, days: list[str], band: pd.DataFrame) -> dict:
    day_index = pd.Index(days)
    n_days = len(days)
    if picked.empty:
        return {
            "n": 0, "days_on": 0, "avg_len": 0.0, "cal_len": 0.0,
            "hit15": None, "pure_up": None, "pure_down": None,
            "ret_v1": None, "sleeve_v1": 0.0, "band_ok": None, "ret_band": None,
            "sleeve_band": 0.0,
        }
    per = picked.groupby("trade_date").size()
    sleeve = picked.groupby("trade_date").ret_v1.sum().reindex(day_index, fill_value=0.0)
    m = picked.merge(band, on=["ts_code", "trade_date"], how="inner")
    if m.empty:
        band_ok = ret_band = None
        sleeve_band = 0.0
    else:
        band_ok = round(100 * float(m.cls_a5_d10.isin([1, 3]).mean()), 1)
        ret_band = round(float(m.ret_a5_b15_d10.mean()), 2)
        sleeve_band = float(
            m.groupby("trade_date").ret_a5_b15_d10.sum().reindex(day_index, fill_value=0.0).mean() / 20.0
        )
    return {
        "n": int(len(picked)),
        "days_on": int(picked.trade_date.nunique()),
        "avg_len": round(float(per.mean()), 2),
        "cal_len": round(len(picked) / n_days, 2),
        "hit15": round(100 * float(picked.hit15.mean()), 1),
        "pure_up": round(100 * float(picked.pure_up.mean()), 1),
        "pure_down": round(100 * float(picked.pure_down.mean()), 1),
        "ret_v1": round(float(picked.ret_v1.mean()), 2),
        "sleeve_v1": round(float(sleeve.mean() / 20.0), 3),
        "band_ok": band_ok,
        "ret_band": ret_band,
        "sleeve_band": round(sleeve_band, 3),
    }


def rules() -> list[dict]:
    """Frozen grid. dn_at 0.50 and the top-half upside cut are not fit on test."""
    r = []

    def add(family, **spec):
        spec["family"] = family
        r.append(spec)

    add("base", name="S20_top20", mode="head", pool="s", cap=20)
    add("base", name="R20_top20", mode="head", pool="r", cap=20)
    for pool, tag, dn in (
        ("r", "R", "p_R5_dn5"), ("s", "S", "p_S5_dn5"),
        ("r", "R7", "p_R5_dn7"), ("s", "S7", "p_S5_dn7"),
    ):
        add("down", name=f"{tag}_dn_top20", mode="head", pool=pool, cap=20, dn=dn, dn_at=0.50)
        if tag in ("R", "S"):
            add("down", name=f"{tag}_dn_fill50", mode="head", pool=pool, cap=50, dn=dn, dn_at=0.50, fill=True)
    add("up", name="S_up_top20", mode="top_score", pool="s", score="p_S5_up", k=20)
    add("up", name="R_up_top20", mode="top_score", pool="r", score="p_R5_up", k=20)
    add("up", name="S_up40_and_S20", mode="up_then_s20", pool="s", up="p_S5_up", k=40)
    add("up", name="R_up40_and_S20", mode="up_then_s20", pool="r", up="p_R5_up", k=40)
    # Hard cross: drop P(break) >= 0.50 and keep the better half of P(+10%).
    for pool, tag, dn, up in (
        ("s", "S", "p_S5_dn5", "p_S5_up"),
        ("r", "R", "p_R5_dn5", "p_R5_up"),
        ("s", "S7", "p_S5_dn7", "p_S5_up"),
        ("r", "R7", "p_R5_dn7", "p_R5_up"),
    ):
        add("cross", name=f"{tag}_dn_up_top20", mode="head", pool=pool, cap=20,
            dn=dn, dn_at=0.50, up=up, up_frac=0.5)
        add("cross", name=f"{tag}_dn_up_fill50", mode="head", pool=pool, cap=50,
            dn=dn, dn_at=0.50, up=up, up_frac=0.5, fill=True)
    add("cross", name="R_dn_up_then_S20", mode="head", pool="r", cap=50,
        dn="p_R5_dn5", dn_at=0.50, up="p_R5_up", up_frac=0.5, fill=True, fill_by="s_rank")
    add("cross", name="both_dn_up_top20", mode="head", pool="both", cap=20,
        dn="p_S5_dn5", dn2="p_R5_dn5", dn_at=0.50,
        up="p_S5_up", up2="p_R5_up", up_frac=0.5)
    add("cross", name="both_dn_up_fill50", mode="head", pool="both", cap=50,
        dn="p_S5_dn5", dn2="p_R5_dn5", dn_at=0.50,
        up="p_S5_up", up2="p_R5_up", up_frac=0.5, fill=True, fill_by="s_rank")
    add("soft", name="S_soft20", mode="soft", pool="s", cap=20, k=20,
        pairs=(("p_S5_up", "p_S5_dn5"),))
    add("soft", name="R_soft20", mode="soft", pool="r", cap=20, k=20,
        pairs=(("p_R5_up", "p_R5_dn5"),))
    add("soft", name="S_soft_of50", mode="soft", pool="s", cap=50, k=20,
        pairs=(("p_S5_up", "p_S5_dn5"),))
    add("soft", name="R_soft_of50", mode="soft", pool="r", cap=50, k=20,
        pairs=(("p_R5_up", "p_R5_dn5"),))
    add("soft", name="both_soft_of50", mode="soft", pool="both", cap=50, k=20,
        pairs=(("p_S5_up", "p_S5_dn5"), ("p_R5_up", "p_R5_dn5")))
    add("soft", name="S_soft_keep10", mode="soft", pool="s", cap=20, k=10,
        pairs=(("p_S5_up", "p_S5_dn5"),))
    add("soft", name="R_soft_keep10", mode="soft", pool="r", cap=20, k=10,
        pairs=(("p_R5_up", "p_R5_dn5"),))
    return r


def choose(table: pd.DataFrame, family: set[str], min_len: float) -> pd.Series | None:
    cand = table[(table.split == "valid") & table.family.isin(family) & (table.cal_len >= min_len)]
    if cand.empty:
        return None
    cand = cand.sort_values(
        ["sleeve_v1", "pure_up", "cal_len"], ascending=[False, False, False]
    )
    return cand.iloc[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    days = sorted(frame.trade_date.unique())
    vcut = days[int(len(days) * 0.55)]
    cut = days[int(len(days) * 0.70)]
    valid_days = [d for d in days if vcut <= d < cut]
    test_days = [d for d in days if d >= cut]
    print(
        f"rows {len(frame):,} train<{vcut} valid {vcut}..{cut} ({len(valid_days)}d) "
        f"test>={cut} ({len(test_days)}d)",
        flush=True,
    )
    scored = frame.copy()
    for name, rank_col, labels in (
        ("S5", "s_rank", ("y_dn5", "y_dn7", "y_up")),
        ("R5", "r_rank", ("y_dn5", "y_dn7", "y_up")),
    ):
        pool = frame[frame[rank_col] <= 200]
        tr = pool[pool.trade_date < vcut]
        va = pool[(pool.trade_date >= vcut) & (pool.trade_date < cut)]
        hold = pool[pool.trade_date >= vcut]
        for lab, tag in zip(labels, ("dn5", "dn7", "up")):
            print(f"fit {name} {tag} train {tr[lab].notna().sum()} pos {tr[lab].mean():.3f}", flush=True)
            model, cols = fit_cls(tr[tr[lab].notna()], va[va[lab].notna()], lab)
            p = model.predict_proba(hold[cols])[:, 1]
            scored.loc[hold.index, f"p_{name}_{tag}"] = p
            te = hold[hold.trade_date >= cut]
            p_te = scored.loc[te.index, f"p_{name}_{tag}"]
            print(
                f"  test AUC {auc(te[lab], p_te):.3f} "
                f"valid AUC {auc(va[lab], scored.loc[va.index, f'p_{name}_{tag}']):.3f} "
                f"base {te[lab].mean():.3f}",
                flush=True,
            )

    band = pd.read_parquet(
        EXP / "band_panel.parquet",
        columns=["ts_code", "trade_date", "cls_a5_d10", "ret_a5_b15_d10"],
    )
    band["trade_date"] = band.trade_date.astype(str)
    band["ts_code"] = band.ts_code.astype(str)
    band = band[band.trade_date >= vcut]

    rows = []
    for spec in rules():
        for split, split_days in (("valid", valid_days), ("test", test_days)):
            part = scored[scored.trade_date.isin(split_days)]
            picked = pick_rule(part, split_days, spec)
            stat = score_picks(picked, split_days, band)
            stat.update(name=spec["name"], family=spec["family"], split=split)
            rows.append(stat)
            print(f"{split:5s} {spec['family']:5s} {spec['name']:22s} {stat}", flush=True)

    table = pd.DataFrame(rows)
    account = choose(table, {"cross", "soft"}, min_len=5)
    success = table[(table.split == "valid") & table.family.isin(["cross", "soft"]) & (table.cal_len >= 5)]
    success_row = None if success.empty else success.sort_values(
        ["pure_up", "sleeve_v1", "cal_len"], ascending=[False, False, False]
    ).iloc[0]
    any_row = choose(table, {"base", "down", "up", "cross", "soft"}, min_len=5)

    def line_for(title: str, row: pd.Series | None) -> str:
        if row is None:
            return f"{title}: none"
        test = table[(table.name == row["name"]) & (table.split == "test")].iloc[0]
        return (
            f"{title}: {row['name']}  valid sleeve {row['sleeve_v1']} pure {row['pure_up']} "
            f"band {row['band_ok']} len {row['cal_len']}  |  "
            f"test sleeve {test['sleeve_v1']} pure {test['pure_up']} hit15 {test['hit15']} "
            f"band {test['band_ok']} ret {test['ret_v1']} band_ret {test['ret_band']} "
            f"len {test['cal_len']} down {test['pure_down']}"
        )

    header = [
        f"train < {vcut}  choose-on-valid {vcut}..{cut} ({len(valid_days)}d)  test >= {cut} ({len(test_days)}d)",
        "Account objective: mean over days of (sum of U15D10 returns / 20); a missing name is 0.",
        "Eligible cross/soft rules need calendar length >= 5 on the valid window.",
        "Winner is frozen on valid before reading the test line below.",
        line_for("account winner among cross/soft", account),
        line_for("pure-up winner among cross/soft", success_row),
        line_for("account winner among every family", any_row),
        "",
    ]
    text = "\n".join(header)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    table.to_csv(OUT / "grid.csv", index=False)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
