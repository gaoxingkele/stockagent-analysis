#!/usr/bin/env python
"""Risk gates (unlocks, holder reductions, negative pre-announcements, earnings window)
and analyst EPS revisions, evaluated on the S20 lists. Rules fixed BEFORE running.

Point in time: an event is visible on signal day t only if its announcement date
(ann_date / report_date) <= t. Outcomes: band exit +5/+15/-10, 0.3% cost.

Features on day t (hypothesis in brackets):
  unlock30    unlocks with float_date in (t, t+30d], announced <= t, sum float_ratio >= 1%   [supply shock -> down]
  reduce60    holder reduction (in_de = DE) announced in [t-60d, t]                          [insider selling -> down]
  fcst_neg    latest pre-announcement within 120d is 预减/首亏/续亏/略减/续亏               [earnings deterioration]
  fcst_pos    latest pre-announcement within 120d is 预增/扭亏/略增/续盈                    (control)
  earn10      scheduled report date (pre_date) in (t, t+14d], schedule announced <= t       [jump risk]
  rev         analyst consensus EPS revision for the nearest fiscal-year quarter:
              mean eps of reports in [t-30d, t] vs [t-90d, t-31d]; up > +2%, down < -2%     [revision drift -> direction]

Decision rules (dev = signal days <= 20260126; confirm / shadow only describe):
  gate G adopted if, on dev, removing G-flagged picks and refilling from the pool
    lowers the bad rate (-10% first) by >= 1.0pp, does not lower per-trade by > 0.1pp,
    and beats a random-removal control of the same size (mean of 200 draws) on bad rate.
  revision adopted as a tilt if, on dev, its in-pool daily direction AUC (pure_up vs
    pure_down, stage1 Top100) > 0.52 and the tilted list (up-revisions first, then
    stage1) has a higher per-trade than the base list.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.s20_pure import three_state  # noqa: E402
from analyze_s20_pure_r11_safe_up import load  # noqa: E402

EV = ROOT / "output/tushare_cache/events"
OUT = ROOT / "output/experiments/risk_gates"
NEG = {"预减", "首亏", "续亏", "略减"}
POS = {"预增", "扭亏", "略增", "续盈"}


def read(api: str) -> pd.DataFrame:
    fs = sorted((EV / api).glob("*.parquet"))
    if api == "report_rc":   # rate-limited endpoint, fetched by month windows instead of by day
        fs += sorted((EV / "report_rc_month").glob("*.parquet"))
    parts = [pd.read_parquet(f) for f in fs if f.stat().st_size > 0]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    for c in ("ann_date", "float_date", "report_date", "end_date", "pre_date", "actual_date"):
        if c in df.columns:
            df[c] = df[c].astype(str).str[:8]
    return df


def features(keys: pd.DataFrame) -> pd.DataFrame:
    """keys: ts_code, trade_date. Returns the event features per key (point in time)."""
    k = keys[["ts_code", "trade_date"]].drop_duplicates().copy()
    k["t"] = pd.to_datetime(k.trade_date)
    out = k.copy()

    sf = read("share_float").dropna(subset=["float_date"])
    sf = sf.groupby(["ts_code", "ann_date", "float_date"], as_index=False).float_ratio.sum()
    sf["fd"], sf["ad"] = pd.to_datetime(sf.float_date), pd.to_datetime(sf.ann_date, errors="coerce")
    m = k.merge(sf, on="ts_code")
    m = m[(m.ad <= m.t) & (m.fd > m.t) & (m.fd <= m.t + pd.Timedelta(days=30))]
    u = m.groupby(["ts_code", "trade_date"]).float_ratio.sum().rename("unlock30_ratio")
    out = out.merge(u, on=["ts_code", "trade_date"], how="left")
    out["unlock30"] = out.unlock30_ratio.fillna(0) >= 1.0

    ht = read("stk_holdertrade")
    ht["ad"] = pd.to_datetime(ht.ann_date, errors="coerce")
    for flag, side in (("reduce60", "DE"), ("increase60", "IN")):
        h = ht[ht.in_de == side][["ts_code", "ad"]]
        m = k.merge(h, on="ts_code")
        m = m[(m.ad <= m.t) & (m.ad >= m.t - pd.Timedelta(days=60))]
        out[flag] = out.set_index(["ts_code", "trade_date"]).index.isin(
            m.set_index(["ts_code", "trade_date"]).index)

    fc = read("forecast")
    fc["ad"] = pd.to_datetime(fc.ann_date, errors="coerce")
    m = k.merge(fc[["ts_code", "ad", "type"]], on="ts_code")
    m = m[(m.ad <= m.t) & (m.ad >= m.t - pd.Timedelta(days=120))].sort_values("ad")
    last = m.groupby(["ts_code", "trade_date"]).type.last().rename("fcst_type")
    out = out.merge(last, on=["ts_code", "trade_date"], how="left")
    out["fcst_neg"] = out.fcst_type.isin(NEG)
    out["fcst_pos"] = out.fcst_type.isin(POS)

    dd = read("disclosure_date")
    dd["pre"], dd["ad"] = pd.to_datetime(dd.pre_date, errors="coerce"), pd.to_datetime(dd.ann_date, errors="coerce")
    m = k.merge(dd[["ts_code", "pre", "ad"]], on="ts_code")
    m = m[(m.ad <= m.t) & (m.pre > m.t) & (m.pre <= m.t + pd.Timedelta(days=14))]
    out["earn10"] = out.set_index(["ts_code", "trade_date"]).index.isin(m.set_index(["ts_code", "trade_date"]).index)

    em_dir = ROOT / "output/news/research_reports_em"
    if em_dir.exists() and any(em_dir.glob("*.parquet")):
        # Eastmoney research-report history: one row per report with its date and FY EPS
        # forecasts. FY2026 is forecast by reports from 2024 on, so it covers the whole window
        # (next-year EPS for 2025 signal days, current-year EPS for 2026). Strictly before t.
        em = pd.concat([pd.read_parquet(f) for f in em_dir.glob("*.parquet")], ignore_index=True)
        if "日期" in em.columns:
            em = em.dropna(subset=["日期"])
            em["rd"] = pd.to_datetime(em["日期"], errors="coerce")
            em["eps"] = pd.to_numeric(em.get("2026-盈利预测-收益"), errors="coerce")
            em = em.dropna(subset=["rd", "eps"])
            m = k.merge(em[["ts_code", "rd", "eps"]], on="ts_code")
            m = m[(m.rd < m.t) & (m.rd >= m.t - pd.Timedelta(days=90))]
            m["recent"] = m.rd >= m.t - pd.Timedelta(days=30)
            g = m.groupby(["ts_code", "trade_date", "recent"]).eps.mean().unstack().reindex(columns=[True, False])
            g = g.rename(columns={True: "eps_new", False: "eps_old"})
            g["rev"] = (g.eps_new - g.eps_old) / g.eps_old.abs()
            out = out.merge(g[["rev"]].reset_index(), on=["ts_code", "trade_date"], how="left")
            out["rev_up"], out["rev_dn"] = out.rev > 0.02, out.rev < -0.02
            cov = m.groupby(["ts_code", "trade_date"]).size().rename("n_reports90")
            out = out.merge(cov, on=["ts_code", "trade_date"], how="left")
            return out.drop(columns="t")
    rc = read("report_rc")
    if rc.empty or "eps" not in rc.columns:
        out["rev"], out["rev_up"], out["rev_dn"], out["n_reports90"] = np.nan, False, False, np.nan
        return out.drop(columns="t")
    rc = rc.dropna(subset=["eps"])
    rc["rd"] = pd.to_datetime(rc.report_date, errors="coerce")
    rc = rc[rc.quarter.astype(str).str.endswith("Q4")]
    rc["fy"] = rc.quarter.str[:4].astype(int)
    m = k.merge(rc[["ts_code", "rd", "fy", "eps"]], on="ts_code")
    m = m[(m.rd <= m.t) & (m.rd >= m.t - pd.Timedelta(days=90)) & (m.fy == m.t.dt.year)]
    m["recent"] = m.rd >= m.t - pd.Timedelta(days=30)
    g = m.groupby(["ts_code", "trade_date", "recent"]).eps.mean().unstack().reindex(columns=[True, False])
    g = g.rename(columns={True: "eps_new", False: "eps_old"})
    g["rev"] = (g.eps_new - g.eps_old) / g.eps_old.abs()
    out = out.merge(g[["rev"]].reset_index(), on=["ts_code", "trade_date"], how="left")
    out["rev_up"], out["rev_dn"] = out.rev > 0.02, out.rev < -0.02
    cov = m.groupby(["ts_code", "trade_date"]).size().rename("n_reports90")
    out = out.merge(cov, on=["ts_code", "trade_date"], how="left")
    return out.drop(columns="t")


def v1_funnel(p: pd.DataFrame, drop_col: str | None = None, drop_idx=None, order=None) -> pd.DataFrame:
    f = p.copy()
    f["pr"] = f.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    f = f[f.pr <= 100].copy()
    f["np"] = f.groupby("trade_date").natr14.rank(pct=True)
    f = f[f.np.isna() | (f.np <= 0.6)]
    if drop_col:
        f = f[~f[drop_col]]
    if drop_idx is not None:
        f = f[~f.index.isin(drop_idx)]
    key = f.stage1_probability if order is None else order.loc[f.index]
    f = f.assign(_k=key)
    f["lr"] = f.groupby("trade_date")._k.rank(ascending=False, method="first")
    return f[f.lr <= 20]


def stats(L: pd.DataFrame) -> dict:
    d = L.groupby("trade_date")
    return {"per_trade%": round(float(d.ret.mean().mean()), 3), "bad%": round(100 * (L.cls_a5_d10 == 2).mean(), 2),
            "success%": round(100 * L.cls_a5_d10.isin([1, 3]).mean(), 1)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p = load()
    p = p[p.trade_date <= "20260805"].copy()
    path = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/path_panel.parquet",
                           columns=["ts_code", "trade_date", "up15_day", "dn10_day"])
    p = p.merge(path, on=["ts_code", "trade_date"], how="left")
    p["ret"] = p.ret_a5_b15_d10 - 0.3
    p["state"] = three_state(p.up15_day.fillna(0), p.dn10_day.fillna(0))
    pool_keys = p.assign(pr=p.groupby("trade_date").stage1_probability.rank(ascending=False))
    F = features(pool_keys[pool_keys.pr <= 150])
    p = p.merge(F, on=["ts_code", "trade_date"], how="left")
    for c in ("unlock30", "reduce60", "increase60", "fcst_neg", "fcst_pos", "earn10", "rev_up", "rev_dn"):
        p[c] = p[c].fillna(False).astype(bool)
    p["period"] = np.where(p.trade_date <= "20260126", "dev", "confirm")
    p = p.reset_index(drop=True)
    lines = [f"days {p.trade_date.nunique()}; pool coverage: analyst reports in 90d for "
             f"{100 * p.loc[p.n_reports90.notna()].shape[0] / max(1, p.n_reports90.shape[0]):.1f}% of scored rows"]

    # 1. flag vs no-flag inside the v1 list and inside the pool
    p["any_risk"] = p.unlock30 | p.reduce60 | p.fcst_neg | p.earn10
    base = v1_funnel(p)
    rows = []
    for c in ("unlock30", "reduce60", "increase60", "fcst_neg", "fcst_pos", "earn10", "rev_up", "rev_dn"):
        for per in ("dev", "confirm"):
            L = base[base.period == per]
            a, b = L[L[c]], L[~L[c]]
            rows.append({"flag": c, "period": per, "share_of_picks%": round(100 * L[c].mean(), 1),
                         "bad%_flag": round(100 * (a.cls_a5_d10 == 2).mean(), 1) if len(a) else np.nan,
                         "bad%_rest": round(100 * (b.cls_a5_d10 == 2).mean(), 1),
                         "ret_flag": round(float(a.ret.mean()), 2) if len(a) else np.nan,
                         "ret_rest": round(float(b.ret.mean()), 2), "n_flag": len(a)})
    lines.append("## inside the S20 v1 list: flagged vs other picks\n" + pd.DataFrame(rows).to_string(index=False))

    # 2. gates: remove flagged picks, refill from the pool; random-removal control
    rng = np.random.default_rng(20261001)
    pool_idx = p.index[p.groupby("trade_date").stage1_probability.rank(ascending=False) <= 100]
    rows = []
    for g in ("unlock30", "reduce60", "fcst_neg", "earn10", "any_risk"):
        col = g
        gated = v1_funnel(p, drop_col=col)
        removed = base[base[col]]
        ctrl = []
        for _ in range(200):
            drop = []
            for d, n in removed.groupby("trade_date").size().items():
                cand = base.index[base.trade_date == d]
                drop += list(rng.choice(cand, min(n, len(cand)), replace=False))
            ctrl.append(v1_funnel(p, drop_idx=set(drop)))
        for per in ("dev", "confirm"):
            s0, s1 = stats(base[base.period == per]), stats(gated[gated.period == per])
            sc = [stats(c[c.period == per]) for c in ctrl]
            rows.append({"gate": g, "period": per, "removed_per_day": round(len(removed[removed.period == per]) /
                         max(1, base[base.period == per].trade_date.nunique()), 2),
                         "bad%_base": s0["bad%"], "bad%_gated": s1["bad%"],
                         "bad%_random": round(float(np.mean([x["bad%"] for x in sc])), 2),
                         "ret_base": s0["per_trade%"], "ret_gated": s1["per_trade%"],
                         "ret_random": round(float(np.mean([x["per_trade%"] for x in sc])), 3)})
    G = pd.DataFrame(rows)
    dev = G[G.period == "dev"]
    G["adopt_rule_dev"] = G.gate.map(dict(zip(dev.gate, (dev["bad%_base"] - dev["bad%_gated"] >= 1.0)
                                                & (dev.ret_gated >= dev.ret_base - 0.1)
                                                & (dev["bad%_gated"] < dev["bad%_random"]))))
    lines.append("\n## gates: remove flagged picks and refill (random removal = control)\n" + G.to_string(index=False))

    # 3. analyst revision as a direction signal inside the stage1 pool
    pool = p.loc[pool_idx]
    mv = pool[pool.state.isin(["pure_up", "pure_down"]) & pool.rev.notna()]
    aucs = {}
    for per, g in mv.groupby("period"):
        daily = [roc_auc_score(x.state.eq("pure_up"), x.rev) for _, x in g.groupby("trade_date")
                 if x.state.nunique() == 2 and len(x) >= 6]
        aucs[per] = round(float(np.mean(daily)), 3) if daily else np.nan
    order = p.stage1_probability + 10 * p.rev_up.astype(float) - 10 * p.rev_dn.astype(float)
    tilt = v1_funnel(p, order=order)
    rows = []
    for per in ("dev", "confirm"):
        rows.append({"period": per, "in_pool_direction_AUC": aucs.get(per), **{f"base_{k}": v for k, v in stats(base[base.period == per]).items()},
                     **{f"tilt_{k}": v for k, v in stats(tilt[tilt.period == per]).items()},
                     "tilt_rev_up_share%": round(100 * tilt[tilt.period == per].rev_up.mean(), 1)})
    R = pd.DataFrame(rows)
    d = R[R.period == "dev"].iloc[0]
    auc_dev = d.in_pool_direction_AUC
    adopt = bool(auc_dev is not None and pd.notna(auc_dev) and auc_dev > 0.52
                 and d["tilt_per_trade%"] > d["base_per_trade%"])
    lines.append("\n## analyst revisions: direction AUC in the pool and an up-first tilt of the list\n" + R.to_string(index=False)
                 + f"\nadopt revision tilt by the dev rule: {adopt}")
    text = "\n".join(lines)
    (OUT / "risk_gates.txt").write_text(text, encoding="utf-8")
    p[["ts_code", "trade_date", "unlock30", "unlock30_ratio", "reduce60", "increase60", "fcst_type", "earn10", "rev",
       "n_reports90"]].to_parquet(OUT / "event_features.parquet", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
