#!/usr/bin/env python
"""Backtest LLM-tagged announcements as a direction signal and as a negative-event veto.

Point in time: announcements dated in [t-4, t] (calendar days) feed signal day t; entry is
the next open, so an announcement published on the evening of t is already public.
Tags come from tag_notices_history.py (MiniMax), one per announcement text.

Per stock-day features: net = sum(sentiment * max(materiality, 1)); pos_major = any
sentiment >= 1 with materiality >= 2; neg_major = any sentiment <= -1 with materiality >= 2.

Rules fixed before running (dev = signal days <= 20260126; confirm only describes):
  direction: pooled AUC of `net` for pure_up vs pure_down among stage1-pool stock-days that
             have a tagged announcement > 0.52 on dev, AND the up-first tilt (pos_major first,
             neg_major last, then stage1) raises the dev per-trade of the v1 list.
  veto:      removing neg_major picks from the v1 list and refilling lowers the dev bad rate by
             >= 1.0pp, keeps per-trade within 0.1pp, and beats random removal of the same count.
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
from eval_risk_gates_and_revisions import stats, v1_funnel  # noqa: E402
from tag_notices_history import relevant  # noqa: E402

OUT = ROOT / "output/experiments/notice_sentiment"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tags = pd.read_parquet(ROOT / "output/news/tags/notices_hist.parquet")
    tags = tags.dropna(subset=["sentiment"]).drop_duplicates("h")
    x = relevant().merge(tags[["h", "sentiment", "materiality", "event_type"]], on="h")
    x["sentiment"] = pd.to_numeric(x.sentiment, errors="coerce")
    x["materiality"] = pd.to_numeric(x.materiality, errors="coerce").fillna(0)
    x["nd"] = pd.to_datetime(x.notice_date)

    p = load()
    p = p[p.trade_date <= "20260805"].copy()
    path = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/path_panel.parquet",
                           columns=["ts_code", "trade_date", "up15_day", "dn10_day"])
    p = p.merge(path, on=["ts_code", "trade_date"], how="left")
    p["ret"] = p.ret_a5_b15_d10 - 0.3
    p["state"] = three_state(p.up15_day.fillna(0), p.dn10_day.fillna(0))
    p["pr"] = p.groupby("trade_date").stage1_probability.rank(ascending=False)
    keys = p[p.pr <= 150][["ts_code", "trade_date"]].copy()
    keys["t"] = pd.to_datetime(keys.trade_date)
    m = keys.merge(x[["ts_code", "nd", "sentiment", "materiality"]], on="ts_code")
    m = m[(m.nd <= m.t) & (m.nd >= m.t - pd.Timedelta(days=4))]
    m["w"] = m.sentiment * m.materiality.clip(lower=1)
    f = m.groupby(["ts_code", "trade_date"]).agg(
        net=("w", "sum"), n_tagged=("w", "size"),
        pos_major=("sentiment", lambda s: bool(((s >= 1) & (m.loc[s.index, "materiality"] >= 2)).any())),
        neg_major=("sentiment", lambda s: bool(((s <= -1) & (m.loc[s.index, "materiality"] >= 2)).any())))
    p = p.merge(f.reset_index(), on=["ts_code", "trade_date"], how="left")
    p["pos_major"] = p.pos_major.fillna(False).astype(bool)
    p["neg_major"] = p.neg_major.fillna(False).astype(bool)
    p["period"] = np.where(p.trade_date <= "20260126", "dev", "confirm")
    p = p.reset_index(drop=True)
    covered = sorted(set(x.notice_date))
    lines = [f"tagged announcements matched: {len(x):,}; notice days covered {covered[0]}..{covered[-1]} ({len(covered)}); "
             f"pool stock-days with a tagged announcement: {int(p.n_tagged.notna().sum()):,}"]

    # direction
    pool = p[(p.pr <= 100) & p.n_tagged.notna() & p.state.isin(["pure_up", "pure_down"])]
    rows = []
    for per, g in pool.groupby("period"):
        auc = roc_auc_score(g.state.eq("pure_up"), g.net) if g.state.nunique() == 2 else np.nan
        rows.append({"period": per, "n": len(g), "AUC_net": round(float(auc), 3),
                     "up_share_net>0": round(100 * g[g.net > 0].state.eq("pure_up").mean(), 1),
                     "up_share_net<0": round(100 * g[g.net < 0].state.eq("pure_up").mean(), 1)})
    D = pd.DataFrame(rows)
    lines.append("## direction inside the stage1 Top100 (stock-days with a tagged announcement)\n" + D.to_string(index=False))

    base = v1_funnel(p)
    order = p.stage1_probability + 10 * p.pos_major.astype(float) - 10 * p.neg_major.astype(float)
    tilt = v1_funnel(p, order=order)
    gated = v1_funnel(p, drop_col="neg_major")
    removed = base[base.neg_major]
    rng = np.random.default_rng(20261002)
    ctrl = []
    for _ in range(200):
        drop = []
        for d, n in removed.groupby("trade_date").size().items():
            cand = base.index[base.trade_date == d]
            drop += list(rng.choice(cand, min(n, len(cand)), replace=False))
        ctrl.append(v1_funnel(p, drop_idx=set(drop)))
    rows = []
    for per in ("dev", "confirm"):
        b, t, g = (stats(z[z.period == per]) for z in (base, tilt, gated))
        sc = [stats(c[c.period == per]) for c in ctrl]
        rows.append({"period": per, "neg_major_in_list_per_day": round(len(removed[removed.period == per]) /
                     max(1, base[base.period == per].trade_date.nunique()), 2),
                     "ret_base": b["per_trade%"], "ret_tilt": t["per_trade%"], "ret_veto": g["per_trade%"],
                     "ret_random": round(float(np.mean([s["per_trade%"] for s in sc])), 3),
                     "bad_base": b["bad%"], "bad_tilt": t["bad%"], "bad_veto": g["bad%"],
                     "bad_random": round(float(np.mean([s["bad%"] for s in sc])), 2)})
    R = pd.DataFrame(rows)
    d, rd = D[D.period == "dev"], R[R.period == "dev"].iloc[0]
    adopt_dir = bool(len(d) and d.AUC_net.iloc[0] > 0.52 and rd.ret_tilt > rd.ret_base)
    adopt_veto = bool((rd.bad_base - rd.bad_veto >= 1.0) and (rd.ret_veto >= rd.ret_base - 0.1)
                      and (rd.bad_veto < rd.bad_random))
    lines.append("\n## S20 v1 list: up-first tilt and negative-event veto (random removal = control)\n" + R.to_string(index=False)
                 + f"\nadopt direction tilt (dev rule): {adopt_dir}\nadopt negative veto (dev rule): {adopt_veto}")
    text = "\n".join(lines)
    (OUT / "notice_sentiment.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
