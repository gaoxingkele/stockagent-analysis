#!/usr/bin/env python
"""Outcomes of the frozen model's lists and the fold models' lists on the same days.

Run after diagnose_s20_model_shift_scores.py; appends to the same report.txt. Descriptive only.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_pure_r09_funnel import natr14
from analyze_s20_scissors import dn7_panel
from experiment_cross_filter_v2 import load_paths, nw_se
from experiment_r5_s5_cross_filter import add_events
OUT = ROOT / "output/experiments/s20_composition_diag_20261005"
lines = open(OUT / "report.txt", encoding="utf-8").read().rstrip("\n").split("\n")
def say(x=""):
    print(x, flush=True); lines.append(str(x))
paths = load_paths(); nat = natr14().rename(columns={"natr14_self": "natr14"}); nat["trade_date"] = nat.trade_date.astype(str)
y7 = dn7_panel()[["ts_code", "trade_date", "y_dn7"]]
say("")
say("== outcomes of each scorer's list on the same days ==")
say("list = scorer's top 100, cut the 40% with the highest NATR, keep 20 (the frozen funnel). ret = +15%/-10% exit, list basis.")
say("The frozen residual model was fitted on 2025-09..10 and tuned on 2025-11..2026-01, so wf1 and wf2 days are outside its ranking fit; wf3 days are its tuning window.")
for label in ("dev", "confirm"):
    fr = pd.read_parquet(OUT / f"scores_{label}.parquet")
    fr = add_events(fr.merge(paths, on=["ts_code", "trade_date"], how="inner")).merge(nat, on=["ts_code", "trade_date"], how="left").merge(y7, on=["ts_code", "trade_date"], how="left")
    for fold, g in fr.groupby("fold"):
        say(f"[{label} {fold}: {g.trade_date.nunique()} days]")
        days = {}
        for name in ("p_frozen", "p_wf1", "p_wf2", "p_wf3"):
            r = g.groupby("trade_date")[name].rank(ascending=False, method="first")
            pool = g[r <= 100].copy()
            pct = pool.groupby("trade_date").natr14.rank(pct=True)
            pool = pool[pct.isna() | (pct <= 0.6)]
            lst = pool[pool.groupby("trade_date")[name].rank(ascending=False, method="first") <= 20]
            raw = g[r <= 20]
            days[name] = lst.groupby("trade_date").ret_v1.mean()
            own = " <- the model that produced the saved scores here" if name == ("p_frozen" if fold == "confirm" else f"p_{fold}") else ""
            say(f"    {name:9s}: funnel list ret {days[name].mean():+5.2f} pure_up {100 * lst.pure_up.mean():4.1f} pure_down {100 * lst.pure_down.mean():4.1f} "
                f"dn7 {100 * lst.y_dn7.mean():4.1f} hit15 {100 * lst.hit15.mean():4.1f} | raw top 20 ret {raw.groupby('trade_date').ret_v1.mean().mean():+5.2f} "
                f"pure_down {100 * raw.pure_down.mean():4.1f} | overlap with frozen list {100 * len(set(zip(lst.ts_code, lst.trade_date)) & keyset) / max(len(lst), 1) if name != 'p_frozen' else 100:5.1f}%{own}"
                if name != "p_frozen" else
                f"    {name:9s}: funnel list ret {days[name].mean():+5.2f} pure_up {100 * lst.pure_up.mean():4.1f} pure_down {100 * lst.pure_down.mean():4.1f} "
                f"dn7 {100 * lst.y_dn7.mean():4.1f} hit15 {100 * lst.hit15.mean():4.1f} | raw top 20 ret {raw.groupby('trade_date').ret_v1.mean().mean():+5.2f} "
                f"pure_down {100 * raw.pure_down.mean():4.1f}{own}")
            if name == "p_frozen":
                keyset = set(zip(lst.ts_code, lst.trade_date))
        for name in ("p_wf1", "p_wf2", "p_wf3"):
            d = (days["p_frozen"] - days[name]).dropna(); se = nw_se(d.to_numpy())
            month = d.groupby(d.index.str[:6]).mean()
            say(f"      frozen minus {name}: {d.mean():+.2f} (t {d.mean() / se:+.2f}, months up {int((month > 0).sum())}/{len(month)})")
(OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
