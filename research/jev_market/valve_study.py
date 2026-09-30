#!/usr/bin/env python
"""Sell-off valve study: risk model choice + how the valve should act on the lists.

Part 1 - risk model (target: any broad sell-off session in the next 5 sessions).
  Expanding window, refit monthly, predict the next month (no look-ahead). Candidates:
  logistic, gradient boosting (LightGBM, small), and transparent rules.
  Rule fixed before running: pick the highest AUC over dev months (<= 202601);
  if a transparent rule is within 0.02 AUC of the best model, prefer the rule.

Part 2 - valve actions on the frozen lists (v1 aggressive U15/D10, v1.1 safe), each
  signal day = one sleeve of 1/20 capital, outcome = band exit a5/b15/D10 (v1.1) or
  U15/D10 (v1). Policies when risk p >= threshold:
    skip   : no new sleeve today (cash)
    half   : half-size sleeve
    safe   : v1 users get the v1.1 safe list instead
    tight  : stop moved from -10% to -7% (band a5/b15/D7) for today's sleeve
  Threshold rule fixed before running: choose on dev days maximising mean sleeve
  return (cash = 0) subject to <= 35% of days affected; confirm + shadow reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_market_crash import FEATS, market_snapshot  # noqa: E402

OUT = ROOT / "output/jev_market"
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=7, max_depth=3, min_data_in_leaf=20,
                  feature_fraction=1.0, bagging_fraction=1.0, lambda_l2=1.0, verbosity=-1, seed=20260929,
                  deterministic=True, num_threads=1)
N_ROUNDS = 150


def expanding_predictions(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    for c in ("p_logit", "p_lgb"):
        m[c] = np.nan
    for mo in sorted(m.date.str[:6].unique())[6:]:
        tr = m[m.date.str[:6] < mo].iloc[:-5].dropna(subset=FEATS + ["crash_next5"])
        te = (m.date.str[:6] == mo) & m[FEATS].notna().all(axis=1)
        y = tr.crash_next5.astype(int)
        if y.nunique() < 2:
            continue
        m.loc[te, "p_logit"] = LogisticRegression(max_iter=2000).fit(tr[FEATS], y).predict_proba(m.loc[te, FEATS])[:, 1]
        bst = lgb.train(LGB_PARAMS, lgb.Dataset(tr[FEATS], label=y), num_boost_round=N_ROUNDS)
        m.loc[te, "p_lgb"] = bst.predict(m.loc[te, FEATS])
    m["rule_ld5"] = m.limit_down_5d
    m["rule_ret5"] = -m.median_ret_5d
    m["rule_combo"] = m.limit_down_5d.rank(pct=True) + (-m.median_ret_5d).rank(pct=True)
    return m


def list_outcomes() -> pd.DataFrame:
    """Per signal day: mean sleeve return of v1 (U15/D10) and v1.1 safe (band) Top lists."""
    from stockagent_analysis.s20_pure import PureConfig, SafeConfig, select, select_safe
    from analyze_s20_pure_r11_safe_up import load
    p = load()
    # load() already carries the band panel (incl. ret20, ret_a5_b15_d8); add first-touch days for v1
    path = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/path_panel.parquet",
                           columns=["ts_code", "trade_date", "up15_day", "dn10_day", "dn8_day"])
    p = p.merge(path, on=["ts_code", "trade_date"], how="left")
    from stockagent_analysis.s20_pure import exit_return
    v1 = select(p, PureConfig(), "U15D10")
    r = PureConfig().rule("U15D10")
    v1["ret"] = exit_return(v1.up15_day, v1.dn10_day, v1.ret20, r)
    v1["ret_tight"] = exit_return(v1.up15_day, v1.dn8_day, v1.ret20, r.__class__("t", 15.0, 8.0, 0.4))
    safe = select_safe(p, SafeConfig())
    safe["ret"] = safe.ret_a5_b15_d10 - 0.3
    safe["ret_tight"] = safe.ret_a5_b15_d8 - 0.3
    agg = lambda L: L.groupby("trade_date").agg(ret=("ret", "mean"), ret_tight=("ret_tight", "mean"))  # noqa: E731
    out = agg(v1).add_prefix("v1_").join(agg(safe).add_prefix("safe_"), how="outer")
    out["period"] = np.where(out.index <= "20260126", "dev", "confirm")
    return out


def main() -> int:
    m = market_snapshot()
    m = m[m.amount_z60.notna()].reset_index(drop=True)
    m = expanding_predictions(m)
    lab = m[m.crash_next5.notna() & m.p_lgb.notna()]
    lines = ["## Part 1: risk model, expanding window (AUC)"]
    rows = []
    for name, g in (("dev (<=202601)", lab[lab.date <= "20260131"]), ("2026 (>=202602)", lab[lab.date > "20260131"]),
                    ("all", lab)):
        y = g.crash_next5.astype(int)
        rows.append({"subset": name, "n": len(g), "pos%": round(100 * y.mean(), 1),
                     **{c: round(roc_auc_score(y, g[c]), 3) for c in ("p_logit", "p_lgb", "rule_ld5", "rule_ret5", "rule_combo")}})
    t = pd.DataFrame(rows)
    lines.append(t.to_string(index=False))
    dev = t.iloc[0]
    models = {"p_logit": dev.p_logit, "p_lgb": dev.p_lgb}
    rules = {"rule_ld5": dev.rule_ld5, "rule_ret5": dev.rule_ret5, "rule_combo": dev.rule_combo}
    best_model = max(models, key=models.get)
    best_rule = max(rules, key=rules.get)
    chosen = best_rule if rules[best_rule] >= models[best_model] - 0.02 else best_model
    lines.append(f"chosen risk score (dev rule): {chosen}  [best model {best_model} {models[best_model]}, "
                 f"best rule {best_rule} {rules[best_rule]}]")

    # risk expressed as the dev-period percentile of the chosen score (comparable across refits)
    ref = m.loc[m.date <= "20260131", chosen].dropna()
    m["risk_pct"] = m[chosen].apply(lambda v: np.nan if pd.isna(v) else float((ref <= v).mean()))
    m[["date", "p_logit", "p_lgb", "rule_ld5", "rule_ret5", "rule_combo", "risk_pct", "crash_next5"]].to_csv(
        OUT / "valve_risk_history.csv", index=False)

    # Part 2
    L = list_outcomes().join(m.set_index("date")[["risk_pct"]], how="left")
    L = L[L.risk_pct.notna()]
    lines.append(f"\n## Part 2: valve policies on sleeves (days with lists and risk: dev {int((L.period=='dev').sum())}, "
                 f"confirm {int((L.period=='confirm').sum())})")
    rows = []
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01):
        on = L.risk_pct >= thr
        for per in ("dev", "confirm"):
            g, o = L[L.period == per], on[L.period == per]
            for lst in ("v1", "safe"):
                base = g[f"{lst}_ret"]
                pol = {"none": base,
                       "skip": base.where(~o, 0.0),
                       "half": base.where(~o, 0.5 * base),
                       "tight": base.where(~o, g[f"{lst}_ret_tight"])}
                if lst == "v1":
                    pol["safe"] = base.where(~o, g["safe_ret"])
                for k, s in pol.items():
                    if thr > 1 and k != "none":
                        continue
                    month = s.groupby(s.index.str[:6]).mean()
                    cum = s.fillna(0).cumsum() / 20          # each sleeve = 1/20 capital
                    rows.append({"thr": thr, "period": per, "list": lst, "policy": k, "affected%": round(100 * o.mean(), 1),
                                 "mean_sleeve": round(float(s.mean()), 3), "worst_month": round(float(month.min()), 2),
                                 "neg_months": int((month < 0).sum()), "n_months": len(month),
                                 "maxDD_book": round(float((cum - cum.cummax()).min()), 2)})
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "valve_policies.csv", index=False)
    d = P[(P.period == "dev") & (P["affected%"] <= 35)]
    for lst in ("v1", "safe"):
        best = d[d.list == lst].sort_values("mean_sleeve", ascending=False).iloc[0]
        lines.append(f"selected for {lst} on dev: thr={best.thr} policy={best.policy} "
                     f"(affected {best['affected%']}%, mean {best.mean_sleeve}, worst month {best.worst_month}, maxDD {best.maxDD_book})")
        for per in ("dev", "confirm"):
            x = P[(P.period == per) & (P.list == lst) & (((P.thr == best.thr) & (P.policy == best.policy)) | (P.policy == "none") & (P.thr > 1))]
            lines.append(x[["period", "policy", "thr", "affected%", "mean_sleeve", "worst_month", "neg_months", "maxDD_book"]].to_string(index=False))
    lines.append("\n## policy grid (dev), sorted\n" + d.sort_values(["list", "mean_sleeve"], ascending=[True, False])
                 .groupby("list").head(8).to_string(index=False))
    text = "\n".join(lines)
    (OUT / "valve_study.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
