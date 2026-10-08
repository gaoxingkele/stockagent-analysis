#!/usr/bin/env python
"""Why the S20 list looks different after 2026-01-26: re-score both periods with every model.

Scores the development days and the confirmation days with the frozen stage1 model and with each
walk-forward fold model, checks that every saved score is reproduced, and compares what each
model's daily top 50 looks like. Writes scores_dev.parquet, scores_confirm.parquet and report.txt
under output/experiments/s20_composition_diag_20261005. Reads nothing after 2026-08-05.
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = sys.argv[:1]
import confirm_s20_20r_portable as C
from stockagent_analysis.s20 import anchored_residual_probability
from stockagent_analysis.s20_pure import _frozen_models, MAX_DELTA_LOGIT
from train_s20_v2_multitarget import S20_V2_FOLDS
from train_s20_20r_residual import NESTED_PLANS
import analyze_s20_trend_filters as T
OUT = ROOT / "output/experiments/s20_composition_diag_20261005"; OUT.mkdir(parents=True, exist_ok=True)
lines = []
def say(x=""):
    print(x, flush=True); lines.append(str(x))

say("== how each score was produced ==")
for fold in S20_V2_FOLDS:
    p = NESTED_PLANS[fold.name]
    say(f"{fold.name}: anchor fit <= {p.anchor_fit_end}, anchor tune {p.anchor_tune_start}..{p.anchor_tune_end}; "
        f"residual fit {p.residual_fit_start}..{fold.fit_end}, residual tune {fold.tune_start}..{fold.tune_end}; "
        f"calibration {fold.calibration_start}..{fold.calibration_end}; test {fold.test_start}..{fold.test_end}")
say("frozen (confirm run): anchor fit <= 20250630, anchor tune 20250701..20250831, anchor refit on all matured dev; "
    "residual fit 20250901..20251031, residual tune 20251101..20260126; applied from 20260127")

anchor, residuals, features = _frozen_models()
models = {"frozen": (anchor, residuals)}
for name in ("wf1", "wf2", "wf3"):
    d = ROOT / "output/experiments/s20_20r_residual_portable" / name
    models[name] = (lgb.Booster(model_file=str(d / "r20_anchor_refit.txt")),
                    [lgb.Booster(model_file=f) for f in sorted(glob.glob(str(d / "s20_residual_seed*.txt")))])
say("")
say("== residual boosters: trees and the features carrying most gain ==")
for name, (a, rs) in models.items():
    g = sum(pd.Series(r.feature_importance("gain"), index=r.feature_name()) for r in rs)
    g = (g / g.sum()).sort_values(ascending=False)
    say(f"{name}: anchor trees {a.num_trees()}, residual trees {[r.num_trees() for r in rs]}; top gain: "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in g.head(8).items()))

def score(frame, a, rs):
    x = frame[features]
    delta = np.mean([r.predict(x, raw_score=True) for r in rs], axis=0)
    return anchored_residual_probability(a.predict(x), delta, max_absolute_residual=MAX_DELTA_LOGIT), delta

dev = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/feat_cache_bps5000.parquet")
dev["trade_date"] = dev.trade_date.astype(str)
oof = pd.read_parquet(ROOT / "output/experiments/s20_20r_residual_portable/predictions.parquet", columns=["ts_code", "trade_date", "fold", "s20_20r_platt"])
oof["trade_date"] = oof.trade_date.astype(str)
dev = dev.merge(oof, on=["ts_code", "trade_date"], how="inner").rename(columns={"s20_20r_platt": "saved"})
conf = C._load_confirmation(C.parse_args(), features)
saved = pd.read_parquet(ROOT / "output/experiments/s20_20r_confirmation/run_v1/predictions.parquet", columns=["ts_code", "trade_date", "stage1_probability"])
conf = conf.merge(saved.rename(columns={"stage1_probability": "saved"}), on=["ts_code", "trade_date"], how="inner")
conf["fold"] = "confirm"
ts = T.trend_states()[["ts_code", "trade_date", "ma20_up"]]
keep = ["ts_code", "trade_date", "fold", "saved", "roc_20", "macd_signal", "atr_pct", "kama_20"]
frames = {}
for label, fr in (("dev", dev), ("confirm", conf)):
    for name, (a, rs) in models.items():
        fr[f"p_{name}"], fr[f"d_{name}"] = score(fr, a, rs)
    frames[label] = fr[keep + [c for c in fr.columns if c.startswith(("p_", "d_"))]].merge(ts, on=["ts_code", "trade_date"], how="left")
    frames[label].to_parquet(OUT / f"scores_{label}.parquet", index=False)
del dev, conf

def top(fr, col, k=50):
    return fr[fr.groupby("trade_date")[col].rank(ascending=False, method="first") <= k]

say("")
say("== sanity: each saved score against its own model re-run (mean share of the daily top 20 in common) ==")
for label, col in (("dev", None), ("confirm", "p_frozen")):
    fr = frames[label]
    for fold, g in fr.groupby("fold"):
        c = col or f"p_{fold}"
        ov = g.groupby("trade_date").apply(lambda q: len(set(q.nlargest(20, "saved").ts_code) & set(q.nlargest(20, c).ts_code)) / 20).mean()
        say(f"{label} {fold}: saved vs {c}: {100 * ov:.1f}%")

say("")
say("== composition of each scorer's daily top 50 (no volatility cut) ==")
say("columns: share with MA20 rising | median past-20-session return | median MACD signal | median price proxy (kama_20) | median atr_pct")
for label in ("dev", "confirm"):
    fr = frames[label]
    for fold, g in fr.groupby("fold"):
        mk = g
        say(f"[{label} {fold}: {g.trade_date.min()}..{g.trade_date.max()}, {g.trade_date.nunique()} days] whole scored universe: "
            f"MA20 rising {100 * mk.ma20_up.mean():.0f}% | roc20 {mk.roc_20.median():+.1f} | macd_sig {mk.macd_signal.median():+.2f} | price {mk.kama_20.median():.1f} | atr {100 * mk.atr_pct.median():.1f}%")
        for name in ("saved", "p_frozen", "p_wf1", "p_wf2", "p_wf3"):
            t = top(g, name)
            say(f"    top 50 by {name:9s}: MA20 rising {100 * t.ma20_up.mean():5.1f}% | roc20 {t.roc_20.median():+6.1f} | macd_sig {t.macd_signal.median():+6.2f} | "
                f"price {t.kama_20.median():6.1f} | atr {100 * t.atr_pct.median():4.1f}%")

say("")
say("== what each scorer leans on: mean daily rank correlation of the score with three features ==")
for label in ("dev", "confirm"):
    fr = frames[label]
    for fold, g in fr.groupby("fold"):
        for name in ("p_frozen", "p_wf1", "p_wf2", "p_wf3"):
            cs = {f: g.groupby("trade_date").apply(lambda q: q[name].corr(q[f], method="spearman")).mean() for f in ("roc_20", "macd_signal", "atr_pct", "kama_20")}
            say(f"{label} {fold} {name:9s}: " + " | ".join(f"{k} {v:+.2f}" for k, v in cs.items()))
(OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
