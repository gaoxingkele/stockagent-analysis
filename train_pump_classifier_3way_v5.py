"""pump v5, full-feature edition: clean start-up / start-down (method C). RUN ON THE PRODUCTION MACHINE.

Same features as r5_pump_3way_lgbm_v3c (factor_lab + amount, money flow, cogalpha, pyramid, v7 extras,
mfk, long-return), only the label changes (src/stockagent_analysis/pump_labels.py, method C):
  start-up   = over the next 5 sessions after the signal close, close-path efficiency >= 0.6, net rise
               >= 1 ATR(14), lowest low never below close - 0.5 ATR  (a clean move; no fixed % size)
  start-down = the mirror image; everything else neutral. Classes 0 neutral, 1 down, 2 up (as v3c).
Windows match the shape edition (scripts/train_pump_v5_shape.py) so the two can be compared row for row:
  train  signal days 2023-01-03.. whose 5-session horizon ends on or before 2025-09-30
  valid  2025-10-01.. horizon end on or before 2026-01-26  (tree count only, by daily AUC)
  test   2026-01-27..2026-07-28                            (never used for any choice)
Nothing from 2026-08-06 on is read.

Output (NOT output/production; promotion is the user's decision):
  output/experiments/pump_v5_full/{classifier.txt, feature_meta.json, meta.json, test_predictions.parquet}
Send back the folder.

Result of the shape edition (K-line features only, 2026-10-07): validation daily AUC ~0.535 (up) / 0.55
(down) at any tree count; on the test window it did not beat chance for start-up (AUC 0.49) and lost to
v3c even on the clean label. Method C removes volatility, which is the predictable part of v3c's label
(v3c P(up) vs NATR rank correlation 0.88 market-wide). This edition tests whether money-flow, pyramid
and mfk features carry the cleanliness signal that K-line features do not.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.pump_labels import CleanStart, clean_start_labels  # noqa: E402
from train_v15_refresh import EXCLUDE, load_window  # noqa: E402

OUT = ROOT / "output" / "experiments" / "pump_v5_full"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
SPEC = CleanStart()
DATA_START, DATA_END = "20230101", "20260805"
TRAIN_HORIZON_END = "20250930"
VALID_START, VALID_HORIZON_END = "20251001", "20260126"
TEST_START, TEST_END = "20260127", "20260728"
MAX_TREES = 1200
GRID = (50, 100, 200, 400, 800, 1200)


def daily_auc(frame: pd.DataFrame, y: str, s: str) -> float:
    from sklearn.metrics import roc_auc_score
    vals = [roc_auc_score(g[y], g[s]) for _, g in frame.groupby("trade_date") if g[y].nunique() == 2]
    return float(np.mean(vals))


def labels() -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "output" / "tushare_cache" / "daily").glob("*.parquet")):
        if DATA_START <= f.stem <= DATA_END:
            x = pd.read_parquet(f, columns=["ts_code", "trade_date", "high", "low", "close", "pre_close"])
            parts.append(x[x.ts_code.str.endswith((".SH", ".SZ"))])
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px.trade_date.astype(str)
    days = sorted(px.trade_date.unique())
    end = {d: days[i + SPEC.horizon] if i + SPEC.horizon < len(days) else "99999999" for i, d in enumerate(days)}
    lab = clean_start_labels(px, SPEC)
    lab["horizon_end"] = lab.trade_date.map(end)
    return lab


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_window(DATA_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        df = df.merge(lf, on=["ts_code", "trade_date"], how="left")
    lab = labels()
    df = df.merge(lab[["ts_code", "trade_date", "label", "horizon_end"]], on=["ts_code", "trade_date"], how="inner")
    df = df[df.label.notna()]
    industries = pd.Categorical(df["industry"].fillna("unknown"))
    df["industry_id"] = industries.codes
    exc = set(EXCLUDE) | {"label", "horizon_end", "pump_3way", "is_pump_up", "is_pump_down", "is_pump"}
    feats = [c for c in df.columns if c not in exc and pd.api.types.is_numeric_dtype(df[c])]
    for c in feats:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200).astype("float32")

    train = df[df.horizon_end <= TRAIN_HORIZON_END]
    valid = df[(df.trade_date >= VALID_START) & (df.horizon_end <= VALID_HORIZON_END)]
    test = df[(df.trade_date >= TEST_START) & (df.trade_date <= TEST_END)]
    if len(train) > 2_000_000:
        train = train.sample(n=2_000_000, random_state=42)
    report = {name: {"rows": len(p), "days": int(p.trade_date.nunique()),
                     "classes": p.label.value_counts(normalize=True).sort_index().round(4).to_dict()}
              for name, p in (("train", train), ("valid", valid), ("test", test))}
    print(json.dumps(report, indent=1), flush=True)

    # No early stopping on multi_logloss: the class shares drift between windows, and on the shape edition
    # it stopped at 9 trees with no ranking power (2026-10-07). The tree count is chosen on the validation
    # window by the mean daily AUC of P(up) for clean start-up and P(down) for clean start-down.
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=MAX_TREES,
                             learning_rate=0.03, num_leaves=63, min_child_samples=300, feature_fraction=0.7,
                             bagging_fraction=0.8, bagging_freq=5, reg_alpha=0.1, reg_lambda=0.1, max_bin=127,
                             force_col_wise=True, random_state=42, n_jobs=4, verbose=-1)
    clf.fit(train[feats], train.label.astype(int), categorical_feature=["industry_id"])
    curve = {}
    for it in GRID:
        pv = clf.booster_.predict(valid[feats], num_iteration=it)
        v = valid[["trade_date", "label"]].assign(su=pv[:, 2], sd=pv[:, 1], yu=(valid.label == 2).astype(int), yd=(valid.label == 1).astype(int))
        curve[it] = (daily_auc(v, "yu", "su") + daily_auc(v, "yd", "sd")) / 2
        print(f"  valid trees {it}: mean daily AUC {curve[it]:.4f}", flush=True)
    best = max(curve, key=curve.get)
    clf.booster_.save_model(str(OUT / "classifier.txt"), num_iteration=best)
    report["valid_auc_curve"] = {str(k): round(v, 4) for k, v in curve.items()}

    p = clf.booster_.predict(test[feats], num_iteration=best)
    out = test[["ts_code", "trade_date", "label"]].copy()
    out["p_neutral"], out["p_down"], out["p_up"] = p[:, 0], p[:, 1], p[:, 2]
    out.to_parquet(OUT / "test_predictions.parquet", index=False)
    for cls, col in ((2, "p_up"), (1, "p_down")):
        base = (out.label == cls).mean()
        top = out[out.groupby("trade_date")[col].rank(pct=True, ascending=False) <= 0.05]
        report[f"test_class{cls}"] = {"base": round(float(base), 4), "top5pct_precision": round(float((top.label == cls).mean()), 4),
                                      "daily_auc": round(daily_auc(out.assign(y=(out.label == cls).astype(int)), "y", col), 4)}
    (OUT / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feats, "industry_map": {str(s): i for i, s in enumerate(industries.categories)},
        "spec": SPEC.__dict__, "classes": {"0": "neutral", "1": "clean start-down", "2": "clean start-up"},
        "model_type": "multiclass_3way_v5_clean_start"}, ensure_ascii=False, indent=2), encoding="utf-8")
    report["best_iteration"] = int(best)
    report["elapsed_s"] = round(time.time() - t0)
    (OUT / "meta.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k.startswith("test") or k == "best_iteration"}, indent=1))
    print(f"wrote {OUT} - send the folder back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
