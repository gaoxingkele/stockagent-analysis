"""Purged S20-v3 six-class training, same-label baselines, feature audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_r20_target_prob_v2 import _load_dataset
from train_s20_v2_multitarget import S20_V2_FOLDS
from train_s20_20r_residual import PORTABLE_EXCLUDED_FEATURES
from confirm_s20_20r_portable import _load_confirmation
from stockagent_analysis.s20 import purged_walk_forward_masks
from stockagent_analysis.s20_v3 import CLASS_NAMES, probability_outputs, select_low_correlation

OUT = ROOT / "output/experiments/s20_v3"
SEED = 20260912
KEYS = ["ts_code", "trade_date"]
TARGETS = ["s20_class", "immediate", "opportunity", "down_risk", "negative"]
FEATURE_MODES = ["portable_full", "lowcorr24"]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def fit_model(fit, tune, features):
    params = dict(objective="multiclass", num_class=6, metric="multi_logloss",
                  learning_rate=.04, num_leaves=31, min_data_in_leaf=250,
                  feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
                  lambda_l2=1., verbosity=-1, seed=SEED, num_threads=8)
    return lgb.train(params, lgb.Dataset(fit[features], label=fit.s20_class),
                     num_boost_round=350,
                     valid_sets=[lgb.Dataset(tune[features], label=tune.s20_class)],
                     callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)])


def calibrate(model, calibration, features):
    raw = model.predict(calibration[features], num_threads=8)
    calibrator = LogisticRegression(C=1., max_iter=1000, random_state=SEED)
    calibrator.fit(np.log(np.clip(raw, 1e-8, 1)), calibration.s20_class)
    if list(calibrator.classes_) != list(range(6)):
        raise ValueError("calibration needs all six classes")
    return calibrator


def predict(model, calibrator, frame, features):
    raw = model.predict(frame[features], num_threads=8)
    probability = calibrator.predict_proba(np.log(np.clip(raw, 1e-8, 1)))
    outputs = probability_outputs(probability)
    result = frame[KEYS + TARGETS].reset_index(drop=True).copy()
    for column in outputs:
        result[column] = outputs[column].to_numpy()
    for i, name in enumerate(CLASS_NAMES):
        result[f"p_class_{name}"] = probability[:, i]
    return result


def top_rows(frame, column, k=20):
    return frame.sort_values(["trade_date", column, "ts_code"], ascending=[True,False,True]).groupby("trade_date", sort=False).head(k)


def metric_rows(frame, scores, period):
    rows = []
    for candidate, column in scores.items():
        for k in [5,10,20,50]:
            selected = top_rows(frame, column, k)
            row = dict(period=period, candidate=candidate, k=k, rows=len(selected),
                       dates=selected.trade_date.nunique(),
                       immediate_rate=selected.immediate.mean(),
                       opportunity_rate=selected.opportunity.mean(),
                       down_rate=selected.down_risk.mean(), negative_rate=selected.negative.mean(),
                       utility=(selected.immediate - selected.down_risk).mean(),
                       lift=selected.immediate.mean()/frame.immediate.mean(),
                       base_immediate=frame.immediate.mean(),
                       unique_symbols=selected.ts_code.nunique())
            rows.append(row)
    return rows


def probability_audit(frame, period, mode):
    rows = []
    for target, column in [("immediate","p_immediate"),("down_risk","p_down"),("negative","p_negative")]:
        y, p = frame[target], frame[column]
        base = y.mean()
        brier = brier_score_loss(y,p)
        selected = top_rows(frame,"score")
        rows.append(dict(period=period,mode=mode,target=target,auc=roc_auc_score(y,p),
                         brier=brier,brier_skill=1-brier/(base*(1-base)),
                         mean_predicted=p.mean(),actual_rate=base,
                         top20_actual=selected[target].mean(),top20_predicted=selected[column].mean(),
                         top20_abs_error=abs(selected[target].mean()-selected[column].mean())))
    return rows


def save_bundle(folder, model, calibrator, features):
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(str(folder / "model.txt"))
    save_json(folder / "schema.json", dict(features=features,classes=CLASS_NAMES,
              research_only=True, probability_status="requires held-out calibration validation",
              calibration_coefficients=calibrator.coef_.tolist(),
              calibration_intercepts=calibrator.intercept_.tolist(),
              calibration_input="log(clip(raw_class_probability,1e-8,1))",
              score="50*(1+p_immediate-p_down); not a probability"))


def train_pair(parts, features, name):
    fit, tune, calibration = (parts[s] for s in ("fit", "tune", "calibration"))
    if set(fit.s20_class.unique()) != set(range(6)):
        raise ValueError("training requires all classes")
    print(f"{name} fit={len(fit):,} tune={len(tune):,} cal={len(calibration):,}", flush=True)
    full = fit_model(fit,tune,features)
    gain = dict(zip(features, full.feature_importance(importance_type="gain")))
    # Numeric category codes have arbitrary order, so not eligible for Spearman screening.
    numeric = [f for f in features if f not in {"industry_id", "regime_id"}]
    sample = fit[numeric].sample(min(60000,len(fit)),random_state=SEED)
    corr = sample.corr(method="spearman").fillna(0)
    low = select_low_correlation(numeric,gain,corr)
    folder = OUT / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    corr.loc[low,low].to_csv(folder / "lowcorr_matrix.csv")
    save_json(folder / "split_audit.json", {s:dict(rows=len(f), date_min=f.trade_date.min(),date_max=f.trade_date.max(),
              max_label_maturity=f.horizon_end_date.max()) for s,f in parts.items()})
    save_json(folder / "feature_selection.json", dict(features=low,fit_only=True,abs_spearman_limit=.7))
    models = {}
    for mode, fs, model in [("portable_full",features,full),("lowcorr24",low,None)]:
        if model is None:
            model = fit_model(fit,tune,fs)
        cal = calibrate(model,calibration,fs)
        save_bundle(folder / mode,model,cal,fs)
        models[mode] = (model,cal,fs)
        print(f"{name} {mode}: features={len(fs)} rounds={model.best_iteration}",flush=True)
    return models


def block_delta_ci(frame, score_a, score_b, target="immediate", repeats=2000):
    a = top_rows(frame,score_a).groupby("trade_date")[target].mean()
    b = top_rows(frame,score_b).groupby("trade_date")[target].mean()
    delta = (a-b).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    n, block = len(delta), min(20,len(delta))
    draws = []
    for _ in range(repeats):
        starts = rng.integers(0,n,size=int(np.ceil(n/block)))
        indices = ((starts[:,None]+np.arange(block))%n).ravel()[:n]
        draws.append(delta[indices].mean())
    return dict(delta=float(delta.mean()), ci95=np.quantile(draws,[.025,.975]).tolist(),
                method="paired circular 20-session block bootstrap; exploratory, not independent trades")


def feature_audit(frame, model_tuple, name):
    model, cal, fs = model_tuple
    sample = frame.sample(min(12000,len(frame)),random_state=SEED).reset_index(drop=True)
    prediction = predict(model,cal,sample,fs)
    baseline_up = brier_score_loss(sample.immediate,prediction.p_immediate)
    baseline_down = brier_score_loss(sample.down_risk,prediction.p_down)
    rows = []
    rng = np.random.default_rng(SEED)
    for f in fs:
        up_delta,down_delta = [],[]
        for _ in range(3):
            perm = sample.copy()
            # Daily market variables: shuffle entire dates. Stock variables: within date.
            if sample.groupby("trade_date")[f].nunique().max() <= 1:
                values = sample.groupby("trade_date")[f].first()
                mapping = dict(zip(values.index,rng.permutation(values.to_numpy())))
                perm[f] = perm.trade_date.map(mapping)
            else:
                for _, idx in sample.groupby("trade_date").groups.items():
                    perm.loc[idx,f] = rng.permutation(sample.loc[idx,f].to_numpy())
            pred = predict(model,cal,perm,fs)
            up_delta.append(brier_score_loss(sample.immediate,pred.p_immediate)-baseline_up)
            down_delta.append(brier_score_loss(sample.down_risk,pred.p_down)-baseline_down)
        valid = sample[f].notna()
        # Daily percentiles remove cross-month level effects for stock-specific factors.
        rank = sample.groupby("trade_date")[f].rank(pct=True)
        rows.append(dict(period=name,feature=f,
                         up_brier_increase=np.mean(up_delta),down_brier_increase=np.mean(down_delta),
                         up_permutation_std=np.std(up_delta),down_permutation_std=np.std(down_delta),
                         positive_median=sample.loc[sample.immediate==1,f].median(),
                         negative_median=sample.loc[sample.negative==1,f].median(),
                         down_median=sample.loc[sample.down_risk==1,f].median(),
                         daily_rank_positive_minus_negative=rank[sample.immediate==1].mean()-rank[sample.negative==1].mean(),
                         univariate_spearman_up=sample.loc[valid,f].corr(sample.loc[valid,"immediate"],method="spearman"),
                         univariate_spearman_down=sample.loc[valid,f].corr(sample.loc[valid,"down_risk"],method="spearman")))
    print(f"{name} feature audit complete ({len(fs)} features)",flush=True)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    labels = pd.read_parquet(OUT / "labels.parquet")
    cache = OUT / "development.parquet"
    if cache.exists():
        data = pd.read_parquet(cache)
        features = json.loads((OUT / "feature_schema.json").read_text())["features"]
    else:
        data, features, source_audit = _load_dataset(5000,20260830)
        features = [f for f in features if f not in PORTABLE_EXCLUDED_FEATURES]
        data = data[KEYS + features].merge(labels,on=KEYS,validate="one_to_one")
        data = data[data.s20_class>=0].reset_index(drop=True)
        data.to_parquet(cache,index=False)
        save_json(OUT / "feature_schema.json",dict(features=features,source_audit=source_audit))
    old = pd.read_parquet(ROOT / "output/experiments/s20_20r_ranker_portable/predictions.parquet")
    old = old[KEYS+["hybrid_rank","ensemble_p20","s20_20_raw"]].rename(columns={"hybrid_rank":"old_s20","ensemble_p20":"old_r20","s20_20_raw":"old_s20_v2"})
    metrics, probs, audits, all_predictions = [],[],[],[]
    for fold in S20_V2_FOLDS:
        masks = purged_walk_forward_masks(data.trade_date,data.horizon_end_date,fold)
        parts = {s:data[m].copy() for s,m in masks.items()}
        models = train_pair(parts,features,fold.name)
        test = parts["test"]
        common = test[KEYS+TARGETS].merge(old,on=KEYS,validate="one_to_one")
        scores = {"old_s20":"old_s20","old_r20":"old_r20","old_s20_v2":"old_s20_v2"}
        for mode, tup in models.items():
            pred = predict(*tup[:2],test,tup[2])
            pred["mode"], pred["period"] = mode,fold.name
            all_predictions.append(pred)
            probs.extend(probability_audit(pred,fold.name,mode))
            common = common.merge(pred[KEYS+["score","p_immediate"]].rename(columns={"score":mode,"p_immediate":mode+"_up"}),on=KEYS,validate="one_to_one")
            scores[mode],scores[mode+"_up"] = mode,mode+"_up"
        metrics.extend(metric_rows(common,scores,fold.name))
        common["period"] = fold.name
        common.to_parquet(OUT / f"comparison_{fold.name}.parquet",index=False)
        audits.extend(feature_audit(test,models["lowcorr24"],fold.name))
        pd.DataFrame(metrics).to_csv(OUT / "metrics.csv",index=False)
        pd.DataFrame(probs).to_csv(OUT / "probability_metrics.csv",index=False)
        pd.DataFrame(audits).to_csv(OUT / "feature_audit.csv",index=False)

    development = pd.concat([pd.read_parquet(OUT / f"comparison_{f.name}.parquet") for f in S20_V2_FOLDS],ignore_index=True)
    metrics.extend(metric_rows(development,scores,"development_all"))
    dev_metric = pd.DataFrame(metrics)
    candidates = dev_metric[(dev_metric.period=="development_all") & (dev_metric.k==20) & dev_metric.candidate.isin(FEATURE_MODES)]
    chosen = candidates.sort_values(["utility","immediate_rate"],ascending=False).iloc[0].candidate
    save_json(OUT / "development_selection.json",dict(candidate=chosen,criterion="Top20 immediate-minus-down utility on development only",candidates=candidates.to_dict("records")))
    print(f"development selected {chosen}",flush=True)

    # Already inspected 2026 window: diagnostic comparison, never promotion evidence.
    d, e = data.trade_date,data.horizon_end_date
    parts = {
        "fit":data[(d<="20250831")&(e<"20250901")],
        "tune":data[d.between("20250901","20251031")&(e<"20251101")],
        "calibration":data[d.between("20251101","20260126")&(e<"20260127")],
    }
    models = train_pair(parts,features,"diagnostic2026")
    args = SimpleNamespace(labels=ROOT / "output/experiments/s20_v2_labels/labels.parquet",
                           factor_dir=ROOT / "output/experiments/s20_20r_confirmation/factor_groups")
    confirmation = _load_confirmation(args,features)
    confirmation = confirmation[KEYS+features].merge(labels,on=KEYS,validate="one_to_one")
    confirmation = confirmation[confirmation.s20_class>=0].reset_index(drop=True)
    old_confirm = pd.read_parquet(ROOT / "output/experiments/s20_20r_confirmation/run_v1/predictions.parquet")
    old_confirm = old_confirm[KEYS+["s20_20r_rank","r20_p20_reference"]].rename(columns={"s20_20r_rank":"old_s20","r20_p20_reference":"old_r20"})
    common = confirmation[KEYS+TARGETS].merge(old_confirm,on=KEYS,validate="one_to_one")
    scores = {"old_s20":"old_s20","old_r20":"old_r20"}
    for mode,tup in models.items():
        pred = predict(*tup[:2],confirmation,tup[2])
        pred["mode"], pred["period"] = mode,"diagnostic2026"
        all_predictions.append(pred)
        probs.extend(probability_audit(pred,"diagnostic2026",mode))
        common = common.merge(pred[KEYS+["score","p_immediate"]].rename(columns={"score":mode,"p_immediate":mode+"_up"}),on=KEYS,validate="one_to_one")
        scores[mode],scores[mode+"_up"] = mode,mode+"_up"
    metrics.extend(metric_rows(common,scores,"diagnostic2026"))
    for month,g in common.groupby(common.trade_date.str[:6]):
        metrics.extend(metric_rows(g,scores,"month_"+month))
    common.to_parquet(OUT / "comparison_diagnostic2026.parquet",index=False)
    audits.extend(feature_audit(confirmation,models["lowcorr24"],"diagnostic2026"))
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv",index=False)
    pd.DataFrame(probs).to_csv(OUT / "probability_metrics.csv",index=False)
    pd.DataFrame(audits).to_csv(OUT / "feature_audit.csv",index=False)
    pd.concat(all_predictions,ignore_index=True).to_parquet(OUT / "predictions.parquet",index=False)
    top_rows(common,chosen).merge(labels[KEYS+["reason","max_gain20","window_mae20"]],on=KEYS).to_csv(OUT / "diagnostic_top20.csv",index=False,encoding="utf-8-sig")
    report = dict(status="research_completed_requires_new_confirmation",candidate=chosen,
                  development_rows=len(development),diagnostic_rows=len(common),
                  diagnostic_dates=common.trade_date.nunique(),
                  dev_delta_vs_old_s20=block_delta_ci(development,chosen,"old_s20"),
                  diagnostic_delta_vs_old_s20=block_delta_ci(common,chosen,"old_s20"),
                  diagnostic_down_delta_vs_old_s20=block_delta_ci(common,chosen,"old_s20","down_risk"),
                  metrics=[r for r in metrics if r["period"] in ["development_all","diagnostic2026"] and r["k"]==20],
                  probability_metrics=probs,
                  note="new label change is not model improvement; all comparisons use same labels and same rows; delayed opportunity never immediate success")
    save_json(OUT / "report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="probability_metrics"},indent=2),flush=True)


if __name__ == "__main__":
    main()
