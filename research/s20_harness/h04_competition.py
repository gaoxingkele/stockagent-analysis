"""H04 executor: registered bounded competition on the H03 split.

Every configuration is declared before it is evaluated, the fit budget is
charged explicitly, and a zero-fit frequency control plus a reused reference
head stay in the comparison set. Nothing here is formal H04 acceptance: the H01
admission failure is inherited through the H02/H03 chain.
"""
from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd

from .abcds_model import CLASS_ORDER, TARGET_ID, run as fit_abcds, run_frequency
from .baseline_model import run as fit_binary
from .policy_support import build_candidates, evaluate_selection, gate_threshold
from .runtime import atomic_json, digest, now
from .splits import training_ids

FEATURE_COLUMNS = ("raw_return20", "raw_ma20_distance", "raw_mean_tr14_pct",
                   "raw_return_vol20", "volume_ratio20")
FROZEN_AT = "2024-05-01T00:00:00+08:00"
MODEL_FIT_CAP = 18          # registered ceiling for this executor

# Declared before evaluation. `fits` is the model-level charge, not an estimate.
CONFIGS = (
    dict(config_id="abcds_frequency_control", family="mature_frequency", fits=0),
    dict(config_id="binary_A_reference", family="binary_reference_reused", fits=0),
    dict(config_id="independent_risk_head", family="binary_risk", fits=1),
    dict(config_id="abcds_multinomial_C1.0", family="abcds_multinomial", fits=1,
         regularisation=1.0),
    dict(config_id="abcds_multinomial_C0.1", family="abcds_multinomial", fits=1,
         regularisation=0.1),
)

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; competition inherits the frozen diagnostic sample",
    "single seed and one chronological split; no cross-fold or cross-seed robustness",
    "forward reference predictions only; fit-segment predictions are withheld, not out-of-fold",
    "the binary A reference is reused from H03 rather than refitted here",
    "no legacy reproduction, no execution pilot, no absolute-probability claim",
)


def _frozen_tables(root: Path):
    source = root / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"
    samples = pd.read_parquet(source / "samples.parquet")
    features = pd.read_parquet(source / "features.parquet")
    return source, samples, features


def build(root: Path, directory: Path, *, upstream_directories: dict[str, Path]) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    upstream = {name: Path(path) for name, path in upstream_directories.items()}
    for required in ("H02", "H03"):
        if required not in upstream:
            raise ValueError("H04 requires upstream outputs: " + required)
    labels = pd.read_parquet(upstream["H02"] / "labels.parquet")
    split_manifest = json.loads((upstream["H03"] / "split_manifest.json").read_text(encoding="utf-8"))
    boundaries = split_manifest["segments"]
    h03_oof = pd.read_parquet(upstream["H03"] / "baseline_oof.parquet")

    source, samples, features = _frozen_tables(root)
    resolved = labels[labels.target.notna()].copy()
    dropped = int(len(labels) - len(resolved))
    keep = set(resolved.sample_id)
    samples = samples[samples.sample_id.isin(keep)].reset_index(drop=True)
    features = features[features.sample_id.isin(keep)].reset_index(drop=True)
    if split_manifest["rows"] != len(samples):
        raise ValueError("H04 sample universe does not match the H03 split manifest")

    from .splits import assign_segments
    assignment, split = assign_segments(samples, boundaries)
    if split["split_protocol_sha256"] != split_manifest["split_protocol_sha256"]:
        raise ValueError("H04 rebuilt a different split protocol")
    fit_ids = training_ids(assignment, "fit")
    calibration_ids = training_ids(assignment, "calibration")
    fit_five = resolved[resolved.sample_id.isin(fit_ids)][["sample_id", "target"]].copy()
    if set(fit_five.sample_id) != set(fit_ids):
        raise ValueError("H04 fit labels must cover exactly the eligible fit rows")
    calibration_five = resolved[resolved.sample_id.isin(calibration_ids)][
        ["sample_id", "target"]].copy()
    contract = {"role": "prediction_features", "columns": list(FEATURE_COLUMNS),
                "source_provenance_verified": True}
    values = features[["sample_id", *FEATURE_COLUMNS]]

    predictions: dict[str, pd.DataFrame] = {}
    cards: dict[str, dict] = {}
    charged = 0
    for config in CONFIGS:
        config_id, family = config["config_id"], config["family"]
        if family == "mature_frequency":
            output, card = run_frequency(samples, values, fit_five, boundaries, contract)
        elif family == "abcds_multinomial":
            output, card = fit_abcds(samples, values, fit_five, boundaries, contract,
                                     target_id=TARGET_ID, random_seed=20,
                                     regularisation=config["regularisation"])
        elif family == "binary_reference_reused":
            # Reused from H03 on purpose: the fit was already charged there.
            reused = h03_oof[["sample_id", "segment", "feature_ready", "raw_probability",
                              "prediction_status"]].copy()
            output = reused.rename(columns={"raw_probability": "score"})
            output["p_A"] = output["score"]
            card = {"model": "binary_A_reference_reused_from_H03",
                    "target_id": TARGET_ID + ".A", "model_level_fits": 0,
                    "reused_from": "H03 baseline_oof.parquet",
                    "absolute_probability_validated": False,
                    "formal_H04_accepted": False}
        elif family == "binary_risk":
            risk_labels = fit_five.copy()
            risk_labels["target"] = risk_labels.target.isin(["B", "D"])
            output, card = fit_binary(samples, values, risk_labels, boundaries, contract,
                                      target_id=TARGET_ID + ".risk",
                                      model_family="logistic", random_seed=20)
            output = output.rename(columns={"raw_probability": "risk_head"})
        else:
            raise ValueError("undeclared H04 family " + str(family))
        charged += int(config["fits"])
        predictions[config_id] = output
        cards[config_id] = dict(card, config_id=config_id, family=family,
                                fits_charged=int(config["fits"]))
    if charged > MODEL_FIT_CAP:
        raise ValueError("H04 fit budget exceeded: " + str(charged))

    risk_head = predictions["independent_risk_head"].set_index("sample_id")["risk_head"]
    risk_cut = gate_threshold(risk_head.loc[
        risk_head.index.isin(set(calibration_ids))].dropna().to_numpy(), 0.5)
    identity = samples[["sample_id", "entity_id", "signal_date", "prediction_at"]]
    realized = resolved.set_index("sample_id")["target"]
    calendar = sorted(samples.signal_date.astype(str).unique().tolist())
    evaluation_segments = {"selection-policy", "outer-test"}
    in_evaluation = assignment.segment.isin(evaluation_segments).to_numpy()
    evaluation_days = int(samples.loc[in_evaluation, "signal_date"].nunique())

    registry_rows, frontier_rows, long_rows = [], [], []
    for config in CONFIGS:
        config_id = config["config_id"]
        output = predictions[config_id]
        control_only = config["family"] == "binary_risk"
        if control_only:
            # The risk head defines the gate; it is not a competing selector.
            score_values = pd.Series(np.nan, index=output.index)
            own_risk_values = output["risk_head"]
        elif "p_hit15" in output.columns:
            score_values, own_risk_values = output["p_hit15"], output["p_risk10"]
        else:
            column = "score" if "score" in output.columns else "p_A"
            score_values = output[column]
            own_risk_values = pd.Series(np.nan, index=output.index)
        # Align strictly by sample id so a row-order change can never silently
        # attach one stock's probability to another.
        score = pd.Series(np.asarray(score_values, dtype=float),
                          index=output["sample_id"].to_numpy())
        own_risk = pd.Series(np.asarray(own_risk_values, dtype=float),
                             index=output["sample_id"].to_numpy())
        status = pd.Series(output["prediction_status"].to_numpy(),
                           index=output["sample_id"].to_numpy())
        frame = identity.copy()
        frame["score"] = score.reindex(frame.sample_id).to_numpy()
        frame["risk"] = risk_head.reindex(frame.sample_id).to_numpy()
        frame["own_risk"] = own_risk.reindex(frame.sample_id).to_numpy()
        for name in CLASS_ORDER:
            column = "p_" + name
            frame[column] = (pd.Series(np.asarray(output[column], dtype=float),
                                       index=output["sample_id"].to_numpy())
                             .reindex(frame.sample_id).to_numpy()
                             if column in output.columns else np.nan)
        frame["segment"] = assignment.segment.to_numpy()
        frame["feature_ready"] = np.asarray(output["feature_ready"], dtype=bool)
        frame["prediction_status"] = status.reindex(frame.sample_id).to_numpy()
        frame["config_id"] = config_id
        frame["realized_target"] = realized.reindex(frame.sample_id).to_numpy()
        long_rows.append(frame)

        if not control_only:
            # Id-indexed series: the policy layer must never rely on row position.
            score_by_id = pd.Series(frame["score"].to_numpy(), index=frame["sample_id"].to_numpy())
            risk_by_id = pd.Series(frame["risk"].to_numpy(), index=frame["sample_id"].to_numpy())
            for policy_id, use_gate in (("score_only_top20", False), ("risk_gated_top20", True)):
                selected, report = evaluate_selection(
                    identity, score_by_id, risk_by_id, calendar,
                    policy_id=f"{config_id}::{policy_id}",
                    use_gate=use_gate, risk_cut=risk_cut, n_cap=20, frozen_at=FROZEN_AT)
                chosen = frame.set_index("sample_id").reindex(selected)
                chosen = chosen[chosen.segment.isin(evaluation_segments)]
                if chosen.empty:
                    frontier_rows.append(dict(config_id=config_id, policy_id=policy_id,
                                              selected=0, coverage=0.0, realized_A=None,
                                              realized_hit15=None, realized_risk10=None))
                    continue
                frontier_rows.append(dict(
                    config_id=config_id, policy_id=policy_id, selected=int(len(chosen)),
                    coverage=float(len(chosen) / (20 * max(evaluation_days, 1))),
                    realized_A=float(chosen.realized_target.eq("A").mean()),
                    realized_hit15=float(chosen.realized_target.isin(["A", "B"]).mean()),
                    realized_risk10=float(chosen.realized_target.isin(["B", "D"]).mean()),
                    mean_score=float(chosen.score.mean()), mean_risk=float(chosen.risk.mean())))
        registry_rows.append(dict(
            config_id=config_id, family=config["family"], fits_charged=int(config["fits"]),
            status="CONTROL_ONLY" if control_only else "COMPLETED",
            predicted_rows=int(score.notna().sum()) if not control_only
            else int(own_risk.notna().sum()),
            reuse_note=cards[config_id].get("reused_from")))

    frontier = pd.DataFrame(frontier_rows)
    gated = frontier[frontier.policy_id == "risk_gated_top20"].copy()
    gated["non_dominated"] = _non_dominated(gated)
    frontier = frontier.merge(
        gated[["config_id", "non_dominated"]], on="config_id", how="left")
    frontier["non_dominated"] = frontier["non_dominated"].fillna(False)

    long = pd.concat(long_rows, ignore_index=True)
    long["prediction_kind"] = "forward_reference_not_out_of_fold"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("trial_registry.sqlite", "oof_predictions.parquet", "model_cards.json",
                 "development_frontier.csv"):
        if (directory / name).exists():
            raise ValueError("H04 refuses existing outputs: " + name)
    long.to_parquet(directory / "oof_predictions.parquet", index=False)
    frontier.to_csv(directory / "development_frontier.csv", index=False)
    atomic_json(directory / "model_cards.json", {
        "at": now(), "target_id": TARGET_ID, "classes": list(CLASS_ORDER),
        "configs": cards, "model_fit_cap": MODEL_FIT_CAP, "fits_charged": charged,
        "registered_configs": [config["config_id"] for config in CONFIGS],
        "formal_H04_accepted": False, "formal_training_authorized": False,
        "production_eligible": False, "acceptance_gaps": list(ACCEPTANCE_GAPS)})
    with sqlite3.connect(directory / "trial_registry.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS trials (config_id TEXT PRIMARY KEY, "
                   "family TEXT, fits_charged INTEGER, status TEXT, predicted_rows INTEGER, "
                   "reuse_note TEXT, formal_H04_accepted INTEGER)")
        db.executemany("INSERT INTO trials VALUES(?,?,?,?,?,?,0)", [
            (row["config_id"], row["family"], row["fits_charged"], row["status"],
             row["predicted_rows"], row["reuse_note"]) for row in registry_rows])
        db.commit()
    return {
        "formal_gate_passed": False,
        "formal_H04_accepted": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "configs": [config["config_id"] for config in CONFIGS],
        "fits_charged": charged,
        "fit_cap": MODEL_FIT_CAP,
        "risk_gate_threshold": risk_cut,
        "evaluation_signal_days": evaluation_days,
        "frontier": frontier_rows,
        "excluded_unresolved_rows": dropped,
        "source_sha256": {"samples": digest(source / "samples.parquet"),
                          "features": digest(source / "features.parquet")},
        "validation_scope": "single registered diagnostic competition; not formal H04",
    }


def _non_dominated(gated: pd.DataFrame) -> list[bool]:
    """Maximise realized safe-target rate while minimising realized risk."""
    flags = []
    a = gated.realized_A.fillna(-np.inf).to_numpy()
    r = gated.realized_risk10.fillna(np.inf).to_numpy()
    for index in range(len(gated)):
        dominated = False
        for other in range(len(gated)):
            if other == index:
                continue
            if a[other] >= a[index] and r[other] <= r[index] and (a[other] > a[index] or r[other] < r[index]):
                dominated = True
                break
        flags.append(not dominated)
    return flags
