"""H06 executor: feature-group ablation and guarded activation decisions.

Relative-information tests, not importance theatre. Each declared group is
removed from the base five-feature contract, and the added groups include a
time-shuffled control so a gain that a shuffled twin also reproduces cannot be
reported as an increment.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .abcds_model import CLASS_ORDER, TARGET_ID, run as fit_abcds
from .policy_support import evaluate_selection, gate_threshold
from .runtime import atomic_json, now
from .splits import assign_segments, training_ids

FROZEN_AT = "2024-05-01T00:00:00+08:00"
N_CAP = 20
MODEL_FIT_CAP = 18

BASE_FEATURES = ("raw_return20", "raw_ma20_distance", "raw_mean_tr14_pct",
                 "raw_return_vol20", "volume_ratio20")

# Declared groups. The shuffled twin is a control, never a candidate feature set.
GROUPS = {
    "trend": ("raw_return20", "raw_ma20_distance"),
    "volatility": ("raw_mean_tr14_pct", "raw_return_vol20"),
    "liquidity": ("volume_ratio20",),
}
RSI_COLUMNS = ("rsi14_wilder", "rsi_avg_gain14", "rsi_avg_loss14", "rsi_down_tr14")
CONTROL_COLUMNS = tuple("shuffle_" + name for name in RSI_COLUMNS)

# Declared before evaluation.
CONFIGS = (
    dict(config_id="base_five", keep=("trend", "volatility", "liquidity"), add=()),
    dict(config_id="drop_trend", keep=("volatility", "liquidity"), add=()),
    dict(config_id="drop_volatility", keep=("trend", "liquidity"), add=()),
    dict(config_id="drop_liquidity", keep=("trend", "volatility"), add=()),
    dict(config_id="add_rsi_wilder", keep=("trend", "volatility", "liquidity"), add=RSI_COLUMNS),
    dict(config_id="add_shuffled_control", keep=("trend", "volatility", "liquidity"),
         add=CONTROL_COLUMNS),
)

# Guarded advanced families: none may activate without residual evidence.
ADVANCED_FAMILIES = (
    ("sequence_models", "no residual evidence that time order adds information over summary features"),
    ("market_graph_relations", "no ex-ante relation graph admitted; a backward-looking graph would leak"),
    ("joint_path_generation", "synthetic paths cannot add independent market evidence"),
    ("cost_sensitive_weighting", "output rescaling was not yet separated from true weighted training"),
)

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; ablation inherits the diagnostic chain",
    "single chronological split and a single seed; group increments are not cross-fold validated",
    "the shuffled twin is a control, not proof that the added family is inert",
    "no formal H06 activation of any advanced family",
)


def _columns_for(config) -> list[str]:
    columns = [name for group in config["keep"] for name in GROUPS[group]]
    return columns + list(config["add"])


def build(root: Path, directory: Path, *, upstream_directories: dict[str, Path]) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    upstream = {name: Path(path) for name, path in upstream_directories.items()}
    for required in ("H02", "H03", "H04"):
        if required not in upstream:
            raise ValueError("H06 requires upstream outputs: " + required)
    labels = pd.read_parquet(upstream["H02"] / "labels.parquet")
    split_manifest = json.loads(
        (upstream["H03"] / "split_manifest.json").read_text(encoding="utf-8"))
    boundaries = split_manifest["segments"]
    h04 = pd.read_parquet(upstream["H04"] / "oof_predictions.parquet")

    source = root / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"
    samples = pd.read_parquet(source / "samples.parquet")
    features = pd.read_parquet(source / "features.parquet")
    resolved = labels[labels.target.notna()].copy()
    keep = set(resolved.sample_id)
    samples = samples[samples.sample_id.isin(keep)].reset_index(drop=True)
    features = features[features.sample_id.isin(keep)].reset_index(drop=True)
    assignment, split = assign_segments(samples, boundaries)
    if split["split_protocol_sha256"] != split_manifest["split_protocol_sha256"]:
        raise ValueError("H06 rebuilt a different split protocol")
    fit_ids = training_ids(assignment, "fit")
    calibration_ids = training_ids(assignment, "calibration")
    fit_labels = resolved[resolved.sample_id.isin(fit_ids)][["sample_id", "target"]].copy()
    if set(fit_labels.sample_id) != set(fit_ids):
        raise ValueError("H06 fit labels must cover exactly the eligible fit rows")

    risk_rows = h04[h04.config_id.eq("independent_risk_head")]
    risk_head = pd.Series(pd.to_numeric(risk_rows["risk"], errors="coerce").to_numpy(),
                          index=risk_rows["sample_id"].to_numpy())
    risk_cut = gate_threshold(risk_head.reindex(calibration_ids).dropna().to_numpy(), 0.5)

    identity = samples[["sample_id", "entity_id", "signal_date", "prediction_at"]]
    realized = resolved.set_index("sample_id")["target"]
    calendar = sorted(samples.signal_date.astype(str).unique().tolist())
    evaluation_segments = {"selection-policy", "outer-test"}
    in_evaluation = assignment.segment.isin(evaluation_segments).to_numpy()
    evaluation_days = int(samples.loc[in_evaluation, "signal_date"].nunique())

    rows, long_rows, cards = [], [], {}
    charged = 0
    for config in CONFIGS:
        columns = _columns_for(config)
        missing = [name for name in columns if name not in features.columns]
        if missing:
            raise ValueError(f"H06 config {config['config_id']} needs missing columns: {missing}")
        contract = {"role": "prediction_features", "columns": columns,
                    "source_provenance_verified": True}
        dropped = [name for name in GROUPS if name not in config["keep"]]
        output, card = fit_abcds(samples, features[["sample_id", *columns]], fit_labels,
                                 boundaries, contract, target_id=TARGET_ID,
                                 random_seed=20, regularisation=1.0)
        charged += int(card["model_level_fits"])
        cards[config["config_id"]] = dict(card, config_id=config["config_id"],
                                          feature_columns=columns,
                                          dropped_groups=dropped,
                                          added_columns=list(config["add"]))
        score = pd.Series(np.asarray(output["p_hit15"], dtype=float),
                          index=output["sample_id"].to_numpy())
        long_rows.append(pd.DataFrame({
            "sample_id": output["sample_id"], "segment": output["segment"],
            "config_id": config["config_id"], "p_hit15": score.to_numpy(),
            "p_risk10": np.asarray(output["p_risk10"], dtype=float),
            "prediction_status": output["prediction_status"]}))
        selected, _ = evaluate_selection(
            identity, score, risk_head, calendar,
            policy_id=f"{config['config_id']}::risk_gated_top20",
            use_gate=True, n_cap=N_CAP, frozen_at=FROZEN_AT, risk_cut=risk_cut)
        pick = pd.DataFrame({"sample_id": selected})
        pick["segment"] = assignment.set_index("sample_id").segment.reindex(
            pick.sample_id).to_numpy()
        pick["target"] = realized.reindex(pick.sample_id).to_numpy()
        pick["score"] = score.reindex(pick.sample_id).to_numpy()
        evaluated = pick[pick.segment.isin(evaluation_segments)]
        rows.append(dict(
            config_id=config["config_id"], n_features=len(columns),
            feature_columns="|".join(columns),
            dropped_groups="|".join(dropped),
            added_columns="|".join(config["add"]),
            calibration_log_loss=_calibration_log_loss(output, assignment, resolved,
                                                       calibration_ids, columns),
            selected=int(len(evaluated)),
            coverage=float(len(evaluated) / (N_CAP * max(evaluation_days, 1))),
            realized_A=None if evaluated.empty else float(evaluated.target.eq("A").mean()),
            realized_hit15=None if evaluated.empty else
            float(evaluated.target.isin(["A", "B"]).mean()),
            realized_risk10=None if evaluated.empty else
            float(evaluated.target.isin(["B", "D"]).mean()),
            risk_gate_threshold=risk_cut))
    if charged > MODEL_FIT_CAP:
        raise ValueError("H06 fit budget exceeded")

    frame = pd.DataFrame(rows)
    baseline = frame[frame.config_id.eq("base_five")].iloc[0]
    frame["delta_log_loss_vs_base"] = frame.calibration_log_loss - baseline.calibration_log_loss
    frame["delta_realized_A_vs_base"] = (
        frame.realized_A - baseline.realized_A) if baseline.realized_A is not None else np.nan
    reshuffle = frame[frame.config_id.eq("add_shuffled_control")]
    control_loss = float(reshuffle.calibration_log_loss.iloc[0]) if len(reshuffle) else None

    long = pd.concat(long_rows, ignore_index=True)
    long["prediction_kind"] = "forward_reference_not_out_of_fold"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("ablation.csv", "factor_cards.json", "advanced_activation_decisions.json",
                 "advanced_oof.parquet"):
        if (directory / name).exists():
            raise ValueError("H06 refuses existing outputs: " + name)
    long.to_parquet(directory / "advanced_oof.parquet", index=False)
    frame.to_csv(directory / "ablation.csv", index=False)
    factor_cards = {
        "at": now(), "target_id": TARGET_ID, "classes": list(CLASS_ORDER),
        "registered_fit_cap": MODEL_FIT_CAP, "fits_charged": charged,
        "risk_gate_threshold": risk_cut, "shuffled_control_log_loss": control_loss,
        "groups": {name: {"members": list(members), "role": "declared_group"}
                   for name, members in GROUPS.items()},
        "evidence": rows,
        "reading_rule": ("a group is only credited with an increment if its removal worsens "
                         "the calibration loss AND the shuffled twin does not reproduce the gain"),
        "formal_H06_accepted": False, "formal_training_authorized": False,
        "production_eligible": False, "acceptance_gaps": list(ACCEPTANCE_GAPS),
    }
    activation = {
        "at": now(),
        "decisions": [{"family": name, "activated": False, "reason": reason,
                       "requires": "documented residual error and a registered budget"}
                      for name, reason in ADVANCED_FAMILIES],
        "activated_families": [], "concurrent_advanced_families_max": 2,
        "note": "no advanced family is activated by this diagnostic run",
        "formal_H06_accepted": False,
    }
    atomic_json(directory / "factor_cards.json", factor_cards)
    atomic_json(directory / "advanced_activation_decisions.json", activation)
    return {
        "formal_gate_passed": False,
        "formal_H06_accepted": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "fits_charged": charged,
        "fit_cap": MODEL_FIT_CAP,
        "configs": [config["config_id"] for config in CONFIGS],
        "ablation": rows,
        "activated_families": [],
        "validation_scope": "single-split group ablation; not formal H06 activation",
    }


def _calibration_log_loss(output, assignment, resolved, calibration_ids, columns):
    """Multiclass log loss on the mature calibration segment only."""
    index = {name: position for position, name in enumerate(CLASS_ORDER)}
    frame = output.set_index("sample_id")
    target = resolved.set_index("sample_id")["target"]
    ids = [sample for sample in calibration_ids if sample in frame.index]
    if not ids:
        raise ValueError("no calibration rows available for ablation scoring")
    probabilities = frame.loc[ids, ["p_" + name for name in CLASS_ORDER]].to_numpy(dtype=float)
    truth = np.array([index[target.loc[sample]] for sample in ids])
    if not np.isfinite(probabilities).all():
        raise ValueError("ablation calibration probabilities must be finite")
    clipped = np.clip(probabilities, 1e-12, 1)
    return float(-np.mean(np.log(clipped[np.arange(len(ids)), truth])))
