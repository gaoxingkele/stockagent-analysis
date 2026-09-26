"""H03 executor: chronological splits plus fixed reference fits.

Consumes the H02 five-class labels and the frozen sample/feature tables, builds
the five ordered segments, fits only on the ``fit`` segment, and reports
forward-segment reference probabilities. Fit-segment predictions are withheld
rather than passed off as out-of-fold.

Nothing here is formal H03 acceptance: the H01 admission failure is inherited.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_model import run as fit_baseline
from .feature_pipeline import prepare
from .runtime import atomic_json, digest, now
from .splits import SEGMENTS, assign_segments, training_ids

FEATURE_COLUMNS = ("raw_return20", "raw_ma20_distance", "raw_mean_tr14_pct",
                   "raw_return_vol20", "volume_ratio20")

SEGMENT_FRACTIONS = (0.5, 0.625, 0.75, 0.875)

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; splits inherit the frozen diagnostic sample",
    "outer-test labels stay sealed; only forward reference predictions are produced",
    "reference fits are uncalibrated and are not a baseline coverage acceptance",
    "feature provenance is declared by this contract, not independently certified",
    "no real-market pilot or legacy reproduction is claimed by this executor",
)


def _stamp(day: str) -> str:
    return f"{day[:4]}-{day[4:6]}-{day[6:]}T21:00:00+08:00"


def _plus_one_day(day: str) -> str:
    value = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}") + pd.Timedelta(days=1)
    return value.strftime("%Y-%m-%dT21:00:00+08:00")


def build_boundaries(dates: list[str]) -> list[dict]:
    """Deterministic calendar-only cut points; no outcome information is used."""
    ordered = sorted(set(dates))
    if len(ordered) < 10:
        raise ValueError("too few signal dates for five ordered segments")
    cuts = [0]
    for fraction in SEGMENT_FRACTIONS:
        cuts.append(int(len(ordered) * fraction))
    cuts.append(len(ordered))
    index = [min(max(cut, 0), len(ordered) - 1) for cut in cuts]
    for left, right in zip(index, index[1:]):
        if left >= right:
            raise ValueError("degenerate split segment; not enough distinct signal dates")
    starts = [_stamp(ordered[cut]) for cut in index]
    starts[-1] = _plus_one_day(ordered[-1])
    return [{"name": name, "start_at": start, "end_at": starts[position + 1]}
            for position, (name, start) in enumerate(zip(SEGMENTS, starts[:-1]))]


def build(root: Path, directory: Path, *, upstream_directories: dict[str, Path]) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    root_names = {name: Path(path) for name, path in upstream_directories.items()}
    missing = [name for name in ("H02",) if name not in root_names]
    if missing:
        raise ValueError("H03 requires H02 outputs: " + ", ".join(missing))
    h02 = root_names["H02"]
    labels_path = h02 / "labels.parquet"
    if not labels_path.is_file():
        raise ValueError("H02 labels.parquet missing")

    source = root / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"
    samples_path, features_path = source / "samples.parquet", source / "features.parquet"
    for path in (samples_path, features_path):
        if not path.is_file():
            raise ValueError("missing frozen sample table: " + path.name)

    samples = pd.read_parquet(samples_path)
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    if not {"sample_id", "target"}.issubset(labels.columns):
        raise ValueError("H02 labels require sample_id and target")

    resolved = labels[labels.target.notna()].copy()
    dropped = int(len(labels) - len(resolved))
    keep = set(resolved.sample_id)
    samples = samples[samples.sample_id.isin(keep)].reset_index(drop=True)
    features = features[features.sample_id.isin(keep)].reset_index(drop=True)

    boundaries = build_boundaries(samples.signal_date.astype(str).tolist())
    assignment, split_report = assign_segments(samples, boundaries)
    fit_ids = training_ids(assignment, "fit")
    if not fit_ids:
        raise ValueError("empty fit segment; refusing to fit on later segments")
    fit_labels = resolved[resolved.sample_id.isin(fit_ids)][["sample_id", "target"]].copy()
    fit_labels["target"] = fit_labels.target.eq("A")
    if set(fit_labels.sample_id) != set(fit_ids):
        raise ValueError("fit labels must cover exactly the eligible fit rows")

    # Declaration of lineage for this run only; it is echoed into the card and the
    # bundle, where the "not independently certified" gap is recorded explicitly.
    contract = {"role": "prediction_features", "columns": list(FEATURE_COLUMNS),
                "source_provenance_verified": True}
    output, card = fit_baseline(samples, features[["sample_id", *FEATURE_COLUMNS]],
                                fit_labels, boundaries, contract,
                                target_id="S20.hit15_risk10_silence8.v1.A",
                                model_family="logistic", random_seed=20)
    frequency_output, frequency_card = fit_baseline(
        samples, features[["sample_id", *FEATURE_COLUMNS]], fit_labels, boundaries, contract,
        target_id="S20.hit15_risk10_silence8.v1.A", model_family="mature_frequency", random_seed=20)

    oof = output.merge(labels[["sample_id", "target"]], on="sample_id", how="left")
    oof["reference_control_probability"] = frequency_output["raw_probability"].to_numpy()
    oof["reference_control_status"] = frequency_output["prediction_status"].to_numpy()
    oof["realized_A"] = oof.target.eq("A")

    forward = oof[oof.segment.isin(["tune", "calibration", "selection-policy", "outer-test"])]
    predicted = forward[forward.prediction_status.eq("uncalibrated_reference_prediction")]
    metrics = []
    for segment in SEGMENTS:
        rows = oof[oof.segment.eq(segment)]
        scored = rows[rows.prediction_status.eq("uncalibrated_reference_prediction")]
        metrics.append({
            "segment": segment,
            "rows": int(len(rows)),
            "scored_rows": int(len(scored)),
            "realized_A_rate": float(rows.realized_A.mean()) if len(rows) else None,
            "scored_realized_A_rate": float(scored.realized_A.mean()) if len(scored) else None,
            "mean_reference_probability": float(scored.raw_probability.mean()) if len(scored) else None,
        })

    split_manifest = {
        "at": now(),
        "segments": boundaries,
        "split_protocol_sha256": split_report["split_protocol_sha256"],
        "segment_counts": split_report["counts"],
        "rows": int(len(samples)),
        "excluded_unresolved_rows": dropped,
        "outer_labels_sealed": True,
        "evaluation_at": None,
        "cut_rule": "calendar quantiles of distinct signal dates: " + ", ".join(
            str(fraction) for fraction in SEGMENT_FRACTIONS),
        "fit_only": True,
        "formal_training_authorized": False,
    }
    bundle = {
        "at": now(),
        "label_version": str(labels.label_version.iloc[0]) if "label_version" in labels.columns else None,
        "reference_card": card,
        "frequency_card": frequency_card,
        "source_hashes": {"samples": digest(samples_path), "features": digest(features_path),
                          "labels": digest(labels_path)},
        "metrics": metrics,
        "forward_rows": int(len(forward)),
        "scored_forward_rows": int(len(predicted)),
        "formal_H03_accepted": False,
        "formal_training_authorized": False,
        "production_eligible": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "note": "reference fits only; fit-segment predictions are withheld, not OOF",
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("split_manifest.json", "baseline_metrics.csv", "baseline_oof.parquet",
                 "baseline_bundle.json"):
        if (directory / name).exists():
            raise ValueError("H03 refuses existing outputs: " + name)
    oof.to_parquet(directory / "baseline_oof.parquet", index=False)
    pd.DataFrame(metrics).to_csv(directory / "baseline_metrics.csv", index=False)
    atomic_json(directory / "split_manifest.json", split_manifest)
    atomic_json(directory / "baseline_bundle.json", bundle)
    return {
        "formal_gate_passed": False,
        "formal_H03_accepted": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "segments": {name: split_report["counts"][name] for name in SEGMENTS},
        "scored_forward_rows": int(len(predicted)),
        "fit_rows": len(fit_ids),
        "excluded_unresolved_rows": dropped,
        "validation_scope": "chronological split and reference fits on the frozen diagnostic sample",
    }
