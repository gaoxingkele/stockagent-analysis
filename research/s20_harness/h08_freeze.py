"""H08 executor: champion freeze, sealing and power planning.

Freezing a champion is a gate, not a formality. G2 requires consistent direction
across folds and seeds; a single chronological split with a single seed cannot
meet it, so this executor records that no champion is eligible instead of
manufacturing one.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .abcds_model import TARGET_ID
from .runtime import atomic_json, digest, now

MINIMUM_PRACTICAL_EFFECT = 0.05
ALPHA_ONE_SIDED = 0.05
Z_ONE_SIDED = 1.959963985
OUTER_SEGMENT = "outer-test"

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; no champion may be frozen on this chain",
    "G2 not met: one chronological split and one seed, so fold/seed direction is untested",
    "the outer-test segment was consumed by the diagnostic evaluation and is no longer a clean holdout",
    "no prospective evaluation has started, so no unopened evidence exists yet",
)


def _required_days(per_day_differences: np.ndarray) -> dict:
    """One-sided power planning for a paired per-date mean difference."""
    known = per_day_differences[np.isfinite(per_day_differences)]
    if known.size < 2:
        return {"observed_dates": int(known.size), "observed_sd": None,
                "required_dates": None, "note": "insufficient paired dates to plan power"}
    sd = float(known.std(ddof=1))
    if sd == 0:
        return {"observed_dates": int(known.size), "observed_sd": 0.0,
                "required_dates": None,
                "note": "zero observed dispersion cannot support a sample-size plan"}
    required = (Z_ONE_SIDED * sd / MINIMUM_PRACTICAL_EFFECT) ** 2
    return {"observed_dates": int(known.size), "observed_sd": sd,
            "required_dates": float(np.ceil(required)),
            "minimum_practical_effect": MINIMUM_PRACTICAL_EFFECT,
            "alpha_one_sided": ALPHA_ONE_SIDED,
            "note": "planning aid for a paired per-date mean difference, not a promise"}


def build(root: Path, directory: Path, *, upstream_directories: dict[str, Path]) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    upstream = {name: Path(path) for name, path in upstream_directories.items()}
    for required in ("H03", "H06", "H07"):
        if required not in upstream:
            raise ValueError("H08 requires upstream outputs: " + required)
    split_manifest = json.loads(
        (upstream["H03"] / "split_manifest.json").read_text(encoding="utf-8"))
    activation = json.loads(
        (upstream["H06"] / "advanced_activation_decisions.json").read_text(encoding="utf-8"))
    report = json.loads(
        (upstream["H07"] / "locked_development_report.json").read_text(encoding="utf-8"))
    episodes = pd.read_parquet(upstream["H07"] / "episodes.parquet")

    date_counts = split_manifest["segment_counts"]
    splits_used = int(sum(1 for name, value in date_counts.items()
                          if name != OUTER_SEGMENT and value["assigned_rows"] > 0))
    seeds_used = 1
    g2_met = splits_used >= 2 and seeds_used >= 2

    candidate = episodes[episodes.role.eq("candidate")]
    control = episodes[episodes.role.eq("control")]
    left = candidate.groupby("signal_date").target.apply(lambda s: s.eq("A").mean())
    right = control.groupby("signal_date").target.apply(lambda s: s.eq("A").mean())
    joined = pd.concat([left.rename("candidate"), right.rename("control")], axis=1).dropna()
    differences = (joined.candidate - joined.control).to_numpy(dtype=float)
    power = _required_days(differences)
    power.update(preregistered=True,
                 review_trigger="mature prospective signal dates and a complete "
                                 "20-trading-day label maturity window after the last signal")

    sealed = {
        "at": now(),
        "segments": split_manifest["segments"],
        "outer_segment": OUTER_SEGMENT,
        "outer_labels_were_sealed_in_H03": split_manifest["outer_labels_sealed"],
        "outer_segment_consumed_by": "H04/H05/H06/H07 diagnostic evaluation",
        "clean_retrospective_holdout_available": False,
        "reason": ("the outer-test rows were scored and evaluated during the diagnostic chain, "
                   "so they can no longer serve as an unopened holdout"),
        "only_clean_path": "prospective collection after a frozen champion",
        "sealed_at": now(),
        "formal_H08_accepted": False,
        "production_eligible": False,
    }
    champion = {
        "at": now(),
        "target_id": TARGET_ID,
        "champion": None,
        "frozen": False,
        "reason": "G2 not met: single chronological split and single seed",
        "gate_evidence": {
            "G0_artifact_hashes": "recorded per run manifest",
            "G2_folds_used": splits_used,
            "G2_seeds_used": seeds_used,
            "G2_met": g2_met,
        },
        "candidate_evaluated": report.get("candidate"),
        "control": report.get("control"),
        "advanced_families_activated": activation.get("activated_families", []),
        "diagnostic_numbers": {
            "candidate_safe_target_rate": report.get("realized_safe_target_rate"),
            "candidate_paired_risk_rate": report.get("realized_paired_risk_rate"),
            "control_safe_target_rate": report.get("control_safe_target_rate"),
            "control_paired_risk_rate": report.get("control_paired_risk_rate"),
        },
        "artifact_hashes": {
            "episodes": digest(upstream["H07"] / "episodes.parquet"),
            "paired_intervals": digest(upstream["H07"] / "paired_intervals.json"),
            "split_manifest": digest(upstream["H03"] / "split_manifest.json"),
        },
        "failure_action": "complete research negative report if no eligible champion",
        "formal_H08_accepted": False,
        "formal_training_authorized": False,
        "production_eligible": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("champion_manifest.json", "sealing_manifest.json", "power_plan.json",
                 "holdout_access.jsonl"):
        if (directory / name).exists():
            raise ValueError("H08 refuses existing outputs: " + name)
    atomic_json(directory / "champion_manifest.json", champion)
    atomic_json(directory / "sealing_manifest.json", sealed)
    atomic_json(directory / "power_plan.json", {
        "at": now(), "target_id": TARGET_ID, "gate": "G3 relative promotion",
        "minimum_practical_effect_pp": MINIMUM_PRACTICAL_EFFECT * 100,
        "paired_metric": "same-signal-date safe-target-rate difference", **power,
        "holdout_definition": "prospective only; the retrospective outer segment is consumed",
        "formal_H08_accepted": False})
    (directory / "holdout_access.jsonl").write_text("", encoding="utf-8")
    return {
        "formal_gate_passed": False,
        "formal_H08_accepted": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "champion": None,
        "g2_met": g2_met,
        "folds_used": splits_used,
        "seeds_used": seeds_used,
        "power_plan": power,
        "clean_holdout_available": False,
        "validation_scope": "gate evaluation only; no champion frozen",
    }
