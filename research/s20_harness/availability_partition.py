"""Observed-now availability evidence, never backdated historical receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import pandas as pd

from .label_availability import _instant, label_maturity
from .label_partition_verify import verify
from .runtime import atomic_json, digest, load_plan, now


def attach(labels, observed_at):
    observed = _instant(observed_at)
    records = []
    for row in labels.itertuples(index=False):
        horizon = pd.to_datetime(str(row.horizon_end), format="%Y%m%d").tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        gross = label_maturity(horizon_close_at=horizon,
                               input_available_at=[observed], unresolved=not row.label_realized)
        tax_resolved = (getattr(row, "taxed_status", None) == "tax_bounds_diagnostic" and
                        not pd.isna(getattr(row, "taxed_p_class", None)))
        taxed = label_maturity(horizon_close_at=horizon,
                               input_available_at=[observed], unresolved=not tax_resolved)
        records.append({"sample_id": row.sample_id, "entity_id": row.entity_id,
                        "signal_date": row.signal_date, "horizon_close_at": horizon.isoformat(),
                        "observed_available_at": observed.isoformat(),
                        "gross_diagnostic_label_available_at": gross["label_available_at"],
                        "tax_class_available_at": taxed["label_available_at"],
                        "gross_availability_reason": gross["reason"], "tax_availability_reason": taxed["reason"],
                        "historical_replay_availability_proven": False,
                        "dependency_coverage_proven": False, "formal_training_eligible": False})
    return pd.DataFrame(records)


def build(root, partition, summary_sha256):
    root, partition = Path(root).resolve(), Path(partition).resolve()
    labels, validation = verify(partition, summary_sha256)
    inputs = load_plan(partition / "inputs.json")
    dependencies = []
    for name, sha in inputs.items():
        source = partition / "code_blobs" / sha if Path(name).suffix == ".py" else Path(name)
        if digest(source) != sha:
            raise ValueError("dependency changed before observation")
        dependencies.append({"logical_path": name, "observed_path": str(source), "sha256": sha})
    # Observation follows complete verification, not source mtime or event date.
    observed_at = now()
    frame = attach(labels, observed_at)
    output = root / "output/experiments/s20_safe_v4/sources" / ("availability-" + uuid.uuid4().hex)
    if not output.resolve().is_relative_to(root):
        raise ValueError("output escapes repository")
    output.mkdir(parents=True)
    frame.to_parquet(output / "label_availability.parquet", index=False)
    atomic_json(output / "observation.json", {"observed_at": observed_at, "partition": str(partition),
                "partition_summary_sha256": summary_sha256, "dependencies": dependencies,
                "basis": "bytes verified present now; not earliest acquisition or historical PIT proof"})
    result = {"at": now(), "directory": str(output), "rows": len(frame),
              "resolved_gross_diagnostic_availability": int(frame.gross_diagnostic_label_available_at.notna().sum()),
              "resolved_tax_class_availability": int(frame.tax_class_available_at.notna().sum()),
              "unresolved_rows_retained": True, "source_validation": validation,
              "historical_replay_availability_proven": False, "formal_training_authorized": False,
              "artifact_hashes": {name: digest(output / name) for name in ("label_availability.parquet", "observation.json")},
              "code_sha256": digest(Path(__file__))}
    atomic_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[2], args.partition, args.summary_sha256), indent=2))
