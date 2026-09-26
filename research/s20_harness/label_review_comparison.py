"""Paired evidence-only diagnostic label comparison, never model performance."""
import json
from pathlib import Path
import uuid

import pandas as pd

from .label_partition_verify import verify
from .runtime import atomic_json, digest, now


def compare(root, before_path, before_sha, after_path, after_sha):
    before_path, after_path = Path(before_path).resolve(), Path(after_path).resolve()
    before, before_check = verify(before_path, before_sha)
    after, after_check = verify(after_path, after_sha)
    summaries = [json.loads((p / "summary.json").read_text(encoding="utf-8")) for p in (before_path, after_path)]
    settings = ("tax_bounds_requested", "tax_context_basis", "instrument_scope_basis", "candidate_scope")
    if any(summaries[0].get(k) != summaries[1].get(k) for k in settings):
        raise ValueError("comparison changes label settings other than reviews")
    manifests = [json.loads((p / "inputs.json").read_text(encoding="utf-8")) for p in (before_path, after_path)]
    for s, m in zip(summaries, manifests):
        if m.get(s["review_path"]) != s["review_sha256"]:
            raise ValueError("review is not pinned to partition")
    for s, m in zip(summaries, manifests):
        m.pop(s["review_path"])
    if manifests[0] != manifests[1]:
        raise ValueError("comparison changes inputs other than reviews")
    if summaries[0]["signal_date"] != summaries[1]["signal_date"] or summaries[0]["artifact_hashes"]["candidates.parquet"] != summaries[1]["artifact_hashes"]["candidates.parquet"]:
        raise ValueError("unpaired candidates")
    if before.sample_id.tolist() != after.sample_id.tolist():
        raise ValueError("sample order mismatch")
    old, new = before.p_class.fillna("UNKNOWN"), after.p_class.fillna("UNKNOWN")
    transition = pd.DataFrame(dict(sample_id=before.sample_id, old_class=old, new_class=new,
                                   old_status=before.label_status, new_status=after.label_status))
    changed = old.ne(new) | before.label_status.ne(after.label_status)
    resolved = old.eq("UNKNOWN") & new.ne("UNKNOWN")
    known = old.ne("UNKNOWN")
    same_known_payloads = before.loc[known, "payload_json"].tolist() == after.loc[known, "payload_json"].tolist()
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("label-review-comparison-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    transition.to_parquet(out / "transitions.parquet", index=False)
    report = dict(directory=str(out), at=now(), candidate_rows=len(before),
                  before_verification=before_check, after_verification=after_check,
                  before_path=str(before_path), after_path=str(after_path),
                  review_only_input_change=True, resolved_outcomes=int(resolved.sum()),
                  changed_outcomes=int(changed.sum()), known_gross_payloads_unchanged=same_known_payloads,
                  class_transitions=transition.groupby(["old_class", "new_class"]).size().rename("count").reset_index().to_dict("records"),
                  model_performance_comparison=False, formal_training_authorized=False,
                  transitions_sha256=digest(out / "transitions.parquet"), code_sha256=digest(Path(__file__)))
    atomic_json(out / "summary.json", report)
    return report
