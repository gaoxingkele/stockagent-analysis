"""Explicit retrospective review overlay, never a historical feature override."""
import json
from pathlib import Path
import uuid

import pandas as pd

from .label_availability import _instant
from .runtime import atomic_json, digest, now


def resolve(rows, review, rows_sha, resolved_at):
    resolved = _instant(resolved_at)
    if review["audit_rows_sha256"] != rows_sha:
        raise ValueError("review source pin mismatch")
    if review.get("automatic_override_enabled") is not False or review.get("historical_receipt_availability_proven") is not False:
        raise ValueError("historical/automatic overrides unsupported")
    keys = ["ts_code", "trade_date"]
    required = {*keys, "suspend_type", "suspend_timing", "quote_present", "received_at", "receipt_sha256"}
    if not required.issubset(rows) or rows[list(required - {"suspend_timing"})].isna().any().any():
        raise ValueError("missing raw evidence")
    if not rows.suspend_type.isin(["S", "R"]).all() or not rows.quote_present.map(lambda v: isinstance(v, bool)).all():
        raise ValueError("invalid raw types")
    if any(_instant(v) > resolved for v in rows.received_at):
        raise ValueError("resolution precedes source receipt")
    cases = review["reviewed_cases"]
    all_keys = [(c["ts_code"], c["trade_date"]) for c in cases + review["remaining_cases"]]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("duplicate review key")
    additions = []
    for c in cases:
        group = rows[rows.ts_code.eq(c["ts_code"]) & rows.trade_date.eq(c["trade_date"])]
        actual = sorted((r.suspend_type, "" if pd.isna(r.suspend_timing) else r.suspend_timing) for r in group.itertuples())
        expected = sorted((r["type"], r["timing"] or "") for r in c["provider_rows"])
        if not actual or actual != expected or type(c["quote_present"]) is not bool or not group.quote_present.eq(c["quote_present"]).all():
            raise ValueError("review raw event mismatch")
        state = c["retrospective_state"]
        if state not in {"full_day_suspended", "resumption_announced_quote_observed"}:
            raise ValueError("unsupported retrospective state")
        if (state == "full_day_suspended") == c["quote_present"]:
            raise ValueError("review state contradicts quote evidence")
        for field in ["source_url", "source_notice", "source_publication_date", "semantic_finding", "scope_limit"]:
            if not isinstance(c.get(field), str) or not c[field].strip():
                raise ValueError("missing review provenance")
        publication = pd.Timestamp(c["source_publication_date"])
        if pd.isna(publication) or publication.date() > resolved.date():
            raise ValueError("invalid publication date")
        additions.append({**{k: c[k] for k in keys}, "review_state": state,
                          "review_source_url": c["source_url"], "review_notice": c["source_notice"],
                          "review_publication_date": c["source_publication_date"],
                          "review_scope_limit": c["scope_limit"], "review_resolved_at": resolved.isoformat()})
    columns = [*keys, "review_state", "review_source_url", "review_notice", "review_publication_date", "review_scope_limit", "review_resolved_at"]
    if (set(columns) - set(keys)) & set(rows):
        raise ValueError("review already applied")
    result = rows.merge(pd.DataFrame(additions, columns=columns), on=keys, how="left", sort=False, validate="many_to_one")
    if not result[rows.columns].equals(rows.reset_index(drop=True)):
        raise ValueError("raw rows/order changed")
    result["review_state"] = result.review_state.fillna("not_event_reviewed")
    result["historical_prediction_eligible"] = False
    result["executable_fill_proven"] = False
    return result


def build(root, rows_path, rows_sha, review_path, review_sha):
    pins = [(Path(rows_path), rows_sha), (Path(review_path), review_sha)]
    for path, sha in pins:
        if digest(path) != sha:
            raise ValueError("input pin mismatch")
    review = json.loads(pins[1][0].read_text(encoding="utf-8"))
    table = resolve(pd.read_parquet(pins[0][0]), review, rows_sha, now())
    for path, sha in pins:
        if digest(path) != sha:
            raise ValueError("input changed during resolution")
    output = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("suspension-semantics-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    table.to_parquet(output / "rows.parquet", index=False)
    atomic_json(output / "inputs.json", [{"path": str(p.resolve()), "sha256": s} for p, s in pins])
    report = dict(directory=str(output), rows=len(table), reviewed_cases=len(review["reviewed_cases"]),
                  row_states=table.review_state.value_counts().to_dict(), raw_rows_preserved=True,
                  formal_training_authorized=False, historical_prediction_availability_proven=False,
                  local_notice_archives_verified=False, at=now(), code_sha256=digest(Path(__file__)),
                  table_sha256=digest(output / "rows.parquet"), inputs_sha256=digest(output / "inputs.json"))
    atomic_json(output / "summary.json", report)
    return report
