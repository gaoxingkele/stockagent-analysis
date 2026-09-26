from copy import deepcopy

import pandas as pd
import pytest

from research.s20_harness.suspension_semantics import resolve, build


def fixture():
    rows = pd.DataFrame([
        dict(ts_code=code, trade_date="20260115", suspend_type=kind, suspend_timing=None,
             quote_present=False, received_at="2026-09-13T00:00:00Z", receipt_sha256="b" * 64)
        for code, kind in [("688005.SH", "R"), ("688005.SH", "S"), ("000001.SZ", "S")]
    ])
    case = dict(ts_code="688005.SH", trade_date="20260115",
                provider_rows=[dict(type="S", timing=None), dict(type="R", timing=None)],
                quote_present=False, retrospective_state="full_day_suspended",
                source_url="https://example.test/issuer.pdf", source_notice="2026-005",
                source_publication_date="2026-01-16", semantic_finding="continued suspension",
                scope_limit="retrospective evidence only")
    review = dict(audit_rows_sha256="a" * 64, automatic_override_enabled=False,
                  historical_receipt_availability_proven=False, reviewed_cases=[case], remaining_cases=[])
    return rows, review


def test_retrospective_overlay_preserves_rows_and_never_grants_historical_use():
    rows, review = fixture()
    result = resolve(rows, review, "a" * 64, "2026-09-14T00:00:00Z")
    pd.testing.assert_frame_equal(result[rows.columns], rows)
    assert result.review_state.tolist() == ["full_day_suspended", "full_day_suspended", "not_event_reviewed"]
    assert not result.historical_prediction_eligible.any()
    assert not result.executable_fill_proven.any()
    assert result.loc[0, "review_resolved_at"].startswith("2026-09-14")
    assert pd.isna(result.loc[2, "review_resolved_at"])


@pytest.mark.parametrize("mutation,match", [
    ("pin", "pin mismatch"), ("override", "overrides unsupported"),
    ("duplicate", "duplicate review"), ("row", "raw event mismatch"),
    ("state", "contradicts quote"), ("missing_source", "provenance"),
])
def test_invalid_reviews_fail_closed(mutation, match):
    rows, review = fixture()
    if mutation == "pin": review["audit_rows_sha256"] = "c" * 64
    if mutation == "override": review["automatic_override_enabled"] = True
    if mutation == "duplicate": review["remaining_cases"] = deepcopy(review["reviewed_cases"])
    if mutation == "row": review["reviewed_cases"][0]["provider_rows"].pop()
    if mutation == "state": review["reviewed_cases"][0]["retrospective_state"] = "resumption_announced_quote_observed"
    if mutation == "missing_source": review["reviewed_cases"][0]["source_url"] = ""
    with pytest.raises(ValueError, match=match):
        resolve(rows, review, "a" * 64, "2026-09-14T00:00:00Z")


def test_resolution_cannot_predate_receipts():
    rows, review = fixture()
    with pytest.raises(ValueError, match="precedes"):
        resolve(rows, review, "a" * 64, "2026-01-15T00:00:00Z")


def test_build_pins_and_materializes(tmp_path):
    from research.s20_harness.runtime import atomic_json, digest
    rows, review = fixture()
    p, r = tmp_path / "raw.parquet", tmp_path / "review.json"
    rows.to_parquet(p, index=False)
    review["audit_rows_sha256"] = digest(p)
    atomic_json(r, review)
    report = build(tmp_path, p, digest(p), r, digest(r))
    assert report["rows"] == 3
    assert report["reviewed_cases"] == 1
    assert report["formal_training_authorized"] is False
    with pytest.raises(ValueError, match="pin mismatch"):
        build(tmp_path, p, "0" * 64, r, digest(r))
