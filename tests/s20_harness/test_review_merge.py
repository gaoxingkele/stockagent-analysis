from copy import deepcopy
import pytest

from research.s20_harness.review_merge import merge
from research.s20_harness import review_merge
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_distribution_adapter import frame, review


def test_agreeing_reviews_keep_both_sources_and_preserve_first_metadata():
    data = frame()
    left = review(data)
    right = deepcopy(left)
    right["e"].update(source="second_source", gross_cash_per_share=1.)
    result, decisions = merge(data, [("manual", left), ("template", right)])
    assert result["e"]["source"] == "reviewed_fixture"
    assert result["e"]["gross_cash_per_share"] == 1.
    assert result["e"]["merge_source_ids"] == ["manual", "template"]
    assert len(decisions[0]["lineage"]) == 2
    assert left == review(data) and not result["e"]["formal_training_eligible"]


def test_conflict_is_not_first_or_last_writer_wins():
    data = frame()
    a = review(data)
    b = deepcopy(a)
    b["e"]["beneficiary_scope"] = "registered_cdr_holders_verified"
    for bundles in [[("a", a), ("b", b)], [("b", b), ("a", a)]]:
        result, ds = merge(data, bundles)
        assert not result and "review_conflict:beneficiary_scope" in ds[0]["reasons"]


def test_even_single_review_must_match_full_source_and_explicit_terms():
    data = frame()
    for key, value in [("row_sha256", "wrong"), ("gross_cash_per_share", 2.), ("record_date", "20240102")]:
        r = review(data)
        r["e"][key] = value
        combined, ds = merge(data, [("a", r)])
        assert not combined and ds[0]["reasons"]
    r = review(data)
    r["missing"] = r.pop("e")
    assert merge(data, [("a", r)])[1][0]["reasons"] == ["source_event_missing"]


def test_build_rejects_modified_bundle_and_midrun_source_change(tmp_path, monkeypatch):
    data = frame()
    source, bundle = tmp_path / "source.parquet", tmp_path / "review.json"
    data.to_parquet(source)
    atomic_json(bundle, {"reviews": review(data)})
    source_sha, review_sha = digest(source), digest(bundle)
    changed = review(data)
    changed["e"]["source"] = "modified"
    atomic_json(bundle, {"reviews": changed})
    with pytest.raises(ValueError, match="pin mismatch"):
        review_merge.build(tmp_path, source, source_sha, [(bundle, review_sha)])
    review_sha = digest(bundle)
    real_merge = review_merge.merge
    def mutate_after_read(frame, bundles):
        result = real_merge(frame, bundles)
        atomic_json(bundle, {"reviews": {}})
        return result
    monkeypatch.setattr(review_merge, "merge", mutate_after_read)
    with pytest.raises(ValueError, match="pin mismatch"):
        review_merge.build(tmp_path, source, source_sha, [(bundle, review_sha)])
    assert not (tmp_path / "output").exists()
