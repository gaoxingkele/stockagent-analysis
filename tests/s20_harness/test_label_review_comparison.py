import pandas as pd
import pytest

from research.s20_harness import label_review_comparison as comparison
from research.s20_harness.runtime import atomic_json


@pytest.mark.parametrize("difference", ["tax_bounds_requested", "source_code", None])
def test_review_only_comparison_rejects_other_settings_and_inputs(tmp_path, monkeypatch, difference):
    before, after = tmp_path/"before", tmp_path/"after"
    for p in (before, after):
        p.mkdir()
        atomic_json(p/"summary.json", dict(review_path=str(p/"review.json"), review_sha256="review",
                    signal_date="20240101", tax_bounds_requested=(p == after and difference == "tax_bounds_requested"),
                    artifact_hashes={"candidates.parquet": "same_candidates"}))
        atomic_json(p/"inputs.json", {str(p/"review.json"): "review",
                                     "code.py": "changed" if p == after and difference == "source_code" else "same"})
    def verified(p, sha):
        cls = None if p == before else "D"
        return pd.DataFrame([dict(sample_id="s", p_class=cls, label_status="unknown" if cls is None else "complete",
                                  payload_json="{}")]), {}
    monkeypatch.setattr(comparison, "verify", verified)
    if difference:
        with pytest.raises(ValueError, match="changes .*other than reviews"):
            comparison.compare(tmp_path, before, "", after, "")
        assert not (tmp_path/"output").exists()
    else:
        r = comparison.compare(tmp_path, before, "", after, "")
        assert r["resolved_outcomes"] == 1 and r["candidate_rows"] == 1
        assert not r["model_performance_comparison"]
