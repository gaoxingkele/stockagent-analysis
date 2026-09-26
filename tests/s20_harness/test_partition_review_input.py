import pytest

from research.s20_harness.label_partition import review_input
from research.s20_harness.runtime import atomic_json, digest


def test_custom_review_requires_external_hash_and_preserves_default(tmp_path):
    p = tmp_path / "reviews.json"
    atomic_json(p, {"reviews": {}})
    assert review_input(tmp_path) == (tmp_path / "config/s20_v4_distribution_reviews.json", None)
    for path, sha in [(p, None), (None, digest(p)), (p, "wrong")]:
        with pytest.raises(ValueError):
            review_input(tmp_path, path, sha)
    expected = digest(p)
    assert review_input(tmp_path, p, expected) == (p, expected)
    atomic_json(p, {"reviews": {"changed": {}}})
    with pytest.raises(ValueError, match="hash mismatch"):
        review_input(tmp_path, p, expected)
