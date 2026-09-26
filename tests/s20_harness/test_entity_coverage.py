import pandas as pd
import pytest

from research.s20_harness.entity_coverage import map_metadata


def test_stable_entity_does_not_silently_merge_multiple_listing_intervals():
    aliases = [dict(old_code="old", new_code="new", entity_id="entity", share_conversion_ratio=1.0)]
    original = pd.DataFrame([dict(ts_code="new", list_date="20200101")])
    result = map_metadata(original, aliases)
    assert result.ts_code.tolist() == ["entity"]
    assert result.metadata_code.tolist() == ["new"]
    assert original.ts_code.tolist() == ["new"]
    with pytest.raises(ValueError, match="multiple metadata"):
        map_metadata(pd.DataFrame([dict(ts_code="old"), dict(ts_code="new")]), aliases)
