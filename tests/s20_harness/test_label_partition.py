import pytest

from research.s20_harness.label_partition import event_codes


def test_alias_event_scope_is_not_only_current_ticker():
    aliases = [{"entity_id": "issuer", "old_code": "old", "new_code": "new", "share_conversion_ratio": 1}]
    assert event_codes("issuer", "new", aliases) == ["new", "old"]
    assert event_codes("issuer", "old", aliases) == ["new", "old"]
    assert event_codes("unrelated", "other", aliases) == ["other"]
    with pytest.raises(ValueError, match="inconsistent"):
        event_codes("issuer", "other", aliases)
    with pytest.raises(ValueError, match="multiple"):
        event_codes("issuer", "new", aliases * 2)
