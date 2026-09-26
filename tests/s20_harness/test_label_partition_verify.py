import json

import pandas as pd
import pytest

from research.s20_harness.label_partition_verify import validate_rows, verify


def test_partition_consumer_label_consistency():
    candidates = pd.DataFrame([{"sample_id": "s", "entity_id": "e", "signal_date": "20260529"}])
    labels = candidates.assign(p_class="A", label_realized=True, formal_training_eligible=False,
                               event_coverage_proven=False, recommendation_kept=True, label_status="diagnostic")
    payload = {"p_class": "A", "label_realized": True, "up_event": True, "b5": False, "terminal_net": .02}
    labels["payload_json"] = json.dumps(payload)
    summary = {"candidate_rows": 1, "output_rows": 1, "class_counts": {"A": 1}, "status_counts": {"diagnostic": 1}}
    validate_rows(candidates, labels, summary)
    payload["b5"] = True
    labels["payload_json"] = json.dumps(payload)
    with pytest.raises(ValueError, match="economic class"):
        validate_rows(candidates, labels, summary)
    labels["sample_id"] = "other"
    with pytest.raises(ValueError, match="denominator"):
        validate_rows(candidates, labels, summary)


def test_consumer_requires_external_pin(tmp_path):
    with pytest.raises(ValueError, match="external expected"):
        verify(tmp_path, None)


def test_tax_sidecar_cannot_assign_favorable_class_to_bounds():
    candidates = pd.DataFrame([{"sample_id": "s", "entity_id": "e", "signal_date": "20260529"}])
    payload = {"p_class": "A", "label_realized": True, "up_event": True, "b5": False, "terminal_net": .02}
    tax = {"status": "tax_bounds_diagnostic", "p_class": None, "formal_training_eligible": False,
           "terminal_net_lower": -.01, "terminal_net_upper": .02, "b5_certain": False,
           "b5_possible": False, "class_outer_set": ["A", "C"]}
    labels = candidates.assign(p_class="A", label_realized=True, formal_training_eligible=False,
        event_coverage_proven=False, recommendation_kept=True, label_status="diagnostic",
        payload_json=json.dumps(payload), taxed_p_class=None, taxed_status="tax_bounds_diagnostic",
        taxed_payload_json=json.dumps(tax))
    summary = {"candidate_rows": 1, "output_rows": 1, "class_counts": {"A": 1},
               "status_counts": {"diagnostic": 1}, "tax_bounds_requested": True,
               "taxed_status_counts": {"tax_bounds_diagnostic": 1}}
    validate_rows(candidates, labels, summary)
    tax["p_class"] = "A"
    labels["taxed_p_class"] = "A"
    labels["taxed_payload_json"] = json.dumps(tax)
    with pytest.raises(ValueError, match="tax class set"):
        validate_rows(candidates, labels, summary)
