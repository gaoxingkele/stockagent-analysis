import json
from copy import deepcopy

import pandas as pd
import pytest

from research.s20_harness.label_event_audit import route


def fixture():
    rows = [dict(sample_id=s, entity_id="stock", signal_date=day,
                 label_status="unknown_event_terms" if ids else "unknown_path",
                 payload_json=json.dumps(dict(unresolved_event_ids=ids)))
            for s, day, ids in [("a", "20240624", ["e", "f"]),
                                ("b", "20240701", ["e"]), ("c", "20240701", [])]]
    ds = [dict(normalized_event_id=e, ts_code="stock", accepted_for_accounting=False,
               reasons=rs) for e, rs in [("e", ["scope", "amount"]), ("f", ["scope"])]]
    return pd.DataFrame(rows), ds


def test_preserves_all_samples_and_separates_event_and_sample_denominators():
    frame, ds = fixture()
    original = deepcopy(ds)
    samples, links, r = route(frame, ds)
    assert [s["sample_id"] for s in samples] == ["a", "b", "c"]
    assert len(links) == 3 and r["affected_candidates"] == 2
    assert r["unique_unresolved_events"] == 2 and r["unique_entities"] == 1
    assert r["affected_candidate_reasons"] == {"amount": 2, "scope": 2}
    assert r["unique_event_reasons"] == {"amount": 1, "scope": 2}
    assert ds == original and not r["formal_training_authorized"]


@pytest.mark.parametrize("fault", ["duplicate_sample", "duplicate_event", "missing", "accepted", "no_reason", "duplicate_link", "wrong_status"])
def test_refuses_inconsistent_links(fault):
    f, ds = fixture()
    if fault == "duplicate_sample": f.loc[1, "sample_id"] = "a"
    if fault == "duplicate_event": ds.append(ds[0])
    if fault == "missing": ds.pop()
    if fault == "accepted": ds[0]["accepted_for_accounting"] = True
    if fault == "no_reason": ds[0]["reasons"] = []
    if fault == "duplicate_link": f.loc[0, "payload_json"] = json.dumps(dict(unresolved_event_ids=["e", "e"]))
    if fault == "wrong_status": f.loc[0, "label_status"] = "complete"
    with pytest.raises(ValueError): route(f, ds)


@pytest.mark.parametrize("fault", [None, "source", "pin", "mutation", "duplicate_date"])
def test_build_pins_ledger_and_partition_sources(tmp_path, monkeypatch, fault):
    from research.s20_harness import label_event_audit as audit
    from research.s20_harness.runtime import atomic_json, digest
    frame, ds = fixture()
    frame = frame.iloc[:1].copy()
    ledger, part = tmp_path/"ledger", tmp_path/"part"
    ledger.mkdir(); part.mkdir()
    source = tmp_path/"source.json"
    atomic_json(source, {})
    atomic_json(ledger/"events.json", [])
    atomic_json(ledger/"decisions.json", ds)
    atomic_json(ledger/"summary.json", dict(source_rows=2, accepted_gross_accounting_events=0,
        unresolved_events=2, events_sha256=digest(ledger/"events.json"),
        decisions_sha256=digest(ledger/"decisions.json"), source_path=str(source), source_sha256=digest(source)))
    atomic_json(part/"inputs.json", {str(source): "wrong" if fault == "source" else digest(source)})
    atomic_json(part/"summary.json", dict(signal_date="20240624", artifact_hashes={"inputs.json": digest(part/"inputs.json")}))
    lh, ph = digest(ledger/"summary.json"), digest(part/"summary.json")
    monkeypatch.setattr(audit, "verify", lambda p, h: (frame, {}))
    if fault == "pin": atomic_json(ledger/"decisions.json", [])
    if fault == "mutation":
        real = audit.route
        def mutate(*args):
            result = real(*args)
            atomic_json(source, {"changed": True})
            return result
        monkeypatch.setattr(audit, "route", mutate)
    parts = [(part, ph)]
    if fault == "duplicate_date": parts += [(part, ph)]
    if fault:
        with pytest.raises(ValueError): audit.build(tmp_path, parts, ledger, lh)
        assert not (tmp_path/"output").exists()
    else:
        r = audit.build(tmp_path, parts, ledger, lh)
        assert r["candidate_rows"] == 1 and r["sample_event_links"] == 2
