import pandas as pd

from research.s20_harness.oof_audit import audit, membership_hash
from research.s20_harness.runtime import digest


def fixture(tmp_path):
    model = tmp_path / "model.txt"
    model.write_text("test artifact", encoding="utf-8")
    rows = pd.DataFrame([
        dict(sample_id="train", entity_id="issuer", prediction_at="2024-01-02T21:00:00+08:00",
             feature_available_at="2024-01-02T20:00:00+08:00", horizon_close_at="2024-01-30T15:00:00+08:00",
             label_available_at="2024-01-30T21:00:00+08:00"),
        dict(sample_id="future", entity_id="issuer", prediction_at="2024-02-02T21:00:00+08:00",
             feature_available_at="2024-02-02T20:00:00+08:00", horizon_close_at="2024-03-01T15:00:00+08:00",
             label_available_at=None)])
    models = {"m": dict(model_path=str(model), model_sha256=digest(model), dependency_sha256=membership_hash(["train"]),
                         information_cutoff_at="2024-02-01T21:00:00+08:00")}
    predictions = pd.DataFrame([dict(sample_id="future", model_id="m", score=.6)])
    return rows, predictions, models, {"m": ["train"]}


def test_future_same_stock_is_allowed_without_future_outcome(tmp_path):
    result, report = audit(*fixture(tmp_path))
    assert result.recorded_oof_provenance_valid.all()
    assert report["valid_recorded_provenance"] == 1
    assert not report["formal_training_authorized"]


def test_alias_same_entity_time_cannot_hide_in_sample_anchor(tmp_path):
    rows, predictions, models, deps = fixture(tmp_path)
    alias = rows.iloc[0].copy()
    alias["sample_id"] = "alias"
    rows = pd.concat([rows, alias.to_frame().T], ignore_index=True)
    predictions.loc[0, "sample_id"] = "alias"
    result, _ = audit(rows, predictions, models, deps)
    assert "prediction_entity_time_used_as_model_dependency" in result.reasons.iloc[0]


def test_unmatured_dependency_and_tampered_artifact_rejected(tmp_path):
    rows, predictions, models, deps = fixture(tmp_path)
    rows.loc[0, "label_available_at"] = "2024-02-01T21:00:00+08:00"
    models["m"]["model_sha256"] = "bad"
    result, _ = audit(rows, predictions, models, deps)
    assert "dependency_label_not_mature_before_cutoff" in result.reasons.iloc[0]
    assert "model_artifact_hash_mismatch" in result.reasons.iloc[0]
