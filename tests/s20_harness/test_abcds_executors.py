"""Executors for the versioned ABCDS track: H02 -> H03 -> H04 -> H05."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.s20_harness import policy_support
from research.s20_harness.abcds_model import CLASS_ORDER, validate_class_order
from research.s20_harness.h06_ablation import CONFIGS as ABLATION_CONFIGS
from research.s20_harness.h06_ablation import GROUPS, _columns_for
from research.s20_harness.h07_execution import _plus_days, _simulate
from research.s20_harness.h08_freeze import _required_days

ROOT = Path(__file__).resolve().parents[2]
SCREEN = ROOT / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"


def test_class_order_is_explicit_and_fail_closed():
    assert validate_class_order(CLASS_ORDER) == CLASS_ORDER
    with pytest.raises(ValueError, match="unexpected ABCDS class order"):
        validate_class_order(("A", "B", "C", "S", "D"))
    with pytest.raises(ValueError, match="unexpected ABCDS class order"):
        validate_class_order(("A", "B", "C", "D"))


def test_candidate_alignment_never_uses_row_position():
    identity = pd.DataFrame({
        "sample_id": ["s2", "s1", "s3"],
        "entity_id": ["e2", "e1", "e3"],
        "signal_date": ["20240102"] * 3,
        "prediction_at": ["2024-01-02T21:00:00+08:00"] * 3,
    })
    # Deliberately shuffled relative to the identity order.
    scores = pd.Series({"s1": 0.1, "s3": 0.3, "s2": 0.2})
    risks = pd.Series({"s1": 0.9, "s3": 0.7, "s2": 0.8})
    candidates = policy_support.build_candidates(identity, scores, risks)
    assert list(candidates.columns) == list(policy_support.CANDIDATE_COLUMNS)
    by_id = candidates.set_index("sample_id")
    assert by_id.loc["s1", "score"] == pytest.approx(0.1)
    assert by_id.loc["s2", "score"] == pytest.approx(0.2)
    assert by_id.loc["s3", "score"] == pytest.approx(0.3)
    assert by_id.loc["s1", "risk"] == pytest.approx(0.9)


def test_gate_threshold_ignores_unavailable_scores():
    values = np.array([np.nan, 0.1, 0.2, np.nan, 0.3])
    assert policy_support.gate_threshold(values, 0.5) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="no finite calibration values"):
        policy_support.gate_threshold(np.array([np.nan, np.nan]), 0.5)
    with pytest.raises(ValueError, match="quantile"):
        policy_support.gate_threshold(values, 1.5)


def test_gated_policy_requires_a_threshold():
    with pytest.raises(ValueError, match="pre-registered risk threshold"):
        policy_support.build_policy("p", use_gate=True, n_cap=20,
                                    frozen_at="2024-01-01T00:00:00+08:00")
    with pytest.raises(ValueError, match="invalid TopN cap"):
        policy_support.build_policy("p", use_gate=False, n_cap=7,
                                    frozen_at="2024-01-01T00:00:00+08:00")
    policy = policy_support.build_policy("p", use_gate=False, n_cap=20,
                                         frozen_at="2024-01-01T00:00:00+08:00")
    assert policy["risk_target_id"] is None and policy["max_risk"] is None


def test_ablation_configs_declare_their_own_columns():
    for config in ABLATION_CONFIGS:
        columns = _columns_for(config)
        assert columns, config["config_id"]
        assert len(set(columns)) == len(columns)
        expected_drops = [name for name in GROUPS if name not in config["keep"]]
        for group in expected_drops:
            for member in GROUPS[group]:
                assert member not in columns
    assert _columns_for(ABLATION_CONFIGS[0]) == [
        "raw_return20", "raw_ma20_distance", "raw_mean_tr14_pct",
        "raw_return_vol20", "volume_ratio20"]


def test_execution_capacity_rejects_rather_than_breaching_the_cap():
    episodes = pd.DataFrame({
        "role": ["candidate"] * 3,
        "sample_id": ["a", "b", "c"],
        "entity_id": ["e1", "e2", "e3"],
        "signal_date": ["20240102"] * 3,
        "entry_date": ["20240103"] * 3,
        "exit_date": ["20240131"] * 3,
        "holding_trading_days": [20] * 3,
        "score": [0.9, 0.8, 0.7],
        "risk": [0.1, 0.1, 0.1],
        "target": ["A", "A", "A"],
        "net": [0.1, 0.1, 0.1],
    })
    calendar = ["20240102"] + [f"202401{d:02d}" for d in range(3, 32)]
    accepted, result = _simulate(episodes, 2, calendar)
    assert result["rejected_by_capacity"] == 1
    assert result["positions"] == 2
    # Highest score wins the scarce slot.
    assert "c" not in set(accepted.sample_id)
    assert result["final_nav"] > 1.0
    # Cost stress must never improve the result.
    _, stressed = _simulate(episodes, 2, calendar, 2.0)
    assert stressed["final_nav"] < result["final_nav"]


def test_trading_day_offset_is_bounded():
    calendar = ["20240102", "20240103", "20240104"]
    assert _plus_days(calendar, "20240102", 1) == "20240103"
    assert _plus_days(calendar, "20240102", 9) is None
    assert _plus_days(calendar, "20240104", 1) is None
    assert _plus_days(calendar, "20231229", 1) is None


def test_power_plan_requires_dispersion_and_scales_with_it():
    tight = _required_days(np.array([0.049, 0.051, 0.050, 0.050]))
    wide = _required_days(np.array([-0.05, 0.15, -0.05, 0.15]))
    assert tight["required_dates"] < wide["required_dates"]
    assert tight["required_dates"] >= 1
    flat = _required_days(np.array([0.0, 0.0, 0.0]))
    assert flat["required_dates"] is None
    assert "cannot support" in flat["note"]


@pytest.mark.skipif(not (SCREEN / "labels_tp15_dd10_screen.parquet").exists(),
                    reason="frozen diagnostic sample is not present")
def test_full_abcds_chain_records_real_evidence(tmp_path):
    from research.s20_harness.abcds_labels import build as h02
    from research.s20_harness.h03_baseline import build as h03
    from research.s20_harness.h04_competition import build as h04
    from research.s20_harness.h05_abcds import build as h05

    d2, d3, d4, d5 = (tmp_path / name for name in ("h02", "h03", "h04", "h05"))
    labels = h02(ROOT, d2)
    assert labels["class_counts"] == {"A": 15098, "B": 1075, "C": 9981, "D": 13690, "S": 15685}
    assert labels["formal_gate_passed"] is False
    contract = json.loads((d2 / "label_contracts.json").read_text(encoding="utf-8"))
    # The two silence definitions must stay distinguishable.
    assert contract["counts"]["S"] != contract["legacy_a_only_silence_rows"]
    assert contract["risk_definition"]["basis"] == "legacy_cost_basis"

    split = h03(ROOT, d3, upstream_directories={"H02": d2})
    assert split["formal_gate_passed"] is False
    manifest = json.loads((d3 / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outer_labels_sealed"] is True
    assert manifest["excluded_unresolved_rows"] == 1
    oof = pd.read_parquet(d3 / "baseline_oof.parquet")
    assert oof.loc[oof.segment.eq("fit"), "raw_probability"].isna().all()

    competition = h04(ROOT, d4, upstream_directories={"H02": d2, "H03": d3})
    assert competition["fits_charged"] <= competition["fit_cap"]
    assert competition["formal_H04_accepted"] is False
    frontier = pd.DataFrame(competition["frontier"])
    gated = frontier[frontier.policy_id.eq("risk_gated_top20")]
    ungated = frontier[frontier.policy_id.eq("score_only_top20")]
    assert gated.selected.gt(0).all()
    # The gate must reduce realized risk on every configuration.
    merged = gated.merge(ungated, on="config_id", suffixes=("_gated", "_ungated"))
    assert (merged.realized_risk10_gated < merged.realized_risk10_ungated).all()
    with sqlite3.connect(d4 / "trial_registry.sqlite") as db:
        rows = db.execute("SELECT config_id, status FROM trials ORDER BY config_id").fetchall()
    assert any(status == "CONTROL_ONLY" for _, status in rows)
    assert any(config_id == "abcds_frequency_control" for config_id, _ in rows)

    calibration = h05(ROOT, d5, upstream_directories={"H02": d2, "H03": d3, "H04": d4})
    assert calibration["formal_H05_accepted"] is False
    assert calibration["calibrator_fits"] <= calibration["calibrator_fit_cap"]
    reliability = pd.read_csv(d5 / "selected_reliability.csv")
    assert not reliability.empty
    # A calibrated subset must report its gap rather than a confidence claim.
    assert reliability.gap.notna().all()
    states = [json.loads(line) for line in
              (d5 / "calibration_states.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert states and all(state["calibrator_fits"] == 1 for state in states)

    from research.s20_harness.h06_ablation import build as h06
    from research.s20_harness.h07_execution import build as h07
    from research.s20_harness.h08_freeze import build as h08
    from research.s20_harness.h10_review import build as h10

    d6, d7, d8, d10 = (tmp_path / name for name in ("h06", "h07", "h08", "h10"))
    ablation = h06(ROOT, d6, upstream_directories={"H02": d2, "H03": d3, "H04": d4})
    assert ablation["fits_charged"] <= ablation["fit_cap"]
    assert ablation["activated_families"] == []
    table = pd.DataFrame(ablation["ablation"])
    # The shuffled twin exists so a control can never be mistaken for an increment.
    assert "add_shuffled_control" in set(table.config_id)
    assert table.calibration_log_loss.notna().all()

    execution = h07(ROOT, d7, upstream_directories={"H02": d2, "H03": d3, "H04": d4, "H06": d6})
    assert execution["formal_H07_accepted"] is False
    assert execution["episodes"] > 0
    episodes = pd.read_parquet(d7 / "episodes.parquet")
    assert set(episodes.role) == {"candidate", "control"}
    assert (episodes.exit_date > episodes.entry_date).all()
    # Capacity is enforced, not merely reported.
    at_cap = pd.read_csv(d7 / "stress.csv")
    primary = at_cap[(at_cap.capacity.eq(20)) & (at_cap.cost_multiplier.eq(1.0))]
    assert primary.rejected_by_capacity.ge(0).all()
    assert execution["paired"]["safe_target_rate"]["paired_dates"] > 0

    freeze = h08(ROOT, d8, upstream_directories={"H03": d3, "H06": d6, "H07": d7})
    assert freeze["champion"] is None
    assert freeze["g2_met"] is False
    assert freeze["clean_holdout_available"] is False
    assert (d8 / "holdout_access.jsonl").read_text(encoding="utf-8") == ""
    champion = json.loads((d8 / "champion_manifest.json").read_text(encoding="utf-8"))
    assert champion["frozen"] is False and champion["champion"] is None

    review = h10(ROOT, d10, upstream_directories={"H02": d2, "H03": d3, "H04": d4,
                                                  "H05": d5, "H06": d6, "H07": d7, "H08": d8})
    assert review["promotable"] is False
    assert review["decision_state"] == "PROMOTION_BLOCKED_BY_VALIDITY"
    decision = json.loads((d10 / "promotion_decision.json").read_text(encoding="utf-8"))
    assert decision["promotable"] is False
    assert decision["claim_supported"] is False
    assert len(decision["blocking_reasons"]) >= 3
    # The decision may never borrow a promotable state from the diagnostics.
    assert decision["state"] not in ("SUPPORTED", "PROMOTED", "ACCEPTED")
    report = (d10 / "final_report.md").read_text(encoding="utf-8")
    assert "PROMOTION_BLOCKED_BY_VALIDITY" in report
