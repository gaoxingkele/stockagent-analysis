import json
from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness import pipeline
from research.s20_harness.abcds_labels import CLASSES, classify, golden_checks
from research.s20_harness.h03_baseline import build_boundaries
from research.s20_harness.splits import SEGMENTS

PLAN = json.loads((Path(__file__).resolve().parents[2]
                   / "config/s20_v4_harness_plan.json").read_text(encoding="utf-8"))


class FakeRuntime:
    """Deterministic stand-in that exercises orchestration without real fits."""

    def __init__(self, states):
        self.plan = PLAN
        self.protocol_hash = "0" * 64
        self._states = dict(states)
        self.actions = []

    def status(self):
        return {"protocol_hash": self.protocol_hash, "runs": [], "stages": dict(self._states)}

    def record_gap(self, stage, *, track="diagnostic", kind=None, reason=None):
        terminal = ("WAITING_MATURITY" if pipeline.EXECUTORS[stage].kind == "prospective"
                    else "WAITING_IMPLEMENTATION")
        self._states[stage] = terminal
        self.actions.append(("gap", stage, terminal, kind, reason))
        return {"run_id": stage + "-gap", "terminal_state": terminal,
                "inherited_blockers": {}, "formal_accepted": False}

    def run(self, stage, *, track="formal", mode=None):
        if mode == "intake":
            raise ValueError("missing pinned intake binding")
        # Every local diagnostic executor maps the frozen sample to a non-passing
        # formal gate, exactly as the real compute executors do.
        terminal = ("FAILED_VALIDITY"
                    if stage in ("H01", "H02", "H03", "H04", "H05", "H06", "H07",
                                 "H08", "H10") else "COMPLETED")
        self._states[stage] = terminal
        self.actions.append(("run", stage, terminal, track))
        return {"run_id": stage + "-fake", "terminal_state": terminal,
                "inherited_blockers": {}, "formal_accepted": track == "formal"}


def test_registry_covers_every_stage_and_is_topological():
    stages = [s["id"] for s in PLAN["stages"]]
    assert set(pipeline.EXECUTORS) == set(stages)
    order = pipeline.stage_order(PLAN)
    position = {stage: index for index, stage in enumerate(order)}
    assert len(order) == len(stages)
    for stage in stages:
        for dependency in next(s for s in PLAN["stages"] if s["id"] == stage)["depends_on"]:
            assert position[dependency] < position[stage]
    for route in pipeline.EXECUTORS.values():
        assert route.kind in ("builtin", "compute", "intake", "unimplemented", "prospective")


def test_diagnostic_track_walks_past_a_failed_gate_without_claiming_promotion():
    runtime = FakeRuntime({"H00": "COMPLETED", "H01": "FAILED_VALIDITY"})
    report = pipeline.run_pipeline(runtime, track="diagnostic")
    states = report["stage_states"]
    assert states["H02"] == "FAILED_VALIDITY"
    assert states["H03"] == "FAILED_VALIDITY"
    assert states["H04"] == "FAILED_VALIDITY"
    assert states["H05"] == "FAILED_VALIDITY"
    assert states["H06"] == "FAILED_VALIDITY"
    assert states["H07"] == "FAILED_VALIDITY"
    assert states["H08"] == "FAILED_VALIDITY"
    assert states["H09"] == "WAITING_MATURITY"
    assert states["H10"] == "FAILED_VALIDITY"
    assert report["promotion_eligible"] is False
    assert report["formal_accepted_stages"] == []
    assert "PLANNED" not in states
    executed = [step for step in report["steps"] if step["action"] == "executed"]
    assert [step["stage"] for step in executed] == [
        "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H10"]
    assert all(step["formal_accepted"] is False for step in executed)


def test_formal_track_refuses_to_continue_past_the_failed_gate():
    runtime = FakeRuntime({"H00": "COMPLETED", "H01": "FAILED_VALIDITY"})
    report = pipeline.run_pipeline(runtime, track="formal")
    actions = {step["stage"]: step for step in report["steps"]}
    assert actions["H01"]["action"] == "already_recorded_failure"
    assert actions["H02"]["action"] == "formal_track_stops_on_unmet_prerequisite"
    assert actions["H02"]["blocking_prerequisites"] == {"H01": "FAILED_VALIDITY"}
    assert runtime.actions == []
    assert report["promotion_eligible"] is False


def test_unimplemented_stage_is_recorded_as_a_gap_not_a_completion():
    runtime = FakeRuntime({"H00": "COMPLETED", "H01": "FAILED_VALIDITY"})
    manifest = runtime.record_gap("H06", track="diagnostic")
    assert manifest["terminal_state"] == "WAITING_IMPLEMENTATION"
    assert manifest["formal_accepted"] is False
    assert runtime.status()["stages"]["H06"] == "WAITING_IMPLEMENTATION"


def _frame(rows):
    return pd.DataFrame(rows, columns=["sample_id", "hit15", "path_risk", "max_gain"])


def test_five_class_boundaries_including_equality_and_unknown():
    out = classify(_frame([
        ("a-hit-safe", True, False, 0.20),
        ("b-hit-risk", True, True, 0.16),
        ("c-at-eight", False, False, 0.08),
        ("s-below-eight", False, False, 0.0799),
        ("s-exactly-below", False, False, 0.08 - 1e-9),
        ("d-risk-no-hit", False, True, 0.12),
        ("unknown-dual-touch", None, True, 0.16),
    ]))
    target = out.set_index("sample_id")["target"]
    assert target["a-hit-safe"] == "A"
    assert target["b-hit-risk"] == "B"
    # Exactly 8% is not silence; only strictly below the cut is.
    assert target["c-at-eight"] == "C"
    assert target["s-below-eight"] == "S"
    assert target["s-exactly-below"] == "S"
    assert target["d-risk-no-hit"] == "D"
    assert pd.isna(target["unknown-dual-touch"])
    assert set(target.dropna()) == set(CLASSES)


def test_golden_checks_reject_a_broken_partition():
    out = classify(_frame([
        ("a-hit-safe", True, False, 0.20),
        ("b-hit-risk", True, True, 0.16),
        ("c-at-eight", False, False, 0.08),
        ("d-risk-no-hit", False, True, 0.12),
        ("s-below-threshold", False, False, 0.0799),
    ]))
    assert all(check["passed"] for check in golden_checks(out, legacy_a_only_silence=99))
    broken = out.copy()
    broken.loc[broken.sample_id.eq("a-hit-safe"), "target"] = "S"   # a +15% touch is never silent
    failed = {check["check"] for check in golden_checks(broken, legacy_a_only_silence=99)
              if not check["passed"]}
    assert "hit15_boundary_is_inclusive" in failed
    assert "A_and_B_are_exactly_H" in failed


def test_the_two_silence_definitions_are_distinguishable():
    # Both rows are quiet and safe, so the new contract puts both in S while the
    # legacy a-only split only ever moved legacy A rows.
    out = classify(_frame([
        ("legacy-a-quiet", False, False, 0.05),
        ("legacy-c-quiet", False, False, 0.05),
    ]))
    assert int(out.target.eq("S").sum()) == 2
    checks = {check["check"]: check for check in golden_checks(out, legacy_a_only_silence=1)}
    assert checks["two_silence_definitions_differ"]["passed"]


def test_split_boundaries_are_ordered_disjoint_and_calendar_only():
    dates = [f"2024{m:02d}{d:02d}" for m in range(1, 13) for d in (5, 15, 25)]
    boundaries = build_boundaries(dates)
    assert tuple(b["name"] for b in boundaries) == SEGMENTS
    for previous, current in zip(boundaries, boundaries[1:]):
        assert previous["end_at"] == current["start_at"]
    assert boundaries[0]["start_at"] < boundaries[0]["end_at"]
    assert build_boundaries(list(reversed(dates))) == boundaries
    with pytest.raises(ValueError, match="too few signal dates"):
        build_boundaries(dates[:4])
