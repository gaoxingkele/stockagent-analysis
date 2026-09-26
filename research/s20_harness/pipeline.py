"""End-to-end stage pipeline for the S20-v4 harness.

Two tracks are explicit and never interchangeable:

``formal``
    Frozen semantics. A stage runs only when every prerequisite is ``COMPLETED``.
    Only this track can ever record ``formal_accepted``.
``diagnostic``
    A registered research track that may continue past a settled non-COMPLETED
    prerequisite (``FAILED_VALIDITY`` / ``WAITING_*``). Every downstream stage
    inherits those blockers, is forced to ``formal_accepted = False``, and may
    never emit a promotable decision.

The diagnostic track does not make H01 pass. It records the failure, carries it
forward as evidence, and lets the rest of the chain produce diagnostic
artifacts instead of stopping at a wall that only new data evidence can move.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TRACKS = ("formal", "diagnostic")

# Terminal states that mean "this stage produced no declared outputs".
WAITING_STATES = ("WAITING_IMPLEMENTATION", "WAITING_MATURITY")
# A prerequisite in one of these states does not stop a diagnostic run.
SETTLED = ("COMPLETED", "FAILED_VALIDITY", *WAITING_STATES)
# States that must be carried forward as inherited blockers.
BLOCKING = ("FAILED_VALIDITY", *WAITING_STATES)
# Terminal states that may never carry formal acceptance.
NON_ACCEPTING = BLOCKING

TRACK_HELP = {
    "formal": "frozen semantics; requires COMPLETED prerequisites; may record formal acceptance",
    "diagnostic": "registered research track; continues past settled blockers; formal acceptance forbidden",
}


@dataclass(frozen=True)
class Executor:
    """Declared execution route for one stage.

    ``kind`` is deliberately honest about what exists today:

    ``builtin``       implemented inside :class:`runtime.Runtime`
    ``compute``       real callable producing this stage's declared outputs
    ``intake``        verifies and pins an externally produced evidence bundle
    ``unimplemented`` no implementation exists; the stage is recorded as a gap
    ``prospective``   requires genuinely future data; cannot be computed now
    """

    stage: str
    kind: str
    module: str | None = None
    produces: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    note: str = ""
    # Frozen default route. H02/H03/H05 keep their original intake default so
    # existing run-stage behaviour is unchanged; the pipeline asks for the
    # local compute route explicitly.
    default_mode: str = "compute"


EXECUTORS: dict[str, Executor] = {
    "H00": Executor(
        "H00", "builtin",
        produces=("protocol.json", "source_snapshot.json", "environment.json",
                  "exposure_ledger.json", "budget.json"),
        note="source snapshot, exposure ledger and bounded resource pilot"),
    "H01": Executor(
        "H01", "builtin", module="research.s20_harness.full_data_audit",
        produces=("dataset_manifest.json", "data_audit.json", "anomaly_ledger.parquet",
                  "coverage_report.md"),
        note="dataset admission audit; currently FAILED_VALIDITY pending source evidence"),
    "H02": Executor(
        "H02", "compute", module="research.s20_harness.abcds_labels",
        produces=("label_contracts.json", "labels.parquet", "label_transfer.csv",
                  "execution_contract.json", "golden_cases.json"),
        requires=("frozen diagnostic screen labels with hit15/path_risk/max_gain",),
        note="five-class ABCDS re-mapping from the frozen diagnostic sample",
        default_mode="intake"),
    "H03": Executor(
        "H03", "compute", module="research.s20_harness.h03_baseline",
        produces=("split_manifest.json", "baseline_metrics.csv", "baseline_oof.parquet",
                  "baseline_bundle.json"),
        requires=("H02 labels", "samples and features of the same frozen sample"),
        note="chronological five-segment split plus fixed reference fits",
        default_mode="intake"),
    "H04": Executor(
        "H04", "compute", module="research.s20_harness.h04_competition",
        produces=("trial_registry.sqlite", "oof_predictions.parquet", "model_cards.json",
                  "development_frontier.csv"),
        requires=("H03 split manifest", "H02 labels"),
        note="registered bounded competition with a zero-fit control and an independent risk head"),
    "H05": Executor(
        "H05", "compute", module="research.s20_harness.h05_abcds",
        produces=("calibration_states.jsonl", "policy_candidates.json", "risk_coverage.csv",
                  "selected_reliability.csv"),
        requires=("H04 predictions", "mature calibration labels"),
        note="delayed temperature calibration plus a registered bounded policy grid",
        default_mode="intake"),
    "H06": Executor(
        "H06", "compute", module="research.s20_harness.h06_ablation",
        produces=("ablation.csv", "factor_cards.json", "advanced_activation_decisions.json",
                  "advanced_oof.parquet"),
        requires=("H04 predictions", "H02 labels"),
        note="feature-group ablation with a shuffled control; no advanced family activated"),
    "H07": Executor(
        "H07", "compute", module="research.s20_harness.h07_execution",
        produces=("locked_development_report.json", "stress.csv", "episodes.parquet",
                  "nav.parquet", "paired_intervals.json"),
        requires=("H06 activation decisions", "H04 predictions", "H02 labels"),
        note="concurrency-capped book, cost and capacity stress, paired block intervals"),
    "H08": Executor(
        "H08", "compute", module="research.s20_harness.h08_freeze",
        produces=("champion_manifest.json", "sealing_manifest.json", "power_plan.json",
                  "holdout_access.jsonl"),
        requires=("H07 locked development report",),
        note="G2 gate evaluation, sealing and power planning; no champion on a single split"),
    "H09": Executor(
        "H09", "prospective",
        produces=("daily_predictions.parquet", "orders.parquet", "fills.parquet",
                  "matured_queue.parquet", "events.jsonl"),
        requires=("H08 frozen champion", "genuinely future signal dates"),
        note="cannot be computed from stored history; requires forward collection and label maturity"),
    "H10": Executor(
        "H10", "compute", module="research.s20_harness.h10_review",
        produces=("final_report.md", "risk_coverage.csv", "paired_intervals.json",
                  "promotion_decision.json", "checkpoint.json"),
        requires=("H09 matured prospective queue",),
        note="one-time review; the diagnostic track may only emit a non-promotable decision"),
}


def stage_order(plan: dict) -> list[str]:
    """Deterministic topological order of stage ids."""
    stages = {s["id"]: list(s.get("depends_on") or []) for s in plan["stages"]}
    order: list[str] = []
    pending = dict(stages)
    while pending:
        ready = sorted(sid for sid, deps in pending.items() if all(d in order for d in deps))
        if not ready:
            raise ValueError("stage DAG is cyclic or references unknown stages")
        order.extend(ready)
        for sid in ready:
            pending.pop(sid)
    return order


def inherited_blockers(state: dict, stage: str, plan: dict) -> dict:
    """Map every non-COMPLETED prerequisite of ``stage`` to its recorded state."""
    stages = {s["id"]: s for s in plan["stages"]}
    blockers = {}
    for dependency in stages[stage].get("depends_on") or []:
        actual = state["stages"].get(dependency, "PLANNED")
        if actual != "COMPLETED":
            blockers[dependency] = actual
    return blockers


def stage_route(stage: str) -> Executor:
    executor = EXECUTORS.get(stage)
    if executor is None:
        raise ValueError("no declared executor route for stage " + stage)
    return executor


def execute(runtime, stage: str, *, track: str = "diagnostic", mode: str | None = None) -> dict:
    """Run one stage through its declared route."""
    route = stage_route(stage)
    chosen = mode or ("compute" if route.kind == "compute" else route.default_mode)
    return runtime.run(stage, track=track, mode=chosen)


def run_pipeline(runtime, *, track: str = "diagnostic", until: str | None = None,
                 max_stages: int | None = None) -> dict:
    """Walk the DAG in order, executing every stage that has a route.

    A stage whose prerequisite is not settled stops its own branch but the walk
    continues with the next independent stage, so the returned report always
    names every stage and why it is where it is.
    """
    if track not in TRACKS:
        raise ValueError("unknown track " + str(track))
    order = stage_order(runtime.plan)
    if until is not None:
        if until not in order:
            raise ValueError("unknown stage " + str(until))
        order = order[:order.index(until) + 1]
    steps = []
    ran = 0
    for stage in order:
        state = runtime.status()
        current = state["stages"].get(stage, "PLANNED")
        route = stage_route(stage)
        if current == "COMPLETED":
            steps.append({"stage": stage, "action": "already_verified", "state": current})
            continue
        if current == "FAILED_VALIDITY":
            # Recorded evidence is not a retry queue. A new attempt needs new
            # evidence and an explicit run-stage call, not another identical run.
            steps.append({"stage": stage, "action": "already_recorded_failure", "state": current,
                          "note": "rerun only with new evidence via run-stage"})
            continue
        if current in WAITING_STATES and route.kind in ("unimplemented", "prospective"):
            steps.append({"stage": stage, "action": "gap_already_recorded", "state": current,
                          "executor": route.kind, "module": route.module})
            continue
        blockers = inherited_blockers(state, stage, runtime.plan)
        unmet = {k: v for k, v in blockers.items() if v not in SETTLED}
        if unmet:
            steps.append({"stage": stage, "action": "blocked_by_prerequisite",
                          "state": current, "blocking_prerequisites": unmet})
            continue
        if track == "formal" and blockers:
            steps.append({"stage": stage, "action": "formal_track_stops_on_unmet_prerequisite",
                          "state": current, "blocking_prerequisites": blockers})
            continue
        if max_stages is not None and ran >= max_stages:
            steps.append({"stage": stage, "action": "not_attempted_budget", "state": current})
            continue
        if route.kind in ("unimplemented", "prospective"):
            manifest = runtime.record_gap(stage, track=track, kind=route.kind)
            ran += 1
            steps.append({"stage": stage, "action": "recorded_gap", "state": manifest["terminal_state"],
                          "executor": route.kind, "module": route.module, "note": route.note})
            continue
        try:
            manifest = execute(runtime, stage, track=track)
        except (ValueError, NotImplementedError) as exc:
            if track == "formal":
                # The frozen track stops; a stage that cannot run is not silently
                # converted into a recorded gap.
                steps.append({"stage": stage, "action": "not_executable", "state": current,
                              "error": str(exc)})
                continue
            # A self-contained diagnostic walk records the missing route and keeps
            # going, so downstream stages report the real inherited limitation
            # instead of an unreachable PLANNED branch.
            manifest = runtime.record_gap(stage, track=track, kind="not_executable",
                                          reason=str(exc))
            ran += 1
            steps.append({"stage": stage, "action": "recorded_gap", "state": manifest["terminal_state"],
                          "executor": "not_executable", "error": str(exc)})
            continue
        ran += 1
        steps.append({"stage": stage, "action": "executed", "state": manifest["terminal_state"],
                      "run_id": manifest["run_id"],
                      "inherited_blockers": manifest.get("inherited_blockers", {}),
                      "formal_accepted": manifest.get("formal_accepted", False)})
    final = runtime.status()["stages"]
    return {
        "track": track,
        "protocol_hash": runtime.protocol_hash,
        "stages_attempted": ran,
        "stage_states": {stage: final.get(stage, "PLANNED") for stage in order},
        "steps": steps,
        "formal_accepted_stages": [stage for stage, value in final.items() if value == "COMPLETED"]
        if track == "formal" else [],
        "promotion_eligible": track == "formal" and all(v == "COMPLETED" for v in final.values()),
        "note": ("diagnostic track records blockers instead of hiding them; "
                 "no stage on this track may carry formal acceptance"),
    }
