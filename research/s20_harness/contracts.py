"""Frozen S20-v4 plan validation and stage gating. Does not start training."""
from __future__ import annotations

import json
from pathlib import Path

ALLOWED_TOP_N = (1, 3, 5, 10, 20)
FOUNDATION_STAGES = ("H00", "H01", "H02")
TRAINING_STAGES = tuple(f"H{i:02d}" for i in range(3, 11))
EXPERIMENT_IDS = tuple(f"E{i:02d}" for i in range(20))
STAGE_IDS = tuple(f"H{i:02d}" for i in range(11))


def load_plan(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("plan must be a JSON object")
    return payload


def _dag_errors(stages: list) -> list[str]:
    errors = []
    by_id = {}
    for stage in stages:
        sid = stage.get("id")
        if sid in by_id:
            errors.append(f"duplicate stage {sid}")
        by_id[sid] = stage
    missing = [sid for sid in STAGE_IDS if sid not in by_id]
    if missing:
        errors.append(f"missing stages {missing}")
    extra = [sid for sid in by_id if sid not in STAGE_IDS]
    if extra:
        errors.append(f"unexpected stages {extra}")
    graph = {sid: list(by_id.get(sid, {}).get("depends_on") or []) for sid in by_id}
    for sid, deps in graph.items():
        for dep in deps:
            if dep not in by_id:
                errors.append(f"{sid} depends on unknown {dep}")
    incoming = {sid: 0 for sid in graph}
    for sid, deps in graph.items():
        for dep in deps:
            if dep in incoming:
                incoming[sid] += 1
    queue = [sid for sid, n in incoming.items() if n == 0]
    seen = 0
    adj = {sid: [] for sid in graph}
    for sid, deps in graph.items():
        for dep in deps:
            if dep in adj:
                adj[dep].append(sid)
    while queue:
        node = queue.pop()
        seen += 1
        for child in adj[node]:
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    if graph and seen != len(graph):
        errors.append("DAG is cyclic")
    return errors


def validate_plan(plan: dict) -> dict:
    errors = []
    selection = plan.get("selection") or {}
    runner = plan.get("runner") or {}
    budget = plan.get("budget_proposal") or {}
    catalog = plan.get("experiment_catalog") or []
    stages = plan.get("stages") or []

    caps = tuple(selection.get("top_n_caps") or ())
    if caps != ALLOWED_TOP_N:
        errors.append(f"top_n_caps must be {list(ALLOWED_TOP_N)}, got {list(caps)}")
    if selection.get("allow_abstain") is not True:
        errors.append("abstain must be allowed")
    if selection.get("zero_selection_metrics") != "null precision/risk; zero coverage":
        errors.append("empty selection must use null precision/risk and zero coverage")
    if runner.get("production_actions_allowed") is not False:
        errors.append("production actions must be disabled")
    if runner.get("real_orders_allowed") is not False:
        errors.append("real orders must be disabled")
    joint = str(selection.get("joint_probability") or "")
    if "never assume marginal independence" not in joint.lower():
        errors.append("joint probability must forbid marginal independence")
    ids = [item.get("id") for item in catalog]
    if tuple(ids) != EXPERIMENT_IDS:
        errors.append(f"experiment catalog must be {list(EXPERIMENT_IDS)}")
    errors.extend(_dag_errors(stages))
    slots = budget.get("initial_model_config_slots")
    seeds = budget.get("initial_seeds_per_config")
    folds = budget.get("outer_folds")
    cap = budget.get("initial_model_level_fit_cap")
    if slots != 12 or seeds != 2 or folds != 3 or cap != 72:
        errors.append("budget must be 12 configs x 2 seeds x 3 folds = 72 fit cap")
    if slots and seeds and folds and slots * seeds * folds > cap:
        errors.append("fit product exceeds cap")

    valid = not errors
    return {
        "valid": valid,
        "errors": errors,
        "dag_valid": valid or not any("DAG" in e or "stage" in e.lower() or "depends" in e for e in errors),
        "n_stages": len(stages),
        "n_experiments": len(catalog),
        "top_n_caps": list(caps),
        "abstain_allowed": selection.get("allow_abstain") is True,
        "production_actions_allowed": bool(runner.get("production_actions_allowed")),
        "real_orders_allowed": bool(runner.get("real_orders_allowed")),
        "training_started": False,
        "foundation_stages": list(FOUNDATION_STAGES),
        "training_stages_gated": list(TRAINING_STAGES),
    }


def load_stage_state(path: str | Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {sid: "PLANNED" for sid in STAGE_IDS}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def foundation_passed(state: dict) -> bool:
    return all(state.get(sid) == "COMPLETED" for sid in FOUNDATION_STAGES)


def stage_allowed(stage: str, state: dict) -> tuple[bool, str]:
    if stage not in STAGE_IDS:
        return False, f"unknown stage {stage}"
    if stage in TRAINING_STAGES and not foundation_passed(state):
        return False, "H03+ training is not started; H00-H02 checks have not passed"
    return True, "allowed"


def format_validate_report(result: dict) -> str:
    dag_line = "DAG is valid" if result["valid"] else "DAG is invalid"
    abstain = "abstain allowed" if result["abstain_allowed"] else "abstain forbidden"
    prod = "disabled" if not result["production_actions_allowed"] else "ENABLED"
    orders = "disabled" if not result["real_orders_allowed"] else "ENABLED"
    caps = "/".join(str(n) for n in result["top_n_caps"])
    lines = [
        "S20-v4 plan valid" if result["valid"] else "S20-v4 plan invalid",
        dag_line,
        f"TopN caps: {caps}",
        abstain,
        f"production actions {prod}",
        f"real orders {orders}",
        "H03+ training is not started",
    ]
    if result["errors"]:
        lines.append("errors: " + "; ".join(result["errors"]))
    return "\n".join(lines) + "\n"
