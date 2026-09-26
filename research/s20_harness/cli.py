"""Launchable S20-v4 research entry. Training stages stay gated until H00-H02 pass."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research.s20_harness.contracts import (
    format_validate_report,
    load_plan,
    load_stage_state,
    stage_allowed,
    validate_plan,
)
from research.s20_harness.data_audit import audit_local_cache
from research.s20_harness.probability_audit import audit_saved_v3
from research.s20_harness.runtime import Runtime

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "config" / "s20_v4_harness_plan.json"


def _repo_root() -> Path:
    return ROOT


def cmd_validate_plan(plan_path: Path) -> int:
    result = validate_plan(load_plan(plan_path))
    sys.stdout.write(format_validate_report(result))
    return 0 if result["valid"] else 1


def cmd_run_stage(plan_path: Path, stage: str, state_path: Path | None,
                  track: str = "formal", mode: str | None = None) -> int:
    result = validate_plan(load_plan(plan_path))
    if not result["valid"]:
        sys.stdout.write(format_validate_report(result))
        return 1
    if state_path is not None:
        sys.stdout.write("external state files cannot authorize stage execution\n")
        return 2
    runtime = Runtime(_repo_root(), plan_path)
    try:
        manifest = runtime.run(stage, track=track, mode=mode)
    except (ValueError, NotImplementedError) as exc:
        sys.stdout.write("H03+ training is not started\n")
        sys.stdout.write("H00-H02 and all DAG dependencies require verified artifacts: " + str(exc) + "\n")
        sys.stdout.write("use run-pipeline --track diagnostic to record blockers and continue instead\n")
        return 2
    sys.stdout.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return 0 if manifest["terminal_state"] == "COMPLETED" else 2


def cmd_run_pipeline(plan_path: Path, track: str, until: str | None, max_stages: int | None) -> int:
    result = validate_plan(load_plan(plan_path))
    if not result["valid"]:
        sys.stdout.write(format_validate_report(result))
        return 1
    from research.s20_harness.pipeline import run_pipeline
    report = run_pipeline(Runtime(_repo_root(), plan_path), track=track, until=until,
                          max_stages=max_stages)
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    states = set(report["stage_states"].values())
    if track == "formal":
        return 0 if states <= {"COMPLETED"} else 2
    return 0 if "PLANNED" not in states else 2


def cmd_pipeline_status(plan_path: Path) -> int:
    result = validate_plan(load_plan(plan_path))
    if not result["valid"]:
        sys.stdout.write(format_validate_report(result))
        return 1
    from research.s20_harness.pipeline import EXECUTORS, stage_order
    runtime = Runtime(_repo_root(), plan_path)
    state = runtime.status()
    rows = []
    for stage in stage_order(runtime.plan):
        route = EXECUTORS[stage]
        rows.append({"stage": stage, "state": state["stages"].get(stage, "PLANNED"),
                     "executor": route.kind, "module": route.module, "note": route.note})
    sys.stdout.write(json.dumps({
        "protocol_hash": runtime.protocol_hash,
        "stages": rows,
        "locally_executable": [r["stage"] for r in rows if r["executor"] in ("builtin", "compute")],
        "promotion_eligible": all(r["state"] == "COMPLETED" for r in rows),
    }, ensure_ascii=False, indent=2) + "\n")
    return 0


def cmd_audit(plan_path: Path) -> int:
    result = validate_plan(load_plan(plan_path))
    if not result["valid"]:
        sys.stdout.write(format_validate_report(result))
        return 1
    audit = audit_local_cache(_repo_root())
    sys.stdout.write(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    v3_pred = _repo_root() / "output/experiments/s20_v3/predictions.parquet"
    if v3_pred.exists():
        fail = audit_saved_v3(_repo_root())
        sel = fail["selection_immediate_half_risk"]
        uni = fail["diagnostic_prediction_universe"]
        sys.stdout.write(
            "v3 probability failure: "
            f"universe p_immediate {uni['mean_p_immediate']:.3f} vs actual {uni['actual_immediate']:.3f}; "
            f"p_down {uni['mean_p_down']:.3f} vs actual {uni['actual_down']:.3f}; "
            f"Top20 half-risk p_immediate {sel['topn_mean_p']:.3f} vs actual {sel['topn_actual']:.3f}\n"
        )
        sys.stdout.write("predicted probabilities are not usable as hit rates\n")
    sys.stdout.write("H03+ training is not started\n")
    return 0


def cmd_status(plan_path: Path, state_path: Path | None) -> int:
    result = validate_plan(load_plan(plan_path))
    sys.stdout.write(format_validate_report(result))
    state = Runtime(_repo_root(), plan_path).status()
    sys.stdout.write("stage_state=" + json.dumps(state, sort_keys=True) + "\n")
    return 0 if result["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research.s20_harness.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-plan", "audit", "status", "report", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        if name == "status":
            p.add_argument("--state", type=Path, default=None)
    run = sub.add_parser("run-stage")
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    run.add_argument("--stage", required=True)
    run.add_argument("--state", type=Path, default=None)
    run.add_argument("--track", choices=["formal", "diagnostic"], default="formal")
    run.add_argument("--mode", choices=["compute", "intake"], default=None)
    pipeline_run = sub.add_parser("run-pipeline")
    pipeline_run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    pipeline_run.add_argument("--track", choices=["formal", "diagnostic"], default="diagnostic")
    pipeline_run.add_argument("--until", default=None)
    pipeline_run.add_argument("--max-stages", type=int, default=None)
    pipeline_status = sub.add_parser("pipeline-status")
    pipeline_status.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    evaluation = sub.add_parser("verify-campaign-evaluation")
    evaluation.add_argument("--directory", type=Path, required=True)
    evaluation.add_argument("--summary-sha256", required=True)
    bundle = sub.add_parser("build-baseline-bundle")
    bundle.add_argument("--directory", type=Path, required=True)
    bundle.add_argument("--summary-sha256", required=True)
    verify_bundle = sub.add_parser('verify-baseline-bundle')
    verify_bundle.add_argument('--directory',type=Path,required=True)
    verify_bundle.add_argument('--summary-sha256',required=True)
    for name in ['build-development-frontier','verify-development-frontier','build-seed-summary','verify-seed-summary']:
        frontier=sub.add_parser(name)
        frontier.add_argument('--directory',type=Path,required=True)
        frontier.add_argument('--summary-sha256',required=True)
    inventory=sub.add_parser('inspect-attempts')
    inventory.add_argument('--budget-path',type=Path,required=True)
    inventory.add_argument('--contract-sha256',required=True)
    selection=sub.add_parser('inspect-selection-plan')
    selection.add_argument('--input',type=Path,required=True)
    selection.add_argument('--input-sha256',required=True)
    registration=sub.add_parser('register-search')
    registration.add_argument('--input',type=Path,required=True)
    registration.add_argument('--input-sha256',required=True)
    registration_check=sub.add_parser('verify-search-registration')
    registration_check.add_argument('--directory',type=Path,required=True)
    registration_check.add_argument('--plan-sha256',required=True)
    search_run=sub.add_parser('run-registered-synthetic-search')
    search_run.add_argument('--directory',type=Path,required=True)
    search_run.add_argument('--plan-sha256',required=True)
    search_run.add_argument('--max-new-jobs',type=int,default=1)
    search_receipt=sub.add_parser('verify-search-execution')
    search_receipt.add_argument('--receipt',type=Path,required=True)
    search_receipt.add_argument('--receipt-sha256',required=True)
    search_eval=sub.add_parser('evaluate-registered-search')
    search_eval.add_argument('--receipt',type=Path,required=True)
    search_eval.add_argument('--receipt-sha256',required=True)
    search_eval.add_argument('--outcomes-json',type=Path,required=True)
    search_eval_verify=sub.add_parser('verify-search-evaluation')
    search_eval_verify.add_argument('--directory',type=Path,required=True)
    search_eval_verify.add_argument('--summary-sha256',required=True)
    for name in ['build-search-decision','verify-search-decision']:
        decision=sub.add_parser(name)
        decision.add_argument('--directory',type=Path,required=True)
        decision.add_argument('--summary-sha256',required=True)
    comparison=sub.add_parser('compare-configurations')
    for side in ['left','right']:
        comparison.add_argument('--'+side+'-directory',type=Path,required=True)
        comparison.add_argument('--'+side+'-sha256',required=True)
    joint_policy=sub.add_parser('build-joint-policy-diagnostic')
    joint_policy.add_argument('--run-directory',type=Path,required=True)
    joint_policy.add_argument('--run-sha256',required=True)
    joint_policy.add_argument('--input',type=Path,required=True)
    joint_policy.add_argument('--input-sha256',required=True)
    joint_policy_verify=sub.add_parser('verify-joint-policy-diagnostic')
    joint_policy_verify.add_argument('--directory',type=Path,required=True)
    joint_policy_verify.add_argument('--summary-sha256',required=True)
    for command in ['run-budgeted-joint-policy','verify-budgeted-joint-policy']:
        budgeted=sub.add_parser(command)
        budgeted.add_argument('--budget-path',type=Path,required=True)
        budgeted.add_argument('--contract-sha256',required=True)
        budgeted.add_argument('--trial-id',required=True)
        budgeted.add_argument('--attempt-id',required=True)
        if command=='run-budgeted-joint-policy':
            budgeted.add_argument('--run-directory',type=Path,required=True)
            budgeted.add_argument('--run-sha256',required=True)
            budgeted.add_argument('--input',type=Path,required=True)
            budgeted.add_argument('--input-sha256',required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ['run-budgeted-joint-policy','verify-budgeted-joint-policy']:
        from .budgeted_joint_policy import run_existing,verify
        import sqlite3
        try:
            result=(run_existing(_repo_root(),args.run_directory,args.run_sha256,args.input,args.input_sha256,
                                 args.budget_path,args.contract_sha256,args.trial_id,args.attempt_id)
                    if args.command=='run-budgeted-joint-policy' else
                    verify(_repo_root(),args.budget_path,args.contract_sha256,args.trial_id,args.attempt_id))
        except (ValueError,RuntimeError,AssertionError,OSError,KeyError,TypeError,sqlite3.Error) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc),formal_H05_accepted=False))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command in ['build-joint-policy-diagnostic','verify-joint-policy-diagnostic']:
        from .joint_policy_run import build,verify
        try:
            result=(build(_repo_root(),args.run_directory,args.run_sha256,args.input,args.input_sha256)
                    if args.command=='build-joint-policy-diagnostic'
                    else verify(_repo_root(),args.directory,args.summary_sha256))
        except (ValueError,RuntimeError,AssertionError,OSError,KeyError,TypeError) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc),formal_H05_accepted=False))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command in ['build-search-decision','verify-search-decision']:
        from .search_decision import build,verify
        import sqlite3
        try:
            operation=build if args.command=='build-search-decision' else verify
            result=operation(_repo_root(),args.directory,args.summary_sha256)
        except (ValueError,RuntimeError,AssertionError,OSError,KeyError,TypeError,sqlite3.Error) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc),formal_H04_accepted=False))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command in ['evaluate-registered-search','verify-search-evaluation']:
        from .search_evaluation import build,verify,load_list
        import sqlite3
        try:
            result=(build(_repo_root(),args.receipt,args.receipt_sha256,load_list(args.outcomes_json))
                if args.command=='evaluate-registered-search' else verify(_repo_root(),args.directory,args.summary_sha256))
        except (ValueError,RuntimeError,AssertionError,OSError,KeyError,TypeError,sqlite3.Error) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc)))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command=='verify-search-execution':
        from .registered_search_run import verify_receipt
        import sqlite3
        try:
            result=verify_receipt(_repo_root(),args.receipt,args.receipt_sha256)
        except (ValueError,RuntimeError,AssertionError,OSError,KeyError,TypeError,sqlite3.Error) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc)))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command=='run-registered-synthetic-search':
        from .registered_search_run import run
        import sqlite3
        try:
            result=run(_repo_root(),args.directory,args.plan_sha256,max_new_jobs=args.max_new_jobs)
        except (ValueError,RuntimeError,AssertionError,OSError,KeyError,TypeError,sqlite3.Error) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc),formal_training_authorized=False))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command in ['register-search','verify-search-registration']:
        from .search_registration import register,verify
        import sqlite3
        try:
            result=(register(_repo_root(),args.input,args.input_sha256) if args.command=='register-search'
                else verify(_repo_root(),args.directory,args.plan_sha256))
        except (ValueError,AssertionError,OSError,KeyError,TypeError,sqlite3.Error) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc)))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command=='inspect-selection-plan':
        from .selection_plan import inspect
        try:
            result=inspect(_repo_root(),args.input,args.input_sha256)
        except (ValueError,AssertionError,OSError,KeyError,TypeError) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc)))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command=='compare-configurations':
        from .configuration_comparison import inspect
        try:
            result=inspect(_repo_root(),args.left_directory,args.left_sha256,args.right_directory,args.right_sha256)
        except (ValueError,AssertionError,OSError,KeyError,TypeError) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc)))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command=='inspect-attempts':
        from .attempt_inventory import inspect
        import sqlite3
        try:
            result=inspect(_repo_root(),args.budget_path,args.contract_sha256)
        except (ValueError,OSError,KeyError,TypeError,sqlite3.Error) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc)))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command in ['build-development-frontier','verify-development-frontier','build-seed-summary','verify-seed-summary']:
        if args.command.endswith('seed-summary'):
            from .seed_summary import build,verify
        else:
            from .development_frontier import build,verify
        operation=build if args.command.startswith('build-') else verify
        try:
            result=operation(_repo_root(),args.directory,args.summary_sha256)
        except (ValueError,AssertionError,OSError,KeyError,TypeError) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc),formal_H04_accepted=False))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command=='verify-baseline-bundle':
        from .baseline_bundle import verify
        try:
            result=verify(_repo_root(),args.directory,args.summary_sha256)
        except (ValueError,AssertionError,OSError,KeyError,TypeError) as exc:
            sys.stdout.write(json.dumps(dict(valid=False,error=str(exc),formal_H03_accepted=False))+'\n')
            return 2
        sys.stdout.write(json.dumps(dict(valid=True,**result))+'\n')
        return 0
    if args.command == "build-baseline-bundle":
        from research.s20_harness.baseline_bundle import build
        try:
            result = build(_repo_root(), args.directory, args.summary_sha256)
        except (ValueError, AssertionError, OSError, KeyError, TypeError) as exc:
            sys.stdout.write(json.dumps({"built": False, "error": str(exc),
                "formal_H03_accepted": False}, ensure_ascii=False) + "\n")
            return 2
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        return 0
    if args.command == "verify-campaign-evaluation":
        from research.s20_harness.campaign_evaluation import verify
        try:
            result = verify(_repo_root(), args.directory, args.summary_sha256)
        except (ValueError, AssertionError, OSError, KeyError, TypeError) as exc:
            sys.stdout.write(json.dumps({"valid": False, "error": str(exc),
                "formal_promotion_authorized": False}, ensure_ascii=False) + "\n")
            return 2
        sys.stdout.write(json.dumps({"valid": True, **result}, ensure_ascii=False) + "\n")
        return 0
    plan = args.plan
    if args.command == "validate-plan":
        return cmd_validate_plan(plan)
    if args.command == "audit":
        return cmd_audit(plan)
    if args.command == "status":
        return cmd_status(plan, getattr(args, "state", None))
    if args.command == "report":
        from research.s20_harness.report import build_report
        sys.stdout.write(json.dumps(build_report(Runtime(_repo_root(), plan)),
                                   ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.command == "resume":
        from research.s20_harness.recovery import resume
        sys.stdout.write(json.dumps(resume(Runtime(_repo_root(), plan)), indent=2) + "\n")
        return 0
    if args.command == "run-stage":
        return cmd_run_stage(plan, args.stage, args.state, args.track, args.mode)
    if args.command == "run-pipeline":
        return cmd_run_pipeline(plan, args.track, args.until, args.max_stages)
    if args.command == "pipeline-status":
        return cmd_pipeline_status(plan)
    return 1


if __name__ == "__main__":
    sys.exit(main())
