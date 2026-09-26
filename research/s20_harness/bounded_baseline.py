"""Owned child baseline execution after parent-side budget reservation."""
import argparse
import json
import math
from pathlib import Path
import sys
import uuid

from .baseline_run import build, verify
from .process_runner import Limits, run_job
from .runtime import atomic_json, digest


def _backend(pipeline):
    if pipeline == 'baseline':
        return 'research.s20_harness.bounded_baseline', verify
    if pipeline == 'joint':
        from .joint_run import verify as verify_joint
        return 'research.s20_harness.bounded_joint', verify_joint
    raise ValueError('unknown bounded pipeline identity')


def verify_process(root, artifact, input_path, input_sha, limits, *, pipeline='baseline'):
    """Verify recorded completion, not current liveness or receipt authenticity."""
    module, verify_artifact = _backend(pipeline)
    if artifact.get('pipeline_kind', 'baseline') != pipeline:
        raise ValueError('process pipeline identity mismatch')
    root, input_path = Path(root).resolve(), Path(input_path).resolve()
    process_dir = Path(artifact["process_directory"]).resolve()
    scope = root/"output/experiments/s20_safe_v4/sources"
    if not process_dir.is_relative_to(scope) or not Path(artifact["directory"]).resolve().is_relative_to(scope):
        raise ValueError("process/artifact outside research scope")
    result_path = process_dir/"process_result.json"
    if digest(result_path) != artifact["process_summary_sha256"]:
        raise ValueError("process receipt pin mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["exit_code"] != 0 or result["termination_reason"] is not None:
        raise ValueError("process did not finish successfully")
    if result["limits"] != limits.__dict__:
        raise ValueError("process limits differ from registered limits")
    if type(result["pid"]) is not int or result["pid"] <= 0:
        raise ValueError("invalid process identity")
    for key in ("process_created", "wall_seconds", "peak_rss_bytes", "sampled_cpu_seconds"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid recorded process measurement")
    receipt = process_dir.parent/"receipt.json"
    command = [sys.executable, "-m", module, "--worker",
               "--root", str(root), "--input", str(input_path), "--input-sha256", input_sha,
               "--receipt", str(receipt)]
    if result["command"] != command:
        raise ValueError("process command/input binding mismatch")
    child = json.loads(receipt.read_text(encoding="utf-8"))
    if child != {"directory": artifact["directory"], "summary_sha256": artifact["summary_sha256"]}:
        raise ValueError("worker receipt/model mismatch")
    verify_artifact(artifact["directory"], artifact["summary_sha256"])
    summary = json.loads((Path(artifact["directory"])/"summary.json").read_text(encoding="utf-8"))
    if summary["artifacts"]["input.json"] != input_sha:
        raise ValueError("worker model input mismatch")
    if digest(result_path) != artifact["process_summary_sha256"]:
        raise ValueError("process receipt changed")
    return {"recorded_completion_verified": True, "limits_and_input_bound": True,
            "os_receipt_authenticity_proven": False, "formal_training_authorized": False}


def run(root, input_path, input_sha, budget, trial_id, attempt_id, limits, *, pipeline='baseline'):
    module, verify_artifact = _backend(pipeline)
    root, input_path = Path(root).resolve(), Path(input_path).resolve()
    if not isinstance(limits, Limits):
        raise ValueError("validated process limits required")
    trial = next((t for t in budget.contract["trials"] if t["trial_id"] == trial_id), None)
    from .baseline_model import pipeline_costs as baseline_costs
    expected = baseline_costs(json.loads(input_path.read_text(encoding='utf-8'))) if pipeline=='baseline' else None
    if pipeline=='joint':
        from .joint_run import pipeline_costs
        expected=pipeline_costs(json.loads(input_path.read_text(encoding='utf-8')))
    if trial is None or trial["costs"] != expected or digest(input_path) != input_sha:
        raise ValueError("bounded baseline cost/input mismatch")
    ticket = budget.reserve(trial_id, attempt_id, input_sha)
    if not ticket["newly_reserved"]:
        status = budget.status()
        if ticket['state'] != 'SUCCEEDED_DIAGNOSTIC':
            return dict(executed=False, reusable=False, reservation=ticket, budget=status)
        attempt = next(a for a in status['attempts']
                       if (a['trial_id'],a['attempt_id']) == (trial_id,attempt_id))
        artifact = attempt['result']
        verify_process(root,artifact,input_path,input_sha,limits,pipeline=pipeline)
        sources = json.loads((Path(artifact['directory'])/'inputs.json').read_text(encoding='utf-8'))
        if any(digest(Path(p)) != h for p,h in sources.items() if Path(p).suffix != '.py'):
            raise ValueError('cached baseline data dependencies changed')
        current = {str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}
        recorded = {str(Path(p)):h for p,h in sources.items() if Path(p).suffix=='.py'}
        if recorded != current:
            raise ValueError('cached baseline computational sources changed')
        return dict(executed=False,reusable=True,artifact=artifact,reservation=ticket,budget=status)
    work = root/"output/experiments/s20_safe_v4/sources"/(pipeline+"-process-"+uuid.uuid4().hex)
    receipt = work/"receipt.json"
    code_root = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-m", module, "--worker",
               "--root", str(root), "--input", str(input_path), "--input-sha256", input_sha,
               "--receipt", str(receipt)]
    try:
        work.mkdir(parents=True)
        process = run_job(command, code_root, work/"process", limits)
        if process["exit_code"] != 0 or process["termination_reason"] is not None:
            raise RuntimeError("bounded baseline process failed: " + str(process["termination_reason"] or process["exit_code"]))
        artifact = json.loads(receipt.read_text(encoding="utf-8"))
        verify_artifact(artifact["directory"], artifact["summary_sha256"])
        if not Path(artifact["directory"]).resolve().is_relative_to(root/"output/experiments/s20_safe_v4/sources"):
            raise ValueError("worker artifact outside research scope")
        if digest(input_path) != input_sha:
            raise ValueError("bounded input changed")
        artifact.update(process_directory=str(work/"process"), process_summary_sha256=digest(work/"process/process_result.json"))
        if pipeline != 'baseline':
            artifact['pipeline_kind'] = pipeline
        verify_process(root, artifact, input_path, input_sha, limits,pipeline=pipeline)
        budget.finish(trial_id, attempt_id, "SUCCEEDED_DIAGNOSTIC", artifact)
    except Exception as exc:
        budget.finish(trial_id, attempt_id, "FAILED", {"type": type(exc).__name__, "message": str(exc),
                       "process_directory": str(work/"process")})
        raise
    return {"executed": True, "artifact": artifact, "process": process, "budget": budget.status()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, action="store_true")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.root, args.input, args.input_sha256)
    atomic_json(args.receipt, {"directory": report["directory"],
                "summary_sha256": digest(Path(report["directory"])/"summary.json")})
