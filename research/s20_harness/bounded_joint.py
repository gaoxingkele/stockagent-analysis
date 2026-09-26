"""Budgeted owned-process entry for the diagnostic joint pipeline."""
import argparse
from pathlib import Path

from .bounded_baseline import run as _run, verify_process as _verify_process
from .joint_run import build
from .joint_replay import replay
from .runtime import atomic_json, digest


def run(root, input_path, input_sha, budget, trial_id, attempt_id, limits):
    return _run(root, input_path, input_sha, budget, trial_id, attempt_id, limits, pipeline='joint')


def verify_process(root, artifact, input_path, input_sha, limits):
    return _verify_process(root, artifact, input_path, input_sha, limits, pipeline='joint')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, action='store_true')
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    report = build(args.root, args.input, args.input_sha256)
    sha = digest(Path(report['directory'])/'summary.json')
    # Numerical verification runs inside the monitored process as well.
    replay(report['directory'], sha)
    atomic_json(args.receipt, dict(directory=report['directory'], summary_sha256=sha))
