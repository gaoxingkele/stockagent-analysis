"""Run the bounded S20-v4 P-track diagnostic campaign. Not formal H04."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.s20_harness.diagnostic_train import run


if __name__ == "__main__":
    summary = run(ROOT)
    print(json.dumps({
        "status": summary["status"],
        "formal_training_authorized": summary["formal_training_authorized"],
        "dataset": summary["dataset"],
        "outer": [dict(family=m["family"], **m["outer"]) for m in summary["models"]],
    }, ensure_ascii=False, indent=2), flush=True)
