"""Build ABCDS labels from saved diagnostic outcomes, without fitting a model."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.s20_harness.silent_profit_labels import CLASSES, LABEL_VERSION, split_silent_profit

SOURCE = ROOT / "output/experiments/s20_safe_v4/sources/v4-paired-campaign/labels_tp15_dd10_screen.parquet"
OUT = ROOT / "output/experiments/s20_safe_v4/sources/v4-abcds-a-only-v1"


def main():
    frame = split_silent_profit(pd.read_parquet(SOURCE))
    OUT.mkdir(parents=True, exist_ok=True)
    labels_path, report_path = OUT / "labels.parquet", OUT / "label_summary.json"
    if labels_path.exists() or report_path.exists():
        raise FileExistsError("versioned label artifacts already exist; do not overwrite")
    frame.to_parquet(labels_path, index=False)
    counts = frame.target.value_counts().reindex(CLASSES, fill_value=0)
    transitions = frame.groupby(["legacy_target", "target"], dropna=False).size()
    report = dict(
        label_version=LABEL_VERSION,
        source=str(SOURCE.relative_to(ROOT)),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        n_rows=len(frame), n_realized=int(frame.target.notna().sum()),
        n_unknown=int(frame.target.isna().sum()),
        class_counts={k: int(v) for k, v in counts.items()},
        transitions=[dict(old=None if pd.isna(old) else old,
                          new=None if pd.isna(new) else new, n=int(n))
                     for (old, new), n in transitions.items()],
        definition="S = legacy A and full-window max_gain < 0.08; B/C/D unchanged",
        profit_classes=["A", "B", "S"], risk_classes=["B", "D"],
        specified_rise="use hit15 separately; A does not imply +15%",
        no_rows_removed=True, model_refit=False, frozen_policy_changed=False,
        formal_training_authorized=False, production_eligible=False,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
