"""H02 executor: five-class ABCDS labels from the frozen diagnostic sample.

This is a *re-mapping* of already saved diagnostic outcomes, not a recomputation
from raw quotes. The saved screen file carries ``hit15`` (sellable-window touch of
+15%), ``path_risk`` (pre-exit break of the cost-basis risk line) and
``max_gain`` (full-window best price gain), which is exactly the evidence the
five-class contract needs.

It never claims H01 admission: ``formal_gate_passed`` is always ``False``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import atomic_json, digest, now

DEFAULT_SOURCE = "output/experiments/s20_safe_v4/sources/v4-paired-campaign/labels_tp15_dd10_screen.parquet"

LABEL_VERSION = "S20.hit15_risk10_silence8.v1"
CLASSES = ("A", "B", "C", "D", "S")

HIT15 = 0.15
SILENCE_MAX_GAIN = 0.08
LEGACY_RISK_MULTIPLE = 0.90
LEGACY_COST_BASIS = 1.001
BUY_COST = 0.001
SELL_COST = 0.0015

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; this mapping inherits the frozen diagnostic sample",
    "source outcomes were produced from saved screen labels, not a fresh economic-path recomputation",
    "dividend/split economic ledger and tradability for the source window are not re-audited here",
    "class counts are a re-mapping of existing diagnostic rows, not a full-market distribution",
)


def _boolean(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(name + " contains unresolved values; unknown rows must be handled explicitly")
    return series.astype("boolean").astype(bool)


def classify(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the five classes plus the unknown mask from H/R/M evidence."""
    required = {"sample_id", "hit15", "path_risk", "max_gain"}
    if not required.issubset(frame.columns):
        raise ValueError("screen labels require sample_id, hit15, path_risk, max_gain")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be present and unique")
    hit = frame.hit15.astype("boolean")
    risk = frame.path_risk.astype("boolean")
    gain = pd.to_numeric(frame.max_gain, errors="coerce")
    resolved = hit.notna() & risk.notna() & gain.notna() & np.isfinite(gain)
    out = frame.copy()
    out["hit15_flag"] = hit
    out["path_risk_flag"] = risk
    out["max_gain_value"] = gain
    out["resolved"] = resolved
    target = pd.Series(pd.NA, index=out.index, dtype="string")
    safe = ~risk.fillna(True)
    target[resolved & hit.fillna(False) & safe] = "A"
    target[resolved & hit.fillna(False) & ~safe] = "B"
    target[resolved & ~hit.fillna(False) & ~safe] = "D"
    quiet = resolved & ~hit.fillna(False) & safe
    target[quiet & gain.ge(SILENCE_MAX_GAIN)] = "C"
    target[quiet & gain.lt(SILENCE_MAX_GAIN)] = "S"
    out["target"] = target
    if out.loc[resolved, "target"].isna().any():
        raise ValueError("resolved rows must receive a class")
    if not out.loc[~resolved, "target"].isna().all():
        raise ValueError("unresolved rows must stay unknown")
    return out


def golden_checks(out: pd.DataFrame, legacy_a_only_silence: int) -> list[dict]:
    """Invariants that must hold for the contract to be usable."""
    def same(left: pd.Series, right: pd.Series, mask: pd.Series | None = None) -> bool:
        # Nullable boolean dtypes must not defeat an equality check on values.
        left_values = left.fillna(False).astype(bool)
        right_values = right.fillna(False).astype(bool)
        if mask is not None:
            left_values, right_values = left_values[mask], right_values[mask]
        return bool((left_values.to_numpy() == right_values.to_numpy()).all())

    gain = out["max_gain_value"]
    target = out["target"]
    hit = out["hit15_flag"].fillna(False)
    risk = out["path_risk_flag"].fillna(False)
    safe = ~risk
    resolved = out["resolved"]
    classes = set(target.dropna().unique())
    checks: list[tuple[str, bool, str]] = [
        ("classes_are_declared_order",
         classes.issubset(set(CLASSES)) and classes == set(CLASSES),
         "observed classes " + ",".join(sorted(classes))),
        ("A_and_B_are_exactly_H",
         same(target.isin(["A", "B"]), hit),
         "A+B must equal the sellable-window +15% touch"),
        ("B_and_D_are_exactly_R",
         same(target.isin(["B", "D"]), risk, resolved),
         "on resolved rows B+D must equal the paired risk event; an unresolved "
         "dual-touch row keeps its partial evidence mask and stays unclassified"),
        ("S_is_exactly_quiet_and_safe",
         same(target.eq("S"), gain.lt(SILENCE_MAX_GAIN) & safe),
         "S must be max_gain<8% with no risk"),
        ("silence_boundary_uses_strict_less_than",
         bool(target[gain.eq(SILENCE_MAX_GAIN)].ne("S").all()),
         "exactly 8% must not be silence"),
        ("hit15_boundary_is_inclusive",
         bool(target[hit].isin(["A", "B"]).all()),
         "a realized +15% touch can never be silent or non-hit"),
        ("unresolved_rows_are_retained",
         bool(out.loc[~out["resolved"], "target"].isna().all()),
         "unknown rows are preserved rather than dropped or imputed"),
        ("two_silence_definitions_differ",
         int(target.eq("S").sum()) != int(legacy_a_only_silence),
         f"new S={int(target.eq('S').sum())} legacy a-only S={int(legacy_a_only_silence)}"),
    ]
    return [{"check": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]


def build(root: Path, directory: Path, *, source: str = DEFAULT_SOURCE,
          expected_source_sha256: str | None = None) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    source_path = (root / source).resolve()
    if not source_path.is_file():
        raise ValueError("missing frozen diagnostic screen labels: " + source)
    source_sha = digest(source_path)
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError("H02 source pin mismatch")
    raw = pd.read_parquet(source_path)
    out = classify(raw)
    target = out["target"]
    legacy = raw["target"] if "target" in raw.columns else pd.Series(pd.NA, index=raw.index)
    gain = out["max_gain_value"]
    legacy_a_only_silence = int((legacy.eq("A") & gain.lt(SILENCE_MAX_GAIN)).sum())
    checks = golden_checks(out, legacy_a_only_silence)
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise ValueError("H02 golden checks failed: " + ", ".join(failed))

    counts = {name: int(target.eq(name).sum()) for name in CLASSES}
    unknown = int(target.isna().sum())
    if sum(counts.values()) + unknown != len(out):
        raise ValueError("five-class partition does not cover the sample")

    labels = out[["sample_id", "target", "hit15_flag", "path_risk_flag", "max_gain_value",
                  "resolved"]].copy()
    labels = labels.rename(columns={"hit15_flag": "hit15", "path_risk_flag": "path_risk",
                                    "max_gain_value": "max_gain"})
    labels["label_version"] = LABEL_VERSION
    if "target" in raw.columns:
        labels["legacy_target"] = raw["target"].to_numpy()
    if "net" in raw.columns:
        labels["net"] = raw["net"].to_numpy()

    transfer = pd.crosstab(labels.get("legacy_target", pd.Series(index=labels.index, dtype="object")),
                           labels["target"].fillna("UNKNOWN"), dropna=False)

    contract = {
        "label_version": LABEL_VERSION,
        "target_id": LABEL_VERSION,
        "classes": list(CLASSES),
        "class_conditions": {
            "A": "hit15 and no paired risk",
            "B": "hit15 and paired risk",
            "C": "no hit15, no risk, max_gain >= 8%",
            "D": "no hit15 with paired risk",
            "S": "max_gain < 8% and no paired risk",
        },
        "thresholds": {"hit15": HIT15, "silence_max_gain": SILENCE_MAX_GAIN,
                       "silence_boundary": "strict_less_than; exactly 8% is not silence"},
        "risk_definition": {
            "condition": "low < entry * 1.001 * 0.90",
            "basis": "legacy_cost_basis",
            "warning": "approximately -9.91% of raw entry, not exactly -10%",
        },
        "costs": {"buy": BUY_COST, "sell": SELL_COST},
        "horizon": "signal day D; D+1 open entry; D+1 not sellable; D+2 earliest exit; "
                   "first sellable touch of entry*1.15 exits; otherwise D+20 close",
        "profit_classes": ["A", "B", "S"],
        "risk_classes": ["B", "D"],
        "silent_classes": ["S"],
        "unknown_policy": "unresolved rows are retained and never imputed",
        "source": {"path": source, "sha256": source_sha,
                   "kind": "frozen diagnostic screen labels, not a new price-path recomputation"},
        "counts": counts,
        "unknown_rows": unknown,
        "total_rows": int(len(out)),
        "legacy_a_only_silence_rows": legacy_a_only_silence,
        "formal_training_authorized": False,
        "production_eligible": False,
    }
    execution = {
        "entry": "next market trading day open after the signal day",
        "exit": {"primary": "first sellable touch of entry*1.15", "fallback": "close of D+20"},
        "costs": {"buy": BUY_COST, "sell": SELL_COST},
        "risk_line": {"multiple": LEGACY_RISK_MULTIPLE, "cost_basis": LEGACY_COST_BASIS,
                      "label": "legacy_cost_basis"},
        "same_day_dual_touch": "unknown; never resolved by assuming an order",
        "unfillable_or_suspended": "retained as unknown, never dropped from the candidate ledger",
        "topn_caps": [1, 3, 5, 10, 20],
        "abstain_allowed": True,
        "zero_selection_metrics": "null precision/risk; zero coverage",
        "formal_training_authorized": False,
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("label_contracts.json", "execution_contract.json", "golden_cases.json"):
        if (directory / name).exists():
            raise ValueError("H02 refuses existing outputs: " + name)
    labels.to_parquet(directory / "labels.parquet", index=False)
    transfer.to_csv(directory / "label_transfer.csv")
    atomic_json(directory / "label_contracts.json", contract)
    atomic_json(directory / "execution_contract.json", execution)
    atomic_json(directory / "golden_cases.json", {"at": now(), "checks": checks, "all_passed": True,
                                                 "definitions_version": LABEL_VERSION})
    return {
        "formal_gate_passed": False,
        "formal_training_authorized": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "label_version": LABEL_VERSION,
        "class_counts": counts,
        "unknown_rows": unknown,
        "total_rows": int(len(out)),
        "source_sha256": source_sha,
        "golden_checks": len(checks),
        "validation_scope": "five-class re-mapping of the frozen diagnostic sample; not a new label engine",
    }
