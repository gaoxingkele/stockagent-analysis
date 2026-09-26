"""Identity-aware research panel, preserving raw inputs and unresolved rows."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, load_plan, now
from .security_identity import canonicalize_aliases


def build(root: Path) -> dict:
    root = Path(root).resolve()
    config = root / "config/s20_v4_security_aliases.json"
    config_hash = digest(config)
    aliases = load_plan(config)["aliases"]
    sources = sorted((root / "output/tushare_cache/daily").glob("*.parquet"))
    if not sources:
        raise ValueError("no daily inputs")
    output = root / "output/experiments/s20_safe_v4/sources" / ("identity-panel-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    (output / "daily").mkdir()
    receipts = []
    raw_count = canonical_count = conflict_count = duplicate_count = 0
    for path in sources:
        source_hash = digest(path)
        raw = pd.read_parquet(path)
        if digest(path) != source_hash:
            raise ValueError("daily input changed while reading")
        raw["trade_date"] = raw.trade_date.astype(str)
        if not raw.trade_date.eq(path.stem).all():
            raise ValueError("daily filename/date mismatch")
        raw["daily_source_row"] = range(len(raw))
        raw["daily_source_file"] = path.name
        canonical, lineage, conflicts = canonicalize_aliases(raw, aliases)
        duplicate = canonical.duplicated(["entity_id", "trade_date"], keep=False)
        # Conflicting aliases are absent from canonical rows but preserved by
        # the source hash, lineage and conflict ledger. The panel is never
        # authorized as an ex-ante selection universe from this absence.
        destination = output / "daily" / path.name
        canonical.to_parquet(destination, index=False)
        lineage_path = output / (path.stem + "-lineage.parquet")
        conflicts_path = output / (path.stem + "-conflicts.parquet")
        lineage.to_parquet(lineage_path, index=False)
        conflicts.to_parquet(conflicts_path, index=False)
        receipts.append({"source": path.relative_to(root).as_posix(), "source_sha256": source_hash,
                         "canonical": destination.relative_to(output).as_posix(), "canonical_sha256": digest(destination),
                         "lineage_sha256": digest(lineage_path), "conflicts_sha256": digest(conflicts_path),
                         "raw_rows": len(raw), "canonical_rows": len(canonical),
                         "conflict_groups": len(conflicts), "duplicate_entity_rows": int(duplicate.sum())})
        raw_count += len(raw)
        canonical_count += len(canonical)
        conflict_count += len(conflicts)
        duplicate_count += int(duplicate.sum())
    if digest(config) != config_hash:
        raise ValueError("identity config changed during build")
    atomic_json(output / "inputs_outputs.json", receipts)
    result = {"at": now(), "directory": str(output), "daily_files": len(receipts),
              "raw_rows": raw_count, "canonical_rows": canonical_count,
              "conflict_groups": conflict_count, "duplicate_entity_rows": duplicate_count,
              "identity_consistent": conflict_count == 0 and duplicate_count == 0,
              "config_sha256": config_hash, "code_sha256": digest(Path(__file__)),
              "identity_code_sha256": digest(Path(__file__).with_name("security_identity.py")),
              "receipt_sha256": digest(output / "inputs_outputs.json"),
              "formal_training_eligible": False, "ex_ante_universe_verified": False}
    atomic_json(output / "summary.json", result)
    return result


def entity_window_frame(panel: pd.DataFrame, entity_id: str) -> pd.DataFrame:
    """Stable identity for label windows; trading code stays date-effective.

    The internal label key is not an input feature or a historical ticker.
    """
    selected = panel.loc[panel.entity_id.eq(entity_id)].copy()
    if selected.empty:
        raise ValueError("unknown entity")
    if selected.trade_date.duplicated().any():
        raise ValueError("unresolved duplicate entity/date")
    selected["ts_code"] = entity_id
    return selected.sort_values("trade_date")


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parents[2]), indent=2))
