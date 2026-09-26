"""Date-effective verified aliases; never infer identity from equal prices."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
import json
import uuid

from .runtime import atomic_json, digest, load_plan, now


def canonicalize_aliases(frame, aliases):
    data = frame.reset_index(drop=True).copy()
    data["source_row"] = data.index
    data["entity_id"] = data.ts_code.astype(str)
    data["trading_code"] = data.ts_code.astype(str)
    claimed = set()
    touched = set()
    canonical, lineage, conflicts = [], [], []
    comparisons = [c for c in ("open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount") if c in data]
    if not comparisons:
        raise ValueError("no observed quote fields to verify")
    for alias in aliases:
        codes = {alias["old_code"], alias["new_code"]}
        if claimed & codes or alias["share_conversion_ratio"] != 1.0:
            raise ValueError("ambiguous alias chain or non-unit conversion requires separate handling")
        claimed |= codes
        selected = data.loc[data.ts_code.isin(codes)].copy()
        for date, group in selected.groupby("trade_date", sort=True):
            touched.update(group.source_row.tolist())
            expected = alias["old_code"] if str(date) < alias["effective_date"] else alias["new_code"]
            # DataFrame equality treats matching nulls as equal without replacing
            # them; missing market data remains the responsibility of G0 audit.
            agree = len(group[comparisons].drop_duplicates()) == 1
            if agree:
                eligible = group.loc[group.ts_code.eq(expected)]
                representative = (eligible.iloc[0] if len(eligible) else group.iloc[0]).to_dict()
                representative.update(entity_id=alias["entity_id"], trading_code=expected, ts_code=expected)
                canonical.append(representative)
            for row in group.itertuples(index=False):
                lineage.append({"source_row": row.source_row, "source_code": row.ts_code,
                                "trade_date": str(date), "entity_id": alias["entity_id"], "trading_code": expected,
                                "status": "canonicalized" if agree else "unresolved_quote_conflict"})
            if not agree:
                conflicts.append({"entity_id": alias["entity_id"], "trade_date": str(date),
                                  "source_rows": group.source_row.tolist(), "reason": "alias_quotes_disagree"})
    untouched = data.loc[~data.source_row.isin(touched)]
    result = pd.concat([untouched, pd.DataFrame(canonical)], ignore_index=True)
    return result, pd.DataFrame(lineage), pd.DataFrame(conflicts)


def audit_local_aliases(root):
    config = root / "config/s20_v4_security_aliases.json"
    aliases = load_plan(config)["aliases"]
    codes = {alias[key] for alias in aliases for key in ("old_code", "new_code")}
    frames, inputs = [], []
    for path in sorted((root / "output/tushare_cache/daily").glob("*.parquet")):
        before = digest(path)
        frame = pd.read_parquet(path)
        frame["daily_source_row"] = frame.index
        frame = frame.loc[frame.ts_code.isin(codes)].copy()
        frame["daily_source_file"] = path.name
        if digest(path) != before:
            raise ValueError("source changed during alias audit")
        frames.append(frame)
        inputs.append({"path": path.relative_to(root).as_posix(), "sha256": before})
    raw = pd.concat(frames, ignore_index=True)
    canonical, lineage, conflicts = canonicalize_aliases(raw, aliases)
    output = root / "output/experiments/s20_safe_v4/sources" / ("identity-" + uuid.uuid4().hex)
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe output")
    output.mkdir(parents=True)
    for name, frame in [("alias_source_rows", raw), ("canonical_alias_rows", canonical), ("lineage", lineage), ("conflicts", conflicts)]:
        frame.to_parquet(output / (name + ".parquet"), index=False)
    result = {"at": now(), "directory": str(output), "source_rows": len(raw), "canonical_rows": len(canonical),
              "conflicting_groups": len(conflicts), "collapsed_equal_alias_rows": len(raw) - len(canonical) if conflicts.empty else None,
              "code_hash": digest(Path(__file__)), "config_hash": digest(config),
              "formal_H01_gate_passed": False, "exhaustive_alias_master": False,
              "production_source_changed": False}
    atomic_json(output / "summary.json", result)
    atomic_json(output / "inputs.json", inputs)
    return result


if __name__ == "__main__":
    print(json.dumps(audit_local_aliases(Path(__file__).resolve().parents[2]), indent=2))
