"""Left join date-effective name proxies without changing the candidate rows."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, load_plan, now


def attach_names(panel, timeline):
    required = {"source_code", "valid_from", "valid_until_exclusive", "name", "name_st_proxy", "status"}
    if not required.issubset(timeline.columns):
        raise ValueError("missing timeline fields")
    reserved = {"name_asof", "name_st_proxy", "name_proxy_status", "_name_join_order"}
    if reserved & set(panel.columns):
        raise ValueError("name output fields already exist")
    states = timeline[list(required)].copy()
    states = states.sort_values(["source_code", "valid_from"])
    prior_end = states.groupby("source_code").valid_until_exclusive.shift()
    if states.valid_from.ge(states.valid_until_exclusive).any() or prior_end.gt(states.valid_from).any():
        raise ValueError("overlapping or invalid name intervals")
    data = panel.copy()
    data["_name_join_order"] = range(len(data))
    chunks = []
    for date, group in data.groupby("trade_date", sort=False, dropna=False):
        if pd.isna(date):
            raise ValueError("missing panel date")
        active = states.loc[states.valid_from.le(str(date)) & states.valid_until_exclusive.gt(str(date))]
        active = active.rename(columns={"source_code": "trading_code", "name": "name_asof", "status": "name_proxy_status"})
        result = group.merge(active[["trading_code", "name_asof", "name_st_proxy", "name_proxy_status"]],
                             on="trading_code", how="left", validate="many_to_one")
        result["name_proxy_status"] = result.name_proxy_status.fillna("unknown")
        result["name_st_proxy"] = result.name_st_proxy.astype("boolean")
        result.loc[result.name_proxy_status.eq("unknown"), "name_st_proxy"] = pd.NA
        chunks.append(result)
    if not chunks:
        return data.drop(columns="_name_join_order").assign(name_asof=pd.Series(dtype="string"),
                  name_st_proxy=pd.Series(dtype="boolean"), name_proxy_status=pd.Series(dtype="string"))
    result = pd.concat(chunks, ignore_index=True).sort_values("_name_join_order").drop(columns="_name_join_order")
    result.index = panel.index
    if len(result) != len(panel):
        raise ValueError("state join changed recommendation population")
    return result


def audit_panel(root, panel_dir, timeline_path, expected_timeline_sha256):
    root, panel_dir, timeline_path = Path(root).resolve(), Path(panel_dir).resolve(), Path(timeline_path).resolve()
    inventory = load_plan(root / "config/s20_v4_data_sources.json")
    registered = next(s for s in inventory["sources"] if s["role"] == "identity_aware_daily_panel")
    if panel_dir != (root / registered["path"]).resolve() or digest(panel_dir / "summary.json") != registered["summary_sha256"]:
        raise ValueError("unregistered panel")
    summary = load_plan(panel_dir / "summary.json")
    if digest(panel_dir / "inputs_outputs.json") != summary["receipt_sha256"]:
        raise ValueError("panel receipt changed")
    if digest(timeline_path) != expected_timeline_sha256:
        raise ValueError("timeline hash mismatch")
    timeline = pd.read_parquet(timeline_path)
    receipts = json.loads((panel_dir / "inputs_outputs.json").read_text(encoding="utf-8"))
    rows = []
    for receipt in receipts:
        path = (panel_dir / receipt["canonical"]).resolve()
        if not path.is_relative_to(panel_dir) or digest(path) != receipt["canonical_sha256"]:
            raise ValueError("panel partition changed")
        joined = attach_names(pd.read_parquet(path), timeline)
        if digest(path) != receipt["canonical_sha256"]:
            raise ValueError("partition changed during join")
        unknown = joined.name_proxy_status.eq("unknown")
        rows.append({"date": path.stem, "rows": len(joined), "unknown_rows": int(unknown.sum()),
                     "st_proxy_rows": int(joined.name_st_proxy.fillna(False).sum()),
                     "unknown_codes": sorted(joined.loc[unknown, "trading_code"].unique().tolist())})
    if digest(timeline_path) != expected_timeline_sha256:
        raise ValueError("timeline changed during join")
    output = root / "output/experiments/s20_safe_v4/sources" / ("state-join-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    atomic_json(output / "date_coverage.json", rows)
    report = {"at": now(), "directory": str(output), "dates": len(rows),
              "rows": sum(r["rows"] for r in rows), "unknown_rows": sum(r["unknown_rows"] for r in rows),
              "st_proxy_rows": sum(r["st_proxy_rows"] for r in rows),
              "timeline_sha256": expected_timeline_sha256, "panel_summary_sha256": registered["summary_sha256"],
              "code_sha256": digest(Path(__file__)), "formal_training_eligible": False,
              "official_ST_status_proven": False,
              "unknown_policy": "preserved; no future-code or current-name fallback"}
    atomic_json(output / "summary.json", report)
    return report
