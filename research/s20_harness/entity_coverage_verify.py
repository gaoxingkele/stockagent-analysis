"""Fresh reconstruction of retrospective entity/listing coverage, not PIT."""
import json
from pathlib import Path

import pandas as pd

from .entity_coverage import map_metadata
from .identity_panel_verify import verify as verify_panel, _within
from .runtime import digest
from .universe_coverage import listing_coverage


def verify(root, directory, summary_sha):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if digest(directory/"summary.json") != summary_sha:
        raise ValueError("coverage summary pin mismatch")
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    for name, key in (("inputs.json", "inputs_sha256"), ("coverage.parquet", "table_sha256")):
        if digest(directory/name) != summary[key]:
            raise ValueError("coverage artifact pin mismatch")
    if digest(Path(__file__).with_name("entity_coverage.py")) != summary["code_sha256"]:
        raise ValueError("coverage builder changed")
    refs = json.loads((directory/"inputs.json").read_text(encoding="utf-8"))
    if len(refs) != 5 or len({r["path"] for r in refs}) != 5:
        raise ValueError("exact five coverage input bindings required")
    pins = [(Path(r["path"]).resolve(), r["sha256"]) for r in refs]
    def check():
        if any(digest(p) != h for p, h in pins):
            raise ValueError("coverage source pin mismatch")
        if digest(directory/"summary.json") != summary_sha or digest(directory/"inputs.json") != summary["inputs_sha256"]:
            raise ValueError("coverage manifest changed")
    check()
    basic_path, calendar_path, aliases_path, panel_summary, panel_receipt = [p for p, _ in pins]
    panel = panel_summary.parent
    if (aliases_path != (root/"config/s20_v4_security_aliases.json").resolve()
            or panel_summary.name != "summary.json" or panel_receipt != panel/"inputs_outputs.json"):
        raise ValueError("coverage source roles mismatch")
    identity = verify_panel(root, panel, pins[3][1])
    if identity["conflict_groups"] or identity["duplicate_entity_rows"]:
        raise ValueError("coverage identity unresolved")
    panel_meta = json.loads(panel_summary.read_text(encoding="utf-8"))
    if pins[4][1] != panel_meta["receipt_sha256"]:
        raise ValueError("coverage panel receipt mismatch")
    receipts = json.loads(panel_receipt.read_text(encoding="utf-8"))
    frames = []
    partition_pins = []
    for receipt in receipts:
        path = _within(panel, receipt["canonical"])
        if digest(path) != receipt["canonical_sha256"]:
            raise ValueError("coverage partition changed")
        frame = pd.read_parquet(path, columns=["entity_id", "trading_code", "trade_date"])
        frames.append(frame.loc[frame.trading_code.str.endswith((".SH", ".SZ"))])
        partition_pins.append((path, receipt["canonical_sha256"]))
    observed = pd.concat(frames, ignore_index=True).rename(columns={"entity_id": "ts_code"})
    basic = pd.read_parquet(basic_path)
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))["aliases"]
    mapped = map_metadata(basic.loc[basic.ts_code.str.endswith((".SH", ".SZ"))], aliases)
    calendar = pd.read_parquet(calendar_path)
    dates = [sorted(calendar.loc[calendar.exchange.eq(ex) & calendar.is_open.astype(int).eq(1), "cal_date"].astype(str)) for ex in ("SSE", "SZSE")]
    if dates[0] != dates[1] or dates[0] != sorted(Path(r["source"]).stem for r in receipts):
        raise ValueError("coverage calendar scope mismatch")
    rebuilt, metrics = listing_coverage(mapped, observed, dates[0])
    try:
        pd.testing.assert_frame_equal(rebuilt, pd.read_parquet(directory/"coverage.parquet"), check_exact=True,
                                      check_dtype=False, check_column_type=False)
    except AssertionError as exc:
        raise ValueError("coverage semantic table mismatch") from exc
    for key, value in metrics.items():
        if summary[key] != value:
            raise ValueError("coverage semantic summary mismatch: " + key)
    if summary["observed_entity_dates"] != len(observed):
        raise ValueError("coverage denominator mismatch")
    check()
    if digest(directory/"coverage.parquet") != summary["table_sha256"] or any(digest(p) != h for p, h in partition_pins):
        raise ValueError("coverage partition/table changed")
    missing = rebuilt.loc[rebuilt.status.eq("no_observations_in_listing_window")]
    return {"semantic_reconstruction_verified": True, "identity_verification": identity,
            "metadata_rows": len(rebuilt), "observed_entity_dates": len(observed),
            "status_counts": metrics["status_counts"], "unobserved_listing_entities": missing.ts_code.tolist(),
            "observed_entities_missing_metadata": metrics["observed_codes_missing_metadata"],
            "calendar_days": len(dates[0]), "formal_universe_verified": False,
            "limitation": "current metadata listing intervals are retrospective; missing quotes are not presumed suspensions"}
