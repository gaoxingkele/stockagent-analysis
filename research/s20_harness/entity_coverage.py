"""Listing coverage on verified stable identities, not a PIT selection universe."""
import json
from pathlib import Path
import uuid

import pandas as pd

from .identity_panel_verify import verify, _within
from .runtime import atomic_json, digest, now
from .universe_coverage import listing_coverage


def map_metadata(basic, aliases):
    mapping = {}
    for a in aliases:
        if a["share_conversion_ratio"] != 1.0:
            raise ValueError("non-unit alias unsupported")
        for code in [a["old_code"], a["new_code"]]:
            if code in mapping:
                raise ValueError("ambiguous alias chain")
            mapping[code] = a["entity_id"]
    result = basic.copy()
    result["metadata_code"] = result.ts_code
    result["ts_code"] = result.ts_code.map(lambda code: mapping.get(code, code))
    if result.ts_code.duplicated().any():
        raise ValueError("multiple metadata intervals for entity require review")
    return result


def build(root, panel, panel_sha, basic_path, basic_sha, calendar_path, calendar_sha):
    root, panel = Path(root).resolve(), Path(panel).resolve()
    config = root / "config/s20_v4_security_aliases.json"
    pins = [(Path(basic_path), basic_sha), (Path(calendar_path), calendar_sha), (config, digest(config))]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input pin mismatch")
    verified = verify(root, panel, panel_sha)
    manifest = json.loads((panel / "summary.json").read_text(encoding="utf-8"))
    if manifest["conflict_groups"] or manifest["duplicate_entity_rows"]:
        raise ValueError("unresolved panel identity conflicts")
    receipt_path = panel / "inputs_outputs.json"
    if digest(receipt_path) != manifest["receipt_sha256"]:
        raise ValueError("panel receipt changed")
    receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    frames = []
    for r in receipts:
        p = _within(panel, r["canonical"])
        if digest(p) != r["canonical_sha256"]:
            raise ValueError("panel partition changed")
        frame = pd.read_parquet(p, columns=["entity_id", "trading_code", "trade_date"])
        frames.append(frame.loc[frame.trading_code.str.endswith((".SH", ".SZ"))])
        if digest(p) != r["canonical_sha256"]:
            raise ValueError("partition changed while reading")
    observed = pd.concat(frames, ignore_index=True).rename(columns={"entity_id": "ts_code"})
    basic = pd.read_parquet(basic_path)
    basic = basic.loc[basic.ts_code.str.endswith((".SH", ".SZ"))]
    aliases = json.loads(config.read_text(encoding="utf-8"))["aliases"]
    mapped = map_metadata(basic, aliases)
    cal = pd.read_parquet(calendar_path)
    dates = [sorted(cal.loc[cal.exchange.eq(ex) & cal.is_open.astype(int).eq(1), "cal_date"].astype(str)) for ex in ["SSE", "SZSE"]]
    if dates[0] != dates[1] or sorted(Path(r["source"]).stem for r in receipts) != dates[0]:
        raise ValueError("panel/calendar scope mismatch")
    table, report = listing_coverage(mapped, observed, dates[0])
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input changed during coverage audit")
    out = root / "output/experiments/s20_safe_v4/sources" / ("entity-coverage-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    table.to_parquet(out / "coverage.parquet", index=False)
    atomic_json(out / "inputs.json", [{"path": str(p.resolve()), "sha256": sha} for p, sha in pins] +
                [{"path": str(panel / "summary.json"), "sha256": panel_sha},
                 {"path": str(receipt_path), "sha256": manifest["receipt_sha256"]}])
    report.update(directory=str(out), at=now(), panel_verification=verified,
                  observed_entity_dates=len(observed), code_sha256=digest(Path(__file__)),
                  table_sha256=digest(out / "coverage.parquet"), inputs_sha256=digest(out / "inputs.json"),
                  scope="SH/SZ reviewed stable identities; current listing metadata, not exhaustive PIT master")
    atomic_json(out / "summary.json", report)
    return report
