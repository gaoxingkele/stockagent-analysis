"""Snapshot registered component evidence without promoting it to PIT truth."""
from __future__ import annotations

from pathlib import Path

from .runtime import digest, load_plan


ROLE_REQUIREMENTS = {
    "reviewed_distribution_ledger": "corporate_action_ledger",
    "entity_listing_coverage": "historical_universe",
    "identity_aware_daily_panel": "security_identity",
    "daily_suspension_events": "historical_status",
    "adjustment_factors": "adjustment_factors",
    "corporate_action_distributions": "corporate_action_ledger",
    "per_code_historical_names": "historical_status",
    "daily_price_limits_fresh": "effective_market_rules",
}


def inspect_sources(root: Path, inventory: dict, *, revalidate_identity=False, revalidate_coverage=False,
                    revalidate_adjustments=False, revalidate_cash_ledger=False, revalidate_limits=False,
                    revalidate_names=False, revalidate_state_join=False, revalidate_suspensions=False):
    records, artifacts = [], []
    for source in inventory.get("sources", []):
        base = root / source["path"]
        item = {"role": source["role"], "path": str(base), "present": base.exists(),
                "declared_status": source.get("status"), "evidence": [],
                "limitations": source.get("limitations", source.get("gaps", [])),
                "formal_requirement": ROLE_REQUIREMENTS.get(source["role"]),
                "semantic_acceptance": False}
        references = []
        for field in ("audit", "reconciliation", "state_join_audit", "special_value_diagnostics", "timeline",
                      "retrospective_event_overlay", "selected_exchange_crosscheck", "gap_validity"):
            if field not in source:
                continue
            value = Path(source[field])
            # Inventory uses root-relative output paths and source-relative child paths.
            path = value if value.is_absolute() else (root / value if value.parts[0] == "output" else base / value)
            references.append((field, path, source.get(field + "_sha256")))
        if "summary_sha256" in source:
            references.append(("summary", base / "summary.json", source["summary_sha256"]))
        if "sha256" in source:
            references.append(("source", base, source["sha256"]))
        for field, path, expected in references:
            evidence = {"field": field, "path": str(path), "present": path.is_file(),
                        "expected_sha256": expected, "hash_verified": False}
            if path.is_file():
                sha = digest(path)
                evidence.update({"observed_sha256": sha, "hash_verified": expected is not None and sha == expected})
                if expected is not None and sha != expected:
                    raise ValueError("registered evidence hash mismatch: " + str(path))
                if path.suffix == ".json":
                    evidence["recorded_summary"] = load_plan(path)
                if digest(path) != sha:
                    raise ValueError("evidence changed during audit: " + str(path))
                artifacts.append({"path": str(path), "sha256": sha,
                                  "role": "component_evidence_snapshot_not_full_revalidation"})
            item["evidence"].append(evidence)
        records.append(item)
        if revalidate_suspensions and source['role'] == 'daily_suspension_events':
            from .suspension_daily_verify import verify as verify_suspensions
            reference = next((e for e in item['evidence'] if e['field'] == 'audit'), None)
            if not reference or not reference['hash_verified']:
                raise ValueError('pinned suspension audit required')
            checked = verify_suspensions(root, Path(reference['path']).parent,
                                         reference['expected_sha256'], base)
            artifacts.extend(checked.pop('evidence_files'))
            item['fresh_suspension_validation'] = checked
            item['validation_scope'] = 'reconstructed suspension receipts; not tradability semantics or historical availability'
        if revalidate_names and source["role"] == "per_code_historical_names":
            from .name_timeline_verify import verify as verify_names
            reference = next((e for e in item["evidence"] if e["field"] == "timeline"), None)
            if not reference or not reference["hash_verified"]:
                raise ValueError("pinned name timeline required")
            checked = verify_names(root, Path(reference["path"]), reference["expected_sha256"])
            artifacts.extend(checked.pop("evidence_files"))
            item["fresh_name_validation"] = checked
            item["validation_scope"] = "reconstructed announcement-gated name proxy; not official ST or historical receipt"
        if revalidate_limits and source["role"] == "daily_price_limits_fresh":
            from .price_limit_verify import verify as verify_limits
            reference = next((e for e in item["evidence"] if e["field"] == "audit"), None)
            if not reference or not reference["present"]:
                raise ValueError("limit audit required")
            checked = verify_limits(root, Path(reference["path"]).parent, reference["observed_sha256"], base)
            artifacts.extend(checked.pop("evidence_files"))
            item["fresh_limit_validation"] = checked
            item["validation_scope"] = "reconstructed vendor limit coverage; not effective rules or auction fills"
        if revalidate_cash_ledger and source["role"] == "reviewed_distribution_ledger":
            from .cash_action_source import load_ledger
            if not source.get("summary_sha256"):
                raise ValueError("pinned cash ledger required")
            _, _, checked = load_ledger(base, source["summary_sha256"], require_code_pin=True)
            for path, sha in checked.pop("source_pins").items():
                artifacts.append({"path": path, "sha256": sha,
                                  "role": "fresh_reviewed_distribution_ledger"})
            item["fresh_cash_ledger_validation"] = checked
            item["validation_scope"] = "reconstructed reviewed gross cash ledger; unresolved rows retained; not exhaustive or PIT"
        if revalidate_adjustments and source["role"] == "adjustment_factors":
            from .adjustment_verify import verify as verify_adjustments
            reference = next((e for e in item["evidence"] if e["field"] == "reconciliation"), None)
            if not reference or not reference["present"]:
                raise ValueError("adjustment reconciliation required")
            checked = verify_adjustments(root, Path(reference["path"]).parent, reference["observed_sha256"])
            artifacts.extend(checked.pop("evidence_files"))
            item["fresh_adjustment_validation"] = checked
            item["validation_scope"] = "reconstructed observed factor ratios; not cash ledger or PIT"
        if revalidate_coverage and source["role"] == "entity_listing_coverage":
            if not source.get("summary_sha256") or not base.is_dir():
                raise ValueError("pinned entity coverage required")
            from .entity_coverage_verify import verify as verify_coverage
            item["fresh_coverage_validation"] = verify_coverage(root, base, source["summary_sha256"])
            item["validation_scope"] = "reconstructed retrospective listing coverage; not PIT universe"
            for name in ("inputs.json", "coverage.parquet"):
                artifacts.append({"path": str(base/name), "sha256": digest(base/name),
                                  "role": "fresh_retrospective_coverage_evidence"})
        if revalidate_identity and source["role"] == "identity_aware_daily_panel":
            if not source.get("summary_sha256") or not base.is_dir():
                raise ValueError("pinned identity panel required for fresh validation")
            from .identity_panel_verify import verify
            checked = verify(root, base, source["summary_sha256"])
            item["fresh_identity_validation"] = checked
            item["identity_mapping_verified"] = (checked["semantic_reconstruction_verified"]
                and checked["conflict_groups"] == 0 and checked["duplicate_entity_rows"] == 0)
            item["validation_scope"] = "full current raw partition set and reviewed alias reconstruction; not PIT universe"
            # Preserve distinction: resolving identity does not certify all data semantics.
            item["semantic_acceptance"] = False
            receipt = base / "inputs_outputs.json"
            artifacts.append({"path": str(receipt), "sha256": digest(receipt),
                              "role": "fresh_identity_reconstruction_receipt"})
    if revalidate_state_join and any(r["role"] == "per_code_historical_names" for r in records):
        from .state_join_verify import reconstruct
        identity = [r for r in records if r["role"] == "identity_aware_daily_panel"]
        names = [r for r in records if r["role"] == "per_code_historical_names"]
        if len(identity) != 1 or len(names) != 1:
            raise ValueError("unique identity and name sources required for state join")
        identity, names = identity[0], names[0]
        if not identity.get("identity_mapping_verified") or not names.get("fresh_name_validation", {}).get("full_timeline_reconstructed"):
            raise ValueError("fresh identity and name reconstruction required before state join")
        panel_ref = next(e for e in identity["evidence"] if e["field"] == "summary")
        timeline_ref = next(e for e in names["evidence"] if e["field"] == "timeline")
        join_ref = next((e for e in names["evidence"] if e["field"] == "state_join_audit"), None)
        if not join_ref or not join_ref["present"]:
            raise ValueError("state join audit required")
        checked = reconstruct(Path(join_ref["path"]).parent, join_ref["observed_sha256"],
            Path(identity["path"]), panel_ref["expected_sha256"],
            Path(timeline_ref["path"]), timeline_ref["expected_sha256"])
        artifacts.extend(checked.pop("evidence_files"))
        checked["upstream_semantics_verified_in_same_source_audit"] = True
        names["fresh_state_join_validation"] = checked
    return records, artifacts
