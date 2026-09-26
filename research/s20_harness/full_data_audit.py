"""Full local OHLC audit. Availability and economic-price gaps fail closed."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .runtime import atomic_json, digest, now, load_plan


REQUIRED = {"ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"}
ANOMALY_COLUMNS = ["file", "ts_code", "trade_date", "issue", "detail"]


def audit_dataset(root: Path, directory: Path) -> dict:
    cache = root / "output/tushare_cache"
    files = sorted((cache / "daily").glob("*.parquet"))
    manifest_files, anomalies, summaries = [], [], []
    last = {}
    observed_codes = set()
    first_seen = {}
    total_rows = 0
    missing_sessions = 0
    discontinuities = 0
    status = Counter()

    def record(path, code, date, issue, detail):
        anomalies.append({"file": path.name, "ts_code": str(code), "trade_date": str(date),
                          "issue": issue, "detail": str(detail)})

    for index, path in enumerate(files):
        before = digest(path)
        frame = pd.read_parquet(path)
        if digest(path) != before:
            raise ValueError("data changed while reading: " + str(path))
        manifest_files.append({"path": path.relative_to(root).as_posix(), "sha256": before,
                               "bytes": path.stat().st_size, "rows": len(frame),
                               "schema": {c: str(t) for c, t in frame.dtypes.items()}})
        missing = REQUIRED - set(frame.columns)
        if missing:
            record(path, "", path.stem, "missing_columns", sorted(missing))
            continue
        total_rows += len(frame)
        dates = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        bad_date = dates.isna() | dates.dt.strftime("%Y%m%d").ne(path.stem)
        codes = frame["ts_code"].astype(str)
        bad_code = ~codes.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", na=False)
        duplicates = frame.duplicated(["ts_code", "trade_date"], keep=False)
        numeric = frame[list(sorted(REQUIRED - {"ts_code", "trade_date"}))].apply(pd.to_numeric, errors="coerce")
        bad_numeric = ~np.isfinite(numeric).all(axis=1)
        bad_price = (numeric[["open", "high", "low", "close", "pre_close"]] <= 0).any(axis=1)
        bad_bounds = ((numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)) |
                      (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)))
        bad_flow = (numeric[["vol", "amount"]] < 0).any(axis=1)
        checks = {"file_date_mismatch": bad_date, "invalid_code": bad_code,
                  "duplicate_stock_date": duplicates, "nonfinite_numeric": bad_numeric,
                  "nonpositive_price": bad_price, "invalid_ohlc_bounds": bad_bounds, "negative_flow": bad_flow}
        for issue, mask in checks.items():
            for i in frame.index[mask]:
                record(path, codes.loc[i], frame.loc[i, "trade_date"], issue, "row=" + str(i))
        valid = ~(bad_date | bad_code | duplicates | bad_numeric | bad_price | bad_bounds | bad_flow)
        for row in frame.loc[valid].itertuples(index=False):
            code = row.ts_code
            observed_codes.add(code)
            first_seen.setdefault(code, path.stem)
            status[code.rsplit(".", 1)[-1]] += 1
            if code in last:
                previous_close, previous_index, previous_date = last[code]
                gap = index - previous_index - 1
                if gap:
                    missing_sessions += gap
                    record(path, code, path.stem, "missing_observed_sessions",
                           f"previous={previous_date};count={gap};not_proven_suspension")
                difference = float(row.pre_close) - previous_close
                # Decimal prices are rounded; this is a review trigger, not a
                # claim that every discontinuity is a corporate action or error.
                if abs(difference) > max(.011, abs(previous_close) * .001):
                    discontinuities += 1
                    record(path, code, path.stem, "unreconciled_reference_close",
                           f"previous_close={previous_close};pre_close={row.pre_close};gap={gap}")
            last[code] = (float(row.close), index, path.stem)
        summaries.append({"date": path.stem, "rows": len(frame), "valid_rows": int(valid.sum())})

    basic_path = cache / "stock_basic.parquet"
    metadata = {"present": basic_path.exists(), "historical_status_verified": False}
    if basic_path.exists():
        basic_hash = digest(basic_path)
        basic = pd.read_parquet(basic_path)
        if digest(basic_path) != basic_hash:
            raise ValueError("stock metadata changed while reading")
        manifest_files.append({"path": basic_path.relative_to(root).as_posix(), "sha256": basic_hash,
                               "bytes": basic_path.stat().st_size, "rows": len(basic),
                               "schema": {c: str(t) for c, t in basic.dtypes.items()}})
        if "ts_code" in basic:
            basic_codes = set(basic["ts_code"].astype(str))
            metadata.update({"rows": len(basic), "daily_codes_missing_from_snapshot": sorted(observed_codes - basic_codes),
                             "list_status_counts": basic["list_status"].value_counts().to_dict() if "list_status" in basic else {}})
        metadata["use_policy"] = "inventory only; current name/ST/industry must not define historical universe"

    # These paths are an explicit input contract, not an assertion that a file
    # with a suggestive name is semantically or point-in-time valid.
    support_paths = {"exchange_calendar": "trade_cal.parquet", "adjustment_factors": "adj_factor.parquet",
                     "corporate_action_ledger": "corporate_actions.parquet", "historical_status": "namechange.parquet",
                     "effective_market_rules": "market_rules.parquet", "availability_ledger": "availability.parquet"}
    support = {k: {"path": (cache / v).relative_to(root).as_posix(), "present": (cache / v).is_file(),
                   "semantic_validation": "not_implemented"} for k, v in support_paths.items()}
    source_config = root / "config/s20_v4_data_sources.json"
    component_evidence = []
    if source_config.is_file() and files:
        from .calendar_source import validate_calendar
        config_hash = digest(source_config)
        source_plan = load_plan(source_config)
        from .source_evidence import inspect_sources
        component_evidence, evidence_files = inspect_sources(root, source_plan, revalidate_identity=True,
                                                            revalidate_coverage=True, revalidate_adjustments=True,
                                                            revalidate_cash_ledger=True, revalidate_limits=True,
                                                            revalidate_names=True, revalidate_state_join=True,
                                                            revalidate_suspensions=True)
        manifest_files.extend(evidence_files)
        for component in component_evidence:
            requirement = component["formal_requirement"]
            if component.get('fresh_suspension_validation'):
                support[requirement]['fresh_suspension_checks'] = component['fresh_suspension_validation']
            if component.get("fresh_state_join_validation"):
                support[requirement]["fresh_state_join_checks"] = component["fresh_state_join_validation"]
            if component.get("fresh_name_validation"):
                support[requirement]["fresh_name_checks"] = component["fresh_name_validation"]
            if component.get("fresh_limit_validation"):
                support[requirement]["fresh_limit_checks"] = component["fresh_limit_validation"]
            if component.get("fresh_cash_ledger_validation"):
                support[requirement]["fresh_cash_ledger_checks"] = component["fresh_cash_ledger_validation"]
            if requirement == "adjustment_factors":
                support[requirement]["fresh_ratio_checks"] = component.get("fresh_adjustment_validation", {})
            if requirement == "historical_universe":
                support[requirement] = {"path": component["path"], "present": component["present"],
                    "semantic_validation": "retrospective_coverage_reconstructed_PIT_unproven",
                    "checks": component.get("fresh_coverage_validation", {}), "PIT_universe_verified": False}
            if requirement == "security_identity":
                checked = component.get("fresh_identity_validation", {})
                support[requirement] = {"path": component["path"], "present": component["present"],
                    "semantic_validation": "validated_identity_mapping" if component.get("identity_mapping_verified") else "failed_identity_mapping",
                    "checks": checked, "PIT_universe_verified": False}
            if requirement in support and component["present"]:
                support[requirement].setdefault("available_components", []).append(component["role"])
                if requirement not in {"security_identity", "historical_universe"}:
                    support[requirement]["semantic_validation"] = "components_present_integration_required"
        if digest(source_config) != config_hash:
            raise ValueError("source inventory changed during audit")
        manifest_files.append({"path": source_config.relative_to(root).as_posix(), "sha256": config_hash})
        for source in source_plan.get("sources", []):
            if source.get("role") != "exchange_calendar" or source.get("status") != "validated_calendar_source":
                continue
            path = root / source["path"]
            if not path.is_file() or digest(path) != source.get("sha256"):
                raise ValueError("registered calendar missing or hash mismatch")
            calendar = pd.read_parquet(path)
            tests = {}
            for exchange in ("SSE", "SZSE"):
                subset = calendar.loc[calendar.exchange.eq(exchange)].copy()
                tests[exchange] = validate_calendar(subset, exchange, source["start"], source["end"])
                open_dates = set(subset.loc[subset.is_open.astype(int).eq(1), "cal_date"].astype(str))
                in_range = {d for d in open_dates if files[0].stem <= d <= files[-1].stem}
                observed = {p.stem for p in files}
                tests[exchange]["missing_daily_dates"] = sorted(in_range - observed)
                tests[exchange]["unexpected_daily_dates"] = sorted(observed - in_range)
                tests[exchange]["valid"] &= (not (observed ^ in_range) and
                                             source["start"] <= files[0].stem and source["end"] >= files[-1].stem)
            support["exchange_calendar"] = {"path": source["path"], "present": True,
                    "semantic_validation": "validated_SH_SZ" if all(t["valid"] for t in tests.values()) else "failed",
                    "checks": tests, "BSE_status": "unverified"}
            manifest_files.append({"path": source["path"], "sha256": source["sha256"], "rows": len(calendar)})
            break
    gaps = [f"{k}: " + ("registered components present; full semantic/PIT integration required" if v.get("available_components")
                         else "semantic audit required" if v["present"] else "missing registered source")
            for k, v in support.items() if v["semantic_validation"] not in {"validated_SH_SZ", "validated_identity_mapping"}]
    if support["exchange_calendar"]["semantic_validation"] == "validated_SH_SZ" and status.get("BJ"):
        gaps.append("BSE exchange calendar not validated; BJ remains separate")
    if not files:
        gaps.append("no daily files")
    counts = Counter(a["issue"] for a in anomalies)
    if anomalies:
        gaps.append("unresolved row/path anomalies; review ledger")
    gaps.append("historical universe/delisting coverage not established")
    report = {"audit_at": now(), "formal_gate_passed": False, "acceptance_gaps": gaps,
              "files_scanned": len(files), "rows_scanned": total_rows, "unique_stocks": len(observed_codes),
              "first_date": files[0].stem if files else None, "last_date": files[-1].stem if files else None,
              "observed_dates_are_not_verified_exchange_calendar": True,
              "market_row_counts": dict(status), "anomaly_counts": dict(counts),
              "missing_observed_sessions": missing_sessions, "reference_close_discontinuities": discontinuities,
              "metadata": metadata, "support_sources": support, "daily_coverage": summaries,
              "component_evidence": component_evidence,
              "component_evidence_policy": "summary snapshots are not fresh full source validation or formal gate approval",
              "stock_row_horizon_equivalent_to_market_horizon": None}
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "dataset_manifest.json", {"frozen_at": now(), "files": manifest_files,
                "storage": "hash-pinned source references; raw source cache may mutate; revalidate before reuse",
                "support_sources": support, "component_evidence": component_evidence,
                "universe_status": "unverified; no formal training"})
    atomic_json(directory / "data_audit.json", report)
    pd.DataFrame(anomalies, columns=ANOMALY_COLUMNS).to_parquet(directory / "anomaly_ledger.parquet", index=False)
    (directory / "coverage_report.md").write_text(
        "# H01 local data audit\n\n"
        f"Scanned {len(files)} files, {total_rows} rows, {len(observed_codes)} stocks.\n\n"
        "Formal data gate: NOT PASSED. Observed file dates are not an exchange calendar.\n\n"
        + "\n".join("- " + g for g in gaps) + "\n\n"
        "Full row/path anomalies are retained in anomaly_ledger.parquet. Missing sessions are not presumed suspensions.\n",
        encoding="utf-8")
    return report
