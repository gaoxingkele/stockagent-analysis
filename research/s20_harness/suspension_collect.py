"""Bounded explicit-source acquisition, preserving failures and raw responses."""
from pathlib import Path
import re
from urllib.parse import urlparse
import uuid

from bs4 import BeautifulSoup
import pandas as pd
import requests

from .runtime import atomic_json, digest, load_plan, now
from .suspension_monthly import parse_table


def collect(root, config_path):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config_sha = digest(config_path)
    plan = load_plan(config_path)
    sources = plan["sources"]
    if not 1 <= len(sources) <= 40 or len({s["month"] for s in sources}) != len(sources):
        raise ValueError("bounded unique-month source list required")
    for source in sources:
        url = urlparse(source["url"])
        if url.scheme != "https" or url.hostname != "docs.static.szse.cn" or not re.fullmatch(r"20\d{2}\.(0[1-9]|1[0-2])", source["month"]):
            raise ValueError("invalid source domain/month")
    output = root / "output/experiments/s20_safe_v4/sources" / ("suspension-collection-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    receipts, tables = [], []
    atomic_json(output / "state.json", {"state": "RUNNING", "at": now()})
    for source in sources:
        record = {**source, "requested_at": now(), "status": "FAILED_INFRA"}
        try:
            response = requests.get(source["url"], timeout=(10, 20), allow_redirects=False)
            record.update(received_at=now(), http_status=response.status_code)
            response.raise_for_status()
            if response.status_code != 200 or len(response.content) > 5_000_000:
                raise ValueError("unexpected status or size")
            archive = output / (source["month"] + ".html")
            archive.write_bytes(response.content)
            record.update(archive_path=str(archive), sha256=digest(archive), status="FAILED_VALIDITY")
            frame = None
            for encoding in ("utf-8", "gb18030"):
                try:
                    decoded = response.content.decode(encoding, errors="strict")
                except UnicodeDecodeError:
                    continue
                text = BeautifulSoup(decoded, "html.parser").get_text(" ", strip=True)
                if source["month"] not in text or "证券停牌情况" not in text:
                    continue
                frame = parse_table(response.content, encoding=encoding, source_url=source["url"],
                                    observed_at=record["received_at"], expected_sha256=record["sha256"])
                record["encoding"] = encoding
                break
            if frame is None:
                raise ValueError("month/title/encoding validation failed")
            frame["report_month"] = source["month"]
            frame.to_parquet(output / (source["month"] + ".parquet"), index=False)
            record.update(status="PARSED", rows=len(frame),
                          parquet_sha256=digest(output / (source["month"] + ".parquet")))
            tables.append(frame)
        except (requests.RequestException, ValueError, OSError) as exc:
            record["error"] = str(exc)
        receipts.append(record)
        atomic_json(output / "receipts.json", receipts)
    merged = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if tables:
        merged.to_parquet(output / "intervals.parquet", index=False)
    report = {"directory": str(output), "at": now(), "requested_months": len(sources),
              "parsed_months": len(tables), "rows": len(merged),
              "states": merged.interval_status.value_counts().to_dict() if tables else {},
              "config_sha256": config_sha, "code_sha256": digest(Path(__file__)),
              "parser_sha256": digest(Path(__file__).with_name("suspension_monthly.py")),
              "receipts_sha256": digest(output / "receipts.json"),
              "intervals_sha256": digest(output / "intervals.parquet") if tables else None,
              "full_history_coverage_verified": False, "formal_training_authorized": False}
    if digest(config_path) != config_sha:
        raise ValueError("config changed during acquisition")
    atomic_json(output / "summary.json", report)
    atomic_json(output / "state.json", {"state": "COMPLETED_DIAGNOSTIC", "at": now(),
                                        "failed_sources": len(sources)-len(tables)})
    return report
