"""Bounded public disclosure search; candidate links are not reviewed evidence."""
import json
import re
from pathlib import Path
import uuid

import pandas as pd
import requests

from .runtime import atomic_json, digest, now


def validate_page(payload):
    rows = payload.get("announcements")
    total, more = payload.get("totalAnnouncement"), payload.get("hasMore")
    # Provider's explicit empty response uses null rather than an empty array.
    # Accept only the unambiguous terminal zero-count case.
    if rows is None and type(total) is int and total == 0 and more is False:
        rows = []
    if not isinstance(rows, list) or type(total) is not int or total < 0 or type(more) is not bool:
        raise ValueError("invalid disclosure page schema")
    ids = [str(r.get("announcementId", "")) for r in rows]
    if any(not re.fullmatch(r"\d+", x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("missing/duplicate page announcement IDs")
    return rows, total, more


def collect(root, dates, *, max_pages=10, keyword="权益分派实施"):
    if keyword not in ("权益分派实施", "分红派息"):
        raise ValueError("unsupported bounded discovery keyword")
    if not dates or len(dates) > 5 or len(dates) != len(set(dates)) or not 1 <= max_pages <= 10:
        raise ValueError("bounded unique date/page scope required")
    for date in dates:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            raise ValueError("ISO date required")
        pd.Timestamp(date)
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("announcement-discovery-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    receipts, results, all_rows = [], [], []
    atomic_json(out / "plan.json", dict(dates=dates, max_pages_per_date=max_pages, searchkey=keyword,
                                        code_sha256=digest(Path(__file__)), at=now()))
    for date in dates:
        seen, declared_total, complete = set(), None, False
        for page in range(1, max_pages + 1):
            params = dict(pageNum=page, pageSize=30, column="sse", tabName="fulltext",
                          searchkey=keyword, seDate=date + "~" + date)
            receipt = dict(date=date, page=page, requested_at=now(), params=params)
            try:
                response = requests.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=params,
                                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                receipt["received_at"] = now()
                response.raise_for_status()
                raw = out / (date + "-" + str(page) + ".json")
                raw.write_bytes(response.content)
                receipt.update(raw_file=raw.name, raw_sha256=digest(raw))
                rows, total, more = validate_page(response.json())
                if declared_total is not None and total != declared_total:
                    raise ValueError("search total changed during pagination")
                declared_total = total
                ids = {str(r["announcementId"]) for r in rows}
                if ids & seen:
                    raise ValueError("repeated page or overlapping results")
                seen |= ids
                all_rows.extend(dict(query_date=date, query_page=page, raw_sha256=receipt["raw_sha256"], **r) for r in rows)
                receipt.update(rows=len(rows), declared_total=total, has_more=more, status="ACQUIRED_NOT_REVIEWED")
                if not more:
                    complete = len(seen) == total
                    if not complete:
                        receipt["coverage_issue"] = "terminal_page_count_disagrees"
                    break
                if not rows or len(seen) > total:
                    raise ValueError("pagination count contradiction")
            except Exception as exc:
                receipt.update(status="FAILED", error_type=type(exc).__name__)
                break
            finally:
                receipts.append(receipt)
                atomic_json(out / "receipts.json", receipts)
        results.append(dict(date=date, retrieved_unique=len(seen), declared_total=declared_total, complete_search=complete))
    atomic_json(out / "announcements.json", all_rows)
    report = dict(directory=str(out), queries=results, keyword=keyword, retrieved_rows=len(all_rows), requests=len(receipts),
                  announcements_sha256=digest(out / "announcements.json"), receipts_sha256=digest(out / "receipts.json"),
                  formal_event_acceptance=False, historical_availability_proven=False,
                  scope="keyword search discovery only; column parameter does not prove market filtering; titles may be unrelated")
    atomic_json(out / "summary.json", report)
    return report
