"""Parse archived SZSE suspension tables without inferring open-ended intervals."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
import uuid

from bs4 import BeautifulSoup
import pandas as pd

from .label_availability import _instant
from .runtime import atomic_json, digest, now


def parse_table(content, *, encoding, source_url, observed_at, expected_sha256):
    sha = hashlib.sha256(content).hexdigest()
    if sha != expected_sha256:
        raise ValueError("source hash mismatch")
    acquired = _instant(observed_at).isoformat()
    soup = BeautifulSoup(content.decode(encoding, errors="strict"), "html.parser")
    tables = [t for t in soup.find_all("table")
              if "Susp. Time" in t.get_text() and "Resume Time" in t.get_text()]
    if len(tables) != 1:
        raise ValueError("one recognizable suspension table required")
    rows = []
    for index, tr in enumerate(tables[0].find_all("tr")):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
        if not cells:
            continue
        if len(cells) != 5 or not re.fullmatch(r"\d{6}", cells[0]):
            raise ValueError("unexpected suspension table row")
        code, name, reason, start, end = cells
        if not re.fullmatch(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}", start):
            raise ValueError("invalid suspension timestamp")
        begin = pd.Timestamp(start).tz_localize("Asia/Shanghai")
        missing = end in ("", "9999/12/31 00:00")
        if not missing and not re.fullmatch(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}", end):
            raise ValueError("invalid resumption timestamp")
        finish = None if missing else pd.Timestamp(end).tz_localize("Asia/Shanghai")
        if finish is not None and finish <= begin:
            raise ValueError("nonpositive suspension interval")
        rows.append({"source_row": index, "ts_code": code + ".SZ", "source_name": name,
                     "reason": reason, "suspended_at": begin.isoformat(),
                     "resumed_at": None if finish is None else finish.isoformat(),
                     "raw_resume": end, "interval_status": "unknown_end" if missing else "closed_reported",
                     "source_url": source_url, "source_sha256": sha,
                     "source_observed_at": acquired, "historical_available_at": None})
    if not rows:
        raise ValueError("empty table needs explicit source coverage review")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["ts_code", "suspended_at"]).any():
        raise ValueError("duplicate suspension identity requires review")
    return frame


def materialize(root, source):
    root = Path(root).resolve()
    path = root / source["archive_path"]
    content = path.read_bytes()
    frame = parse_table(content, encoding=source["archive_encoding"], source_url=source["url"],
                        observed_at=source["archive_observed_by"], expected_sha256=source["archive_sha256"])
    output = root / "output/experiments/s20_safe_v4/sources" / ("suspension-table-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    frame.to_parquet(output / "intervals.parquet", index=False)
    report = {"at": now(), "directory": str(output), "rows": len(frame),
              "states": frame.interval_status.value_counts().to_dict(), "source": source,
              "parser_sha256": digest(Path(__file__)), "table_sha256": digest(output / "intervals.parquet"),
              "formal_training_authorized": False, "historical_availability_proven": False,
              "scope": "one retrospective monthly table; unknown ends not extended to future dates"}
    atomic_json(output / "summary.json", report)
    return report
