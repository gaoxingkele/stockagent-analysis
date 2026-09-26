"""Hash-pinned raw daily closes observed now, never backdated receipts."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import uuid

import pandas as pd

from .portfolio_valuation import PriceMark
from .exit_book import positive
from .label_availability import _instant
from .runtime import atomic_json, digest, now


def read_marks(path, expected_sha, trade_date, codes):
    path = Path(path).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") or digest(path) != expected_sha:
        raise ValueError("close source pin mismatch")
    if not re.fullmatch(r"\d{8}", trade_date or ""):
        raise ValueError("invalid close date")
    close_at = pd.Timestamp(trade_date + " 15:00", tz="Asia/Shanghai").tz_convert("UTC")
    if not codes or len(set(codes)) != len(codes) or any(not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", c) for c in codes):
        raise ValueError("unique explicit trading codes required")
    frame = pd.read_parquet(path)
    if not {"ts_code", "trade_date", "close"}.issubset(frame):
        raise ValueError("raw daily close fields missing")
    if frame.ts_code.isna().any() or frame.trade_date.isna().any() or not frame.trade_date.astype(str).eq(trade_date).all():
        raise ValueError("source date or identity mismatch")
    selected = frame.loc[frame.ts_code.isin(codes)]
    if selected.ts_code.duplicated().any():
        raise ValueError("duplicate selected close identity")
    indexed = selected.set_index("ts_code")
    values = []
    for code in codes:
        price, reason = None, "no_row_in_supplied_source"
        if code in indexed.index:
            raw = indexed.loc[code, "close"]
            try:
                if isinstance(raw, bool):
                    raise ValueError("boolean price")
                price = float(raw)
                positive(price, "close")
                reason = "observed_raw_close"
            except (TypeError, ValueError):
                price, reason = None, "invalid_raw_close"
        values.append((code, price, reason))
    if digest(path) != expected_sha:
        raise ValueError("close source changed during read")
    observed = _instant(now())
    if close_at > observed:
        raise ValueError("daily close is later than observation")
    marks = [PriceMark(code, close_at.isoformat(), observed.isoformat(),
                       "observed" if price is not None else "missing", price,
                       str(path), expected_sha) for code, price, _ in values]
    return marks, dict(source_path=str(path), source_sha256=expected_sha,
        source_rows=len(frame), requested_codes=list(codes), trade_date=trade_date,
        observed_at=observed.isoformat(), valuation_at=close_at.isoformat(),
        decisions=[dict(ts_code=c, reason=r) for c, _, r in values],
        source_bytes_verified=True, availability_basis="verified present now, not original historical receipt",
        raw_source_semantics_independently_verified=False,
        historical_availability_proven=False, formal_training_authorized=False)


def build(root, path, expected_sha, trade_date, codes):
    marks, report = read_marks(path, expected_sha, trade_date, codes)
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("close-marks-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out / "marks.json", [asdict(m) for m in marks])
    report.update(directory=str(out), code_sha256=digest(Path(__file__)),
                  marks_sha256=digest(out / "marks.json"))
    atomic_json(out / "summary.json", report)
    return marks, report
