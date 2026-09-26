"""Announcement-gated historical name proxy; never a guaranteed ST registry."""
from __future__ import annotations

import pandas as pd
import re


def name_asof(events, codes, signal_date):
    """Day-only announcements enter on the next date, not at assumed 21:00.

    Future end_date fields are not used to infer a change before its announcement.
    This is a retrospective vendor-history proxy, not proof of vintage accuracy.
    """
    signal = pd.to_datetime(str(signal_date), format="%Y%m%d", errors="raise")
    rows = events.loc[events.ts_code.isin(codes)].copy()
    starts = pd.to_datetime(rows.start_date.astype("string"), format="%Y%m%d", errors="coerce")
    announcements = pd.to_datetime(rows.ann_date.astype("string"), format="%Y%m%d", errors="coerce")
    rows = rows.assign(_start=starts, _announcement=announcements)
    known = rows.loc[rows._start.le(signal) & rows._announcement.lt(signal)]
    result = {"signal_date": str(signal_date), "name": None, "name_st_proxy": None,
              "status": "unknown", "historical_revision_availability_proven": False,
              "official_ST_status_proven": False}
    if known.empty:
        result["reason"] = "no_effective_previously_announced_name"
        return result
    latest = known.loc[known._start.eq(known._start.max())]
    latest = latest.loc[latest._announcement.eq(latest._announcement.max())]
    normalized_names = latest.name.astype("string").str.strip()
    values = normalized_names.dropna().unique()
    if len(values) != 1 or normalized_names.isna().any() or normalized_names.eq("").any():
        result["reason"] = "conflicting_or_missing_latest_name"
        return result
    name = values[0]
    result.update(name=name, name_st_proxy=bool(re.match(r"^S?\*?ST", name.upper())),
                  status="known_name_proxy", reason="announcement_gated_not_official_status",
                  effective_start=latest._start.iloc[0].strftime("%Y%m%d"),
                  announcement_date=latest._announcement.iloc[0].strftime("%Y%m%d"))
    return result
