"""Reusable supplied-event index; missing dates remain relevant to every window."""
from bisect import bisect_left, bisect_right

import pandas as pd


class EventIndex:
    def __init__(self, frame, events, decisions):
        required = {"ts_code", "normalized_event_id", "record_date"}
        if not required.issubset(frame) or frame[["ts_code", "normalized_event_id"]].isna().any().any():
            raise ValueError("missing event identity")
        if frame.normalized_event_id.duplicated().any():
            raise ValueError("duplicate source event")
        self.accepted = {}
        for d in decisions:
            key = str(d["normalized_event_id"])
            if key in self.accepted or type(d["accepted_for_accounting"]) is not bool:
                raise ValueError("duplicate decision or invalid acceptance")
            self.accepted[key] = d["accepted_for_accounting"]
        self.events = {}
        self.order = {}
        for i, event in enumerate(events):
            if event.event_id in self.events:
                raise ValueError("duplicate event payload")
            self.events[event.event_id] = event
            self.order[event.event_id] = i
        parsed = pd.to_datetime(frame.record_date.astype("string"), format="%Y%m%d", errors="coerce")
        grouped, unknown = {}, {}
        for code, event_id, date in zip(frame.ts_code, frame.normalized_event_id.astype(str), parsed):
            if pd.isna(date):
                unknown.setdefault(code, set()).add(event_id)
            else:
                grouped.setdefault(code, []).append((date.value, event_id))
        self.unknown = unknown
        self.rows = {code: sorted(rows) for code, rows in grouped.items()}
        self.dates = {code: [date for date, _ in rows] for code, rows in self.rows.items()}

    def window(self, codes, entry_date, end_date):
        if not isinstance(codes, (list, tuple)) or not codes:
            raise ValueError("explicit event codes required")
        start, end = pd.to_datetime(entry_date, format="%Y%m%d"), pd.to_datetime(end_date, format="%Y%m%d")
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("invalid event window")
        ids = set()
        for code in codes:
            ids.update(self.unknown.get(code, ()))
            dates = self.dates.get(code, [])
            rows = self.rows.get(code, [])
            ids.update(event_id for _, event_id in rows[bisect_left(dates, start.value):bisect_right(dates, end.value)])
        unresolved = sorted(key for key in ids if not self.accepted.get(key, False))
        if unresolved:
            return dict(events=[], unresolved_event_ids=unresolved, status="unknown_event_terms", event_coverage_proven=False)
        if not ids.issubset(self.events):
            raise ValueError("accepted event payload missing")
        return dict(events=[self.events[key] for key in sorted(ids, key=self.order.get)], unresolved_event_ids=[],
                    status="supplied_events_resolved", event_coverage_proven=False)
