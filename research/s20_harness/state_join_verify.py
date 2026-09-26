"""Reconstruct state coverage from pinned outputs; upstream semantics verified separately."""
import json
from pathlib import Path

import pandas as pd

from . import historical_state
from .runtime import digest, load_plan


def reconstruct(directory, summary_sha, panel, panel_sha, timeline, timeline_sha):
    directory, panel, timeline = Path(directory).resolve(), Path(panel).resolve(), Path(timeline).resolve()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        sha = digest(path)
        if (expected is not None and sha != expected) or (path in pins and pins[path] != sha):
            raise ValueError('state join pin mismatch')
        pins[path] = sha
        return path

    summary = load_plan(pin(directory/'summary.json', summary_sha))
    if summary['panel_summary_sha256'] != panel_sha or summary['timeline_sha256'] != timeline_sha:
        raise ValueError('state join upstream binding mismatch')
    pin(Path(historical_state.__file__), summary['code_sha256'])
    panel_summary = load_plan(pin(panel/'summary.json', panel_sha))
    receipts = json.loads(pin(panel/'inputs_outputs.json', panel_summary['receipt_sha256']).read_text(encoding='utf-8'))
    states = pd.read_parquet(pin(timeline, timeline_sha))
    saved = json.loads(pin(directory/'date_coverage.json').read_text(encoding='utf-8'))
    rows, dates = [], set()
    for receipt in receipts:
        path = (panel/receipt['canonical']).resolve()
        if not path.is_relative_to(panel) or path.stem in dates:
            raise ValueError('invalid state join partition inventory')
        dates.add(path.stem)
        data = pd.read_parquet(pin(path, receipt['canonical_sha256']))
        if not data.trade_date.astype(str).eq(path.stem).all():
            raise ValueError('state join partition date mismatch')
        joined = historical_state.attach_names(data, states)
        pd.testing.assert_frame_equal(joined[data.columns], data, check_exact=True)
        unknown = joined.name_proxy_status.eq('unknown')
        if joined.loc[unknown, 'name_st_proxy'].notna().any():
            raise ValueError('unknown state converted to known')
        rows.append(dict(date=path.stem, rows=len(joined), unknown_rows=int(unknown.sum()),
                         st_proxy_rows=int(joined.name_st_proxy.fillna(False).sum()),
                         unknown_codes=sorted(joined.loc[unknown, 'trading_code'].unique().tolist())))
    if rows != saved:
        raise ValueError('state join reconstructed daily coverage mismatch')
    counts = dict(dates=len(rows), rows=sum(r['rows'] for r in rows),
                  unknown_rows=sum(r['unknown_rows'] for r in rows),
                  st_proxy_rows=sum(r['st_proxy_rows'] for r in rows))
    if any(summary[k] != value for k, value in counts.items()):
        raise ValueError('state join reconstructed summary mismatch')
    for path, sha in pins.items():
        if digest(path) != sha:
            raise ValueError('state join source changed during reconstruction')
    return dict(counts, full_join_reconstructed=True, upstream_semantics_verified_here=False,
                official_ST_status_proven=False, historical_availability_proven=False,
                formal_training_eligible=False, evidence_files=[dict(path=str(p), sha256=h,
                    role='fresh_state_join_reconstruction') for p,h in pins.items()])
