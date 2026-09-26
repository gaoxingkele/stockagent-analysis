"""Replay pinned suspension acquisition evidence, without approving tradability."""
from pathlib import Path
import json

import pandas as pd

from .runtime import digest, load_plan
from .suspension_daily_audit import audit


def verify(root, directory, summary_sha256, source):
    root, directory, source = Path(root).resolve(), Path(directory).resolve(), Path(source).resolve()
    if digest(directory/'summary.json') != summary_sha256:
        raise ValueError('suspension summary pin mismatch')
    summary = load_plan(directory/'summary.json')
    if Path(summary['source_directory']).resolve() != source:
        raise ValueError('suspension source binding mismatch')
    if summary.get('formal_H01_gate_passed') is not False or summary.get('historical_availability_proven') is not False:
        raise ValueError('suspension audit claims unsupported acceptance')
    pins = {str(directory/'summary.json'): summary_sha256,
            str(directory/'inputs.json'): summary['inputs_sha256'],
            str(source/'collection_plan.json'): summary['plan_sha256']}
    if summary['rows_sha256'] is not None:
        pins[str(directory/'rows.parquet')] = summary['rows_sha256']
    def check():
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise ValueError('suspension input hash mismatch')
    check()
    receipts = json.loads((directory/'inputs.json').read_text(encoding='utf-8'))
    for item in receipts:
        date = item['date']
        pins[str(source/(date+'.json'))] = item['receipt_sha256']
        pins[str(source/(date+'.parquet'))] = item['data_sha256']
        pins[str(root/'output/tushare_cache/daily'/(date+'.parquet'))] = item['reference_sha256']
    from . import suspension_daily_audit, suspension_daily_source
    for module in [suspension_daily_audit, suspension_daily_source]:
        path = Path(module.__file__).resolve(); pins[str(path)] = digest(path)
    pins[str(Path(__file__).resolve())] = digest(Path(__file__))
    check()
    rebuilt = audit(root, source, expected_plan_sha256=summary['plan_sha256'])
    for key in ['committed_dates','total_dates','uncommitted_dates','complete_receipt_coverage',
                'rows','type_counts','full_day_candidates','full_day_candidates_with_quotes']:
        if rebuilt[key] != summary[key]:
            raise ValueError('suspension reconstruction summary mismatch: '+key)
    replay = Path(rebuilt['directory'])
    if json.loads((replay/'inputs.json').read_text(encoding='utf-8')) != receipts:
        raise ValueError('suspension committed receipt set changed')
    if summary['rows_sha256'] is not None:
        pd.testing.assert_frame_equal(pd.read_parquet(directory/'rows.parquet'),
                                      pd.read_parquet(replay/'rows.parquet'), check_exact=True)
    check()
    return dict(full_receipt_reconstruction_verified=True, committed_dates=rebuilt['committed_dates'],
        complete_receipt_coverage=rebuilt['complete_receipt_coverage'], rows=rebuilt['rows'],
        full_day_candidates_with_quotes=rebuilt['full_day_candidates_with_quotes'],
        replay_directory=str(replay), replay_summary_sha256=digest(replay/'summary.json'),
        historical_availability_proven=False, tradability_semantics_accepted=False,
        evidence_files=[dict(path=p, sha256=h, role='fresh_suspension_receipt_reconstruction') for p,h in pins.items()])
