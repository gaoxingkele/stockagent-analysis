"""Rebuild diagnostic labels from pinned market inputs using current code.

This checks numerical reproducibility, not independent algorithm correctness,
historical source availability, or completeness of corporate-action coverage.
"""
from pathlib import Path
import uuid

import pandas as pd

from .label_partition import build
from .label_partition_verify import verify
from .runtime import atomic_json, digest, load_plan, now


def compare(original, rebuilt):
    pd.testing.assert_frame_equal(original,rebuilt,check_exact=True)
    return dict(rows=len(original),all_columns_equal=True,
        label_path_recomputed=True,independent_algorithm_validation=False,
        formal_training_authorized=False)


def replay(root, directory, summary_sha):
    root,directory=Path(root).resolve(),Path(directory).resolve()
    original,validation=verify(directory,summary_sha)
    summary=load_plan(directory/'summary.json')
    sources=load_plan(directory/'inputs.json')
    paths=[Path(p) for p in sources if Path(p).name=='normalized_distributions.parquet']
    if len(paths)!=1: raise ValueError('unique pinned normalized distribution input required')
    normalized=paths[0]
    review=Path(summary['review_path'])
    if sources.get(str(review))!=summary['review_sha256']:
        raise ValueError('review not bound to original sources')
    # The current builder pins current code while original code remains recoverable.
    rebuilt_report=build(root,summary['signal_date'],normalized,sources[str(normalized)],
        tax_bounds=summary['tax_bounds_requested'],review_path=review,
        review_sha256=summary['review_sha256'])
    rebuilt_dir=Path(rebuilt_report['directory'])
    rebuilt_sha=digest(rebuilt_dir/'summary.json')
    rebuilt,_=verify(rebuilt_dir,rebuilt_sha)
    rebuilt_sources=load_plan(rebuilt_dir/'inputs.json')
    data_pins=lambda pins:{p:h for p,h in pins.items() if Path(p).suffix!='.py'}
    if data_pins(sources)!=data_pins(rebuilt_sources):
        raise ValueError('replay changed non-code input scope or values')
    result=compare(original,rebuilt)
    verify(directory,summary_sha)
    verify(rebuilt_dir,rebuilt_sha)
    out=root/'output/experiments/s20_safe_v4/sources'/('label-replay-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    report=dict(directory=str(out),at=now(),original_directory=str(directory),
        original_summary_sha256=summary_sha,rebuilt_directory=str(rebuilt_dir),
        rebuilt_summary_sha256=rebuilt_sha,original_validation=validation,
        data_source_pins_equal=True,current_code_used=True,
        replay_code_sha256=digest(Path(__file__)),**result)
    atomic_json(out/'summary.json',report)
    return report
