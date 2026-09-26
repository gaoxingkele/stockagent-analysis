"""Audit saved old-S20 composition and exact-run model persistence gaps."""
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json,digest,now


def compose(frame):
    keys=['trade_date','ts_code']
    columns=['stage1_probability','lambdarank_score','s20_20r_rank']
    if not set(keys+columns).issubset(frame) or frame.empty:
        raise ValueError('nonempty saved old-S20 components required')
    if frame[keys+columns].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('missing/duplicate old-S20 component rows')
    result=frame[keys].copy()
    result['recomputed_rank']=sum(.5*frame.groupby('trade_date',sort=False)[c].rank(
        pct=True,method='average') for c in columns[:2])
    result['saved_rank']=frame.s20_20r_rank
    result['absolute_error']=(result.recomputed_rank-result.saved_rank).abs()
    return result


def build(root):
    root=Path(root).resolve()
    directory=root/'output/experiments/s20_20r_confirmation/run_v1'
    source=directory/'predictions.parquet'
    script=root/'scripts/confirm_s20_20r_portable.py'
    paths=[p for p in directory.iterdir() if p.is_file()]+[script,Path(__file__)]
    pins={str(p.resolve()):digest(p) for p in paths}
    table=compose(pd.read_parquet(source))
    # These are exact-run provenance requirements, not a claim that no other
    # project directory contains similarly named (possibly unrelated) models.
    gaps=[dict(component='final_anchor',required='exact fitted final anchor model and feature schema',
               evidence='confirmation script fits/refits and consumes anchor but does not serialize it'),
          dict(component='R20_reference',required='ordinal model, binary models, isotonic and Platt states',
               evidence='confirmation script fits these objects in memory without saving their states'),
          dict(component='rank_feature_selection',required='anchor-derived selected features or a bound reconstructable schema',
               evidence='selection computed in memory; ranker files alone do not supply missing anchor predictions')]
    if any(digest(Path(p))!=h for p,h in pins.items()):
        raise ValueError('old-S20 audit source changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('legacy-anchor-audit-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    table.to_parquet(out/'rank_comparison.parquet',index=False)
    atomic_json(out/'inputs.json',pins)
    report=dict(directory=str(out),at=now(),rows=len(table),
        rank_composition_matches=bool(table.absolute_error.le(1e-12).all()),
        max_absolute_error=float(table.absolute_error.max()),model_fits=0,
        component_predictions_recomputed=False,full_legacy_inference_reproduced=False,
        persistence_gaps=gaps,formal_H03_accepted=False,
        next_action='recover exact original states with provenance, or register a new reconstruction trial; never call a refit the original model',
        artifacts={n:digest(out/n) for n in ['rank_comparison.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
