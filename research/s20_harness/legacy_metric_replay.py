"""Replay saved v3 diagnostic rankings, not historical model inference or new labels."""
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .runtime import atomic_json,digest,load_plan,now


SCORES={'old_s20':'old_s20','old_r20_reference':'old_r20',
        'full_equal_risk':'portable_full','lowcorr_equal_risk':'lowcorr24',
        'full_half_risk':'half_risk'}


def replay(frame, k=20):
    required={'trade_date','ts_code','immediate','down_risk','old_s20','old_r20',
              'portable_full','portable_full_up','lowcorr24'}
    if not required.issubset(frame) or frame.empty:
        raise ValueError('complete nonempty legacy comparison required')
    if frame[['trade_date','ts_code']].isna().any().any() or frame.duplicated(['trade_date','ts_code']).any():
        raise ValueError('duplicate/missing legacy identities')
    if not frame[['immediate','down_risk']].isin([0,1]).all().all():
        raise ValueError('legacy binary outcomes must be complete')
    frame=frame.copy()
    score_cols=['old_s20','old_r20','portable_full','portable_full_up','lowcorr24']
    if not np.isfinite(frame[score_cols].to_numpy(dtype=float)).all():
        raise ValueError('legacy score missing/nonfinite; do not silently drop')
    # Preserve the old arithmetic order, including its floating-point tie behavior.
    down=1+frame.portable_full_up-frame.portable_full/50
    frame['half_risk']=100*(.5+frame.portable_full_up-.5*down)/1.5
    rows=[];selections=[]
    for candidate,column in SCORES.items():
        selected=frame.sort_values(['trade_date',column,'ts_code'],ascending=[True,False,True]).groupby('trade_date',sort=False).head(k)
        rows.append(dict(candidate=candidate,rows=len(selected),dates=int(selected.trade_date.nunique()),
            immediate_count=int(selected.immediate.sum()),down_count=int(selected.down_risk.sum()),
            immediate_rate=float(selected.immediate.mean()),down_rate=float(selected.down_risk.mean())))
        ledger=selected[['trade_date','ts_code','immediate','down_risk']].copy()
        ledger['candidate']=candidate;ledger['score']=selected[column].to_numpy();selections.append(ledger)
    return rows,pd.concat(selections,ignore_index=True)


def build(root):
    root=Path(root).resolve()
    source=root/'output/experiments/s20_v3/comparison_diagnostic2026.parquet'
    reference=root/'config/s20_v3_results.json'
    code=Path(__file__).resolve()
    pins={str(p):digest(p) for p in [source,reference,code,
        root/'scripts/train_s20_v3.py',root/'scripts/evaluate_s20_v3_tradeoff.py']}
    frame=pd.read_parquet(source);expected=load_plan(reference)['diagnostic_top20']
    metrics,selected=replay(frame)
    comparisons=[]
    for row in metrics:
        for name in ['immediate_rate','down_rate']:
            comparisons.append(dict(candidate=row['candidate'],metric=name,actual=row[name],
                expected=expected[row['candidate']][name],
                matches=abs(row[name]-expected[row['candidate']][name])<=1e-12))
    coverage_match=(len(frame)==expected['rows'] and frame.trade_date.nunique()==expected['dates']
        and all(r['rows']==expected['selected_rows'] for r in metrics)
        and [str(frame.trade_date.min()),str(frame.trade_date.max())]==expected['window'])
    if any(digest(Path(p))!=h for p,h in pins.items()):
        raise ValueError('legacy replay source changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('legacy-metric-replay-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    selected.to_parquet(out/'selected.parquet',index=False)
    atomic_json(out/'metrics.json',dict(metrics=metrics,comparisons=comparisons))
    atomic_json(out/'inputs.json',pins)
    result=dict(directory=str(out),at=now(),source_rows=len(frame),coverage_matches=bool(coverage_match),
        saved_metric_reproduction_passed=bool(coverage_match and all(c['matches'] for c in comparisons)),
        metric_comparisons=len(comparisons),model_fits=0,model_predictions_recomputed=False,
        raw_price_labels_recomputed=False,original_legacy_label_reproduction_complete=False,
        frozen_v4_label_alignment_complete=False,historical_availability_proven=False,
        formal_H03_accepted=False,independent_confirmation=False,
        artifacts={n:digest(out/n) for n in ['selected.parquet','metrics.json','inputs.json']})
    atomic_json(out/'summary.json',result)
    return result
