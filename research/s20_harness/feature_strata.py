"""Preregistered base-feature strata; missing/late controls retain their rows."""
import bisect
import math
import pandas as pd
from .label_availability import _instant
from .metrics import binary_bounds


def validate(controls,base_columns):
    if not isinstance(controls,list) or not 1<=len(controls)<=4:
        raise ValueError('one to four explicit control strata required')
    names=[]
    for control in controls:
        if not isinstance(control,dict) or set(control)!={'feature','cuts'} or control['feature'] not in base_columns:
            raise ValueError('stratum control must be a base feature')
        cuts=control['cuts'];names.append(control['feature'])
        if not isinstance(cuts,list) or not 1<=len(cuts)<=8 or any(type(v) not in [int,float] or not math.isfinite(v) for v in cuts) or cuts!=sorted(set(cuts)):
            raise ValueError('bounded sorted unique finite cutpoints required')
    if len(names)!=len(set(names)):raise ValueError('unique stratum control features required')


def summarize(plan,rows,controls):
    validate(controls,plan['feature_contract']['columns'])
    values=pd.DataFrame(plan['features']).set_index('sample_id')
    samples=pd.DataFrame(plan['samples']).set_index('sample_id')
    groups=[]
    for sid in rows.sample_id:
        sample=samples.loc[sid];available=sample.feature_available_at
        if pd.isna(available) or _instant(available)>=_instant(sample.prediction_at):
            groups.append('unavailable');continue
        parts=[]
        for c in controls:
            value=values.loc[sid,c['feature']]
            if pd.isna(value):parts.append(c['feature']+'=missing')
            elif isinstance(value,bool) or not math.isfinite(value):raise ValueError('finite numeric control required')
            else:parts.append(c['feature']+'='+str(bisect.bisect_right(c['cuts'],value)))
        groups.append('|'.join(parts))
    frame=rows.copy();frame['control_stratum']=groups;reports=[]
    for name,part in frame.groupby('control_stratum',sort=True):
        picked=part.loc[part.selected]
        events={}
        for event in ['safe_profit','down5']:
            events[event]=binary_bounds([None if pd.isna(v) else bool(v) for v in picked[event+'_target']])
        reports.append(dict(stratum=name,candidates=len(part),selected=len(picked),events=events,
            daily_selected=[dict(signal_date=d,count=int(picked.signal_date.eq(d).sum())) for d in plan['calendar']]))
    if sum(r['candidates'] for r in reports)!=len(rows):raise ValueError('stratum denominator mismatch')
    return dict(controls=controls,strata=reports,all_candidates_retained=True,
        observed_strata_only=True,cutpoints_fitted=False,bin_rule='left_closed_at_cutpoints',
        conditional_information_proven=False,formal_H06_accepted=False)
