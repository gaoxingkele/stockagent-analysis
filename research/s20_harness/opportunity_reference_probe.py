"""Finite exhaustive short-event permutations padded to20days; not market validation."""
from itertools import product
from pathlib import Path
import uuid

import numpy as np

from .opportunity_reference import classify
from .labels import path_labels
from .runtime import atomic_json,digest,now


def evaluate():
    contract=dict(entry_values=[10.,100.],high_ratios=[1.10,1.1499,1.15,1.1999,1.20,1.25],
        low_ratios=[.85,.90,.95],varying_days=3,horizon=20,padding_high_ratio=1.10,padding_low_ratio=.95,
        first_touch_day_grid=list(range(21)),zero_day_means_no_touch=True)
    entries=[];highs=[];lows=[];expected=[]
    levels=list(product(contract['high_ratios'],contract['low_ratios']))
    for entry in contract['entry_values']:
        for prefix in product(levels,repeat=3):
            h=[entry*p[0] for p in prefix]+[entry*1.10]*17
            l=[entry*p[1] for p in prefix]+[entry*.95]*17
            entries.append(entry);highs.append(h);lows.append(l);expected.append(classify(entry,h,l))
        for up_day,down_day in product(range(21),repeat=2):
            h=[entry*1.10]*20;l=[entry*.95]*20
            if up_day: h[up_day-1]=entry*1.20
            if down_day: l[down_day-1]=entry*.85
            entries.append(entry);highs.append(h);lows.append(l);expected.append(classify(entry,h,l))
    actual=path_labels(entries,np.array(highs),np.array(lows))
    mismatches=[]
    for i,(want,got) in enumerate(zip(expected,actual.to_dict('records'))):
        if want!=got: mismatches.append(dict(case_index=i,expected=want,actual=got))
    return contract,dict(cases=len(expected),mismatches=mismatches,all_equal=not mismatches,
        class_counts=actual.s20_class.value_counts().sort_index().to_dict(),
        scope='finite first-three-day permutations and all first-touch day pairs; not exhaustive20day paths or market evidence',
        formal_training_authorized=False)


def build(root):
    from . import opportunity_reference
    root=Path(root).resolve()
    paths=[Path(__file__).resolve(),Path(opportunity_reference.__file__).resolve(),root/'src/stockagent_analysis/s20_v3.py']
    pins={str(p):digest(p) for p in paths}
    contract,result=evaluate()
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('reference probe code changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('opportunity-reference-'+uuid.uuid4().hex);out.mkdir(parents=True)
    atomic_json(out/'probe_contract.json',contract);atomic_json(out/'comparison.json',result)
    atomic_json(out/'inputs.json',dict(code_pins=pins))
    report=dict(directory=str(out),at=now(),cases=result['cases'],mismatches=len(result['mismatches']),
        all_equal=result['all_equal'],formal_training_authorized=False,
        artifacts={n:digest(out/n) for n in ['probe_contract.json','comparison.json','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
