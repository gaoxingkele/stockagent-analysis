"""Scalar reference for the frozen O-label event rules, independent of vector code."""
import math


def classify(entry,highs,lows):
    entry=float(entry);highs=list(highs);lows=list(lows)
    if (not math.isfinite(entry) or entry<=0 or not highs or len(highs)!=len(lows)
        or any(not math.isfinite(h) or not math.isfinite(l) or l>h for h,l in zip(highs,lows))):
        raise ValueError('invalid reference path')
    hit20=stop10=0;touch15=False
    for day,(high,low) in enumerate(zip(highs,lows),1):
        if high>=entry*1.15: touch15=True
        if not hit20 and high>=entry*1.20: hit20=day
        if not stop10 and low<entry*.90: stop10=day
    if hit20:
        if hit20==stop10: cls=-1
        elif stop10 and stop10<hit20: cls=2
        else: cls=0
    elif touch15:
        cls=5 if stop10 else 1
    else:
        cls=4 if stop10 else 3
    names={-1:'ambiguous_same_day',0:'immediate20',1:'immediate15',2:'delayed20',
        3:'negative_flat',4:'negative_down',5:'negative_unsafe15'}
    return dict(s20_class=cls,reason=names[cls],hit20_day=hit20,stop10_day=stop10,
        max_gain20=(max(highs)/entry-1)*100,window_mae20=(min(lows)/entry-1)*100,
        immediate=-1 if cls<0 else int(cls in (0,1)),opportunity=-1 if cls<0 else int(cls in (0,1,2)),
        down_risk=-1 if cls<0 else int(cls in (2,4,5)),negative=-1 if cls<0 else int(cls in (3,4,5)))
