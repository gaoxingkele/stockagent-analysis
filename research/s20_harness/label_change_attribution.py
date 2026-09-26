"""Attribute saved label transitions without equating them with model gains."""
import json
from pathlib import Path
import uuid
import pandas as pd
from .runtime import atomic_json,digest,load_plan,now


def attribute(frame):
    frame=frame.copy()
    for column in ['compat_class','raw_calendar_class','o_class']:
        frame[column]=frame[column].fillna(-99).astype(int)
    rows=[]
    for r in frame.itertuples(index=False):
        raw=json.loads(r.raw_calendar_payload_json);economic=json.loads(r.o_payload_json)
        raw_class=raw.get('o_class');economic_class=economic.get('o_class')
        if (raw_class if raw_class is not None else -99)!=r.raw_calendar_class or (economic_class if economic_class is not None else -99)!=r.o_class:
            raise ValueError('label column/payload mismatch')
        calendar_known=r.compat_class>=0 and r.raw_calendar_class>=0
        economic_known=r.raw_calendar_class>=0 and r.o_class>=0
        events=economic.get('economic_ledger') or []
        changed=economic_known and r.raw_calendar_class!=r.o_class
        ids=sorted({e['event_id'] for e in events if e.get('event_id')})
        rows.append(dict(sample_id=r.sample_id,signal_date=r.signal_date,trading_code=r.trading_code,
            calendar_comparable=calendar_known,calendar_class_changed=calendar_known and r.compat_class!=r.raw_calendar_class,
            economic_comparable=economic_known,economic_class_changed=changed,
            raw_class=r.raw_calendar_class,economic_class=r.o_class,
            event_ids_json=json.dumps(ids),event_record_count=len(events),
            economic_change_has_recorded_events=bool(changed and ids),
            same_entry_and_horizon=raw.get('entry_date')==economic.get('entry_date') and raw.get('horizon_end')==economic.get('horizon_end'),
            raw_max_gain20=raw.get('max_gain20'),economic_max_gain20=economic.get('max_gain20'),
            raw_window_mae20=raw.get('window_mae20'),economic_window_mae20=economic.get('window_mae20')))
    result=pd.DataFrame(rows)
    changed=result.loc[result.economic_class_changed]
    return result,dict(rows=len(result),calendar_comparable=int(result.calendar_comparable.sum()),
        calendar_class_changes=int(result.calendar_class_changed.sum()),
        economic_comparable=int(result.economic_comparable.sum()),economic_class_changes=len(changed),
        changed_without_recorded_events=int((~changed.economic_change_has_recorded_events).sum()),
        changed_with_different_window=int((~changed.same_entry_and_horizon).sum()),
        transitions=[dict(raw_class=int(a),economic_class=int(b),rows=int(n)) for (a,b),n in
            changed.groupby(['raw_class','economic_class']).size().items()],
        model_performance_gain_proven=False,source_event_semantics_independently_verified=False,
        formal_H02_H03_accepted=False)


def build(root,preparation,summary_sha):
    preparation=Path(preparation).resolve()
    if digest(preparation/'summary.json')!=summary_sha: raise ValueError('preparation pin mismatch')
    summary=load_plan(preparation/'summary.json')
    path=preparation/'labels.parquet';expected=summary['artifacts']['labels.parquet']
    if digest(path)!=expected: raise ValueError('preparation labels changed')
    pins={str(preparation/'summary.json'):summary_sha,str(path):expected,str(Path(__file__)):digest(Path(__file__))}
    rows,report=attribute(pd.read_parquet(path))
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('attribution source changed')
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('label-change-attribution-'+uuid.uuid4().hex)
    out.mkdir(parents=True);rows.to_parquet(out/'rows.parquet',index=False)
    atomic_json(out/'inputs.json',pins)
    report.update(directory=str(out),at=now(),artifacts={n:digest(out/n) for n in ['rows.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
