"""Re-run saved v3 models with the legacy loader; no training or PIT approval."""
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd

from .runtime import atomic_json,digest,load_plan,now


def build(root):
    root=Path(root).resolve()
    sys.path.insert(0,str(root/'scripts'))
    from confirm_s20_20r_portable import _load_confirmation
    from score_s20_v3 import score_frame
    base=root/'output/experiments/s20_v3'
    factors=root/'output/experiments/s20_20r_confirmation/factor_groups'
    label_path=root/'output/experiments/s20_v2_labels/labels.parquet'
    source=base/'comparison_diagnostic2026.parquet'
    modes=['portable_full','lowcorr24']
    bundles={m:base/'models/diagnostic2026'/m for m in modes}
    paths=[source,label_path,Path(__file__),root/'output/regimes/daily_regime.parquet',
        root/'output/regime_extra/regime_extra.parquet',root/'output/lgbm_maxgain/feature_meta.json',
        root/'output/tushare_cache/stock_basic.parquet',root/'scripts/confirm_s20_20r_portable.py',
        root/'scripts/score_s20_v3.py',root/'src/stockagent_analysis/s20_v3.py',
        *sorted(factors.glob('group_*.parquet')),
        *[b/n for b in bundles.values() for n in ['schema.json','model.txt']]]
    pins={str(p.resolve()):digest(p) for p in paths}
    saved=pd.read_parquet(source)
    features=load_plan(bundles['portable_full']/'schema.json')['features']
    loaded=_load_confirmation(SimpleNamespace(labels=label_path,factor_dir=factors),features)
    keys=['ts_code','trade_date']
    # The old loader uses present-day ST metadata and legacy label filtering.
    # Preserve that behavior only for reproduction, and disclose missing rows.
    frame=saved[keys].merge(loaded[keys+features],on=keys,how='left',validate='one_to_one',indicator=True)
    absent=frame['_merge'].ne('both')
    audit=frame[keys].copy();audit['legacy_loader_row_present']=~absent
    results=[]
    for mode,bundle in bundles.items():
        pred=score_frame(frame.loc[~absent].drop(columns='_merge'),bundle)
        joined=saved[keys+[mode,mode+'_up']].merge(pred[keys+['score','p_immediate']],
            on=keys,how='left',validate='one_to_one')
        valid=joined.score.notna()
        ds=(joined.score-joined[mode]).abs();dp=(joined.p_immediate-joined[mode+'_up']).abs()
        audit[mode+'_score_abs_error']=ds;audit[mode+'_probability_abs_error']=dp
        results.append(dict(mode=mode,expected_rows=len(saved),recomputed_rows=int(valid.sum()),
            missing_rows=int((~valid).sum()),max_score_abs_error=float(ds.max()) if valid.any() else None,
            max_probability_abs_error=float(dp.max()) if valid.any() else None,
            mismatched_rows=int((valid & ((ds>1e-10)|(dp>1e-12))).sum()),
            passed=bool(valid.all() and (ds<=1e-10).all() and (dp<=1e-12).all())))
    if any(digest(Path(p))!=h for p,h in pins.items()):
        raise ValueError('legacy inference source changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('legacy-inference-replay-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    audit.to_parquet(out/'row_comparison.parquet',index=False)
    atomic_json(out/'inputs.json',pins)
    report=dict(directory=str(out),at=now(),results=results,model_fits=0,
        saved_v3_inference_reproduced=all(r['passed'] for r in results),
        old_s20_r20_inference_reproduced=False,legacy_loader_current_metadata_used=True,
        raw_price_labels_recomputed=False,historical_availability_proven=False,
        formal_H03_accepted=False,independent_confirmation=False,
        artifacts={n:digest(out/n) for n in ['row_comparison.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
