"""Forward joint candidates with recorded fit/calibration provenance."""
import json
from pathlib import Path
import pandas as pd
from .joint_replay import replay
from .joint_run import verify
from .oof_audit import audit, membership_hash
from .label_availability import _instant
from .runtime import digest, load_plan


def collect_one(job):
    directory=Path(job['directory']).resolve()
    replay(directory,job['summary_sha256'])
    plan=load_plan(directory/'input.json');base=load_plan(directory/'joint_card.json')
    cal=load_plan(directory/'calibration_card.json')
    predictions=pd.read_parquet(directory/'calibrated_predictions.parquet')
    outer=predictions.loc[predictions.segment.eq('outer-test')]
    ledger=pd.read_parquet(directory/'candidate_ledger.parquet')
    if outer.sample_id.tolist()!=ledger.sample_id.tolist():
        raise ValueError('joint outer candidate denominator/order mismatch')
    dependencies=base['fit_sample_ids']+cal['calibration_sample_ids']
    cutoff=max(_instant(plan['boundaries'][3]['start_at']),
               _instant(plan['policy']['selection']['frozen_at'])).isoformat()
    model_id=job['job_id']
    models={model_id:dict(model_path=str(directory/'calibration_card.json'),
        model_sha256=digest(directory/'calibration_card.json'),information_cutoff_at=cutoff,
        dependency_sha256=membership_hash(dependencies))}
    checked,_=audit(pd.DataFrame(plan['samples']),
        pd.DataFrame(dict(sample_id=outer.sample_id.tolist(),model_id=model_id,score=outer.cal_p_A.tolist())),
        models,{model_id:dependencies})
    table=ledger.copy()
    table['job_id']=model_id;table['fold_id']=job['fold_id']
    table['model_family']=base['model'];table['target_id']=plan['target_id']
    for c in ['A','B','C','D']:table['raw_p_'+c]=outer['p_'+c].to_numpy()
    table['recorded_oof_provenance_valid']=checked.recorded_oof_provenance_valid.to_numpy()
    table['provenance_reasons_json']=checked.reasons.map(json.dumps).to_numpy()
    table['information_cutoff_at']=cutoff;table['source_summary_sha256']=job['summary_sha256']
    table['historical_availability_proven']=False
    verify(directory,job['summary_sha256'])
    return table
