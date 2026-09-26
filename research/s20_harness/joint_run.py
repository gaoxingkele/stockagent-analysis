"""Pinned diagnostic joint training/calibration/selection artifact chain."""
from pathlib import Path
import json
import uuid
import pandas as pd
from .joint_model import run as train
from .baseline_model import validate_seed
from .joint_calibration import run as calibrate
from .joint_policy import apply as select
from .runtime import atomic_json,digest,now

OUTPUTS={'input.json','inputs.json','joint_predictions.parquet','joint_card.json',
    'calibrated_predictions.parquet','calibration_card.json','candidate_ledger.parquet','selection_report.json'}


def pipeline_costs(plan):
    validate_seed(plan.get('random_seed',20))
    from .joint_policy import validate as validate_policy
    if 'policy' in plan:validate_policy(plan['policy'])
    family=plan.get('model_family','multinomial')
    if family not in ['multinomial','conditional_three','cost_sensitive_joint','mature_frequency','shallow_joint']:
        raise ValueError('unsupported joint model family')
    if family=='cost_sensitive_joint':
        from .joint_model import validate_weights
        validate_weights(plan.get('class_weights'))
    elif 'class_weights' in plan:raise ValueError('class weights require cost-sensitive family')
    method=plan.get('calibration_method','bounded_scalar_temperature')
    if method not in ['bounded_scalar_temperature','identity_raw']:raise ValueError('unsupported joint calibration method')
    if method=='identity_raw' and 'calibration_labels' in plan and plan['calibration_labels']!=[]:
        raise ValueError('identity calibration requires explicit empty labels')
    cal_fits=int(method!='identity_raw')
    return dict(model_fits=0 if family=='mature_frequency' else 1,
                underlying_fits=(0 if family=='mature_frequency' else 3 if family=='conditional_three' else 1)+cal_fits,
                calibrator_fits=cal_fits,policy_evaluations=1)


def build(root,input_path,input_sha):
    input_path=Path(input_path).resolve()
    if digest(input_path)!=input_sha: raise ValueError('joint input pin mismatch')
    raw=input_path.read_bytes();plan=json.loads(raw)
    required={'schema_version','evidence_mode','target_id','samples','features','fit_labels',
        'calibration_labels','boundaries','feature_contract','policy','calendar'}
    if 'model_family' in plan:required.add('model_family')
    if 'random_seed' in plan:required.add('random_seed')
    if 'calibration_method' in plan:required.add('calibration_method')
    if plan.get('model_family')=='cost_sensitive_joint':required.add('class_weights')
    if set(plan)!=required or plan['schema_version']!='joint-1' or plan['target_id']!='P.joint.v4' or plan['evidence_mode'] not in ['synthetic','supplied_reference']:
        raise ValueError('exact joint diagnostic plan required')
    costs=pipeline_costs(plan)
    if plan.get('model_family')=='conditional_three':
        from .conditional_joint_model import run as trainer
    elif plan.get('model_family')=='mature_frequency':
        from .mature_frequency import run as trainer
    else:trainer=train
    if not isinstance(plan['samples'],list) or not 1<=len(plan['samples'])<=100000:
        raise ValueError('bounded joint sample universe required')
    pins={str(input_path):input_sha,**{str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}}
    def check():
        if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('joint source changed')
    check()
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('joint-run-'+uuid.uuid4().hex)
    out.mkdir(parents=True);artifacts={};steps=[]
    def save(name,value):
        if isinstance(value,pd.DataFrame): value.to_parquet(out/name,index=False)
        else: atomic_json(out/name,value)
        artifacts[name]=digest(out/name)
    def checkpoint(status,error=None):
        atomic_json(out/'checkpoint.json',dict(status=status,completed_steps=list(steps),artifacts=dict(artifacts),
            error=error,formal_H04_H05_accepted=False))
    checkpoint('RUNNING')
    try:
        (out/'input.json').write_bytes(raw);artifacts['input.json']=input_sha
        save('inputs.json',pins)
        samples=pd.DataFrame(plan['samples'])
        trainer_args={'class_weights':plan['class_weights']} if plan.get('model_family')=='cost_sensitive_joint' else {}
        trainer_args['random_seed']=plan.get('random_seed',20)
        if plan.get('model_family')=='shallow_joint':trainer_args['model_family']='shallow_joint'
        predictions,card=trainer(samples,pd.DataFrame(plan['features']),pd.DataFrame(plan['fit_labels']),
            plan['boundaries'],plan['feature_contract'],target_id=plan['target_id'],**trainer_args)
        check();save('joint_predictions.parquet',predictions);save('joint_card.json',card)
        steps.append('joint_model');checkpoint('RUNNING')
        calibrated,calcard=calibrate(samples,predictions,pd.DataFrame(plan['calibration_labels'],columns=['sample_id','target'])
            if not plan['calibration_labels'] else pd.DataFrame(plan['calibration_labels']),
            plan['boundaries'],card,target_id=plan['target_id'],method=plan.get('calibration_method','bounded_scalar_temperature'))
        check();save('calibrated_predictions.parquet',calibrated);save('calibration_card.json',calcard)
        steps.append('calibration');checkpoint('RUNNING')
        outer=calibrated.loc[calibrated.segment.eq('outer-test')]
        candidates=samples.set_index('sample_id').loc[outer.sample_id,['entity_id','signal_date','prediction_at']].reset_index()
        for c in ['A','B','C','D']: candidates['p_'+c]=outer['cal_p_'+c].to_numpy()
        rows,policy=select(candidates,plan['policy'],plan['calendar'])
        check();save('candidate_ledger.parquet',rows);save('selection_report.json',policy)
        steps.append('selection');checkpoint('COMPLETED_DIAGNOSTIC')
        check()
        if any(digest(out/n)!=h for n,h in artifacts.items()): raise ValueError('joint artifact changed')
    except Exception as exc:
        checkpoint('FAILED',dict(type=type(exc).__name__,message=str(exc)));raise
    result=dict(directory=str(out),at=now(),status='COMPLETED_DIAGNOSTIC',evidence_mode=plan['evidence_mode'],
        model_level_fits=costs['model_fits'],underlying_fits=costs['underlying_fits'],calibrator_fits=costs['calibrator_fits'],policy_evaluations=1,
        outer_candidates=len(rows),selected=int(rows.selected.sum()),formal_H04_H05_accepted=False,
        absolute_probability_validated=False,production_eligible=False,
        artifacts={**artifacts,'checkpoint.json':digest(out/'checkpoint.json')})
    atomic_json(out/'summary.json',result)
    return result


def verify(directory,summary_sha):
    directory=Path(directory).resolve()
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('joint summary pin mismatch')
    report=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if set(report['artifacts'])!=OUTPUTS|{'checkpoint.json'} or report['status']!='COMPLETED_DIAGNOSTIC':
        raise ValueError('joint completed artifact set mismatch')
    for name,sha in report['artifacts'].items():
        if digest(directory/name)!=sha: raise ValueError('joint artifact pin mismatch')
    cp=json.loads((directory/'checkpoint.json').read_text(encoding='utf-8'))
    if cp['status']!='COMPLETED_DIAGNOSTIC' or cp['completed_steps']!=['joint_model','calibration','selection'] or cp['artifacts']!={n:h for n,h in report['artifacts'].items() if n!='checkpoint.json'}:
        raise ValueError('joint checkpoint mismatch')
    costs=pipeline_costs(json.loads((directory/'input.json').read_text(encoding='utf-8')))
    if report['model_level_fits']!=costs['model_fits'] or any(report[k]!=costs[k] for k in ['underlying_fits','calibrator_fits','policy_evaluations']):
        raise ValueError('joint reported fit costs mismatch')
    return dict(artifact_bytes_verified=True,semantic_replay_performed=False,formal_H04_H05_accepted=False)
