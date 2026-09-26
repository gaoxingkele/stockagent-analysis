"""Pinned saved-prediction policy ablation; diagnostic, not budget admission."""
from pathlib import Path
import uuid
import pandas as pd
from .joint_replay import replay
from .joint_policy_comparison import compare
from .runtime import atomic_json, digest, load_plan
from .trial_budget import canonical
from .policy_coverage_export import csv_text as coverage_csv


def evidence_texts(run_directory,run_sha,input_sha,comparison):
    """Static source-state snapshot, not a fabricated online calibration log."""
    from .reliability_export import table
    source=Path(run_directory).resolve();plan=load_plan(source/'input.json')
    card=load_plan(source/'calibration_card.json')
    probability_source=comparison['probability_source']
    raw=probability_source=='raw'
    state=dict(run_directory=str(source),run_sha256=run_sha,
        source_calibration_card_sha256=digest(source/'calibration_card.json'),source_calibration_state=card,
        probability_source=probability_source,effective_method='identity_raw' if raw else card['method'],
        source_calibration_applied=not raw and card['method']!='identity_raw',
        additional_calibrator_fits=0,online_update_history=False,formal_H05_accepted=False)
    boundary=plan['boundaries'][3]
    jobs=[dict(candidate_id=run_sha,job_id=t['policy_sha256'],fold_id=boundary['start_at']+'/'+boundary['end_at'],
        random_seed=plan.get('random_seed',20),input_sha256=input_sha,metrics=t['metrics']) for t in comparison['trials']]
    reliability=table(jobs)
    policy_ids={t['policy_sha256']:t['policy_id'] for t in comparison['trials']}
    reliability['policy_id']=reliability.job_id.map(policy_ids)
    reliability['probability_source']=probability_source
    return {'calibration_states.jsonl':canonical(state)+'\n',
            'policy_candidates.json':canonical(dict(registry=comparison['registry'],
                probability_source=probability_source,policy_evaluations=comparison['policy_evaluations'],
                selected_policy_id=None,formal_H05_accepted=False))+'\n',
            'selected_reliability.csv':reliability.to_csv(index=False,lineterminator='\n')}


def reconstruct(run_directory, run_sha, input_path, input_sha):
    directory=Path(run_directory).resolve();source=Path(input_path).resolve()
    if digest(source)!=input_sha:raise ValueError('policy comparison input pin mismatch')
    replay(directory,run_sha)
    payload=load_plan(source);plan=load_plan(directory/'input.json')
    required={'registry','selection_calendar','selection_outcomes'}|({'probability_source'} if 'probability_source' in payload else set())
    if set(payload)!=required:
        raise ValueError('exact selection-only policy payload required')
    probability_source=payload.get('probability_source','calibrated')
    if probability_source not in ['raw','calibrated']:raise ValueError('explicit raw or calibrated probability source required')
    if plan['evidence_mode']!='synthetic':
        raise ValueError('policy comparison currently synthetic-only pending formal admission')
    records=payload['selection_outcomes']
    if not isinstance(records,list) or any(not isinstance(r,dict) or set(r)!={'sample_id','target','label_available_at'} for r in records):
        raise ValueError('exact selection outcome records required')
    samples=pd.DataFrame(plan['samples'])
    predictions=pd.read_parquet(directory/'calibrated_predictions.parquet')
    selected=predictions.loc[predictions.segment.eq('selection-policy')]
    candidates=samples.set_index('sample_id').loc[selected.sample_id,['entity_id','signal_date','prediction_at']].reset_index()
    for name in ['p_A','p_B','p_C','p_D']:
        candidates[name]=selected[('cal_' if probability_source=='calibrated' else '')+name].to_numpy()
    ledgers,report=compare(samples,candidates,pd.DataFrame(records,columns=['sample_id','target','label_available_at']),
                           plan['boundaries'],payload['selection_calendar'],payload['registry'])
    from .fixed_selection_calibration import assess
    report['probability_source']=probability_source
    report['fixed_selection_calibration']=assess(samples,predictions,
        pd.DataFrame(records,columns=['sample_id','target','label_available_at']),ledgers,report,payload['selection_calendar'],
        selection_source=probability_source)
    replay(directory,run_sha)
    if digest(source)!=input_sha:raise ValueError('policy comparison input changed')
    return ledgers,report


def build(root,run_directory,run_sha,input_path,input_sha):
    ledgers,comparison=reconstruct(run_directory,run_sha,input_path,input_sha)
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('joint-policy-comparison-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'comparison.json',comparison)
    (out/'risk_coverage.csv').write_text(coverage_csv(ledgers,comparison),encoding='utf-8',newline='')
    names=['comparison.json','risk_coverage.csv']
    for name,content in evidence_texts(run_directory,run_sha,input_sha,comparison).items():
        (out/name).write_text(content,encoding='utf-8',newline='');names.append(name)
    for i,trial in enumerate(comparison['trials']):
        name=f'policy_{i:02d}.parquet';names.append(name)
        ledgers[trial['policy_id']].to_parquet(out/name,index=False)
    report=dict(directory=str(out),run_directory=str(Path(run_directory).resolve()),run_sha256=run_sha,
        input_path=str(Path(input_path).resolve()),input_sha256=input_sha,
        code_sha256=digest(Path(__file__)),policy_evaluations=comparison['policy_evaluations'],
        model_fits=0,calibrator_fits=0,shared_budget_enforced=False,formal_H05_accepted=False,
        artifacts={n:digest(out/n) for n in names})
    atomic_json(out/'summary.json',report)
    verify(root,out,digest(out/'summary.json'))
    return report


def verify(root,directory,sha):
    directory=Path(directory).resolve();scope=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'
    if not directory.is_relative_to(scope) or digest(directory/'summary.json')!=sha:
        raise ValueError('joint policy summary pin/scope mismatch')
    report=load_plan(directory/'summary.json')
    if set(report)!={'directory','run_directory','run_sha256','input_path','input_sha256','code_sha256',
                      'policy_evaluations','model_fits','calibrator_fits','shared_budget_enforced','formal_H05_accepted','artifacts'}:
        raise ValueError('exact joint policy summary required')
    count=report['policy_evaluations']
    if type(count)is not int or not 2<=count<=6 or report['directory']!=str(directory):
        raise ValueError('joint policy count/directory mismatch')
    expected={'comparison.json','risk_coverage.csv','calibration_states.jsonl','policy_candidates.json','selected_reliability.csv',
              *(f'policy_{i:02d}.parquet' for i in range(count))}
    if set(report['artifacts'])!=expected:raise ValueError('joint policy artifact scope mismatch')
    if any(type(report[k])is not int or report[k]!=0 for k in ['model_fits','calibrator_fits']) or any(
            report[k] is not False for k in ['shared_budget_enforced','formal_H05_accepted']):
        raise ValueError('unsupported joint policy claim')
    def check():
        if digest(directory/'summary.json')!=sha or digest(Path(__file__))!=report['code_sha256'] or any(digest(directory/n)!=h for n,h in report['artifacts'].items()):
            raise ValueError('joint policy source/artifact changed')
    check()
    ledgers,comparison=reconstruct(report['run_directory'],report['run_sha256'],report['input_path'],report['input_sha256'])
    if canonical(comparison)!=canonical(load_plan(directory/'comparison.json')) or comparison['policy_evaluations']!=count:
        raise ValueError('joint policy comparison reconstruction mismatch')
    for i,trial in enumerate(comparison['trials']):
        pd.testing.assert_frame_equal(ledgers[trial['policy_id']],pd.read_parquet(directory/f'policy_{i:02d}.parquet'),check_exact=True)
    if (directory/'risk_coverage.csv').read_bytes()!=coverage_csv(ledgers,comparison).encode('utf-8'):
        raise ValueError('policy risk coverage reconstruction mismatch')
    for name,content in evidence_texts(report['run_directory'],report['run_sha256'],report['input_sha256'],comparison).items():
        if (directory/name).read_bytes()!=content.encode('utf-8'):
            raise ValueError('policy evidence reconstruction mismatch: '+name)
    check()
    return dict(policy_comparison_recomputed=True,model_fits=0,calibrator_fits=0,
                historical_registry_receipt_proven=False,shared_budget_enforced=False,formal_H05_accepted=False)
