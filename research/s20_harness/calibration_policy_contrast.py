"""Read-only same-model, same-policy raw/calibrated outcome contrast."""
from pathlib import Path
from .joint_policy_run import verify
from .runtime import load_plan
from .trial_budget import canonical


def compare_reports(raw,calibrated):
    if raw['probability_source']!='raw' or calibrated['probability_source']!='calibrated':
        raise ValueError('ordered raw and calibrated reports required')
    if any(canonical(raw[k])!=canonical(calibrated[k]) for k in ['registry','split','label_cutoff']):
        raise ValueError('same registry split and cutoff required')
    left={r['policy_id']:r for r in raw['trials']};right={r['policy_id']:r for r in calibrated['trials']}
    if not left or set(left)!=set(right) or len(left)!=len(raw['trials']) or len(right)!=len(calibrated['trials']):
        raise ValueError('identical unique policy universe required')
    panels=[]
    for pid,a in left.items():
        b=right[pid]
        if a['policy_sha256']!=b['policy_sha256']:raise ValueError('identical policy pin required')
        counts=lambda trial:[(d['signal_date'],d['selected']) for d in trial['selection']['daily']]
        matched=counts(a)==counts(b)
        events={}
        for event in ['safe_profit','up','down5']:
            x=a['metrics']['events'][event]['selected_candidates']['event_bounds']
            y=b['metrics']['events'][event]['selected_candidates']['event_bounds']
            nonempty=x['denominator']>0 and y['denominator']>0
            events[event]=dict(raw=x,calibrated=y,
                calibrated_minus_raw_bounds=[y['rate_lower']-x['rate_upper'],y['rate_upper']-x['rate_lower']]
                    if matched and nonempty else None)
        panels.append(dict(policy_id=pid,policy_sha256=a['policy_sha256'],same_daily_coverage=matched,
            raw_daily_selected=counts(a),calibrated_daily_selected=counts(b),events=events,
            status='MATCHED_DESCRIPTIVE' if matched else 'COVERAGE_TRADEOFF'))
    return dict(policies=panels,model_fits=0,calibrator_fits=0,policy_evaluations=0,
                bounds_are_confidence_intervals=False,automatic_winner_selected=False,
                risk10_evaluated=False,formal_H05_accepted=False)


def inspect(root,raw_directory,raw_sha,calibrated_directory,calibrated_sha):
    refs=[(Path(raw_directory).resolve(),raw_sha),(Path(calibrated_directory).resolve(),calibrated_sha)]
    if refs[0][0]==refs[1][0]:raise ValueError('distinct policy outputs required')
    reports=[];sources=[];payloads=[]
    for directory,sha in refs:
        verify(root,directory,sha)
        source=load_plan(directory/'summary.json');sources.append(source)
        payload=load_plan(source['input_path']);payloads.append({k:v for k,v in payload.items() if k!='probability_source'})
        reports.append(load_plan(directory/'comparison.json'))
    if any(sources[0][k]!=sources[1][k] for k in ['run_directory','run_sha256']) or canonical(payloads[0])!=canonical(payloads[1]):
        raise ValueError('identical saved model policies calendar and outcomes required')
    result=compare_reports(*reports)
    for directory,sha in refs:verify(root,directory,sha)
    return dict(result,raw_directory=str(refs[0][0]),raw_summary_sha256=raw_sha,
        calibrated_directory=str(refs[1][0]),calibrated_summary_sha256=calibrated_sha,
        same_model_and_evaluation_inputs_verified=True,global_budget_verified=False)
