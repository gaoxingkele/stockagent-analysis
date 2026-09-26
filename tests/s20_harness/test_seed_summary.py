from pathlib import Path
import json
import pandas as pd
import pytest

from research.s20_harness.seed_summary import aggregate,verify
from research.s20_harness.metrics import binary_bounds
from research.s20_harness.runtime import atomic_json,digest,load_plan


def test_prediction_disagreement_alignment_missing_and_changed():
    from research.s20_harness.seed_summary import prediction_disagreement
    rows=[dict(fold_id='f',random_seed=s,job_id=str(s)) for s in [20,71]]
    records=[]
    for seed in [20,71]:
        for sid in ['a','b','c']:
            records.append(dict(job_id=str(seed),fold_id='f',sample_id=sid,entity_id=sid,signal_date='20240503',
                selected=sid=='a',**{prefix+c:(.25 if sid!='c' else None) for prefix in ['p_','raw_p_'] for c in 'ABCD'}))
    frame=pd.DataFrame(records)
    get=lambda value:prediction_disagreement(value,rows,[20,71],['f'])[0]
    same=get(frame.iloc[::-1])
    assert same['same_predictions_and_selection']
    assert same['probabilities']['raw']['both_unavailable']==1
    assert same['selection_jaccard']==1
    frame.loc[3,['p_A','p_B']]=[.3,.2]
    frame.loc[3,'selected']=False;frame.loc[4,'selected']=True
    changed=get(frame)
    assert changed['probabilities']['raw']['exactly_identical']
    assert changed['probabilities']['calibrated']['changed_candidates']==1
    assert changed['selection_disagreements']==2 and changed['selection_jaccard']==0
    frame['selected']=False
    assert get(frame)['selection_jaccard'] is None
    frame.loc[5,['p_'+c for c in 'ABCD']]=.25
    assert get(frame)['probabilities']['calibrated']['availability_mismatch']==1
    with pytest.raises(ValueError,match='universe'):get(frame.iloc[:-1])
    with pytest.raises(ValueError,match='unique'):get(pd.concat([frame,frame.iloc[:1]]))
    frame.loc[5,'p_A']=None
    with pytest.raises(ValueError,match='partial'):get(frame)


def row(fold,seed,values):
    return dict(fold_id=fold,random_seed=seed,job_id=f'{fold}-{seed}',candidates=max(2,len(values)),
        selected=len(values),safe_profit=binary_bounds(values),down5=binary_bounds(values))


def test_aggregation_preserves_unknown_and_zero_folds_without_seed_pooling():
    rows=[row('a',20,[True,None]),row('b',20,[]),row('a',71,[False]),row('b',71,[True]*3)]
    result=aggregate(rows,[20,71],['a','b'])
    assert result[0]['safe_profit']['rate_lower']==.5
    assert result[0]['safe_profit']['rate_upper']==1
    assert result[0]['safe_profit']['equal_fold_lower'] is None
    assert result[0]['zero_selection_folds']==['b']
    assert result[1]['safe_profit']['rate_lower']==.75
    assert result[1]['safe_profit']['equal_fold_lower']==.5
    assert [r['selected'] for r in result]==[2,4]
    empty=aggregate([row('a',20,[])],[20],['a'])[0]
    assert empty['safe_profit']['rate_lower'] is None
    for invalid in [rows[:-1],rows+[rows[0]]]:
        with pytest.raises(ValueError,match='grid'):aggregate(invalid,[20,71],['a','b'])


def test_owned_joint_seed_evaluation_cli_and_rehashed_tamper(tmp_path,monkeypatch,capsys):
    from tests.s20_harness.test_seed_campaign import setup
    from research.s20_harness.multifold_run import run
    from research.s20_harness.campaign_evaluation import build as evaluate
    from research.s20_harness import cli
    path=setup(tmp_path,'joint');campaign=run(tmp_path,path,digest(path))
    receipt=next(Path(campaign['directory']).glob('receipt-*.json'))
    candidates=pd.read_parquet(Path(campaign['jobs'][0]['directory'])/'candidate_ledger.parquet')
    outcomes=tmp_path/'outcomes.json'
    atomic_json(outcomes,dict(target_id='P.joint.v4',evaluation_at='2024-06-01T00:00:00Z',
        outcomes=[dict(sample_id=sid,target='A',label_available_at='2024-05-29T00:00:00Z') for sid in candidates.sample_id]))
    evaluated=evaluate(tmp_path,receipt,digest(receipt),[dict(fold_id='common',path=str(outcomes),sha256=digest(outcomes))])
    directory=Path(evaluated['directory']);monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    assert cli.main(['build-seed-summary','--directory',str(directory),'--summary-sha256',digest(directory/'summary.json')])==0
    report=json.loads(capsys.readouterr().out);report.pop('valid');out=Path(report['directory'])
    assert verify(tmp_path,out,digest(out/'summary.json'))['seed_summary_recomputed']
    result=load_plan(out/'seed_summary.json')
    assert len(result['per_seed'])==2 and len(result['jobs'])==2
    assert not result['seed_pooling_performed'] and not result['finalists_selected']
    assert len(result['prediction_disagreement'])==1
    assert result['prediction_disagreement'][0]['same_predictions_and_selection']
    assert cli.main(['verify-seed-summary','--directory',str(out),'--summary-sha256',digest(out/'summary.json')])==0
    capsys.readouterr()
    for field in ['finalists_selected','independent_market_replicates','bounds_are_confidence_intervals']:
        changed=dict(result);changed[field]=True;atomic_json(out/'seed_summary.json',changed)
        report['artifacts']['seed_summary.json']=digest(out/'seed_summary.json');atomic_json(out/'summary.json',report)
        with pytest.raises(ValueError,match='reconstruction'):verify(tmp_path,out,digest(out/'summary.json'))
    changed=json.loads(json.dumps(result))
    changed['prediction_disagreement'][0]['selection_disagreements']=999
    atomic_json(out/'seed_summary.json',changed)
    report['artifacts']['seed_summary.json']=digest(out/'seed_summary.json');atomic_json(out/'summary.json',report)
    with pytest.raises(ValueError,match='reconstruction'):verify(tmp_path,out,digest(out/'summary.json'))
