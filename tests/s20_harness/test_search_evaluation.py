from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.search_evaluation import build,verify
from research.s20_harness.search_registration import register
from research.s20_harness.registered_search_run import run
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_selection_plan import setup


@pytest.mark.parametrize('rule',['matched_cell_robust_dominance_no_automatic_tiebreak',
                                  'all_registered_nondominated_no_automatic_tiebreak'])
@pytest.mark.parametrize('families',[('multinomial','cost_sensitive_joint'),('mature_frequency','shallow_joint')])
def test_full_shared_search_evaluation_unknowns_and_rehash(tmp_path,monkeypatch,capsys,rule,families):
    source=setup(tmp_path,families);selection=load_plan(source);selection['selection_rule']=rule
    atomic_json(source,selection);sha=digest(source);registration=register(tmp_path,source,sha)
    partial=run(tmp_path,registration['directory'],sha,max_new_jobs=1)
    with pytest.raises(ValueError,match='complete registered'):
        build(tmp_path,partial['receipt_path'],digest(Path(partial['receipt_path'])),[])
    completed=run(tmp_path,registration['directory'],sha,max_new_jobs=12)
    expected_fits=6*sum(f!='mature_frequency' for f in families)
    assert completed['budget']['reserved_counts']['model_fits']==expected_fits
    assert completed['budget']['reserved_counts']['calibrator_fits']==12
    assert completed['budget']['reserved_counts']['underlying_fits']==expected_fits+12
    budget=Path(registration['shared_budget_path']);before=digest(budget)
    refs=[]
    for fold in ['0','1','2']:
        job=next(j for j in completed['jobs'] if j['fold_id']==fold)
        candidates=pd.read_parquet(Path(job['artifact']['directory'])/'candidate_ledger.parquet')
        path=tmp_path/f'outcomes-{fold}.json'
        atomic_json(path,dict(target_id='P.joint.v4',evaluation_at='2026-09-14T00:00:00Z',outcomes=[
            dict(sample_id=sid,target='A' if i==0 else None,label_available_at='2026-06-01T00:00:00Z' if i==0 else None)
            for i,sid in enumerate(candidates.sample_id)]))
        refs.append(dict(fold_id=fold,path=str(path),sha256=digest(path)))
    result=build(tmp_path,completed['receipt_path'],digest(Path(completed['receipt_path'])),refs)
    directory=Path(result['directory']);assert result['jobs']==12 and result['rows']==24
    rows=pd.read_parquet(directory/'evaluated_predictions.parquet')
    assert rows.safe_profit_target.isna().sum()==12
    assert rows.candidate_id.nunique()==2 and rows.job_id.nunique()==12
    reliability_path=directory/'selected_reliability.csv'
    reliability=pd.read_csv(reliability_path)
    assert reliability.job_id.nunique()==12
    assert not reliability.formal_H05_accepted.any()
    reference=reliability.loc[reliability.scope.eq('all_candidates')]
    assert reference.scored_count.sum()+reference.drop_duplicates(['job_id','event']).group_missing_predictions.sum()==24*3
    groups=reference.drop_duplicates(['job_id','event'])
    assert groups.group_unknown_outcomes.sum()==12*3
    # Fixture unknown outcomes have missing predictions, hence no score bin.
    assert reference.unknown_count.sum()==0
    assert groups.group_missing_predictions.sum()==12*3
    assert verify(tmp_path,directory,digest(directory/'summary.json'))['models_refit']==0
    from research.s20_harness import cli
    monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    assert cli.main(['verify-search-evaluation','--directory',str(directory),'--summary-sha256',digest(directory/'summary.json')])==0
    import json
    assert json.loads(capsys.readouterr().out)['search_evaluation_recomputed']
    assert digest(budget)==before
    from research.s20_harness.search_candidates import inspect
    summary=inspect(tmp_path,directory,digest(directory/'summary.json'))
    assert summary['registered_candidate_ids']==['0','1']
    assert len(summary['cells'])==12 and sum(c['candidates'] for c in summary['cells'])==24
    assert len(summary['candidates'])==2 and len(summary['baseline_comparisons'])==1
    assert all(len(c['per_seed'])==2 for c in summary['candidates'])
    assert not summary['finalists_selected'] and digest(budget)==before
    coverage=summary['baseline_coverage']
    assert len(coverage['observed_scope'])==3
    assert not coverage['all_observed_groups_represent_required_families']
    assert all('legacy_r20' in fold['missing_families'] and 'atr_liquidity_control' in fold['missing_families']
               for fold in coverage['observed_scope'])
    if 'mature_frequency' in families:
        assert all('mature_frequency' not in fold['missing_families'] for fold in coverage['observed_scope'])
        assert all('shallow_joint' in fold['other_families'] for fold in coverage['observed_scope'])
    from research.s20_harness.search_decision import build as decide,verify as verify_decision
    assert cli.main(['build-search-decision','--directory',str(directory),'--summary-sha256',digest(directory/'summary.json')])==0
    decision=json.loads(capsys.readouterr().out);assert decision.pop('valid')
    decision_dir=Path(decision['directory'])
    assert cli.main(['verify-search-decision','--directory',str(decision_dir),'--summary-sha256',digest(decision_dir/'summary.json')])==0
    assert json.loads(capsys.readouterr().out)['decision_recomputed']
    saved=load_plan(decision_dir/'decision.json')
    assert not saved['formal_finalists_selected'] and digest(budget)==before
    if rule=='all_registered_nondominated_no_automatic_tiebreak':
        assert saved['baseline_in_comparison']
        assert saved['status'] in ['DIAGNOSTIC_FRONTIER','UNRESOLVED_COMPARABILITY','UNRESOLVED_CAPACITY']
        if saved['status']=='DIAGNOSTIC_FRONTIER' and families==('multinomial','cost_sensitive_joint'):
            # Fixture outcomes do not establish strict superiority over the baseline.
            assert '0' in saved['diagnostic_shortlist']
    saved['formal_finalists_selected']=True;atomic_json(decision_dir/'decision.json',saved)
    decision['artifacts']['decision.json']=digest(decision_dir/'decision.json');atomic_json(decision_dir/'summary.json',decision)
    with pytest.raises(ValueError,match='reconstruction'):verify_decision(tmp_path,decision_dir,digest(decision_dir/'summary.json'))
    # Rehashed reliability edits cannot pass semantic reconstruction.
    original_reliability=reliability_path.read_bytes()
    reliability.loc[0,'known_count']+=1
    reliability.to_csv(reliability_path,index=False)
    original_hash=result['artifacts']['selected_reliability.csv']
    result['artifacts']['selected_reliability.csv']=digest(reliability_path)
    atomic_json(directory/'summary.json',result)
    with pytest.raises(ValueError,match='reliability reconstruction'):
        verify(tmp_path,directory,digest(directory/'summary.json'))
    reliability_path.write_bytes(original_reliability)
    result['artifacts']['selected_reliability.csv']=original_hash
    atomic_json(directory/'summary.json',result)
    rows.loc[0,'safe_profit_target']=False
    rows.to_parquet(directory/'evaluated_predictions.parquet',index=False)
    result['artifacts']['evaluated_predictions.parquet']=digest(directory/'evaluated_predictions.parquet')
    atomic_json(directory/'summary.json',result)
    with pytest.raises(AssertionError):verify(tmp_path,directory,digest(directory/'summary.json'))
    with pytest.raises(ValueError,match='one shared outcome'):
        build(tmp_path,completed['receipt_path'],digest(Path(completed['receipt_path'])),refs[:-1])
