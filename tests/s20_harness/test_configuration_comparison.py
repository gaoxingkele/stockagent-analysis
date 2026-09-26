import pytest
from research.s20_harness.configuration_comparison import compare_cells
from research.s20_harness.metrics import binary_bounds


def row(seed,safe,risk):
    return dict(fold_id='f',random_seed=seed,selected=len(safe),comparison_scope='a',outcome_sha256='b',
        daily_selected=[dict(signal_date='20240503',count=len(safe))],
        safe_profit=binary_bounds(safe),down5=binary_bounds(risk))


def test_all_seeds_required_for_robust_domination():
    best=[row(s,[True],[False]) for s in [20,71]]
    bad=[row(s,[False],[True]) for s in [20,71]]
    assert compare_cells(best,bad)['dominant']=='left'
    assert compare_cells(bad,best)['dominant']=='right'
    assert compare_cells(best,best)['dominant'] is None
    bad[1]=row(71,[None],[None])
    assert compare_cells(best,bad)['dominant']=='left' # no worse even in unknown best case
    best[1]=row(71,[False],[True])
    assert compare_cells(best,bad)['dominant'] is None


def test_incomplete_unknown_empty_and_mismatched_scopes():
    one=[row(20,[None],[None])];two=[row(20,[True],[False])]
    assert compare_cells(one,two)['dominant'] is None
    assert compare_cells([row(20,[],[])],[row(20,[],[])])['cells'][0]['status']=='NO_SELECTION'
    with pytest.raises(ValueError,match='grid'):compare_cells(one,[row(71,[True],[False])])
    with pytest.raises(ValueError,match='unique'):compare_cells(one+one,two)
    for key in ['comparison_scope','outcome_sha256']:
        with pytest.raises(ValueError,match='scope'):compare_cells(one,[dict(two[0],**{key:'different'})])
    changed=dict(two[0],daily_selected=[dict(signal_date='20240506',count=1)])
    assert compare_cells(one,[changed])['status']=='INCOMPARABLE_COVERAGE'


def test_verified_two_campaigns_cli(tmp_path,monkeypatch,capsys):
    from pathlib import Path
    import json
    import pandas as pd
    from tests.s20_harness.test_seed_campaign import setup
    from research.s20_harness.multifold_run import run
    from research.s20_harness.campaign_evaluation import build as evaluate
    from research.s20_harness.seed_summary import build as summarize
    from research.s20_harness.runtime import atomic_json,digest,load_plan
    from research.s20_harness import cli
    summaries=[];outcomes=tmp_path/'outcomes.json'
    for index in range(2):
        folder=tmp_path/str(index);folder.mkdir()
        source=setup(folder,'joint');manifest=load_plan(source)
        if index:
            for j,job in enumerate(manifest['jobs']):
                path=Path(job['input_path']);plan=load_plan(path)
                plan.update(model_family='cost_sensitive_joint',class_weights=dict(A=1.,B=2.,C=1.,D=3.))
                atomic_json(path,plan);job['input_sha256']=digest(path)
                manifest['budget']['trials'][j]['input_sha256']=digest(path)
            atomic_json(source,manifest)
        campaign=run(tmp_path,source,digest(source))
        receipt=next(Path(campaign['directory']).glob('receipt-*.json'))
        if not index:
            candidates=pd.read_parquet(Path(campaign['jobs'][0]['directory'])/'candidate_ledger.parquet')
            atomic_json(outcomes,dict(target_id='P.joint.v4',evaluation_at='2024-06-01T00:00:00Z',
                outcomes=[dict(sample_id=sid,target='A',label_available_at='2024-05-29T00:00:00Z') for sid in candidates.sample_id]))
        evaluated=evaluate(tmp_path,receipt,digest(receipt),[dict(fold_id='common',path=str(outcomes),sha256=digest(outcomes))])
        directory=Path(evaluated['directory'])
        summarized=summarize(tmp_path,directory,digest(directory/'summary.json'))
        directory=Path(summarized['directory']);summaries.append((directory,digest(directory/'summary.json')))
    monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    args=['compare-configurations','--left-directory',str(summaries[0][0]),'--left-sha256',summaries[0][1],
          '--right-directory',str(summaries[1][0]),'--right-sha256',summaries[1][1]]
    assert cli.main(args)==0
    result=json.loads(capsys.readouterr().out)
    assert result['valid'] and result['models_refit']==0
    assert not result['finalists_selected'] and not result['search_preregistration_verified']
    assert result['status'] in ['NO_ROBUST_DOMINANCE','INCOMPARABLE_COVERAGE','ROBUST_DESCRIPTIVE_DOMINANCE']
    args[-1]='0'*64
    assert cli.main(args)==2
