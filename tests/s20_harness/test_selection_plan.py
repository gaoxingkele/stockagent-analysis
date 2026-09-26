import json
import pytest
from research.s20_harness.selection_plan import inspect
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_run import plan as joint_plan


def setup(tmp_path,families=('multinomial','cost_sensitive_joint')):
    refs=[];costs=dict(model_fits=1,underlying_fits=2,calibrator_fits=1,policy_evaluations=1)
    for config,family in enumerate(families):
        from research.s20_harness.joint_run import pipeline_costs
        family_plan=dict(model_family=family)
        if family=='cost_sensitive_joint':family_plan['class_weights']=dict(A=1.,B=2.,C=1.,D=3.)
        costs=pipeline_costs(family_plan)
        jobs=[];trials=[]
        for fold in range(3):
            for seed in [20,71]:
                value=joint_plan();value['samples'][9]['feature_available_at']=None
                value=json.loads(json.dumps(value).replace('2024',str(2024+fold)))
                value.update(random_seed=seed,**family_plan)
                key=f'{config}-{fold}-{seed}';path=tmp_path/(key+'.json');atomic_json(path,value)
                jobs.append(dict(job_id=key,trial_id=key,attempt_id='one',fold_id=str(fold),pipeline_kind='joint',input_path=str(path),input_sha256=digest(path)))
                trials.append(dict(trial_id=key,input_sha256=digest(path),costs=costs,max_attempts=1))
        manifest=dict(schema_version='7',evidence_mode='synthetic',comparison_contract='fixed_model_seed_control',
            seed_grid=[20,71],fold_grid=['0','1','2'],jobs=jobs,limits=dict(wall_seconds=120,memory_bytes=2147483648,cpu_threads=1),
            budget=dict(budget_id=str(config),trials=trials,limits={k:v*6 for k,v in costs.items()}))
        path=tmp_path/f'campaign{config}.json';atomic_json(path,manifest)
        refs.append(dict(candidate_id=str(config),manifest_path=str(path),manifest_sha256=digest(path)))
    path=tmp_path/'selection.json'
    total={k:sum(load_plan(r['manifest_path'])['budget']['limits'][k] for r in refs) for k in costs}
    atomic_json(path,dict(schema_version='selection-1',baseline_id='0',candidates=refs,limits=total,
        finalist_cap=3,selection_rule='matched_cell_robust_dominance_no_automatic_tiebreak'))
    return path


def test_complete_pinned_initial_search_budget(tmp_path):
    path=setup(tmp_path);result=inspect(tmp_path,path,digest(path))
    assert result['planned_initial_costs']['model_fits']==12
    assert result['planned_initial_costs']['underlying_fits']==24
    assert result['input_contracts_validated'] and not result['training_authorized']
    assert not result['preregistration_before_training_proven']
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('change',[dict(finalist_cap=4),dict(baseline_id='missing'),dict(selection_rule='best_seen'),dict(finalist_cap=True)])
def test_invalid_selection_contract(tmp_path,change):
    path=setup(tmp_path);plan=load_plan(path);plan.update(change);atomic_json(path,plan)
    with pytest.raises(ValueError):inspect(tmp_path,path,digest(path))


def test_total_budget_not_just_each_campaign(tmp_path):
    path=setup(tmp_path);plan=load_plan(path);plan['limits']['model_fits']=11;atomic_json(path,plan)
    with pytest.raises(ValueError,match='total search budget'):inspect(tmp_path,path,digest(path))


def test_changed_source_rejected(tmp_path):
    path=setup(tmp_path);atomic_json(tmp_path/'0-0-20.json',{})
    with pytest.raises(ValueError,match='input pin'):inspect(tmp_path,path,digest(path))


def test_cli_preflight_not_training_authorization(tmp_path,monkeypatch,capsys):
    from research.s20_harness import cli
    path=setup(tmp_path);monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    assert cli.main(['inspect-selection-plan','--input',str(path),'--input-sha256',digest(path)])==0
    assert not json.loads(capsys.readouterr().out)['training_authorized']
    assert cli.main(['inspect-selection-plan','--input',str(path),'--input-sha256','0'*64])==2


def test_missing_fold_rejected_even_with_rehashed_manifest(tmp_path):
    from pathlib import Path
    path=setup(tmp_path);plan=load_plan(path);ref=plan['candidates'][1]
    campaign_path=Path(ref['manifest_path']);manifest=load_plan(campaign_path)
    manifest['jobs']=manifest['jobs'][:-2];manifest['budget']['trials']=manifest['budget']['trials'][:-2]
    atomic_json(campaign_path,manifest);ref['manifest_sha256']=digest(campaign_path);atomic_json(path,plan)
    with pytest.raises(ValueError,match='grid'):inspect(tmp_path,path,digest(path))


@pytest.mark.parametrize('kind',['overallocate','underallocate','boolean','attempts','resource','evidence'])
def test_candidate_budget_and_execution_contract_rejected(tmp_path,kind):
    from pathlib import Path
    path=setup(tmp_path);plan=load_plan(path);ref=plan['candidates'][0]
    target=Path(ref['manifest_path']);manifest=load_plan(target)
    if kind=='overallocate':manifest['budget']['limits']['model_fits']=12
    elif kind=='underallocate':manifest['budget']['limits']['model_fits']=5
    elif kind=='boolean':manifest['budget']['trials'][0]['costs']['model_fits']=True
    elif kind=='attempts':manifest['budget']['trials'][0]['max_attempts']=4
    elif kind=='resource':manifest['limits']['wall_seconds']=-1
    else:manifest['evidence_mode']='supplied_reference'
    atomic_json(target,manifest);ref['manifest_sha256']=digest(target);atomic_json(path,plan)
    with pytest.raises(ValueError):inspect(tmp_path,path,digest(path))


def test_relabelled_overlapping_outer_folds_rejected(tmp_path):
    from pathlib import Path
    path=setup(tmp_path);plan=load_plan(path)
    # Both candidates share the same changed scope, so only interval validation catches this.
    for ref in plan['candidates']:
        target=Path(ref['manifest_path']);manifest=load_plan(target)
        for i,job in enumerate(manifest['jobs']):
            if job['fold_id']!='1':continue
            source=Path(job['input_path']);value=load_plan(source)
            value['boundaries'][4].update(start_at='2024-05-01T00:00:00Z',end_at='2024-06-01T00:00:00Z')
            atomic_json(source,value);job['input_sha256']=digest(source)
            manifest['budget']['trials'][i]['input_sha256']=digest(source)
        atomic_json(target,manifest);ref['manifest_sha256']=digest(target)
    atomic_json(path,plan)
    with pytest.raises(ValueError,match='overlap'):inspect(tmp_path,path,digest(path))
