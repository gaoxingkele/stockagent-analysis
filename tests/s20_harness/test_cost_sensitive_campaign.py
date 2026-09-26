from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.multifold_run import run
from research.s20_harness.runtime import atomic_json,digest,load_plan
from research.s20_harness.baseline_outer_ledger import verify_receipt
from tests.s20_harness.test_joint_family_campaign import setup as family_setup


def setup(tmp_path):
    source=family_setup(tmp_path);manifest=load_plan(source)
    manifest.update(schema_version='4',comparison_contract='joint_loss_weight_control')
    job=manifest['jobs'][1];path=Path(job['input_path']);value=load_plan(path)
    value.update(model_family='cost_sensitive_joint',class_weights=dict(A=1.,B=2.,C=1.,D=3.))
    atomic_json(path,value);job['input_sha256']=digest(path)
    manifest['budget']['trials'][1]['input_sha256']=digest(path)
    manifest['budget']['trials'][1]['costs']['underlying_fits']=2
    manifest['budget']['limits']['underlying_fits']=4
    atomic_json(source,manifest)
    return source


def test_registered_weight_control_reuses_and_verifies_scope(tmp_path):
    source=setup(tmp_path);first=run(tmp_path,source,digest(source))
    assert first['same_fold_variation_allowed']==['model_family','class_weights']
    assert first['distinct_outer_folds']==1 and first['budget']['reserved_counts']['underlying_fits']==4
    rows=pd.read_parquet(first['outer_prediction_ledger']['path'])
    assert len(rows)==4 and set(rows.model_family)=={'fixed_multinomial_logistic','fixed_cost_sensitive_multinomial'}
    again=run(tmp_path,source,digest(source),directory=first['directory'])
    assert not any(j['executed_this_call'] for j in again['jobs'])
    receipt=next(Path(first['directory']).glob('receipt-*.json'));report=load_plan(receipt)
    report['same_fold_variation_allowed']=['model_family']
    altered=Path(first['directory'])/'altered-receipt.json';atomic_json(altered,report)
    with pytest.raises(ValueError,match='contract/scope report'):verify_receipt(tmp_path,altered,digest(altered))


@pytest.mark.parametrize('change',['old_contract','policy','conditional_family'])
def test_unregistered_variation_rejected_before_launch(tmp_path,change):
    source=setup(tmp_path);manifest=load_plan(source)
    if change=='old_contract':
        manifest['schema_version']='3';del manifest['comparison_contract']
    else:
        job=manifest['jobs'][1];path=Path(job['input_path']);value=load_plan(path)
        if change=='policy':value['policy']['selection']['n_cap']+=1
        else:
            value['model_family']='conditional_three';del value['class_weights']
        atomic_json(path,value);job['input_sha256']=digest(path)
        manifest['budget']['trials'][1]['input_sha256']=digest(path)
    atomic_json(source,manifest)
    with pytest.raises(ValueError,match='scope mismatch|requires multinomial'):run(tmp_path,source,digest(source))
    assert not (tmp_path/'output').exists()
