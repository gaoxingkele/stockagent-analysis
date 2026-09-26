import json
from pathlib import Path
import pytest
import pandas as pd
from research.s20_harness.baseline_run import build
from research.s20_harness.baseline_replay import replay
from research.s20_harness.baseline_outer_ledger import collect
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_anchor_training import setup
from tests.s20_harness.test_baseline_run import plan


def test_serialized_anchor_chain_replay_and_model_pin(tmp_path):
    args,context=setup(tmp_path)
    value=plan()
    for sample,template in zip(args[0].to_dict('records'),value['samples']):
        sample['signal_date']=template['signal_date']
        template.update(sample)
    value['features']=args[1].to_dict('records')
    value['feature_contract']=args[4]
    serialized=dict(context)
    serialized['samples']=context['samples'].to_dict('records')
    serialized['predictions']=context['predictions'].to_dict('records')
    value['anchor_context']=serialized
    path=tmp_path/'anchor-plan.json';atomic_json(path,value)
    report=build(tmp_path,path,digest(path));out=Path(report['directory'])
    rows=pd.read_parquet(out/'calibrated_predictions.parquet')
    assert len(rows)==10 and pd.isna(rows.calibrated_probability.iloc[-1])
    assert replay(tmp_path,out,digest(out/'summary.json'))['semantic_replay_performed']
    ledger=collect([dict(job_id='j',fold_id='f',directory=str(out),summary_sha256=digest(out/'summary.json'))])
    assert ledger.recorded_oof_provenance_valid.tolist()==[True,False]
    assert ledger.recorded_label_dependency_count.tolist()==[5,5]
    assert ledger.recorded_anchor_model_count.tolist()==[1,1]
    assert 'prediction_feature_not_prior' in ledger.provenance_reasons_json.iloc[-1]
    from research.s20_harness.bounded_baseline import run as owned
    from research.s20_harness.process_runner import Limits
    from research.s20_harness.trial_budget import Budget
    from tests.s20_harness.test_trial_budget import contract
    pin=digest(path);budget=Budget(tmp_path/'budget.sqlite',contract(pin))
    limits=Limits(120,2*1024**3,1)
    child=owned(tmp_path,path,pin,budget,'baseline','one',limits)
    assert child['process']['exit_code']==0
    reused=owned(tmp_path,path,pin,budget,'baseline','one',limits)
    assert reused['reusable'] and not reused['executed']
    assert budget.status()['reserved_counts']['model_fits']==1
    model=Path(serialized['models']['m']['model_path'])
    model.write_text('modified artifact')
    with pytest.raises(ValueError,match='model pin'):
        build(tmp_path,path,digest(path))
    with pytest.raises(ValueError,match='data dependencies changed'):
        owned(tmp_path,path,pin,budget,'baseline','one',limits)
