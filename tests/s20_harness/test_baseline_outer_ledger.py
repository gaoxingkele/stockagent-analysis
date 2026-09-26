from pathlib import Path
import pandas as pd

from research.s20_harness.baseline_run import build
from research.s20_harness.baseline_outer_ledger import collect
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_baseline_run import plan


def test_unknown_forward_candidate_retained_not_valid_oof(tmp_path):
    value=plan()
    value['samples'][-2]['feature_available_at']=None
    path=tmp_path/'input.json';atomic_json(path,value)
    report=build(tmp_path,path,digest(path));directory=Path(report['directory'])
    ledger=collect([dict(job_id='job',fold_id='fold',directory=str(directory),
        summary_sha256=digest(directory/'summary.json'))])
    assert ledger.sample_id.tolist()==['outer-test0','outer-test1']
    assert pd.isna(ledger.score.iloc[0])
    assert not ledger.recorded_oof_provenance_valid.iloc[0]
    assert 'prediction_feature_not_prior' in ledger.provenance_reasons_json.iloc[0]
    assert ledger.recorded_oof_provenance_valid.iloc[1]
    assert not ledger.historical_availability_proven.any()
