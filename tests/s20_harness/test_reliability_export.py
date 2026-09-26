import copy
import io
import pandas as pd
import pytest
from research.s20_harness.reliability_export import table, csv_text
from tests.s20_harness.test_joint_evaluation import inputs, assess


def jobs():
    _, metrics = assess(inputs())
    return [dict(candidate_id='0',job_id='job-a',fold_id='0',random_seed=20,
                 input_sha256='a'*64,metrics=metrics)]


def test_complete_export_preserves_unknowns_scopes_and_seed_identity():
    source = jobs(); second = copy.deepcopy(source[0])
    second.update(job_id='job-b',random_seed=71)
    result = table(source+[second])
    assert len(result) == 2*3*4*10  # two aggregate scopes plus two calendar days
    assert set(result.random_seed) == {20,71}
    buckets = result.query("job_id == 'job-a' and event == 'safe_profit' and scope == 'selected_candidates'")
    assert buckets.scored_count.sum() == 2
    assert buckets.known_count.sum() == buckets.unknown_count.sum() == 1
    assert not result.formal_H05_accepted.any()
    assert not result.probability_reliability_proven.any()
    assert not result.bounds_are_confidence_intervals.any()
    # Nulls are empty CSV cells, not a fabricated zero success rate.
    restored = pd.read_csv(io.StringIO(csv_text(source)),dtype={'candidate_id':str,'fold_id':str})
    assert set(restored.fold_id) == {'0'}
    assert restored.loc[restored.scored_count.eq(0),'event_rate_lower'].isna().all()


def test_export_retains_empty_selection_and_all_candidate_reference():
    args = inputs(); args[0]['selected'] = False
    _, metrics = assess(args)
    source = jobs();source[0]['metrics'] = metrics
    result = table(source)
    selected = result.loc[result.scope.ne('all_candidates')]
    assert selected.group_candidates.eq(0).all()
    assert selected.mean_prediction_all_scored.isna().all()
    assert result.loc[result.scope.eq('all_candidates'),'scored_count'].sum() == 6


def test_export_refuses_empty_or_duplicate_jobs():
    with pytest.raises(ValueError,match='nonempty'):table([])
    with pytest.raises(ValueError,match='duplicate'):table(jobs()*2)
