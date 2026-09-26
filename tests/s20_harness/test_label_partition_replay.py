import pandas as pd
import pytest

from research.s20_harness.label_partition_replay import compare


def table():
    return pd.DataFrame(dict(sample_id=['a','b'],p_class=['A',None],
        payload_json=['{"terminal_net":0.2}','{"label_status":"unknown_path"}']))


def test_exact_replay_keeps_unknown_rows():
    result=compare(table(),table())
    assert result['rows']==2 and result['label_path_recomputed']
    assert not result['independent_algorithm_validation'] and not result['formal_training_authorized']


@pytest.mark.parametrize('kind',['payload','class','order','missing'])
def test_replay_detects_numeric_and_denominator_changes(kind):
    altered=table()
    if kind=='payload': altered.loc[0,'payload_json']='{"terminal_net":0.21}'
    if kind=='class': altered.loc[0,'p_class']='B'
    if kind=='order': altered=altered.iloc[::-1].reset_index(drop=True)
    if kind=='missing': altered=altered.iloc[:1]
    with pytest.raises(AssertionError): compare(table(),altered)
