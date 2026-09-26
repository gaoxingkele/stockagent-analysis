import pytest
import pandas as pd
from research.s20_harness.h02_intake import audit,OUTPUTS
from research.s20_harness.runtime import atomic_json,digest


@pytest.fixture(autouse=True)
def reference_reconstruction(monkeypatch):
    # Unit-test the intake/replay wiring against a separate immutable reference.
    # Actual market reconstruction is exercised separately, not claimed here.
    def reconstruct(root,partition,sha):
        return pd.read_parquet(root/'expected_o.parquet'),{},{}
    monkeypatch.setattr('research.s20_harness.opportunity_partition.reconstruct',reconstruct)
    monkeypatch.setattr('research.s20_harness.label_partition_replay.replay',
        lambda root,path,sha:dict(label_path_recomputed=True,formal_training_authorized=False))
    monkeypatch.setattr('research.s20_harness.stock_session_compat.replay',
        lambda root,path,sha:dict(current_code_paths_recomputed=True,formal_training_authorized=False))


def protocol(root):
    import hashlib
    from research.s20_harness.runtime import canonical,load_plan
    return hashlib.sha256(canonical(load_plan(root/'config/s20_v4_harness_plan.json'))).hexdigest()


def fixture(root,plan=None):
    source=root/'source';source.mkdir();(root/'config').mkdir()
    for name in OUTPUTS: (source/name).write_bytes(b'fixture integrity only')
    labels=pd.DataFrame(dict(sample_id=['a','b'],entity_id=['A','B'],signal_date=['20240102']*2,
        p_class=['D',None],o_class=pd.array([0,None],dtype='Int64'),o_label_realized=[True,False],
        o_safe_opportunity=pd.array([True,None],dtype='boolean'),o_reason=['safe_touch','unknown'],o_payload_json=['{}','{}']))
    for name in ['compat_class','raw_calendar_class']: labels[name]=pd.array([3,None],dtype='Int64')
    for name in ['compat_reason','raw_calendar_reason']: labels[name]=['negative_flat','unknown']
    for name in ['compat_payload_json','raw_calendar_payload_json']: labels[name]=['{}','{}']
    for name in ['compat_entry_date','compat_horizon_end']: labels[name]=['20240103',None]
    labels['different_horizon']=[False,False];labels['compat_label_realized']=[True,False]
    labels.to_parquet(source/'labels.parquet',index=False)
    labels.to_parquet(root/'expected_o.parquet',index=False)
    collection=root/'collection';collection.mkdir()
    labels[['sample_id','entity_id','signal_date','p_class']].to_parquet(collection/'samples.parquet',index=False)
    atomic_json(collection/'inputs.json',{})
    atomic_json(collection/'summary.json',dict(artifacts={n:digest(collection/n) for n in ['samples.parquet','inputs.json']}))
    plan_path=root/'config/s20_v4_harness_plan.json'
    plan=plan or dict(label_tracks=dict(execution_book_v4={'exit':'fixed'},opportunity={'target':20}))
    tracks=plan['label_tracks']
    atomic_json(plan_path,plan)
    atomic_json(source/'label_contracts.json',dict(frozen_contracts=tracks,formal_training_authorized=False))
    atomic_json(source/'execution_contract.json',dict(contract=tracks['execution_book_v4'],formal_training_authorized=False,actual_execution_evidence=False))
    opportunity=root/'opportunity';opportunity.mkdir()
    labels.to_parquet(opportunity/'labels.parquet',index=False)
    atomic_json(opportunity/'inputs.json',dict(partition='fixture',partition_sha256='fixture'))
    pd.crosstab(labels.o_reason,labels.p_class.fillna('UNKNOWN'),dropna=False).to_csv(opportunity/'op_transfer.csv')
    atomic_json(opportunity/'summary.json',dict(rows=2,o_resolved_rows=1,o_safe_rows=1,
        o_reason_counts=labels.o_reason.value_counts().to_dict(),formal_training_authorized=False,
        artifacts={n:digest(opportunity/n) for n in ['labels.parquet','inputs.json','op_transfer.csv']}))
    compat=root/'compat';compat.mkdir()
    labels.to_parquet(compat/'labels.parquet',index=False)
    atomic_json(compat/'inputs.json',dict(partition='fixture',partition_sha256='fixture'))
    atomic_json(compat/'summary.json',dict(artifacts={n:digest(compat/n) for n in ['labels.parquet','inputs.json']}))
    from research.s20_harness.execution_golden import build
    from research.s20_harness.runtime import load_plan
    from pathlib import Path
    golden=Path(build(root)['directory'])
    atomic_json(source/'golden_cases.json',load_plan(golden/'golden_cases.json'))
    atomic_json(source/'inputs.json',dict(plan_path=str(plan_path),plan_sha256=digest(plan_path),collection=str(collection),collection_sha256=digest(collection/'summary.json'),
        golden_directory=str(golden),golden_summary_sha256=digest(golden/'summary.json'),
        compatibility_partitions=[dict(directory=str(compat),summary_sha256=digest(compat/'summary.json'))],
        opportunity_partitions=[dict(directory=str(opportunity),summary_sha256=digest(opportunity/'summary.json'))]))
    pd.crosstab(labels.o_reason,labels.p_class.fillna('UNKNOWN'),dropna=False).to_csv(source/'label_transfer.csv')
    summary=dict(status='DIAGNOSTIC_PREPARATION',formal_gate_passed=False,formal_training_authorized=False,
        rows=2,dates=['20240102'],o_resolved_rows=1,o_safe_rows=1,
        artifacts={name:digest(source/name) for name in (*OUTPUTS,'inputs.json')},acceptance_gaps=['fixture semantic gap'])
    atomic_json(source/'summary.json',summary)
    atomic_json(root/'config/s20_v4_h02_intake.json',dict(protocol_hash=protocol(root),directory='source',summary_sha256=digest(source/'summary.json')))
    return source


def test_intake_copies_but_cannot_accept(tmp_path):
    source=fixture(tmp_path);result=audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not result['formal_gate_passed'] and not result['formal_training_authorized']
    assert 'fixture semantic gap' in result['acceptance_gaps']
    for name in OUTPUTS: assert digest(source/name)==digest(tmp_path/'out'/name)
    assert len(result['parent_checks']['profit_path_checks'])==1
    assert result['golden_checks']['all_passed'] and not result['golden_checks']['real_fill_evidence']


def test_golden_failure_rejects_before_path_work(tmp_path,monkeypatch):
    fixture(tmp_path)
    def failed(*args): return dict(all_passed=False)
    monkeypatch.setattr('research.s20_harness.execution_golden.replay_current',failed)
    with pytest.raises(ValueError,match='execution case failure'):
        audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not (tmp_path/'out').exists()


def test_rehashed_golden_copy_cannot_change_parent(tmp_path):
    from research.s20_harness.runtime import load_plan
    source=fixture(tmp_path)
    cases=load_plan(source/'golden_cases.json');cases['cases'][0]['actual']['remaining']=99
    atomic_json(source/'golden_cases.json',cases)
    summary=load_plan(source/'summary.json');summary['artifacts']['golden_cases.json']=digest(source/'golden_cases.json')
    atomic_json(source/'summary.json',summary)
    atomic_json(tmp_path/'config/s20_v4_h02_intake.json',dict(protocol_hash=protocol(tmp_path),directory='source',summary_sha256=digest(source/'summary.json')))
    with pytest.raises(ValueError,match='golden copy differs'):
        audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not (tmp_path/'out').exists()


def test_profit_replay_failure_prevents_intake_publication(tmp_path,monkeypatch):
    fixture(tmp_path)
    def fail(root,path,sha):
        raise ValueError('P recomputation differs')
    monkeypatch.setattr('research.s20_harness.label_partition_replay.replay',fail)
    with pytest.raises(ValueError,match='P recomputation differs'):
        audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not (tmp_path/'out').exists()


def test_compatibility_parent_swap_rejected(tmp_path):
    from research.s20_harness.runtime import load_plan
    source=fixture(tmp_path)
    compat=tmp_path/'compat'
    cp=load_plan(compat/'inputs.json');cp['partition']='different-parent'
    atomic_json(compat/'inputs.json',cp)
    meta=load_plan(compat/'summary.json');meta['artifacts']['inputs.json']=digest(compat/'inputs.json')
    atomic_json(compat/'summary.json',meta)
    inputs=load_plan(source/'inputs.json');inputs['compatibility_partitions'][0]['summary_sha256']=digest(compat/'summary.json')
    atomic_json(source/'inputs.json',inputs)
    summary=load_plan(source/'summary.json');summary['artifacts']['inputs.json']=digest(source/'inputs.json')
    atomic_json(source/'summary.json',summary)
    atomic_json(tmp_path/'config/s20_v4_h02_intake.json',dict(protocol_hash=protocol(tmp_path),directory='source',summary_sha256=digest(source/'summary.json')))
    with pytest.raises(ValueError,match='compatibility/P parent mismatch'):
        audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not (tmp_path/'out').exists()


def test_ambiguous_minus_one_remains_unresolved(tmp_path):
    from research.s20_harness.h02_intake import validate_labels
    from research.s20_harness.runtime import load_plan
    source=fixture(tmp_path)
    labels=pd.read_parquet(source/'labels.parquet')
    labels.loc[1,'o_class']=-1
    labels.to_parquet(source/'labels.parquet',index=False)
    result=validate_labels(source,load_plan(source/'summary.json'))
    assert result['o_resolved_rows']==1 and result['o_safe_rows']==1


def test_coordinated_parent_child_rehash_still_requires_path_replay(tmp_path):
    from research.s20_harness.runtime import load_plan
    source=fixture(tmp_path)
    opportunity=tmp_path/'opportunity'
    for directory in (source,opportunity):
        labels=pd.read_parquet(directory/'labels.parquet')
        labels.loc[0,'o_payload_json']='{"fabricated_path":true}'
        labels.to_parquet(directory/'labels.parquet',index=False)
    op_summary=load_plan(opportunity/'summary.json')
    op_summary['artifacts']['labels.parquet']=digest(opportunity/'labels.parquet')
    atomic_json(opportunity/'summary.json',op_summary)
    inputs=load_plan(source/'inputs.json')
    inputs['opportunity_partitions'][0]['summary_sha256']=digest(opportunity/'summary.json')
    atomic_json(source/'inputs.json',inputs)
    summary=load_plan(source/'summary.json')
    summary['artifacts']={n:digest(source/n) for n in (*OUTPUTS,'inputs.json')}
    atomic_json(source/'summary.json',summary)
    atomic_json(tmp_path/'config/s20_v4_h02_intake.json',dict(protocol_hash=protocol(tmp_path),directory='source',summary_sha256=digest(source/'summary.json')))
    with pytest.raises(AssertionError,match='o_payload_json'):
        audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('tamper',['protocol','artifact'])
def test_bad_intake_rejected(tmp_path,tamper):
    source=fixture(tmp_path)
    if tamper=='artifact': (source/'labels.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError): audit(tmp_path,tmp_path/'out','wrong' if tamper=='protocol' else protocol(tmp_path))


@pytest.mark.parametrize('kind',['unknown','counts','transfer'])
def test_rehashed_semantic_corruption_rejected(tmp_path,kind):
    from research.s20_harness.runtime import load_plan
    source=fixture(tmp_path)
    summary=load_plan(source/'summary.json')
    if kind=='unknown':
        labels=pd.read_parquet(source/'labels.parquet')
        labels.loc[1,'o_safe_opportunity']=False
        labels.to_parquet(source/'labels.parquet',index=False)
    elif kind=='counts': summary['rows']=3
    else:
        transfer=pd.read_csv(source/'label_transfer.csv',index_col=0)
        transfer.iloc[0,0]+=1
        transfer.to_csv(source/'label_transfer.csv')
    summary['artifacts']={name:digest(source/name) for name in (*OUTPUTS,'inputs.json')}
    atomic_json(source/'summary.json',summary)
    atomic_json(tmp_path/'config/s20_v4_h02_intake.json',dict(protocol_hash=protocol(tmp_path),directory='source',summary_sha256=digest(source/'summary.json')))
    with pytest.raises((ValueError,AssertionError)): audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['rules','candidate','opportunity_payload'])
def test_consistent_child_cannot_replace_parents(tmp_path,kind):
    from research.s20_harness.runtime import load_plan
    source=fixture(tmp_path)
    if kind=='rules':
        contract=load_plan(source/'label_contracts.json')
        contract['frozen_contracts']['opportunity']['target']=15
        atomic_json(source/'label_contracts.json',contract)
    else:
        labels=pd.read_parquet(source/'labels.parquet')
        if kind=='candidate': labels.loc[0,'sample_id']='replacement'
        else: labels.loc[0,'o_payload_json']='{"changed":true}'
        labels.to_parquet(source/'labels.parquet',index=False)
    summary=load_plan(source/'summary.json')
    summary['artifacts']={n:digest(source/n) for n in (*OUTPUTS,'inputs.json')}
    atomic_json(source/'summary.json',summary)
    atomic_json(tmp_path/'config/s20_v4_h02_intake.json',dict(protocol_hash=protocol(tmp_path),directory='source',summary_sha256=digest(source/'summary.json')))
    with pytest.raises((ValueError,AssertionError)): audit(tmp_path,tmp_path/'out',protocol(tmp_path))
    assert not (tmp_path/'out').exists()
