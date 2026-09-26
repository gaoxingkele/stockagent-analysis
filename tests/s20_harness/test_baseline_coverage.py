import pytest
from research.s20_harness.baseline_coverage import inventory,REQUIRED


def card(job,family,fold='f',target='P.safe.v4'):
    return dict(job_id=job,model_family=family,fold_id=fold,target_id=target,evidence_mode='synthetic')


def test_models_in_other_fold_or_target_do_not_fill_missing_slots():
    r=inventory([card('a','logistic'),card('b','shallow_tree',fold='other'),card('c','mature_frequency',target='O.safe.v4')])
    groups={(g['fold_id'],g['target_id']):g for g in r['observed_scope']}
    p=groups['f','P.safe.v4']
    assert 'shallow_tree' in p['missing_families'] and 'mature_frequency' in p['missing_families']
    assert 'logistic' not in p['missing_families']
    assert not r['all_observed_groups_represent_required_families']


def test_full_family_names_still_do_not_prove_formal_coverage():
    r=inventory([card(str(i),f) for i,f in enumerate(REQUIRED)])
    assert not r['all_observed_groups_represent_required_families']
    assert r['observed_scope'][0]['missing_families']==['atr_liquidity_control']
    assert not r['required_track_market_fold_scope_verified']
    assert not r['legacy_original_metrics_reproduced'] and not r['formal_H03_accepted']


def test_unknown_family_and_duplicate_ids_are_not_silent():
    r=inventory([card('a','novel')]);assert r['observed_scope'][0]['other_families']==['novel']
    with pytest.raises(ValueError):inventory([card('a','logistic'),card('a','shallow_tree')])


def test_atr_is_verified_policy_identity_not_renamed_model():
    c=card('filtered','logistic')
    c.update(policy_mode='atr_liquidity_control',policy_sha256='a'*64)
    r=inventory([c]); group=r['observed_scope'][0]
    assert 'atr_liquidity_control' not in group['missing_families']
    assert 'logistic' not in group['missing_families']
    assert group['policy_inventory'][0]['model_family']=='logistic'
    assert not r['same_universe_and_coverage_comparability_verified']
    c['policy_sha256']='not-a-pin'
    assert 'atr_liquidity_control' in inventory([c])['observed_scope'][0]['missing_families']
