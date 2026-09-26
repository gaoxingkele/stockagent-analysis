from dataclasses import asdict
import json

import pytest

from research.s20_harness import cash_action_source as source
from research.s20_harness.distribution_adapter import adapt
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_distribution_adapter import frame, review


def fixture(tmp_path, reviewed=True):
    data=frame(); path=tmp_path/'normalized.parquet'; data.to_parquet(path,index=False)
    rv=tmp_path/'reviews.json'; reviews=review(data) if reviewed else {}
    atomic_json(rv,dict(reviews=reviews))
    events,decisions=adapt(data,reviews,rate_policy='gross_reference_diagnostic')
    ledger=tmp_path/'ledger'; ledger.mkdir()
    atomic_json(ledger/'events.json',[asdict(e) for e in events]); atomic_json(ledger/'decisions.json',decisions)
    s=dict(source_path=str(path),source_sha256=digest(path),review_path=str(rv),review_sha256=digest(rv),
           unit_review_path=None,source_rows=1,accepted_gross_accounting_events=len(events),
           unresolved_events=1-len(events),events_sha256=digest(ledger/'events.json'),
           decisions_sha256=digest(ledger/'decisions.json'))
    atomic_json(ledger/'summary.json',s)
    return ledger,digest(ledger/'summary.json')


def test_reconstructs_and_binds_code_without_backdating(tmp_path,monkeypatch):
    p,h=fixture(tmp_path)
    monkeypatch.setattr(source,'now',lambda:'2026-09-14T00:00:00+00:00')
    e,r=source.load_event(p,h,'e','A')
    assert e.cash_per_share==1 and r['ts_code']=='A' and r['full_ledger_reconstructed']
    assert r['observed_available_at'].startswith('2026') and not r['historical_availability_proven']
    with pytest.raises(ValueError,match='identity'):source.load_event(p,h,'e','B')


def test_unresolved_not_usable(tmp_path):
    p,h=fixture(tmp_path,False)
    with pytest.raises(ValueError,match='unresolved'):source.load_event(p,h,'e','A')


def test_rehashed_event_edit_still_fails_reconstruction(tmp_path):
    p,h=fixture(tmp_path)
    events=json.loads((p/'events.json').read_text());events[0]['cash_per_share']=99
    atomic_json(p/'events.json',events)
    s=json.loads((p/'summary.json').read_text());s['events_sha256']=digest(p/'events.json');atomic_json(p/'summary.json',s)
    with pytest.raises(ValueError,match='reconstruction'):
        source.build(tmp_path,p,digest(p/'summary.json'),'e','A')
    assert not (tmp_path/'output').exists()


def test_source_mutation_during_reconstruction_rejected(tmp_path,monkeypatch):
    p,h=fixture(tmp_path);real=source.adapt
    def mutate(*args,**kwargs):
        r=real(*args,**kwargs);atomic_json(tmp_path/'reviews.json',dict(reviews={}));return r
    monkeypatch.setattr(source,'adapt',mutate)
    with pytest.raises(ValueError,match='pin mismatch'):source.load_event(p,h,'e','A')


def test_present_receipt_cannot_authorize_old_record_date(tmp_path,monkeypatch):
    from research.s20_harness.portfolio_book import create
    p,h=fixture(tmp_path)
    monkeypatch.setattr(source,'now',lambda:'2026-09-14T00:00:00+00:00')
    b=create(1000,['20240102','20240103','20240104','20240105'])
    with pytest.raises(ValueError,match='unavailable'):
        source.record_from_ledger(b,ledger=p,summary_sha=h,distribution_id='e',ts_code='A',
                                  event_id='claim',processing_at='2024-01-03T15:01:00+08:00')


def test_full_unresolved_ledger_retained_and_formal_code_pin_required(tmp_path):
    p,h=fixture(tmp_path,False)
    events,decisions,evidence=source.load_ledger(p,h)
    assert not events and len(decisions)==1
    assert evidence['source_rows']==evidence['unresolved_events']==1
    assert not evidence['full_corporate_action_coverage_proven']
    with pytest.raises(ValueError,match='code pin required'):
        source.load_ledger(p,h,require_code_pin=True)


def test_inventory_reconstructs_ledger_without_promoting_to_pit(tmp_path):
    from research.s20_harness.source_evidence import inspect_sources
    p,h=fixture(tmp_path)
    summary=json.loads((p/'summary.json').read_text())
    summary['code_sha256']=digest(source.Path(source.__file__).with_name('distribution_adapter.py'))
    atomic_json(p/'summary.json',summary)
    inventory={'sources':[{'role':'reviewed_distribution_ledger','path':str(p),
                           'summary_sha256':digest(p/'summary.json')}]}
    records,pins=inspect_sources(tmp_path,inventory,revalidate_cash_ledger=True)
    check=records[0]['fresh_cash_ledger_validation']
    assert check['accepted_gross_accounting_events']==1 and check['unresolved_events']==0
    assert not check['historical_availability_proven'] and not records[0]['semantic_acceptance']
    assert any(x['role']=='fresh_reviewed_distribution_ledger' for x in pins)
