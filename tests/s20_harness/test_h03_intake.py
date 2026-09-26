import pytest
from research.s20_harness import h03_intake
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_baseline_bundle import bundle


def bind(root,source):
    path=root/'config/s20_v4_h03_intake.json'
    path.parent.mkdir(exist_ok=True)
    atomic_json(path,dict(protocol_hash='protocol',directory=str(source),summary_sha256=digest(source/'summary.json')))
    return path


def test_recomputed_bundle_is_copied_but_never_promoted(bundle):
    root,source=bundle;bind(root,source);out=root/'intake'
    r=h03_intake.audit(root,out,'protocol')
    assert r['bundle_validation']['bundle_values_recomputed'] and r['models_refit']==0
    assert not r['formal_gate_passed'] and r['acceptance_gaps']
    assert all(digest(out/n)==digest(source/n) for n in h03_intake.OUTPUTS)
    with pytest.raises(ValueError,match='existing outputs'):h03_intake.audit(root,out,'protocol')


def test_protocol_mismatch_and_failed_verification_publish_no_receipt(bundle,monkeypatch):
    root,source=bundle;bind(root,source);out=root/'intake'
    with pytest.raises(ValueError,match='protocol'):h03_intake.audit(root,out,'wrong')
    def fail(*a):raise ValueError('invalid bundle')
    monkeypatch.setattr(h03_intake,'verify',fail)
    with pytest.raises(ValueError,match='invalid bundle'):h03_intake.audit(root,out,'protocol')
    assert not (out/'h03_intake.json').exists()


def test_binding_changed_during_reverification_rejects(bundle,monkeypatch):
    root,source=bundle;path=bind(root,source);real=h03_intake.verify;calls=[]
    def change(*a):
        result=real(*a);calls.append(1)
        if len(calls)==2:atomic_json(path,dict(changed=True))
        return result
    monkeypatch.setattr(h03_intake,'verify',change)
    with pytest.raises(ValueError,match='binding/source changed'):h03_intake.audit(root,root/'intake','protocol')
    assert not (root/'intake/h03_intake.json').exists()
