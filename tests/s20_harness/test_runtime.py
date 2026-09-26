import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from research.s20_harness.runtime import Runtime, atomic_json
from tests.s20_harness.test_baseline_bundle import bundle

PLAN = Path(__file__).resolve().parents[2] / "config/s20_v4_harness_plan.json"


@pytest.fixture
def runtime(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "commit", "--allow-empty", "-qm", "fixture"], check=True)
    (tmp_path / "research").mkdir()
    (tmp_path / "research/example.py").write_text("# untracked research\n", encoding="utf-8")
    return Runtime(tmp_path, PLAN)


def test_h00_captures_untracked_sources_and_hashes(runtime):
    result = runtime.run("H00")
    assert runtime.verify(result["run_id"], require_completion=False)["valid"]
    assert runtime.verify(result["run_id"])["valid"]
    assert runtime.status()["stages"]["H00"] == "COMPLETED"
    snapshot = json.loads((runtime.base / result["run_id"] / "source_snapshot.json").read_text())
    assert snapshot["files"][0]["path"] == "research/example.py"
    assert not result["acceptance_gaps"]
    with pytest.raises(ValueError, match="already verified"):
        runtime.run("H00")


def test_tampered_artifacts_revoke_gate(runtime):
    result = runtime.run("H00")
    atomic_json(runtime.base / result["run_id"] / "budget.json", {})
    assert not runtime.verify(result["run_id"], require_completion=False)["valid"]
    assert runtime.status()["stages"]["H00"] == "FAILED_VALIDITY"
    with pytest.raises(ValueError, match="prerequisites"):
        runtime.run("H01")


def test_blob_tampering_and_append_only_events(runtime):
    result = runtime.run("H00")
    blob = next((runtime.base / result["run_id"] / "source_blobs").iterdir())
    blob.write_bytes(b"tampered")
    assert not runtime.verify(result["run_id"], require_completion=False)["valid"]
    with runtime.connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM events")


def test_dependencies_and_unimplemented_stages_fail_closed(runtime):
    with pytest.raises(ValueError, match="prerequisites"):
        runtime.run("H03")
    runtime.run("H00")
    audit = runtime.run("H01")
    assert audit["upstream_runs"]["H00"]["run_id"].startswith("H00-")
    assert runtime.verify(audit["run_id"], require_completion=False)["valid"]
    assert audit["terminal_state"] == "FAILED_VALIDITY"
    assert runtime.status()["stages"]["H01"] == "FAILED_VALIDITY"
    with pytest.raises(ValueError, match="prerequisites"):
        runtime.run("H02")


def test_h02_intake_records_failure_and_cannot_unlock_h03(runtime,monkeypatch):
    from research.s20_harness import full_data_audit
    from tests.s20_harness.test_h02_intake import fixture
    from research.s20_harness.runtime import digest
    runtime.run('H00')
    # Synthetic upstream solely exercises runtime routing, not real H01 approval.
    def synthetic_upstream(root,directory):
        for name in runtime.stages['H01']['outputs']:
            (directory/name).write_bytes(b'synthetic upstream fixture')
        return dict(formal_gate_passed=True,acceptance_gaps=[])
    monkeypatch.setattr(full_data_audit,'audit_dataset',synthetic_upstream)
    runtime.run('H01')
    source=fixture(runtime.root,plan=runtime.plan)
    # Only the path engine is isolated here; real path replay has separate checks.
    import pandas as pd
    monkeypatch.setattr('research.s20_harness.opportunity_partition.reconstruct',
        lambda root,partition,sha:(pd.read_parquet(root/'expected_o.parquet'),{},{}))
    monkeypatch.setattr('research.s20_harness.label_partition_replay.replay',
        lambda root,path,sha:dict(label_path_recomputed=True,formal_training_authorized=False))
    monkeypatch.setattr('research.s20_harness.stock_session_compat.replay',
        lambda root,path,sha:dict(current_code_paths_recomputed=True,formal_training_authorized=False))
    atomic_json(runtime.root/'config/s20_v4_h02_intake.json',dict(protocol_hash=runtime.protocol_hash,
        directory='source',summary_sha256=digest(source/'summary.json')))
    result=runtime.run('H02')
    assert result['terminal_state']=='FAILED_VALIDITY'
    assert runtime.verify(result['run_id'],require_completion=False)['valid']
    assert not runtime.verify(result['run_id'])['valid']
    assert result['upstream_runs']['H01']['run_id'].startswith('H01-')
    with pytest.raises(ValueError,match='prerequisites'): runtime.run('H03')


def test_h03_diagnostic_intake_cannot_unlock_h04(runtime,bundle,monkeypatch):
    from research.s20_harness import full_data_audit,h02_intake
    from research.s20_harness.runtime import digest
    runtime.run('H00')
    def upstream(stage):
        def synthetic(root,directory,*args):
            for name in runtime.stages[stage]['outputs']:(directory/name).write_bytes(b'synthetic only')
            return dict(formal_gate_passed=True,acceptance_gaps=[])
        return synthetic
    monkeypatch.setattr(full_data_audit,'audit_dataset',upstream('H01'))
    monkeypatch.setattr(h02_intake,'audit',upstream('H02'))
    runtime.run('H01');runtime.run('H02')
    root,source=bundle
    (root/'config').mkdir(exist_ok=True)
    atomic_json(root/'config/s20_v4_h03_intake.json',dict(protocol_hash=runtime.protocol_hash,
        directory=str(source),summary_sha256=digest(source/'summary.json')))
    result=runtime.run('H03')
    assert result['terminal_state']=='FAILED_VALIDITY'
    assert runtime.verify(result['run_id'],require_completion=False)['valid']
    assert not runtime.verify(result['run_id'])['valid']
    assert result['upstream_runs']['H02']['run_id'].startswith('H02-')
    with pytest.raises(ValueError,match='prerequisites'):runtime.run('H04')


def test_upstream_pin_and_terminal_event_tamper_fail_closed(runtime):
    parent = runtime.run("H00")
    child = runtime.run("H01")
    child_path = runtime.base / child["run_id"] / "manifest.json"
    child["upstream_runs"]["H00"]["manifest_sha256"] = "0" * 64
    atomic_json(child_path, child)
    assert not runtime.verify(child["run_id"], require_completion=False)["valid"]
    child.pop("upstream_runs")
    atomic_json(child_path, child)
    assert "missing or unexpected upstream run bindings" in runtime.verify(child["run_id"], False)["errors"]
    parent["science_state"] = "PROMISING"
    atomic_json(runtime.base / parent["run_id"] / "manifest.json", parent)
    assert "completion event manifest hash mismatch" in runtime.verify(parent["run_id"])["errors"]
    assert runtime.status()["stages"]["H00"] == "FAILED_VALIDITY"


def test_output_escape_rejected(tmp_path):
    runtime = Runtime(tmp_path, PLAN)
    with pytest.raises(ValueError, match="invalid run id"):
        runtime.verify("../elsewhere")


@pytest.mark.parametrize('error,expected', [
    (ValueError('source hash mismatch'), 'FAILED_VALIDITY'),
    (AssertionError('reconstructed table mismatch'), 'FAILED_VALIDITY'),
    (OSError('read failed'), 'FAILED_INFRA'),
    (TypeError('programming error'), 'FAILED_INFRA'),
])
def test_h01_rejected_data_not_infrastructure_retry(runtime, monkeypatch, error, expected):
    from research.s20_harness import full_data_audit
    from research.s20_harness.errors import DataValidityError
    from research.s20_harness.runtime import digest
    runtime.run('H00')
    def fail(*args):
        raise error
    monkeypatch.setattr(full_data_audit, 'audit_dataset', fail)
    with pytest.raises(DataValidityError if expected == 'FAILED_VALIDITY' else type(error)):
        runtime.run('H01')
    with runtime.connect() as db:
        row = db.execute('SELECT * FROM runs WHERE stage=?', ('H01',)).fetchone()
        event = db.execute('SELECT * FROM events WHERE run_id=? ORDER BY seq DESC', (row['run_id'],)).fetchone()
    assert row['state'] == event['state'] == expected
    directory = runtime.base/row['run_id']
    failure = json.loads((directory/'failure.json').read_text())
    assert failure['terminal_state'] == expected and not failure['formal_stage_completed']
    assert not failure['automatic_retry_authorized']
    assert json.loads(event['payload'])['failure_sha256'] == digest(directory/'failure.json')
    assert not (directory/'manifest.json').exists()
    assert not runtime.verify(row['run_id'], require_completion=False)['valid']
    from research.s20_harness.report import build_report
    report = build_report(runtime)
    item = next(x for x in report['run_history'] if x['run_id'] == row['run_id'])
    assert item['failure_evidence']['verified'] and not item['evidence']['valid']
    assert not next(x for x in report['stages'] if x['stage'] == 'H01')['accepted_evidence']
    failure['error'] = 'edited after terminal event'
    atomic_json(directory/'failure.json', failure)
    report = build_report(runtime)
    assert not next(x for x in report['run_history'] if x['run_id'] == row['run_id'])['failure_evidence']['verified']
    with pytest.raises(ValueError, match='prerequisites'):
        runtime.run('H02')


def test_report_read_only_empty_and_event_bound(runtime):
    from research.s20_harness.report import build_report
    report = build_report(runtime)
    assert not runtime.base.exists()
    assert not any(s["accepted_evidence"] for s in report["stages"])
    result = runtime.run("H00")
    report = build_report(runtime)
    assert report["stages"][0]["accepted_evidence"]
    assert report["stages"][1]["unverified_dependencies"] == []
    path = runtime.base / result["run_id"] / "manifest.json"
    result["science_state"] = "PROMISING"
    atomic_json(path, result)
    report = build_report(runtime)
    assert not report["stages"][0]["accepted_evidence"]
    assert report["stages"][1]["unverified_dependencies"] == ["H00"]
    assert "manifest differs from terminal event hash" in report["run_history"][0]["evidence"]["errors"]


@pytest.mark.parametrize("live", [True, None, False])
def test_recovery_owner_identity_and_unanchored_failure(runtime, monkeypatch, live):
    from research.s20_harness import recovery
    from research.s20_harness.runtime import now
    with runtime.connect() as db:
        db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)",
                   ("H00-interrupted", "H00", "RUNNING", now(), None, 123, 1.0, None))
        runtime._event(db, "H00-interrupted", "RUNNING", {})
    monkeypatch.setattr(recovery, "owner_live", lambda *args: live)
    result = recovery.resume(runtime)
    assert not result["training_started"]
    with runtime.connect() as db:
        state = db.execute("SELECT state FROM runs").fetchone()[0]
    assert state == ("FAILED_INFRA" if live is False else "RUNNING")
    if live is False:
        assert recovery.resume(runtime)["actions"] == []


@pytest.mark.parametrize("tampered", [False, True])
def test_recovery_finishes_only_anchored_commit(runtime, monkeypatch, tampered):
    from research.s20_harness import recovery
    from research.s20_harness.runtime import digest
    result = runtime.run("H00")
    run_id = result["run_id"]
    with runtime.connect() as db:
        db.execute("UPDATE runs SET state='VERIFYING' WHERE run_id=?", (run_id,))
        runtime._event(db, run_id, "VERIFYING", {
            "manifest_hash": digest(runtime.base / run_id / "manifest.json")})
    if tampered:
        result["science_state"] = "PROMISING"
        atomic_json(runtime.base / run_id / "manifest.json", result)
    monkeypatch.setattr(recovery, "owner_live", lambda *args: False)
    assert recovery.resume(runtime)["actions"][0]["state"] == ("FAILED_INFRA" if tampered else "COMPLETED")
    from research.s20_harness.report import build_report
    assert build_report(runtime)["stages"][0]["accepted_evidence"] is (not tampered)
