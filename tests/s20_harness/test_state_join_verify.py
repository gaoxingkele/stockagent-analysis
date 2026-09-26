import json
from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness.historical_state import audit_panel
from research.s20_harness.runtime import atomic_json, digest
from research.s20_harness.state_join_verify import reconstruct
from research.s20_harness.source_evidence import inspect_sources
from tests.s20_harness.test_source_evidence import identity_source
from tests.s20_harness.test_name_timeline_verify import fixture as name_fixture


def fixture(root):
    inventory = identity_source(root)
    atomic_json(root/'config/s20_v4_data_sources.json', inventory)
    entry = inventory['sources'][0]
    panel = root/entry['path']
    timeline = name_fixture(root)
    report = audit_panel(root, panel, timeline, digest(timeline))
    directory = Path(report['directory'])
    inventory['sources'].append(dict(role='per_code_historical_names', path=str(timeline.parent.parent),
        timeline=str(timeline), timeline_sha256=digest(timeline), state_join_audit=str(directory/'summary.json')))
    return inventory, directory, panel, entry['summary_sha256'], timeline


def test_full_upstream_and_join_reconstruction(tmp_path):
    inventory, directory, panel, sha, timeline = fixture(tmp_path)
    checked = reconstruct(directory, digest(directory/'summary.json'), panel, sha, timeline, digest(timeline))
    assert checked['rows'] == 3 and checked['unknown_rows'] > 0
    assert not checked['upstream_semantics_verified_here']
    records, pins = inspect_sources(tmp_path, inventory, revalidate_identity=True,
                                    revalidate_names=True, revalidate_state_join=True)
    result = records[-1]['fresh_state_join_validation']
    assert result['upstream_semantics_verified_in_same_source_audit'] and pins
    assert not result['formal_training_eligible']
    with pytest.raises(ValueError, match='fresh identity'):
        inspect_sources(tmp_path, inventory, revalidate_state_join=True)


def test_false_daily_coverage_rejected(tmp_path):
    _, directory, panel, sha, timeline = fixture(tmp_path)
    path = directory/'date_coverage.json'
    rows = json.loads(path.read_text())
    rows[0]['unknown_rows'] = 99
    atomic_json(path, rows)
    with pytest.raises(ValueError, match='daily coverage mismatch'):
        reconstruct(directory, digest(directory/'summary.json'), panel, sha, timeline, digest(timeline))
