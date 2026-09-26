"""Persist chronological usability evidence for a pinned diagnostic collection.

Passing timestamp checks is necessary, not sufficient for formal H01/H02.
This adapter never changes source timestamps or drops unavailable candidates.
"""
from pathlib import Path
import uuid

from .dataset_collection import verify_collection
from .runtime import atomic_json, digest, load_plan, now
from .splits import assign_segments


def build(root, collection, collection_sha, split_path, split_sha):
    root,collection,split_path=Path(root).resolve(),Path(collection).resolve(),Path(split_path).resolve()
    if digest(split_path)!=split_sha: raise ValueError('admission split pin mismatch')
    contract=load_plan(split_path)
    if set(contract)!={'boundaries','evaluation_at'}: raise ValueError('exact admission split contract required')
    samples,validation=verify_collection(root,collection,collection_sha)
    assignments,split_report=assign_segments(samples,contract['boundaries'],evaluation_at=contract['evaluation_at'])
    if digest(split_path)!=split_sha or digest(collection/'summary.json')!=collection_sha:
        raise ValueError('admission input changed')
    # A diagnostic timestamp-ready row still lacks formal source/economic admission.
    assignments['formal_training_authorized']=False
    out=root/'output/experiments/s20_safe_v4/sources'/('dataset-admission-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    assignments.to_parquet(out/'assignments.parquet',index=False)
    atomic_json(out/'inputs.json',dict(collection=str(collection),collection_sha256=collection_sha,
        split_path=str(split_path),split_sha256=split_sha,code_sha256=digest(Path(__file__))))
    report=dict(directory=str(out),at=now(),rows=len(samples),collection_validation=validation,
        split=split_report,reason_counts=assignments.reason.value_counts().to_dict(),
        supervised_timestamp_ready_rows=int(assignments.supervised_eligible.sum()),
        evaluation_timestamp_ready_rows=int(assignments.evaluation_eligible.sum()),
        diagnostic_only=True,formal_training_authorized=False,
        unresolved_requirements=['H01 source/universe/availability admission','H02 economic labels and execution admission'],
        artifacts={n:digest(out/n) for n in ['assignments.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
