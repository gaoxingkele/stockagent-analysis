"""Strict date collection of replayed diagnostic assemblies; no row filtering."""
from pathlib import Path
import uuid

import pandas as pd

from .dataset_assembly import verify_saved
from .runtime import atomic_json, digest, now


def verify_collection(root, directory, summary_sha):
    """Consume an externally pinned collection and replay all member assemblies."""
    from .runtime import load_plan
    directory=Path(directory).resolve()
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('collection summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    if set(summary['artifacts'])!={'samples.parquet','inputs.json'}:
        raise ValueError('unexpected collection artifacts')
    def check():
        if digest(directory/'summary.json')!=summary_sha or any(
            digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('saved collection changed')
    check()
    inputs=load_plan(directory/'inputs.json')
    frames=[]
    for item in inputs['assemblies']:
        if set(item)!={'directory','summary_sha256'}: raise ValueError('invalid collection member')
        frame,_=verify_saved(root,item['directory'],item['summary_sha256'])
        frames.append(frame)
    rebuilt=combine(frames)
    saved=pd.read_parquet(directory/'samples.parquet')
    pd.testing.assert_frame_equal(rebuilt,saved,check_exact=True)
    if (summary['rows']!=len(saved) or summary['dates']!=[str(f.signal_date.iloc[0]) for f in frames]
        or summary['rows_by_date']!=saved.groupby('signal_date').size().to_dict()
        or summary['all_candidates_retained'] is not True
        or summary['formal_training_authorized'] is not False
        or summary['historical_availability_proven'] is not False):
        raise ValueError('collection summary semantics mismatch')
    check()
    return saved,dict(rows=len(saved),member_assemblies=len(frames),assembly_values_replayed=True,
        label_path_independently_recomputed=False,formal_training_authorized=False)


def combine(frames):
    if not frames: raise ValueError('nonempty date collection required')
    dates=[]
    schema=[(c,str(t)) for c,t in frames[0].dtypes.items()]
    for frame in frames:
        if frame.empty or frame.signal_date.isna().any() or frame.signal_date.nunique()!=1:
            raise ValueError('exactly one nonempty signal date per assembly required')
        date=str(frame.signal_date.iloc[0])
        if pd.to_datetime(date,format='%Y%m%d').strftime('%Y%m%d')!=date:
            raise ValueError('canonical signal date required')
        dates.append(date)
        if [(c,str(t)) for c,t in frame.dtypes.items()]!=schema:
            raise ValueError('mixed assembly schemas')
        if not frame.formal_training_eligible.eq(False).all():
            raise ValueError('diagnostic collection cannot admit formal eligibility')
    if dates!=sorted(set(dates)): raise ValueError('unique chronological assembly dates required')
    result=pd.concat(frames,ignore_index=True)
    if result.sample_id.isna().any() or result.sample_id.duplicated().any():
        raise ValueError('duplicate or missing collection sample identity')
    return result


def build(root, assemblies):
    """assemblies is an ordered list of exact directory/summary_sha256 bindings."""
    root=Path(root).resolve()
    frames=[]; evidence=[]; bindings=[]
    for item in assemblies:
        if set(item)!={'directory','summary_sha256'}: raise ValueError('invalid assembly binding')
        directory=Path(item['directory']).resolve()
        frame,check=verify_saved(root,directory,item['summary_sha256'])
        frames.append(frame); evidence.append(check)
        bindings.append(dict(directory=str(directory),summary_sha256=item['summary_sha256']))
    result=combine(frames)
    # Recheck all input bytes after the last reconstruction, before publication.
    from .runtime import load_plan
    for item in bindings:
        directory=Path(item['directory'])
        if digest(directory/'summary.json')!=item['summary_sha256']:
            raise ValueError('collection input changed')
        summary=load_plan(directory/'summary.json')
        if any(digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('collection artifact changed')
        inputs=load_plan(directory/'inputs.json')
        if any(digest(Path(p))!=h for p,h in inputs['source_pins'].items()):
            raise ValueError('collection upstream source changed')
        from .label_partition_verify import verify
        verify(inputs['label_directory'],inputs['label_summary_sha256'])
    out=root/'output/experiments/s20_safe_v4/sources'/('dataset-collection-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    result.to_parquet(out/'samples.parquet',index=False)
    atomic_json(out/'inputs.json',dict(assemblies=bindings,replay_evidence=evidence,
        code_sha256=digest(Path(__file__))))
    report=dict(directory=str(out),at=now(),rows=len(result),dates=[str(f.signal_date.iloc[0]) for f in frames],
        rows_by_date=result.groupby('signal_date').size().to_dict(),all_candidates_retained=True,
        formal_training_authorized=False,historical_availability_proven=False,
        artifacts={n:digest(out/n) for n in ['samples.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
