"""Assemble retained real diagnostic candidates, never silently train a complete-case subset."""
from pathlib import Path
import uuid

import pandas as pd

from .availability_partition import attach
from .factor_price_features import compute
from .label_partition_verify import verify
from .runtime import atomic_json, digest, load_plan, now


FEATURES = ['vendor_factor_return20','vendor_factor_ma20_distance','vendor_factor_mean_tr14_pct',
            'vendor_factor_return_vol20','volume_ratio20']


def assemble(labels, features, observed_at):
    keys=['sample_id','entity_id','signal_date']
    for table in (labels,features):
        if table[keys].isna().any().any() or table.sample_id.duplicated().any():
            raise ValueError('unique nonmissing assembly identities required')
    if labels[keys].to_dict('records') != features[keys].to_dict('records'):
        raise ValueError('feature/label candidate identity or order mismatch')
    if not labels.p_class.dropna().isin(['A','B','C','D']).all() or not labels.label_realized.eq(labels.p_class.notna()).all():
        raise ValueError('invalid or unresolved diagnostic class')
    availability=attach(labels,observed_at)
    result=features[keys+['trading_code',*FEATURES,'price_window_status']].copy()
    result=result.rename(columns={'price_window_status':'feature_window_status'})
    result['label_status']=labels.label_status.to_numpy()
    result['p_class']=labels.p_class.to_numpy()
    result['gross_safe_diagnostic_target']=pd.array([None if pd.isna(c) else c=='A' for c in labels.p_class],dtype='boolean')
    result['gross_b5_diagnostic_target']=pd.array([None if pd.isna(c) else c in ('B','D') for c in labels.p_class],dtype='boolean')
    result['prediction_at']=[(pd.to_datetime(d,format='%Y%m%d').tz_localize('Asia/Shanghai')+pd.Timedelta(hours=21)).isoformat()
                             for d in result.signal_date.astype(str)]
    result['horizon_close_at']=availability.horizon_close_at.to_numpy()
    result['feature_available_at']=observed_at
    result['label_available_at']=availability.gross_diagnostic_label_available_at.to_numpy()
    result['all_feature_values_present']=result[FEATURES].notna().all(axis=1)
    result['diagnostic_label_resolved']=labels.label_realized.to_numpy()
    result['formal_training_eligible']=False
    return result


def reconstruct(root, label_dir, label_sha, feature_dir, feature_sha, observed=None):
    root,label_dir,feature_dir=Path(root).resolve(),Path(label_dir).resolve(),Path(feature_dir).resolve()
    labels,label_check=verify(label_dir,label_sha)
    if digest(feature_dir/'summary.json')!=feature_sha: raise ValueError('feature summary pin mismatch')
    summary=load_plan(feature_dir/'summary.json')
    if set(summary['artifacts'])!={'features.parquet','inputs.json','factor_receipts.json'}:
        raise ValueError('unexpected feature artifacts')
    pins={label_dir/'summary.json':label_sha,feature_dir/'summary.json':feature_sha}
    for name,sha in summary['artifacts'].items(): pins[feature_dir/name]=sha
    if any(digest(p)!=h for p,h in pins.items()): raise ValueError('assembly artifact mismatch')
    pins.update({Path(p):h for p,h in load_plan(feature_dir/'inputs.json').items()})
    if any(digest(p)!=h for p,h in pins.items()): raise ValueError('assembly feature source changed')
    candidates=pd.read_parquet(label_dir/'candidates.parquet')
    quote_paths=[p for p in pins if p.parent.name=='daily' and p.suffix=='.parquet' and p.stem in summary['window']]
    if len(quote_paths)!=21 or {p.stem for p in quote_paths}!=set(summary['window']): raise ValueError('incomplete feature window')
    receipts=load_plan(feature_dir/'factor_receipts.json')
    from .dependency_receipts import bind
    _,bound=bind(root,pd.DataFrame(receipts['bindings']))
    if bound['source_pins']!=receipts['evidence']['source_pins']: raise ValueError('factor binding changed')
    factor_paths=sorted({Path(b['artifact_path']) for b in receipts['bindings']})
    quotes=pd.concat([pd.read_parquet(p) for p in sorted(quote_paths)],ignore_index=True)
    factors=pd.concat([pd.read_parquet(p) for p in factor_paths],ignore_index=True)
    rebuilt=compute(candidates,quotes,factors,summary['window'])
    features=pd.read_parquet(feature_dir/'features.parquet')
    pd.testing.assert_frame_equal(rebuilt,features,check_exact=True)
    observed=now() if observed is None else observed
    result=assemble(labels,features,observed)
    # Recheck complete label input provenance after feature reconstruction.
    verify(label_dir,label_sha)
    if any(digest(p)!=h for p,h in pins.items()): raise ValueError('assembly sources changed')
    return result, label_check, pins, observed


def verify_saved(root, directory, summary_sha):
    """Replay upstream semantics, not just the saved artifact hashes."""
    directory=Path(directory).resolve()
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('assembly summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    if set(summary['artifacts'])!={'samples.parquet','inputs.json'}:
        raise ValueError('unexpected assembly artifacts')
    def check():
        if digest(directory/'summary.json')!=summary_sha or any(
            digest(directory/n)!=h for n,h in summary['artifacts'].items()
        ): raise ValueError('saved assembly artifact changed')
    check()
    inputs=load_plan(directory/'inputs.json')
    rebuilt,_,pins,_=reconstruct(root,inputs['label_directory'],inputs['label_summary_sha256'],
        inputs['feature_directory'],inputs['feature_summary_sha256'],inputs['observed_at'])
    if {str(p):h for p,h in pins.items()}!=inputs['source_pins']:
        raise ValueError('assembly source contract mismatch')
    saved=pd.read_parquet(directory/'samples.parquet')
    pd.testing.assert_frame_equal(rebuilt,saved,check_exact=True)
    cross=saved.groupby(['all_feature_values_present','diagnostic_label_resolved']).size().rename('rows').reset_index().to_dict('records')
    if (summary['rows']!=len(saved) or summary['class_counts']!=saved.p_class.fillna('UNKNOWN').value_counts().to_dict()
        or summary['cross_counts']!=cross or summary['all_candidates_retained'] is not True
        or summary['formal_training_authorized'] is not False
        or summary['feature_economic_semantics_proven'] is not False):
        raise ValueError('assembly summary semantics mismatch')
    check()
    return saved, dict(replayed=True, rows=len(saved), formal_training_authorized=False,
        validation_scope='feature computation and assembly replay; label integrity/consistency only, not independent label path recomputation',
        historical_code_replayed=False, replay_code_sha256=digest(Path(__file__)))


def build(root, label_dir, label_sha, feature_dir, feature_sha):
    root,label_dir,feature_dir=Path(root).resolve(),Path(label_dir).resolve(),Path(feature_dir).resolve()
    result,label_check,pins,observed=reconstruct(root,label_dir,label_sha,feature_dir,feature_sha)
    out=root/'output/experiments/s20_safe_v4/sources'/('dataset-assembly-'+uuid.uuid4().hex); out.mkdir(parents=True)
    result.to_parquet(out/'samples.parquet',index=False)
    atomic_json(out/'inputs.json',dict(label_directory=str(label_dir),label_summary_sha256=label_sha,
        feature_directory=str(feature_dir),feature_summary_sha256=feature_sha,
        source_pins={str(p):h for p,h in pins.items()},observed_at=observed,code_sha256=digest(Path(__file__))))
    report=dict(directory=str(out),at=now(),rows=len(result),all_candidates_retained=True,
        cross_counts=result.groupby(['all_feature_values_present','diagnostic_label_resolved']).size().rename('rows').reset_index().to_dict('records'),
        class_counts=result.p_class.fillna('UNKNOWN').value_counts().to_dict(),label_validation=label_check,
        feature_computation_reconstructed=True,observation_basis='verified now; not backdated historical receipt',
        feature_economic_semantics_proven=False,formal_training_authorized=False,
        artifacts={n:digest(out/n) for n in ['samples.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
