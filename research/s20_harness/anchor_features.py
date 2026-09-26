"""Admit recorded chronological anchors for downstream fit/tune/calibration.

An invalid or missing anchor remains a candidate with an unavailable feature;
never salvage it using the final refit's in-sample prediction.
"""
import json
import hashlib
from pathlib import Path

import pandas as pd

from .oof_audit import audit
from .runtime import digest


def decode(root, payload):
    """Decode bounded JSON evidence; bind existing models, never fit anchors."""
    required={'samples','predictions','models','dependencies','feature_name'}
    if not isinstance(payload,dict) or set(payload)!=required:
        raise ValueError('exact serialized anchor context required')
    for name in ['samples','predictions']:
        if not isinstance(payload[name],list) or len(payload[name])>100000:
            raise ValueError('bounded serialized anchor rows required')
    if not isinstance(payload['models'],dict) or len(payload['models'])>100:
        raise ValueError('bounded anchor models required')
    if not isinstance(payload['dependencies'],dict) or set(payload['dependencies'])!=set(payload['models']):
        raise ValueError('exact anchor model dependencies required')
    root=Path(root).resolve();pins={}
    for model_id,model in payload['models'].items():
        if not isinstance(model,dict) or set(model)!={'model_path','model_sha256','dependency_sha256','information_cutoff_at'}:
            raise ValueError('exact serialized anchor model required')
        path=Path(model['model_path'])
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError('anchor model must be absolute and workspace scoped')
        if digest(path)!=model['model_sha256']:
            raise ValueError('anchor model pin mismatch')
        pins[path.resolve()]=model['model_sha256']
        ids=payload['dependencies'][model_id]
        if not isinstance(ids,list) or len(ids)>100000 or any(not isinstance(i,str) for i in ids):
            raise ValueError('bounded anchor dependency identities required')
    context=dict(payload)
    context['samples']=pd.DataFrame(payload['samples'])
    context['predictions']=pd.DataFrame(payload['predictions'],columns=['sample_id','model_id','score'])
    if any(not isinstance(r,dict) or set(r)!={'sample_id','model_id','score'} for r in payload['predictions']):
        raise ValueError('exact serialized anchor prediction required')
    return context,pins


def effective_samples(samples, context):
    """Recheck admission before downstream label consumers derive eligibility."""
    if set(context)!={'samples','predictions','models','dependencies','feature_name'}:
        raise ValueError('exact anchor context required')
    columns=['entity_id','prediction_at','feature_available_at','horizon_close_at','label_available_at']
    history=context['samples'].set_index('sample_id')
    pd.testing.assert_frame_equal(samples.set_index('sample_id')[columns],
        history.loc[samples.sample_id,columns],check_dtype=False,check_names=False)
    admitted,report=attach(samples[['sample_id']],context['samples'],context['predictions'],
        context['models'],context['dependencies'],feature_name=context['feature_name'])
    unavailable=set(admitted.loc[~admitted.anchor_provenance_valid,'sample_id'])
    result=samples.copy()
    result.loc[result.sample_id.isin(unavailable),'feature_available_at']=None
    return result,report


def attach(candidates, samples, predictions, models, dependencies, *, feature_name='anchor_score'):
    if not isinstance(feature_name,str) or not feature_name.isidentifier():
        raise ValueError('valid anchor feature name required')
    generated={feature_name,'anchor_provenance_valid','anchor_provenance_reasons'}
    if generated & set(candidates.columns):
        raise ValueError('anchor feature/provenance overwrite forbidden')
    if 'sample_id' not in candidates or candidates.sample_id.isna().any() or candidates.sample_id.duplicated().any():
        raise ValueError('unique downstream candidate identity required')
    if not set(candidates.sample_id).issubset(set(samples.sample_id)):
        raise ValueError('unknown downstream sample')
    if not set(predictions.sample_id).issubset(set(candidates.sample_id)):
        raise ValueError('anchor predictions outside downstream candidates')
    checked,summary=audit(samples,predictions,models,dependencies)
    # audit's empty output has no columns; an empty prediction set is ordinary
    # missing evidence, not permission to drop the entire candidate set.
    records={}
    if len(checked):
        scores=predictions.set_index('sample_id').score
        for row in checked.itertuples(index=False):
            records[row.sample_id]=(float(scores.loc[row.sample_id]) if row.recorded_oof_provenance_valid else None,
                                    bool(row.recorded_oof_provenance_valid),row.reasons)
    out=candidates.copy()
    values=[records.get(s,(None,False,['anchor_prediction_missing'])) for s in out.sample_id]
    out[feature_name]=pd.array([r[0] for r in values],dtype='Float64')
    out['anchor_provenance_valid']=[r[1] for r in values]
    out['anchor_provenance_reasons']=[json.dumps(r[2]) for r in values]
    def exact_score(value):
        try:
            return float(value).hex()
        except (TypeError,ValueError):
            return type(value).__name__+':'+repr(value)
    evidence=json.dumps(dict(samples=samples.to_json(orient='split',date_format='iso'),
        predictions=predictions.to_json(orient='split',date_format='iso'),models=models,
        prediction_scores_exact=[exact_score(v) for v in predictions.score],
        dependencies=dependencies,feature_name=feature_name),sort_keys=True,allow_nan=False)
    return out,dict(candidates=len(out),admitted=int(out.anchor_provenance_valid.sum()),
        unavailable=int((~out.anchor_provenance_valid).sum()),all_candidates_retained=True,
        all_anchor_features_admitted=bool(len(out) and out.anchor_provenance_valid.all()),
        dependency_completeness_proven=False,formal_training_authorized=False,audit=summary,
        recorded_anchor_evidence_sha256=hashlib.sha256(evidence.encode()).hexdigest())
