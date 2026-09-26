"""Audit each outer prediction's recorded upstream label dependency union.

Anchor manifests must list all their labeled dependencies; this cannot discover
hidden dependencies or authenticate caller-provided training membership.
"""
import pandas as pd
from .anchor_features import effective_samples
from .oof_audit import audit,membership_hash
from .label_availability import _instant
from .runtime import digest


def audit_outer(plan,base,cal,outer,artifact,cutoff):
    context=dict(plan['anchor_context'])
    context['samples']=pd.DataFrame(context['samples'])
    context['predictions']=pd.DataFrame(context['predictions'],columns=['sample_id','model_id','score'])
    effective,report=effective_samples(pd.DataFrame(plan['samples']),context)
    if report!=base['preprocessing']['anchor_admission']:
        raise ValueError('outer anchor evidence changed')
    checked,_=audit(context['samples'],context['predictions'],context['models'],context['dependencies'])
    valid=set(checked.loc[checked.recorded_oof_provenance_valid,'sample_id']) if len(checked) else set()
    labeled=base['fit_sample_ids']+cal['calibration_sample_ids']
    if not set(labeled).issubset(valid):
        raise ValueError('downstream labeled sample used invalid anchor')
    mapping=context['predictions'].set_index('sample_id').model_id.to_dict()
    common_models={mapping[s] for s in labeled}
    history=context['samples'].set_index('sample_id').copy()
    for field in ['feature_available_at','label_available_at']:
        history.loc[effective.sample_id,field]=effective[field].to_numpy()
    history=history.reset_index()
    models={};dependencies={};predictions=[];details=[]
    for row in outer.itertuples(index=False):
        used=common_models | ({mapping[row.sample_id]} if row.sample_id in valid else set())
        ids=set(labeled)
        times=[_instant(cutoff)]
        for model_id in used:
            ids.update(context['dependencies'][model_id])
            times.append(_instant(context['models'][model_id]['information_cutoff_at']))
        ids=sorted(ids);instant=max(times).isoformat();key=row.sample_id
        models[key]=dict(model_path=str(artifact),model_sha256=digest(artifact),
            information_cutoff_at=instant,dependency_sha256=membership_hash(ids))
        dependencies[key]=ids
        predictions.append(dict(sample_id=key,model_id=key,score=row.calibrated_probability))
        details.append(dict(information_cutoff_at=instant,recorded_label_dependency_count=len(ids),
            recorded_anchor_model_count=len(used),recorded_dependency_sha256=membership_hash(ids)))
    result,_=audit(history,pd.DataFrame(predictions),models,dependencies)
    return result,pd.DataFrame(details)
