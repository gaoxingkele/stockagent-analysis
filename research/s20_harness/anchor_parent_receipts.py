"""Bind supplied calibrated anchors to pinned saved baseline parent outputs."""
from pathlib import Path
import pandas as pd
from .baseline_run import verify
from .runtime import digest,load_plan
from .label_availability import _instant


def verify_budget_binding(root,binding):
    from .trial_budget import verify_attempt
    from .process_runner import Limits
    ref=binding['budget_reference']
    if not isinstance(ref,dict) or set(ref)!={'path','contract_sha256','trial_id','attempt_id','input_path','input_sha256','artifact','limits'}:
        raise ValueError('exact parent budget reference required')
    if Path(ref['artifact']['directory']).resolve()!=Path(binding['directory']).resolve() or ref['artifact']['summary_sha256']!=binding['summary_sha256']:
        raise ValueError('parent budget points to another model')
    result=verify_attempt(root,ref['path'],ref['contract_sha256'],ref['trial_id'],ref['attempt_id'],
        ref['input_path'],ref['input_sha256'],ref['artifact'],Limits(**ref['limits']))
    # The global ledger may legitimately grow. Freeze the checked attempt,
    # not aggregate counts that would change when another trial is reserved.
    result.pop('reserved_counts_at_review')
    return result


def bind(root,context,bindings):
    scope=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'
    fields={'model_id','directory','summary_sha256'}
    if not isinstance(bindings,list) or any(not isinstance(b,dict) or set(b) not in (fields,fields|{'budget_reference'}) for b in bindings):
        raise ValueError('exact anchor parent bindings required')
    if len(bindings)!=len(context['models']) or {b['model_id'] for b in bindings}!=set(context['models']):
        raise ValueError('one parent binding per anchor model required')
    pins={};count=0;budget_evidence=[]
    for binding in bindings:
        directory=Path(binding['directory']).resolve();mid=binding['model_id']
        if not directory.is_relative_to(scope): raise ValueError('anchor parent outside research scope')
        verify(directory,binding['summary_sha256'])
        summary=load_plan(directory/'summary.json');plan=load_plan(directory/'input.json')
        if 'anchor_context' in plan:
            raise ValueError('nested parent requires recursive receipt verification')
        pins[str(directory/'summary.json')]=binding['summary_sha256']
        pins.update({str(directory/n):h for n,h in summary['artifacts'].items()})
        model=context['models'][mid]
        if Path(model['model_path']).resolve()!=directory/'calibration_card.json' or model['model_sha256']!=digest(directory/'calibration_card.json'):
            raise ValueError('anchor model not bound to parent calibrator')
        base=load_plan(directory/'baseline_card.json');cal=load_plan(directory/'calibration_card.json')
        dependencies=base['fit_sample_ids']+cal['calibration_sample_ids']
        if sorted(context['dependencies'][mid])!=sorted(dependencies):
            raise ValueError('anchor parent label dependency mismatch')
        if _instant(model['information_cutoff_at'])<_instant(plan['boundaries'][3]['start_at']):
            raise ValueError('anchor cutoff precedes parent calibration boundary')
        supplied=context['predictions'].loc[context['predictions'].model_id.eq(mid)]
        saved=pd.read_parquet(directory/'calibrated_predictions.parquet').set_index('sample_id')
        if not set(supplied.sample_id).issubset(saved.index): raise ValueError('unknown parent prediction')
        actual=saved.loc[supplied.sample_id]
        if not actual.segment.isin(['selection-policy','outer-test']).all() or actual.calibrated_probability.isna().any():
            raise ValueError('anchor must be a usable forward parent prediction')
        pd.testing.assert_series_equal(supplied.score.reset_index(drop=True),
            actual.calibrated_probability.reset_index(drop=True),check_names=False,check_dtype=False,check_exact=True)
        ids=list(dict.fromkeys(dependencies+supplied.sample_id.tolist()))
        columns=['entity_id','prediction_at','feature_available_at','horizon_close_at','label_available_at']
        pd.testing.assert_frame_equal(context['samples'].set_index('sample_id').loc[ids,columns],
            pd.DataFrame(plan['samples']).set_index('sample_id').loc[ids,columns],check_dtype=False)
        verify(directory,binding['summary_sha256']);count+=len(supplied)
        if 'budget_reference' in binding:
            budget_evidence.append(dict(model_id=mid,**verify_budget_binding(root,binding)))
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('anchor parent source changed')
    return dict(bound_parent_models=len(bindings),bound_prediction_rows=count,source_pins=pins,
        saved_prediction_binding_verified=True,model_inference_independently_replayed=False,
        upstream_fit_budget_verified=bool(bindings) and len(budget_evidence)==len(bindings),
        budget_attempts=budget_evidence,formal_training_authorized=False)
