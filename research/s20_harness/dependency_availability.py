"""Role-aware dependency timing; a declared receipt is not provenance proof."""
import pandas as pd

from .label_availability import _instant


def resolve(samples, contract, receipts):
    """Keep all samples; derive times only from the full declared dependency set.

    Feature inputs must precede scoring. Labels may arrive afterwards and are
    additionally bounded below by horizon close. Historical training cutoff is
    enforced later by splits; it is not the original prediction timestamp.
    """
    sample_fields = {'sample_id', 'prediction_at', 'horizon_close_at'}
    keys = ['sample_id', 'role', 'dependency_id']
    if not sample_fields.issubset(samples) or samples.sample_id.isna().any() or samples.sample_id.duplicated().any():
        raise ValueError('unique sample identities and timing required')
    if set(contract.columns) != set(keys) or set(receipts.columns) != set(keys+['available_at', 'basis']):
        raise ValueError('exact dependency contract and receipt schema required')
    for table in (contract, receipts):
        if table[keys].isna().any().any() or table.duplicated(keys).any():
            raise ValueError('unique dependency identities required')
        if not table.role.isin(['feature', 'label']).all():
            raise ValueError('unknown dependency role')
        if not table.sample_id.isin(samples.sample_id).all():
            raise ValueError('dependency sample outside universe')
        if not table.dependency_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('named dependencies required')
    # Column order must not alter identity matching.
    declared = set(contract[keys].itertuples(index=False, name=None))
    if not set(receipts[keys].itertuples(index=False, name=None)).issubset(declared):
        raise ValueError('receipt not declared for this role')
    if not receipts.basis.isin(['observed_now', 'historical_receipt', 'unknown']).all():
        raise ValueError('unsupported receipt basis')
    for r in receipts.itertuples(index=False):
        if r.basis == 'unknown' and pd.notna(r.available_at):
            raise ValueError('unknown receipt cannot assert availability')
        if pd.notna(r.available_at):
            _instant(r.available_at)
    ledger = contract.merge(receipts, on=keys, how='left', validate='one_to_one')
    ledger['basis'] = ledger.basis.fillna('unknown')
    derived, states = samples.copy(), []
    feature_times, label_times = [], []
    for sample in samples.itertuples(index=False):
        prediction, horizon = _instant(sample.prediction_at), _instant(sample.horizon_close_at)
        if horizon <= prediction:
            raise ValueError('horizon must follow prediction')
        role_times = {}
        for role in ('feature', 'label'):
            selected = ledger.loc[ledger.sample_id.eq(sample.sample_id) & ledger.role.eq(role)]
            missing = selected.empty or selected.available_at.isna().any()
            at = None if missing else max(_instant(v) for v in selected.available_at)
            if at is not None and role == 'label':
                at = max(at, horizon)
            role_times[role] = at.isoformat() if at is not None else None
            states.append(dict(sample_id=sample.sample_id, role=role, dependencies=len(selected),
                availability_complete=not missing, derived_available_at=role_times[role],
                feature_prior_to_prediction=(at < prediction) if role == 'feature' and at is not None else None))
        feature_times.append(role_times['feature'])
        label_times.append(role_times['label'])
    # Ignore optimistic summary timestamps: derive them again from dependencies.
    derived['feature_available_at'] = feature_times
    derived['label_available_at'] = label_times
    report = dict(rows=len(samples), declared_dependencies=len(contract), supplied_receipts=len(receipts),
                  all_samples_retained=True, dependency_completeness_independently_proven=False,
                  receipt_provenance_independently_verified=False, formal_training_authorized=False,
                  role_diagnostics=states)
    return derived, ledger, report
