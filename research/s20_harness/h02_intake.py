"""H02 diagnostic artifact intake. Formal semantic acceptance is not implemented."""
from pathlib import Path
import pandas as pd

from .runtime import atomic_json,digest,load_plan

OUTPUTS=('label_contracts.json','labels.parquet','label_transfer.csv','execution_contract.json','golden_cases.json')


def validate_golden(root,source):
    from .execution_golden import replay_current
    inputs=load_plan(source/'inputs.json')
    parent=(root/inputs['golden_directory']).resolve()
    if not parent.is_relative_to(root): raise ValueError('H02 golden source escapes root')
    expected=inputs['golden_summary_sha256']
    if digest(parent/'summary.json')!=expected: raise ValueError('H02 golden summary mismatch')
    summary=load_plan(parent/'summary.json')
    if set(summary['artifacts'])!={'golden_cases.json','inputs.json'}:
        raise ValueError('H02 golden artifact set mismatch')
    pins={str(parent/'summary.json'):expected}
    pins.update({str(parent/n):h for n,h in summary['artifacts'].items()})
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('H02 golden artifacts changed')
    if digest(source/'golden_cases.json')!=summary['artifacts']['golden_cases.json']:
        raise ValueError('H02 golden copy differs from parent')
    result=replay_current(parent,expected)
    if not result['all_passed']: raise ValueError('H02 execution case failure')
    pins.update(result['current_code_pins'])
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('H02 golden changed during replay')
    return dict(**result,source_pins=pins)


def validate_parents(root, source, protocol_hash):
    """Bind inherited rows and declared rules; not independent upstream replay."""
    inputs=load_plan(source/'inputs.json')
    pins={}
    def pin(path, expected):
        path=(root/Path(path)).resolve()
        if not path.is_relative_to(root) or digest(path)!=expected:
            raise ValueError('H02 parent path/hash mismatch')
        pins[str(path)]=expected
        return path
    plan_path=pin(inputs['plan_path'],inputs['plan_sha256'])
    if plan_path!=(root/'config/s20_v4_harness_plan.json').resolve():
        raise ValueError('H02 requires active frozen plan')
    plan=load_plan(plan_path)
    import hashlib
    from .runtime import canonical
    if hashlib.sha256(canonical(plan)).hexdigest()!=protocol_hash:
        raise ValueError('H02 active protocol hash mismatch')
    labels_contract=load_plan(source/'label_contracts.json')
    execution_contract=load_plan(source/'execution_contract.json')
    if (labels_contract.get('frozen_contracts')!=plan['label_tracks']
            or execution_contract.get('contract')!=plan['label_tracks']['execution_book_v4']
            or labels_contract.get('formal_training_authorized') is not False
            or execution_contract.get('formal_training_authorized') is not False
            or execution_contract.get('actual_execution_evidence') is not False):
        raise ValueError('H02 frozen rules or diagnostic claims mismatch')
    collection=(root/inputs['collection']).resolve()
    summary=load_plan(pin(collection/'summary.json',inputs['collection_sha256']))
    if set(summary['artifacts'])!={'samples.parquet','inputs.json'}:
        raise ValueError('H02 parent collection artifact set mismatch')
    for name,sha in summary['artifacts'].items(): pin(collection/name,sha)
    parent=pd.read_parquet(collection/'samples.parquet')
    child=pd.read_parquet(source/'labels.parquet')
    if not set(parent.columns).issubset(child):
        raise ValueError('H02 inherited collection fields missing')
    pd.testing.assert_frame_equal(child[parent.columns],parent,check_exact=True)
    refs=inputs.get('opportunity_partitions')
    if not isinstance(refs,list) or not refs:
        raise ValueError('H02 opportunity parent bindings required')
    opportunity=[]
    for ref in refs:
        if set(ref)!={'directory','summary_sha256'}:
            raise ValueError('H02 invalid opportunity parent binding')
        part=(root/ref['directory']).resolve()
        part_summary=load_plan(pin(part/'summary.json',ref['summary_sha256']))
        if set(part_summary['artifacts'])!={'labels.parquet','op_transfer.csv','inputs.json'}:
            raise ValueError('H02 opportunity artifact set mismatch')
        for name,sha in part_summary['artifacts'].items(): pin(part/name,sha)
        opportunity.append(pd.read_parquet(part/'labels.parquet'))
    combined=pd.concat(opportunity,ignore_index=True)
    columns=['sample_id','entity_id','signal_date','p_class','o_class','o_label_realized',
             'o_reason','o_safe_opportunity','o_payload_json']
    if not set(columns).issubset(combined) or not set(columns).issubset(child):
        raise ValueError('H02 complete opportunity fields required')
    pd.testing.assert_frame_equal(child[columns],combined[columns],check_exact=True)
    from .opportunity_partition import replay
    path_checks=[replay(root,root/ref['directory'],ref['summary_sha256']) for ref in refs]
    # P and O are different targets. O replay's P integrity check is not a P
    # price-path recomputation, so invoke the existing P replay separately.
    from .label_partition_replay import replay as replay_p
    p_refs=set()
    for ref in refs:
        op_inputs=load_plan(root/ref['directory']/'inputs.json')
        p_path=(root/op_inputs['partition']).resolve()
        if not p_path.is_relative_to(root): raise ValueError('H02 P source escapes root')
        p_refs.add((str(p_path),op_inputs['partition_sha256']))
    compat_refs=inputs.get('compatibility_partitions')
    if not isinstance(compat_refs,list) or not compat_refs:
        raise ValueError('H02 compatibility bindings required')
    compat_frames=[];compat_parents=[]
    for ref in compat_refs:
        if set(ref)!={'directory','summary_sha256'}:
            raise ValueError('H02 invalid compatibility binding')
        part=(root/ref['directory']).resolve()
        meta=load_plan(pin(part/'summary.json',ref['summary_sha256']))
        if set(meta['artifacts'])!={'labels.parquet','inputs.json'}:
            raise ValueError('H02 compatibility artifact set mismatch')
        for name,sha in meta['artifacts'].items(): pin(part/name,sha)
        cp=load_plan(part/'inputs.json')
        compat_parents.append((str((root/cp['partition']).resolve()),cp['partition_sha256']))
        compat_frames.append(pd.read_parquet(part/'labels.parquet'))
    if len(set(compat_parents))!=len(compat_parents) or set(compat_parents)!=p_refs:
        raise ValueError('H02 compatibility/P parent mismatch')
    compat=pd.concat(compat_frames,ignore_index=True)
    compat_columns=['sample_id','entity_id','signal_date','compat_class','compat_reason','compat_entry_date',
        'compat_horizon_end','compat_payload_json','different_horizon','compat_label_realized',
        'raw_calendar_class','raw_calendar_reason','raw_calendar_payload_json']
    if not set(compat_columns).issubset(child) or not set(compat_columns).issubset(compat):
        raise ValueError('H02 compatibility fields missing')
    pd.testing.assert_frame_equal(child[compat_columns],compat[compat_columns],check_exact=True)
    from .stock_session_compat import replay as replay_compat
    compat_checks=[replay_compat(root,root/r['directory'],r['summary_sha256']) for r in compat_refs]
    p_path_checks=[replay_p(root,path,sha) for path,sha in sorted(p_refs)]
    for path,sha in pins.items():
        if digest(Path(path))!=sha: raise ValueError('H02 parent changed during check')
    return dict(parent_rows=len(parent),inherited_collection_rows_verified=True,
                opportunity_rows_verified=True,opportunity_partitions=len(refs),
                opportunity_path_checks=path_checks,
                profit_path_checks=p_path_checks,
                compatibility_path_checks=compat_checks,
                frozen_contracts_verified=True,parent_path_replay=False,source_pins=pins,
                formal_training_authorized=False)


def validate_labels(source, summary):
    """Reconstruct label consistency and transfer counts, not future paths."""
    labels = pd.read_parquet(source/'labels.parquet')
    keys = ['sample_id', 'entity_id', 'signal_date']
    required = set(keys + ['p_class', 'o_class', 'o_label_realized', 'o_safe_opportunity', 'o_reason'])
    if not required.issubset(labels) or labels.columns.duplicated().any() or labels.empty:
        raise ValueError('H02 label schema required')
    if labels[keys].isna().any().any() or labels.sample_id.duplicated().any():
        raise ValueError('H02 unique nonmissing candidates required')
    if not labels.p_class.dropna().isin(['A','B','C','D']).all():
        raise ValueError('H02 invalid P class')
    if not labels.o_class.dropna().isin([-1,0,1,2,3,4,5]).all():
        raise ValueError('H02 invalid O class')
    resolved = labels.o_class.ge(0).fillna(False)
    if labels.o_label_realized.isna().any() or not labels.o_label_realized.eq(resolved).all():
        raise ValueError('H02 O maturity/class mismatch')
    safe = labels.o_class.isin([0,1])
    if (not labels.o_safe_opportunity.isna().eq(~resolved).all()
            or not labels.loc[resolved,'o_safe_opportunity'].eq(safe[resolved]).all()):
        raise ValueError('H02 O success/unknown mismatch')
    if labels.o_reason.isna().any():
        raise ValueError('H02 outcome reason required')
    transfer = pd.crosstab(labels.o_reason,labels.p_class.fillna('UNKNOWN'),dropna=False)
    saved = pd.read_csv(source/'label_transfer.csv',index_col=0)
    saved.columns.name = transfer.columns.name  # CSV does not retain column-axis name.
    pd.testing.assert_frame_equal(saved,transfer,check_exact=True)
    counts = dict(rows=len(labels), dates=sorted(labels.signal_date.unique().tolist()),
                  o_resolved_rows=int(resolved.sum()),o_safe_rows=int(safe.sum()))
    if any(summary.get(k)!=v for k,v in counts.items()):
        raise ValueError('H02 label summary counts mismatch')
    return dict(**counts, label_consistency_verified=True,transfer_counts_reconstructed=True,
                label_path_independently_recomputed=False,formal_training_authorized=False)


def audit(root,directory,protocol_hash):
    root,directory=Path(root).resolve(),Path(directory).resolve()
    binding_path=root/'config/s20_v4_h02_intake.json'
    if not binding_path.is_file(): raise ValueError('missing pinned H02 intake binding')
    binding_sha=digest(binding_path);binding=load_plan(binding_path)
    if set(binding)!={'protocol_hash','directory','summary_sha256'} or binding['protocol_hash']!=protocol_hash:
        raise ValueError('H02 intake protocol/binding mismatch')
    source=(root/binding['directory']).resolve()
    if not source.is_relative_to(root): raise ValueError('H02 intake source escapes workspace')
    if digest(source/'summary.json')!=binding['summary_sha256']: raise ValueError('H02 preparation pin mismatch')
    summary=load_plan(source/'summary.json')
    if (summary.get('status')!='DIAGNOSTIC_PREPARATION' or summary.get('formal_gate_passed') is not False
        or summary.get('formal_training_authorized') is not False or not (set(OUTPUTS)|{'inputs.json'}).issubset(summary['artifacts'])):
        raise ValueError('H02 preparation contract mismatch')
    def check():
        if digest(binding_path)!=binding_sha or digest(source/'summary.json')!=binding['summary_sha256']:
            raise ValueError('H02 intake source binding changed')
        for name,sha in summary['artifacts'].items():
            path=(source/name).resolve()
            if not path.is_relative_to(source) or digest(path)!=sha:
                raise ValueError('H02 preparation artifact changed')
    check()
    label_checks=validate_labels(source,summary)
    golden_checks=validate_golden(root,source)
    parent_checks=validate_parents(root,source,protocol_hash)
    check()
    directory.mkdir(parents=True,exist_ok=True)
    for name in OUTPUTS:
        (directory/name).write_bytes((source/name).read_bytes())
        if digest(directory/name)!=summary['artifacts'][name]: raise ValueError('H02 copied artifact mismatch')
    check()
    # An intake verifier must never promote a diagnostic summary into acceptance.
    for path,sha in parent_checks['source_pins'].items():
        if digest(Path(path))!=sha: raise ValueError('H02 parent changed during intake')
    for path,sha in golden_checks['source_pins'].items():
        if digest(Path(path))!=sha: raise ValueError('H02 golden changed during intake')
    gaps=list(summary.get('acceptance_gaps',[]))
    gaps.append('H02 formal acceptance not implemented; label consistency is not independent path replay')
    result=dict(formal_gate_passed=False,acceptance_gaps=gaps,source_directory=str(source),
        source_summary_sha256=binding['summary_sha256'],binding_sha256=binding_sha,
        output_hashes={name:digest(directory/name) for name in OUTPUTS},
        label_checks=label_checks,parent_checks=parent_checks,golden_checks=golden_checks,
        formal_training_authorized=False,validation_scope='artifact/parent integrity, label consistency and current-code O/P path replay; not independent algorithm validation or H02 acceptance')
    atomic_json(directory/'h02_intake.json',result)
    return result
