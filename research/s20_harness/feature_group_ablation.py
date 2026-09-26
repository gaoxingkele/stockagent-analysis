"""Explicit diagnostic feature-group controls with identical non-feature scope."""
import copy
import hashlib
from .trial_budget import canonical
from .joint_run import pipeline_costs
from .baseline_model import validate_seed


def contrasts(contract,grouped):
    from .configuration_comparison import compare_cells
    arms=contract['arms']
    pairs=[('augmented','base'),('augmented','noise_control'),('noise_control','base')]
    return dict(comparisons=[dict(left_arm=a,right_arm=b,left_candidate_id=arms[a],right_candidate_id=arms[b],
        **compare_cells(grouped[arms[a]],grouped[arms[b]])) for a,b in pairs],
        conditional_feature_increment_proven=False,noise_is_distribution_matched=False,
        orthogonal_information_proven=False,formal_H06_accepted=False)


def validate_registered_cells(contract,cells,baseline_id):
    required={'groups','base_groups','added_group','noise_seed','arms'}
    if isinstance(contract,dict) and 'control_strata' in contract:required.add('control_strata')
    if not isinstance(contract,dict) or set(contract)!=required or set(contract['arms'])!={'base','augmented','noise_control'}:
        raise ValueError('exact three-arm feature ablation contract required')
    arms=contract['arms']
    if len(set(arms.values()))!=3 or arms['base']!=baseline_id:
        raise ValueError('distinct ablation arms and base reference required')
    if 'control_strata' in contract:
        from .feature_strata import validate
        validate(contract['control_strata'],[c for g in contract['base_groups'] for c in contract['groups'][g]])
    for members in cells.values():
        if set(members)!=set(arms.values()):raise ValueError('complete ablation arms per fold/seed required')
        expected,_=plans(members[arms['augmented']],contract['groups'],contract['base_groups'],contract['added_group'],noise_seed=contract['noise_seed'])
        for name,cid in arms.items():
            if canonical(expected[name])!=canonical(members[cid]):
                raise ValueError('registered feature ablation reconstruction mismatch: '+name)


def plans(source,groups,base_groups,added_group,*,noise_seed=20):
    """Return three candidate plans; never launch or silently allocate fits."""
    if source.get('target_id')!='P.joint.v4' or source.get('evidence_mode')!='synthetic':
        raise ValueError('synthetic joint source required pending formal feature admission')
    seed=validate_seed(noise_seed)
    columns=source['feature_contract']['columns']
    if not isinstance(groups,dict) or not groups or any(not isinstance(k,str) or not k for k in groups):
        raise ValueError('named feature groups required')
    if any(not isinstance(v,list) or not v or any(not isinstance(c,str) or not c for c in v) for v in groups.values()):
        raise ValueError('nonempty named feature lists required')
    flattened=[c for values in groups.values() for c in values]
    if len(set(flattened))!=len(flattened) or set(flattened)!=set(columns):
        raise ValueError('disjoint exhaustive feature groups required')
    if not isinstance(base_groups,list) or not base_groups or len(set(base_groups))!=len(base_groups) or not set(base_groups)<=set(groups):
        raise ValueError('explicit unique base groups required')
    if added_group not in groups or added_group in base_groups:
        raise ValueError('distinct added group required')
    baseline=[c for c in columns if any(c in groups[g] for g in base_groups)]
    added=groups[added_group];combined=[c for c in columns if c in baseline or c in added]
    records=source['features']
    if any(set(row)!={'sample_id',*columns} for row in records) or len({r['sample_id'] for r in records})!=len(records):
        raise ValueError('exact unique source feature records required')
    variants={}
    for name,chosen in [('base',baseline),('augmented',combined),('noise_control',combined)]:
        value=copy.deepcopy(source);value['feature_contract']['columns']=chosen
        value['features']=[{k:r[k] for k in ['sample_id',*chosen]} for r in records]
        if name=='noise_control':
            for row in value['features']:
                for col in added:
                    key=canonical(dict(seed=seed,sample_id=row['sample_id'],feature=col)).encode()
                    # Independent keyed pseudo-uniform control: no observed value,
                    # price, label, future row or row order participates.
                    row[col]=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')/2**64*2-1
            value['feature_contract']['noise_control']=dict(columns=added,seed=seed,
                generator='sha256_keyed_uniform_v1',market_independent=True)
        variants[name]=value
    costs=[pipeline_costs(v) for v in variants.values()]
    return variants,dict(groups=groups,base_groups=base_groups,added_group=added_group,
        candidate_costs={n:pipeline_costs(v) for n,v in variants.items()},
        total_initial_costs={k:sum(c[k] for c in costs) for k in costs[0]},
        non_feature_scope_fixed=True,availability_policy='shared conservative source sample availability',
        noise_is_distribution_matched=False,independent_information_proven=False,
        budget_reserved=False,formal_H06_accepted=False)
