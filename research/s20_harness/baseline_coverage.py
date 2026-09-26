"""E00/E01 diagnostic inventory, not baseline equivalence or formal coverage.

Model and fixed-policy identities verified by the parent campaign are inventoried.
Required label-track/market/fold scope and legacy metric reproduction need
separate evidence; a family name cannot establish either.
"""
REQUIRED=('legacy_r20','legacy_s20','legacy_s20_v3','mature_frequency',
          'logistic','shallow_tree','atr_liquidity_control')


def inventory(cards):
    if not cards: raise ValueError('baseline coverage needs nonempty job cards')
    groups={};seen=set()
    for card in cards:
        key=(card['fold_id'],card['target_id'])
        if card['job_id'] in seen: raise ValueError('duplicate baseline coverage job')
        seen.add(card['job_id'])
        group=groups.setdefault(key,[]);group.append(card)
    rows=[]
    for (fold,target),members in sorted(groups.items()):
        def matches(card, family):
            if family == 'atr_liquidity_control':
                return (card.get('policy_mode') == family
                        and isinstance(card.get('policy_sha256'), str)
                        and len(card['policy_sha256']) == 64
                        and all(v in '0123456789abcdef' for v in card['policy_sha256']))
            return card['model_family'] == family
        represented={family for family in REQUIRED if any(matches(c,family) for c in members)}
        rows.append(dict(fold_id=fold,target_id=target,jobs=[c['job_id'] for c in members],
            family_inventory=[dict(family=family,job_ids=[c['job_id'] for c in members if matches(c,family)],
                status='represented_diagnostic' if family in represented else 'missing') for family in REQUIRED],
            missing_families=[family for family in REQUIRED if family not in represented],
            other_families=sorted({c['model_family'] for c in members}-set(REQUIRED)),
            policy_inventory=[dict(job_id=c['job_id'],model_family=c['model_family'],
                policy_mode=c.get('policy_mode'),policy_sha256=c.get('policy_sha256')) for c in members],
            evidence_modes=sorted({c['evidence_mode'] for c in members})))
    return dict(required_families=list(REQUIRED),observed_scope=rows,
        all_observed_groups_represent_required_families=all(not r['missing_families'] for r in rows),
        required_track_market_fold_scope_verified=False,legacy_original_metrics_reproduced=False,
        same_universe_and_coverage_comparability_verified=False,formal_H03_accepted=False,
        scope='verified parent job inventory only; missing tasks explicit, represented is not accepted')
