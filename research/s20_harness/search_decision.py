"""Bounded diagnostic shortlist; never a formal H04 or production promotion."""
from pathlib import Path
from itertools import combinations
import uuid

from .search_candidates import inspect
from .configuration_comparison import compare_cells
from .runtime import atomic_json,digest,load_plan
from .trial_budget import canonical


def decide(summary):
    if summary['registered_rule']=='all_registered_nondominated_no_automatic_tiebreak':
        return decide_frontier(summary)
    if summary['registered_rule']!='matched_cell_robust_dominance_no_automatic_tiebreak':
        raise ValueError('unsupported registered decision rule')
    cap=summary['registered_finalist_cap']
    if type(cap)is not int or not 1<=cap<=3:raise ValueError('bounded finalist cap required')
    ids=summary['registered_candidate_ids'];baseline=summary['baseline_id']
    if len(ids)!=len(set(ids)) or baseline not in ids:raise ValueError('unique candidates and baseline required')
    grouped={cid:[r for r in summary['cells'] if r['candidate_id']==cid] for cid in ids}
    if sum(len(v) for v in grouped.values())!=len(summary['cells']):raise ValueError('unregistered decision cell')
    contrasts=[];eligible=[];unresolved=[]
    for cid in ids:
        if cid==baseline:continue
        contrast=compare_cells(grouped[cid],grouped[baseline]);contrasts.append(dict(candidate_id=cid,**contrast))
        if contrast['dominant']=='left':eligible.append(cid)
        elif contrast['status']=='INCOMPARABLE_COVERAGE':unresolved.append(cid)
    pairs=[];dominated=set()
    for left,right in combinations(eligible,2):
        contrast=compare_cells(grouped[left],grouped[right]);pairs.append(dict(left=left,right=right,**contrast))
        if contrast['dominant']=='left':dominated.add(right)
        elif contrast['dominant']=='right':dominated.add(left)
    survivors=[cid for cid in eligible if cid not in dominated]
    if len(survivors)>cap:
        status='UNRESOLVED_CAPACITY';shortlist=[]
    elif survivors:
        status='DIAGNOSTIC_SHORTLIST';shortlist=survivors
    else:
        status='NO_ROBUST_CANDIDATE';shortlist=[]
    return dict(status=status,diagnostic_shortlist=shortlist,eligible_against_baseline=eligible,
        non_dominated_eligible=survivors,coverage_unresolved=unresolved,baseline_contrasts=contrasts,
        eligible_pairwise=pairs,registered_finalist_cap=cap,automatic_tiebreak_used=False,
        formal_finalists_selected=False,formal_H04_accepted=False,production_promotion_authorized=False,
        evidence='descriptive unknown-outcome bounds, not sampling confidence intervals')


def decide_frontier(summary):
    """Retain registered controls in the comparison; no superiority prerequisite."""
    cap=summary['registered_finalist_cap'];ids=summary['registered_candidate_ids']
    if type(cap)is not int or not 1<=cap<=3:raise ValueError('bounded finalist cap required')
    if not ids or len(ids)!=len(set(ids)) or summary['baseline_id'] not in ids:
        raise ValueError('unique registered candidates and baseline required')
    grouped={cid:[r for r in summary['cells'] if r['candidate_id']==cid] for cid in ids}
    if sum(map(len,grouped.values()))!=len(summary['cells']):raise ValueError('unregistered frontier cell')
    for cid in ids:compare_cells(grouped[cid],grouped[cid])
    empty=[cid for cid in ids if any(not r['selected'] for r in grouped[cid])]
    eligible=[cid for cid in ids if cid not in empty];pairs=[];dominated=set();incomparable=False
    for left,right in combinations(eligible,2):
        contrast=compare_cells(grouped[left],grouped[right]);pairs.append(dict(left=left,right=right,**contrast))
        incomparable |= contrast['status']=='INCOMPARABLE_COVERAGE'
        if contrast['dominant']=='left':dominated.add(right)
        elif contrast['dominant']=='right':dominated.add(left)
    survivors=[cid for cid in eligible if cid not in dominated]
    if empty or incomparable:status='UNRESOLVED_COMPARABILITY';shortlist=[]
    elif len(survivors)>cap:status='UNRESOLVED_CAPACITY';shortlist=[]
    else:status='DIAGNOSTIC_FRONTIER';shortlist=survivors
    return dict(status=status,diagnostic_shortlist=shortlist,non_dominated_registered=survivors,
        registered_pairwise=pairs,empty_cell_candidates=empty,baseline_in_comparison=True,
        registered_finalist_cap=cap,automatic_tiebreak_used=False,
        formal_finalists_selected=False,formal_H04_accepted=False,production_promotion_authorized=False,
        evidence='descriptive registered frontier; baseline controls not excluded by strict-superiority screening')


def build(root,directory,sha):
    summary=inspect(root,directory,sha);decision=decide(summary)
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('search-decision-'+uuid.uuid4().hex)
    out.mkdir(parents=True);atomic_json(out/'decision.json',decision)
    report=dict(directory=str(out),evaluation_directory=str(Path(directory).resolve()),evaluation_summary_sha256=sha,
        code_sha256=digest(Path(__file__)),artifacts={'decision.json':digest(out/'decision.json')},
        models_refit=0,formal_H04_accepted=False)
    atomic_json(out/'summary.json',report)
    verify(root,out,digest(out/'summary.json'))
    return report


def verify(root,directory,sha):
    directory=Path(directory).resolve();scope=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'
    if not directory.is_relative_to(scope) or digest(directory/'summary.json')!=sha:raise ValueError('decision pin/scope mismatch')
    report=load_plan(directory/'summary.json')
    if set(report)!={'directory','evaluation_directory','evaluation_summary_sha256','code_sha256','artifacts','models_refit','formal_H04_accepted'}:
        raise ValueError('exact decision report required')
    if report['directory']!=str(directory) or set(report['artifacts'])!={'decision.json'}:raise ValueError('decision artifact scope mismatch')
    if type(report['models_refit'])is not int or report['models_refit']!=0 or report['formal_H04_accepted'] is not False:
        raise ValueError('unsupported decision claim')
    def check():
        if digest(directory/'summary.json')!=sha or digest(Path(__file__))!=report['code_sha256'] or digest(directory/'decision.json')!=report['artifacts']['decision.json']:
            raise ValueError('decision source/artifact changed')
    check();expected=decide(inspect(root,report['evaluation_directory'],report['evaluation_summary_sha256']))
    if canonical(expected)!=canonical(load_plan(directory/'decision.json')):raise ValueError('decision reconstruction mismatch')
    check()
    return dict(decision_recomputed=True,models_refit=0,formal_H04_accepted=False)
