"""Registered sequential diagnostic fold jobs, with budgeted reuse."""
import json
import hashlib
from pathlib import Path
import re
import uuid

from .baseline_run import verify
from .label_availability import _instant
from .runtime import atomic_json, digest, now
from .trial_budget import Budget
from .bounded_baseline import run as run_baseline, verify_process
from .process_runner import Limits


def comparison_variation(manifest):
    fields={'schema_version','evidence_mode','budget','jobs','limits'}
    if manifest.get('schema_version')=='7':
        fields.update(['comparison_contract','seed_grid','fold_grid'])
        if manifest.get('comparison_contract')!='fixed_model_seed_control':
            raise ValueError('explicit fixed model seed comparison contract required')
        from .baseline_model import validate_seed
        seeds=manifest.get('seed_grid');folds=manifest.get('fold_grid')
        if not isinstance(seeds,list) or not 2<=len(seeds)<=3:
            raise ValueError('seed grid requires two or three explicit seeds')
        for seed in seeds:validate_seed(seed)
        if len(set(seeds))!=len(seeds):raise ValueError('duplicate seed grid')
        if not isinstance(folds,list) or not 1<=len(folds)<=3 or any(not isinstance(f,str) or not f.strip() for f in folds) or len(set(folds))!=len(folds):
            raise ValueError('explicit unique bounded fold grid required')
        variation=['random_seed']
    elif manifest.get('schema_version')=='6':
        fields.add('comparison_contract')
        if manifest.get('comparison_contract')!='source_bound_atr_policy_control':
            raise ValueError('explicit source-bound ATR comparison contract required')
        variation=['policy','policy_controls','policy_control_sources']
    elif manifest.get('schema_version')=='5':
        fields.add('comparison_contract')
        if manifest.get('comparison_contract')!='fixed_atr_policy_control':
            raise ValueError('explicit fixed ATR policy comparison contract required')
        variation=['policy','policy_controls']
    elif manifest.get('schema_version')=='4':
        fields.add('comparison_contract')
        if manifest.get('comparison_contract')!='joint_loss_weight_control':
            raise ValueError('explicit joint loss-weight comparison contract required')
        variation=['model_family','class_weights']
    else:variation=['model_family']
    if set(manifest)!=fields or manifest.get('schema_version') not in ['2','3','4','5','6','7']:
        raise ValueError('exact campaign schema required')
    if manifest['schema_version'] in ['5','6']:
        counts={}
        for job in manifest['jobs']:
            fold=job['fold_id'];counts[fold]=counts.get(fold,0)+1
        if any(n>6 for n in counts.values()):
            raise ValueError('fixed policy comparison exceeds six policies per fold')
    return variation


def check_seed_grid(manifest, plans):
    """One registered attempt per declared fold/seed; no implicit repeat evidence."""
    if manifest.get('schema_version')!='7':return
    from .baseline_model import validate_seed
    if len(plans)!=len(manifest['jobs']):raise ValueError('seed grid input cardinality mismatch')
    expected={(fold,seed) for fold in manifest['fold_grid'] for seed in manifest['seed_grid']}
    observed=set();model_specs=set()
    for job,plan in zip(manifest['jobs'],plans):
        if 'random_seed' not in plan:raise ValueError('seed comparison requires explicit input seed')
        seed=validate_seed(plan['random_seed']);key=(job['fold_id'],seed)
        if key in observed:raise ValueError('duplicate fold/seed cell')
        observed.add(key)
        family=plan.get('model_family','multinomial' if plan.get('schema_version')=='joint-1' else 'logistic')
        policy=json.loads(json.dumps(plan.get('policy')))
        if isinstance(policy,dict):
            policy.pop('frozen_at',None)
            if isinstance(policy.get('selection'),dict):policy['selection'].pop('frozen_at',None)
        model_specs.add(json.dumps(dict(pipeline=job['pipeline_kind'],family=family,
            weights=plan.get('class_weights'),target=plan.get('target_id'),features=plan.get('feature_contract'),policy=policy),
            sort_keys=True,allow_nan=False))
    if observed!=expected:raise ValueError('incomplete or undeclared fold/seed grid')
    if len(model_specs)!=1:raise ValueError('seed campaign requires fixed model specification across folds')


def comparison_scope(plan,variation):
    if 'class_weights' in variation:
        if plan.get('schema_version')!='joint-1' or plan.get('model_family','multinomial') not in ['multinomial','cost_sensitive_joint']:
            raise ValueError('loss-weight control requires multinomial joint families')
        from .joint_run import pipeline_costs
        pipeline_costs(plan)
    scope={k:v for k,v in plan.items() if k not in variation}
    if 'policy' in variation:
        if plan.get('schema_version') not in ['1','2']:
            raise ValueError('fixed ATR comparison requires binary baseline')
        policy=plan['policy']
        if policy['mode'] not in ['score_only_control','atr_liquidity_control']:
            raise ValueError('fixed ATR comparison cannot vary learned risk policies')
        if ('policy_controls' in plan)!=(policy['mode']=='atr_liquidity_control'):
            raise ValueError('fixed ATR control inputs mismatch')
        if ('policy_control_sources' in variation
                and ('policy_control_sources' in plan)!=(policy['mode']=='atr_liquidity_control')):
            raise ValueError('source-bound ATR comparison requires filter source evidence')
        # Only policy identity/mode and ATR thresholds vary. Target, TopN,
        # score floor, frozen time and risk claims remain identical.
        scope['fixed_policy_fields']={k:v for k,v in policy.items()
            if k not in ['policy_id','mode','max_atr_fraction','min_traded_value_cny']}
    return hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_control_scope(plan, fold, variation, scopes):
    if 'policy' not in variation or 'policy_controls' not in plan:
        return
    data=plan['policy_controls']
    if 'policy_control_sources' in variation:
        data=dict(controls=data,sources=plan['policy_control_sources'])
    sha=hashlib.sha256(json.dumps(data,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if fold in scopes and scopes[fold]!=sha:
        raise ValueError('same-fold policy control data mismatch')
    scopes[fold]=sha


def run(root, manifest_path, manifest_sha, *, directory=None):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha or "") or digest(manifest_path) != manifest_sha:
        raise ValueError("campaign manifest pin mismatch")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    variation=comparison_variation(manifest)
    limits = Limits(**manifest["limits"])
    if manifest["evidence_mode"] not in {"synthetic", "supplied_reference"}:
        raise ValueError("explicit diagnostic evidence mode required")
    jobs = manifest["jobs"]
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 72:
        raise ValueError("bounded registered jobs required")
    job_ids, attempts, folds, inputs = set(), set(), {}, {manifest_path: manifest_sha}
    comparison_scopes = {}
    control_scopes = {}
    input_plans=[]
    for job in jobs:
        required={"job_id", "fold_id", "input_path", "input_sha256", "trial_id", "attempt_id"}
        if manifest['schema_version'] in ['3','4','5','6','7']:required.add('pipeline_kind')
        if set(job) != required:
            raise ValueError("exact registered job required")
        pipeline=job.get('pipeline_kind','baseline')
        if pipeline not in ['baseline','joint']:
            raise ValueError('unknown campaign pipeline identity')
        if any(not isinstance(job[k], str) or not job[k] for k in ("job_id", "fold_id", "trial_id", "attempt_id")):
            raise ValueError("named job/fold/trial/attempt required")
        key = (job["trial_id"], job["attempt_id"])
        if job["job_id"] in job_ids or key in attempts:
            raise ValueError("duplicate campaign job/attempt")
        job_ids.add(job["job_id"]); attempts.add(key)
        path = Path(job["input_path"]).resolve()
        if digest(path) != job["input_sha256"]:
            raise ValueError("job input pin mismatch")
        inputs[path] = job["input_sha256"]
        plan = json.loads(path.read_text(encoding="utf-8"))
        if (plan.get('schema_version')=='joint-1') != (pipeline=='joint'):
            raise ValueError('campaign pipeline/input identity mismatch')
        input_plans.append(plan)
        if plan["evidence_mode"] != manifest["evidence_mode"]:
            raise ValueError("mixed campaign evidence")
        bounds = plan["boundaries"]
        interval = (_instant(bounds[4]["start_at"]), _instant(bounds[4]["end_at"]))
        if interval[0] >= interval[1]:
            raise ValueError("reversed outer fold")
        if job["fold_id"] in folds and folds[job["fold_id"]] != interval:
            raise ValueError("inconsistent same-fold intervals")
        # No arbitrary policy/feature search: legacy contracts vary family;
        # the explicit loss-weight contract also permits class weights.
        scope_sha = comparison_scope(plan,variation)
        check_control_scope(plan,job['fold_id'],variation,control_scopes)
        if job['fold_id'] in comparison_scopes and comparison_scopes[job['fold_id']] != scope_sha:
            raise ValueError('same-fold baseline comparison scope mismatch')
        comparison_scopes[job['fold_id']] = scope_sha
        folds[job["fold_id"]] = interval
        matches = [t for t in manifest["budget"]["trials"] if t["trial_id"] == job["trial_id"]]
        if len(matches) != 1 or matches[0]["input_sha256"] != job["input_sha256"]:
            raise ValueError("job not bound to budget trial")
        if pipeline=='joint':
            from .joint_run import pipeline_costs
            expected_costs=pipeline_costs(plan)
        else:
            from .baseline_model import pipeline_costs as baseline_costs
            expected_costs=baseline_costs(plan)
        if matches[0]['costs']!=expected_costs:
            raise ValueError('campaign job registered costs differ from model family')
    check_seed_grid(manifest,input_plans)
    intervals = sorted(folds.values())
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("different outer folds overlap")
    for counter, limit in manifest["budget"]["limits"].items():
        required = sum(next(t for t in manifest["budget"]["trials"] if t["trial_id"] == j["trial_id"])["costs"][counter] for j in jobs)
        if required > limit:
            raise ValueError("registered campaign exceeds budget")
    out = root/"output/experiments/s20_safe_v4/sources"/("multifold-"+manifest_sha[:24]) if directory is None else Path(directory).resolve()
    fresh = directory is None and not out.exists()
    from .campaign_parent_budget import inspect as inspect_parent_budget
    parent_budget=inspect_parent_budget(root,input_plans,manifest['budget']['limits'],out/'budget.sqlite')
    if not out.resolve().is_relative_to(root/"output/experiments/s20_safe_v4/sources"):
        raise ValueError("campaign output outside research scope")
    if fresh:
        out.mkdir(parents=True)
        (out/"manifest.json").write_bytes(raw)
        inputs.update({p: digest(p) for p in Path(__file__).parent.glob("*.py")})
        atomic_json(out/"inputs.json", {str(p): h for p, h in inputs.items()})
    else:
        if digest(out/"manifest.json") != manifest_sha:
            raise ValueError("resume manifest differs")
        inputs = {Path(p): h for p, h in json.loads((out/"inputs.json").read_text(encoding="utf-8")).items()}

    def check():
        if any(digest(p) != h for p, h in inputs.items()):
            raise ValueError("campaign source changed")
    check()
    budget = Budget(out/"budget.sqlite", manifest["budget"])
    completed = []
    def checkpoint(state, error=None):
        atomic_json(out/"checkpoint.json", {"at": now(), "status": state, "completed_jobs": completed,
                    "budget": budget.status(), "error": error, "formal_training_authorized": False})
    checkpoint("RUNNING")
    try:
        for job in jobs:
            check()
            parent_budget=inspect_parent_budget(root,input_plans,manifest['budget']['limits'],out/'budget.sqlite')
            pipeline=job.get('pipeline_kind','baseline')
            if pipeline=='joint':
                from .bounded_joint import run as runner
            else:runner=run_baseline
            result = runner(root, job["input_path"], job["input_sha256"], budget, job["trial_id"], job["attempt_id"], limits)
            attempt = next(a for a in budget.status()["attempts"]
                           if (a["trial_id"], a["attempt_id"]) == (job["trial_id"], job["attempt_id"]))
            if attempt["state"] != "SUCCEEDED_DIAGNOSTIC":
                raise ValueError("attempt not reusable; reserved/failed requires reconciliation")
            artifact = attempt["result"]
            if pipeline=='joint':
                from .joint_run import verify as verify_artifact
            else:verify_artifact=verify
            verify_artifact(artifact["directory"], artifact["summary_sha256"])
            verify_process(root, artifact, job["input_path"], job["input_sha256"], limits,pipeline=pipeline)
            check()
            completed.append(dict(job_id=job["job_id"], fold_id=job["fold_id"],
                                  executed_this_call=result["executed"], **artifact))
            checkpoint("RUNNING")
    except Exception as exc:
        checkpoint("FAILED", {"type": type(exc).__name__, "message": str(exc)})
        raise
    try:
        from .baseline_outer_ledger import collect
        outer=collect(completed)
        check()
        outer_path=out/('outer-predictions-'+uuid.uuid4().hex+'.parquet')
        outer.to_parquet(outer_path,index=False)
    except Exception as exc:
        checkpoint('FAILED',{'type':type(exc).__name__,'message':str(exc)})
        raise
    checkpoint("VERIFYING")
    report = {"directory": str(out), "manifest_sha256": manifest_sha, "jobs": completed,
              "parent_budget_inventory": parent_budget,
              "same_fold_comparison_scope_sha256": comparison_scopes,
              "same_fold_variation_allowed": variation,
              "outer_prediction_ledger": {"path":str(outer_path),"sha256":digest(outer_path),
                  "rows":len(outer),"recorded_provenance_valid_rows":int(outer.recorded_oof_provenance_valid.sum()),
                  "historical_availability_proven":False},
              "distinct_outer_folds": len(folds), "budget": budget.status(),
              "evidence_mode": manifest["evidence_mode"], "formal_H03_accepted": False,
              "policy_search_performed": False, "outer_performance_evaluated": False}
    # A new call gets a separate receipt; do not overwrite prior completion evidence.
    receipt_path=out/("receipt-"+uuid.uuid4().hex+".json")
    atomic_json(receipt_path, report)
    try:
        from .baseline_outer_ledger import verify_receipt
        verify_receipt(root,receipt_path,digest(receipt_path))
    except Exception as exc:
        checkpoint('FAILED',{'type':type(exc).__name__,'message':str(exc)})
        raise
    checkpoint('COMPLETED_DIAGNOSTIC')
    return report
