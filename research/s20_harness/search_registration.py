"""Immutable local search registration and one shared job budget, no launch."""
from pathlib import Path
import hashlib
import json
import sqlite3

from .selection_plan import inspect as preflight
from .trial_budget import Budget,canonical
from .attempt_inventory import inspect as inspect_budget
from .runtime import digest,load_plan,now


def specification(root,path,sha):
    report=preflight(root,path,sha);plan=load_plan(path)
    pins={str(Path(path).resolve()):sha};trials=[];mapping=[]
    for candidate in plan['candidates']:
        manifest_path=Path(candidate['manifest_path']).resolve()
        pins[str(manifest_path)]=candidate['manifest_sha256'];manifest=load_plan(manifest_path)
        for job in manifest['jobs']:
            source=Path(job['input_path']).resolve();pins[str(source)]=job['input_sha256']
            original=next(t for t in manifest['budget']['trials'] if t['trial_id']==job['trial_id'])
            identity=dict(candidate_id=candidate['candidate_id'],trial_id=job['trial_id'])
            tid=hashlib.sha256(canonical(identity).encode()).hexdigest()
            trials.append(dict(original,trial_id=tid))
            mapping.append(dict(identity,shared_trial_id=tid,job_id=job['job_id'],fold_id=job['fold_id'],
                attempt_id=job['attempt_id'],input_path=str(source),input_sha256=job['input_sha256']))
    for source in Path(__file__).parent.glob('*.py'):pins[str(source.resolve())]=digest(source)
    if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('search registration input changed')
    budget=dict(budget_id='search-'+sha,limits=plan['limits'],trials=trials)
    return dict(plan_sha256=sha,plan_path=str(Path(path).resolve()),source_pins=pins,
        candidate_ids=report['candidate_ids'],mapping=mapping,budget_contract=budget,
        budget_contract_sha256=hashlib.sha256(canonical(budget).encode()).hexdigest())


def register(root,path,sha):
    root=Path(root).resolve();spec=specification(root,path,sha)
    directory=root/'output/experiments/s20_safe_v4/sources'/('search-registration-'+sha)
    directory.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(directory/'registration.sqlite',timeout=30) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.executescript("""
            CREATE TABLE IF NOT EXISTS registration(id INTEGER PRIMARY KEY CHECK(id=1),registered_at TEXT NOT NULL,payload TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS registration_no_update BEFORE UPDATE ON registration
                BEGIN SELECT RAISE(ABORT,'immutable search registration'); END;
            CREATE TRIGGER IF NOT EXISTS registration_no_delete BEFORE DELETE ON registration
                BEGIN SELECT RAISE(ABORT,'immutable search registration'); END;
        """)
        db.execute('BEGIN IMMEDIATE')
        existing=db.execute('SELECT registered_at,payload FROM registration WHERE id=1').fetchone()
        if existing is None:
            db.execute('INSERT INTO registration VALUES(1,?,?)',(now(),canonical(spec)))
        elif existing[1]!=canonical(spec):raise ValueError('search registration changed')
    # A crash after registration but before budget creation can resume here.
    Budget(directory/'budget.sqlite',spec['budget_contract'])
    return verify(root,directory,sha)


def verify(root,directory,sha):
    root=Path(root).resolve();directory=Path(directory).resolve()
    expected=root/'output/experiments/s20_safe_v4/sources'/('search-registration-'+sha)
    if directory!=expected or not (directory/'registration.sqlite').is_file():
        raise ValueError('existing scoped search registration required')
    with sqlite3.connect((directory/'registration.sqlite').as_uri()+'?mode=ro',uri=True,timeout=30) as db:
        rows=db.execute('SELECT id,registered_at,payload FROM registration').fetchall()
    if len(rows)!=1 or rows[0][0]!=1:raise ValueError('exact search registration required')
    registered_at,payload=rows[0][1:];spec=json.loads(payload)
    if spec['plan_sha256']!=sha or canonical(spec)!=payload:raise ValueError('search registration identity mismatch')
    current=specification(root,spec['plan_path'],sha)
    if canonical(current)!=payload:raise ValueError('search registration source/specification mismatch')
    budget=inspect_budget(root,directory/'budget.sqlite',spec['budget_contract_sha256'])
    with sqlite3.connect((directory/'registration.sqlite').as_uri()+'?mode=ro',uri=True,timeout=30) as db:
        if db.execute('SELECT id,registered_at,payload FROM registration').fetchall()!=rows:
            raise ValueError('search registration changed during verification')
    return dict(directory=str(directory),registered_at=registered_at,plan_sha256=sha,
        shared_budget_path=str(directory/'budget.sqlite'),shared_budget_contract_sha256=spec['budget_contract_sha256'],
        candidate_ids=spec['candidate_ids'],mapping=spec['mapping'],budget=budget,
        local_registration_verified=True,preregistration_before_training_proven=False,
        shared_runner_connected=False,formal_training_authorized=False,finalists_selected=False,models_refit=0)
