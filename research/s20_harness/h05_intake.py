"""Budget-bound diagnostic H05 intake; files alone never establish G1/G2."""
from pathlib import Path
from .runtime import digest,load_plan,atomic_json
from .budgeted_joint_policy import verify
from .h03_intake import scope_path

OUTPUTS=('calibration_states.jsonl','policy_candidates.json','risk_coverage.csv','selected_reliability.csv')
ACCEPTANCE_GAPS=(
    'formal H04 shortlist and complete calibration/policy competition not established',
    'historical preregistration authenticity and complete global search accounting not established',
    'real-market point-in-time data and baseline acceptance required',
    'cross-fold/seed selected calibration and risk-coverage robustness not established',
    'mature-feedback calibration/update validation not established',
    'diagnostic static source snapshot is not accepted online calibration history',
)


def audit(root,directory,protocol_hash):
    root=Path(root).resolve();directory=Path(directory).resolve()
    binding_path=root/'config/s20_v4_h05_intake.json'
    if not binding_path.is_file():raise ValueError('missing pinned H05 intake binding')
    binding_sha=digest(binding_path);binding=load_plan(binding_path)
    if set(binding)!={'protocol_hash','budget_path','contract_sha256','trial_id','attempt_id'} or binding['protocol_hash']!=protocol_hash:
        raise ValueError('H05 intake binding/protocol mismatch')
    def check():
        if digest(binding_path)!=binding_sha:raise ValueError('H05 binding changed during intake')
        return verify(root,root/binding['budget_path'],binding['contract_sha256'],binding['trial_id'],binding['attempt_id'])
    checked=check();source=Path(checked['directory']).resolve()
    if not scope_path(directory).is_relative_to(scope_path(root)) or scope_path(source)==scope_path(directory):
        raise ValueError('H05 intake path scope invalid')
    summary=load_plan(source/'summary.json')
    directory.mkdir(parents=True,exist_ok=True)
    if any((directory/n).exists() for n in (*OUTPUTS,'h05_intake.json')):
        raise ValueError('H05 intake refuses existing outputs')
    for name in OUTPUTS:
        (directory/name).write_bytes((source/name).read_bytes())
        if digest(directory/name)!=summary['artifacts'][name]:raise ValueError('H05 copied output differs')
    if check()!=checked:raise ValueError('H05 source/budget changed during intake')
    result=dict(formal_gate_passed=False,formal_training_authorized=False,models_refit=0,
        acceptance_gaps=list(ACCEPTANCE_GAPS),source_directory=str(source),source_summary_sha256=checked['summary_sha256'],
        binding_sha256=binding_sha,budget_validation=checked,
        output_hashes={n:digest(directory/n) for n in OUTPUTS},
        validation_scope='recomputed budgeted diagnostic only; not formal H05 acceptance')
    atomic_json(directory/'h05_intake.json',result)
    return result
