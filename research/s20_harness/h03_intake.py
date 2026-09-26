"""Pinned diagnostic H03 bundle intake; incomplete coverage cannot pass H03."""
from pathlib import Path
import os
from .runtime import load_plan,digest,atomic_json
from .baseline_bundle import verify,ACCEPTANCE_GAPS

OUTPUTS=('split_manifest.json','baseline_metrics.csv','baseline_oof.parquet','baseline_bundle.json')


def scope_path(path):
    """Compare resolved Windows long/ordinary paths without changing I/O paths."""
    value=str(path.resolve())
    if os.name=='nt':
        if value.startswith('\\\\?\\UNC\\'): value='\\\\'+value[8:]
        elif value.startswith('\\\\?\\'): value=value[4:]
    return Path(value)


def audit(root,directory,protocol_hash):
    root,directory=Path(root).resolve(),Path(directory).resolve()
    binding_path=root/'config/s20_v4_h03_intake.json'
    if not binding_path.is_file(): raise ValueError('missing pinned H03 intake binding')
    binding_sha=digest(binding_path);binding=load_plan(binding_path)
    if set(binding)!={'protocol_hash','directory','summary_sha256'} or binding['protocol_hash']!=protocol_hash:
        raise ValueError('H03 intake binding/protocol mismatch')
    source=(root/binding['directory']).resolve()
    if (not scope_path(source).is_relative_to(scope_path(root))
            or not scope_path(directory).is_relative_to(scope_path(root)) or scope_path(source)==scope_path(directory)):
        raise ValueError('H03 intake path scope invalid')
    summary_sha=binding['summary_sha256']
    checked=verify(root,source,summary_sha)
    summary=load_plan(source/'summary.json')
    def pins():
        if digest(binding_path)!=binding_sha or digest(source/'summary.json')!=summary_sha:
            raise ValueError('H03 binding/source changed during intake')
        if any(digest(source/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('H03 bundle changed during intake')
    pins();directory.mkdir(parents=True,exist_ok=True)
    if any((directory/n).exists() for n in (*OUTPUTS,'h03_intake.json')):
        raise ValueError('H03 intake refuses existing outputs')
    for name in OUTPUTS:
        (directory/name).write_bytes((source/name).read_bytes())
        if digest(directory/name)!=summary['artifacts'][name]:raise ValueError('H03 copy differs')
    verify(root,source,summary_sha);pins()
    result=dict(formal_gate_passed=False,formal_training_authorized=False,models_refit=0,
        acceptance_gaps=list(ACCEPTANCE_GAPS),source_directory=str(source),source_summary_sha256=summary_sha,
        binding_sha256=binding_sha,bundle_validation=checked,
        output_hashes={n:digest(directory/n) for n in OUTPUTS},
        validation_scope='recomputed diagnostic bundle; not baseline coverage, legacy reproduction or H03 acceptance')
    atomic_json(directory/'h03_intake.json',result)
    return result
