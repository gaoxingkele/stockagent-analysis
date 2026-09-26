"""Bind every declared portfolio evidence reference to local immutable bytes.

This is byte/identity completeness, not content validation, receipt authenticity,
historical availability, legal eligibility or execution truth.
"""
from pathlib import Path
import re
from .runtime import digest


def bind(root, plan):
    bindings=plan.get('evidence_bindings')
    if bindings is None:
        return {},dict(declared_evidence_bindings_verified=False,declared_evidence_references=0)
    if not isinstance(bindings,list): raise ValueError('evidence bindings must be a list')
    refs={}
    def walk(value,path,index):
        if isinstance(value,dict):
            for key in ('evidence_id','source_id'):
                if key in value:
                    identity=value[key]
                    if not isinstance(identity,str) or not identity.strip(): raise ValueError('named evidence reference required')
                    refs[(index,path+'/'+key)]=(identity,value.get(key.replace('_id','_sha256')))
            for k,v in value.items(): walk(v,path+'/'+k,index)
        elif isinstance(value,list):
            for k,v in enumerate(value): walk(v,path+'/'+str(k),index)
    for i,event in enumerate(plan['events']): walk(event['args'],'args',i)
    root=Path(root).resolve();pins={};seen=set()
    for row in bindings:
        if not isinstance(row,dict) or set(row)!={'event_index','field','identity','path','sha256'}:
            raise ValueError('exact portfolio evidence binding schema required')
        if type(row['event_index']) is not int or not isinstance(row['field'],str): raise ValueError('invalid evidence reference key')
        key=(row['event_index'],row['field'])
        if key not in refs or key in seen: raise ValueError('duplicate or undeclared evidence binding')
        identity,declared_sha=refs[key];sha=row['sha256']
        if row['identity']!=identity or not isinstance(sha,str) or not re.fullmatch(r'[0-9a-f]{64}',sha):
            raise ValueError('evidence identity/hash mismatch')
        if declared_sha is not None and declared_sha!=sha: raise ValueError('event evidence hash differs from binding')
        if not isinstance(row['path'],str) or not row['path'].strip(): raise ValueError('evidence path required')
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or path.suffix.lower()=='.py': raise ValueError('evidence path outside data scope')
        if digest(path)!=sha or (path in pins and pins[path]!=sha): raise ValueError('evidence bytes differ')
        pins[path]=sha;seen.add(key)
    if seen!=set(refs): raise ValueError('missing declared evidence bindings')
    if any(digest(p)!=h for p,h in pins.items()): raise ValueError('evidence changed during binding')
    return pins,dict(declared_evidence_bindings_verified=True,declared_evidence_references=len(refs))
