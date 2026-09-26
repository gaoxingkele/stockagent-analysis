"""Bounded saved depth-three four-class tree inference, no estimator fit."""
import numpy as np


def predict(tree,x):
    x=np.asarray(x,dtype=np.float32)
    if set(tree)!={'left','right','feature','threshold','values'} or x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('exact finite tree inference inputs required')
    n=len(tree['left'])
    if not 1<=n<=15 or any(len(v)!=n for v in tree.values()):raise ValueError('bounded aligned tree state required')
    values=np.asarray(tree['values'],dtype=float)
    if values.shape!=(n,4) or not np.isfinite(values).all() or (values<0).any() or (values.sum(axis=1)<=0).any():
        raise ValueError('valid four-class node values required')
    seen=set()
    def walk(node,depth):
        if type(node)is not int or not 0<=node<n or node in seen or depth>3:raise ValueError('invalid tree topology')
        seen.add(node);left,right=tree['left'][node],tree['right'][node]
        if left==right==-1:return
        feature=tree['feature'][node];threshold=tree['threshold'][node]
        if type(feature)is not int or not 0<=feature<x.shape[1] or isinstance(threshold,bool) or not np.isfinite(threshold):
            raise ValueError('invalid tree split')
        walk(left,depth+1);walk(right,depth+1)
    walk(0,0)
    if len(seen)!=n:raise ValueError('unreachable tree nodes')
    result=[]
    for sample in x:
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if sample[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        result.append(values[node]/values[node].sum())
    return np.asarray(result,dtype=float).reshape(-1,4)
