# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:16:41 2022

The big cahuna.

@author: Daniel
"""

from itertools import chain
from .tn import Tensor, Index, make_bonds, bond_by_name
from .contract_internals import _contract_many, _contract_sequential


#%% The main one:
def contract(*tensors, mode='bond', out_order=[], skip=set(), extra_edges=(), \
             newnames=(), merge=(),optimize=False, warnbig=True): # will add options
    """
    This version: Use numpy, minimize overhead, no optimization. Hand off to contract_big 
    otherwise.
    TODO: Write proper documentation! 
    mode can be 'name', 'bond', 'explicit'
    When mode is 'explicit', exactly the indices listed in extra_edges are contracted. 
        skip or out_order will not prevent contraction in explicit mode.
        
    """        
    
    if not tensors:
        return Tensor(1)
    # Need to special-case the trivial contraction
    
    tensors, out_order, skip, extra_edges = \
        parse_args(tensors, mode, out_order, skip, extra_edges, newnames, merge)
        
    # Rearrange bonds to match the input description
    need_to_restore = arrange_bonds(tensors, mode, skip, extra_edges, out_order)
    
    # Do the contraction
    out_arr,out_indices = _contract_many(tensors,out_order=out_order,merge=merge,\
                                         optimize=optimize,warnbig=warnbig)
    8
    # Restore the bonds
    if need_to_restore: [T.restore_bonds() for T in tensors]
    
    # Construct the output
    out = Tensor(out_arr, out_indices) 
            
    for i,name in enumerate(newnames):
        out.indices[i].name = name
        
    return out

def parse_args(tensors, mode, out_order, skip, extra_edges,newnames, merge):
    if isinstance(skip,Index): 
        skip = {skip} # flexible input   
    else:
        skip = set(skip)
        
    if hasattr(extra_edges,'__len__') and len(extra_edges) == 2 \
        and isinstance(extra_edges[0],Index):
        extra_edges = (extra_edges,)
        
    if merge and isinstance(merge[0],Index): merge = (merge,)
    out_order = list(out_order)
    
    tensors = list(tensors)
    for i,T in enumerate(tensors):
        if isinstance(T,Tensor):
            pass
        elif isinstance(T,Index):
            tensors[i] = T0 = T.tensor
            skip |= {j for j in T0.indices if j is not T} # add all the other indices of T to skip
        elif isinstance(T,(tuple,list)) and T:
            tensors[i] = T0 = T[0].tensor
            assert all(t.tensor is T0 for t in T) # also ensures they are indices
            T = set(T)
            skip |= {j for j in T0.indices if j not in T}
        else:
            raise Exception(f'Unrecognized input to contract: Object {T} of type {type(T)}')
    return tensors, out_order, skip, extra_edges
            
            
def arrange_bonds(tensors, mode, skip, extra_edges, out_order):
    from .tn import Bond
    
    # Check for bad input:
    tens_ids_mentioned = {id(i.tensor) for i in chain(skip,out_order)} \
                        | {id(i.tensor) for j in extra_edges for i in j}

    assert tens_ids_mentioned.issubset({id(T) for T in tensors}), 'Bad input to contract'

    
    need_to_restore = True 
    if mode == 'bond' and not (skip or extra_edges or out_order):
        need_to_restore = False # No need to mess with the bonds
    elif mode == 'bond':
        [T.save_bonds() for T in tensors]
    elif mode == 'name':
        [T.clear_bonds() for T in tensors]
        bond_by_name(*tensors)
    elif mode == 'explicit':
        [T.clear_bonds() for T in tensors]
    else:
        raise Exception('Mode not recognized: ' + str(mode))
        
    skip_bonds = {b for b in skip if isinstance(b,Bond)}
    skip = {i for i in skip if isinstance(i,Index)} \
        | {idx for T in tensors for b in skip_bonds if (idx := T[b]) is not None}
    
        
    # Rearrange the bonds into the desired pattern:
    if extra_edges: make_bonds(*extra_edges) # as needed
    [idx.clear() for idx in chain(skip,out_order)] # move bonds as needed
    
    return need_to_restore

#%%
def contract_sequence(*tensors,optimize=False):
    # Goal: Write code specialized for doing sequential matmuls,
    # but letting opt_einsum handle the order.
    # Question: Should we include operator-specific handling?
    # Yes, we should.
    # We want to be able to do U @ U @ U @ U, ideally.
    # I think it would make the most sense to do some network rearranging outside
    # and then just call contract_many? Or sequential, at that point it ceases to matter.
    out_arr,out_indices = _contract_sequential(*tensors,optimize=optimize)
    return Tensor(out_arr,out_indices)