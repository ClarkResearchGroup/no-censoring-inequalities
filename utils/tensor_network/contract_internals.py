# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:48:01 2022

Holds various simpler contraction utils.

TODO: If any of the input arrays has dtype=object,
try again with opt_einsum in object mode. 
For that matter, could have the same strategy be used to handle big contractions...
Hmm...
try/except seems like a perfectly fine approach here.

@author: Daniel
"""

from numpy import arange,einsum,cumsum
from collections import defaultdict
from itertools import count, chain
# from opt_einsum import shared_intermediates
from .tn_utils import twine

#def profile(func): return func # to disable


def lrange(*args):
    return list(arange(*args)) # handy...

#%%
def _index_index(i1,i2):
    """ Contracts only the two specified indices together."""
    T1 = i1.tensor
    T2 = i2.tensor
        
    L1 = len(T1.indices)    
    labels1 = lrange(L1)
    labels2 = lrange(L1,L1 + len(T2.indices))
    
    labels2[i2.int] = i1.int # match the one label
    
    out_indices = tuple(i for i in T1.indices if i is not i1) \
        + tuple(i for i in T2.indices if i is not i2)
    
    out_arr = _do_contraction((T1.arr,T2.arr),(labels1,labels2))
    
    return out_arr, out_indices

#%%
def _tensor_tensor(T1,T2):
    """
    Contracts only bonds between the two, not self-bonds.

    """
    #if len(T1.indices) > len(T2.indices):
        #T1,T2 = T2,T1 
        # TODO: Decide if we're willing to sacrifice index-order-reliability or not.
        # in principle I think we should, but it's kinda nice to have...
        
    L = len(T1.indices)
    labels1 = lrange(L)
    labels2 = lrange(L,L + len(T2.indices))
    out_set1 = dict.fromkeys(T1.indices) # using as ordered set, TODO: OrderedSet class instead?
    out_set2 = dict.fromkeys(T2.indices)
    
    for idx1,bond in zip(T1.indices,T1.bonds):
        if (idx2 := T2[bond.dual]): # if there is a paired index
            # remove the contracted-over indices:
            try:
                del out_set1[idx1]
                del out_set2[idx2]
            except KeyError:
                raise Exception('Repeated bond in network?')
                
            labels2[idx2.int] = idx1.int
        
    out_indices = out_set1 | out_set2
    
    out_arr = _do_contraction((T1.arr,T2.arr),(labels1,labels2))
    return out_arr, out_indices
    
#%%
def _tensor_tuple(T,indices):
    """
    Contract T with a tuple of indices. Requires existing bonds between every index in the tuple
    and some index of T.
    """
    L = len(T.indices)
    labelsT = lrange(L)
    
    tensor_labels = {} # map from tensor id to label list
    arrs = []
    out_indices = dict.fromkeys(T.indices) # using as ordered set
    for idx2 in indices:
        X = idx2.tensor
        if (Xid := id(X)) not in tensor_labels:
            arrs.append(X.arr)
            Lnew = L + len(X.indices)
            labels_i = lrange(L,Lnew)
            tensor_labels[Xid] = labels_i
            L = Lnew
            out_indices |= dict.fromkeys(X.indices)
        else:
            labels_i = tensor_labels[Xid] # if we have already seen it, retrieve the label list.
            
        idx1 = T[idx2.bond.dual]
        # assert idx1 is not None, "No bond to contract between this index and this tensor"
        # TODO: Is this necessary? 
        # Hmm...
        if idx1 is not None:
            labels_i[idx2.int] = idx1.int # Change label of X to match label of T
        
            del out_indices[idx1] # remove both from output...
            del out_indices[idx2]
        
    tensor_labels = list(tensor_labels.values())
    out_arr = _do_contraction([T.arr,] + arrs,[labelsT,] + tensor_labels)

    #out_arr = einsum(*twine([T.arr,] + arrs, \
    #                        [labelsT,] + tensor_labels))
    
    return out_arr, out_indices


#%%
def _contract_self(T,skip):
    """ TODO: Add option to skip? """
    labels = lrange(len(T.indices))
    out_indices = []
    for i in T.indices:
        if (j := T[i.bond.dual]) and j not in skip and i not in skip:
            labels[i.int] = min(i.int,j.int)
        else:
            out_indices.append(i)
    # out_arr = einsum(T.arr,labels)
    out_arr = _do_contraction((T.arr,),(labels,))
    return out_arr, out_indices

#%%
def _contract_sequential(*tensors,optimize):
    """ Contract a sequence of tensors for which it is guaranteed that
    bonds will exist only between each tensor and the next. Should we allow
    a circular sequence/sequence trace too?
    Also should permit optimization.
    """
    L = [len(T.indices) for T in tensors]
    first_label = [0,*cumsum(L)] # gives start of labels for each tensor
    labels = [lrange(i,j) for i,j in zip(first_label[:-1],first_label[1:])]
    out_sets = [dict.fromkeys(T.indices) for T in tensors] # using dict as ordered set because I'm lazy.
    
    for i,(T1,T2) in enumerate(zip(tensors[:-1],tensors[1:])):
        
        for idx1,bond in zip(tensors[i].indices,tensors[i].bonds):
            if (idx2 := tensors[i+1][bond.dual]): # if there is a paired index
                # remove the contracted-over indices:
                try:
                    del out_sets[i][idx1]
                    del out_sets[i+1][idx2]
                except KeyError:
                    raise Exception('Repeated bond in network?')
                    
                labels[i+1][idx2.int] = labels[i][idx1.int] # match the labels together
            
    # NOTE: Breaks if a tensor has a single bond that is joined from both sides.
    # Oh well, sucks to be you if you manage to trigger that behavior.
    # Question: Can I make this handle operators correctly? And, well, do I want to?
        
    out_indices = chain(*[os.keys() for os in out_sets])
    
    arrs = [T.arr for T in tensors]
    
    out_arr = _do_contraction(arrs,labels,None,optimize) # no need for explicit output label requests, right?
    return out_arr, out_indices
    


#%% The big one
# @profile
def _contract_many(tensors,*,out_order,merge,optimize,warnbig):
    """
    Flexible option for any more complicated tensor network.
    """
    #if not optimize:
        #optimize = 'ordered'
        
    # Assign labels to each index:
    label_dict, default_out_order, num_labels = _assemble_label_dict(tensors)
    
    if out_order: # account for user-requested output order
        if len(default_out_order) != len(out_order): # if it's incomplete
            out_order_set = set(out_order)
            out_order = out_order + \
                [idx for idx in default_out_order if idx not in out_order_set] # add in the unspecified ones at the end
    else:
        out_order = default_out_order # just use the default

    # line up the labels with the tensors
    labels = [[label_dict[i.bond] for i in T.indices] for T in tensors]
    arrs = [T.arr for T in tensors]
        
    if merge: 
        new_shape, merged_indices = _prepare_merge(merge,out_order) # !! mutates out_order to put stuff adjacent
    
    out_labels = [label_dict[i.bond] for i in out_order]
    
    if not optimize and num_labels >= 52:
        optimize = 'ordered' # Use oe_contract, but still do it in order.
        if warnbig:
            print('Warning: Big contraction without optimization requested')
    
    # Do the sum and return
    out_arr = _do_contraction(arrs,labels,out_labels,optimize)
        
    if merge:
        out_arr = out_arr.reshape(new_shape) # TODO: This reshape call requires a copy. Can we make it faster?
        out_order = merged_indices 
        
    return out_arr, out_order

def _do_contraction(arrs,labels,out_labels=None,optimize=False):
    """ Dispatches to oe_contract or numpy contract as appropriate. Forces
    numpy to do contractions in order. 
    
    TODO: Use try/except to fallback to oe_contract for either (a) symbolic arrays
    or (b) too many indices.
    
    TODO: Try to use opt_einsum caching.
    """
    
    N = len(arrs)
    einargs = twine(arrs,labels)
    if out_labels is not None:
        einargs.append(out_labels)
    
    if optimize: 
        from opt_einsum import contract as oe_contract
        if optimize=='ordered': 
            optimize = [(0,1),] + [(0,i) for i in range(N-2,0,-1)] if N > 2 else False
            
        out_arr = oe_contract(*einargs,optimize=optimize)
    else:
        path = ['einsum_path',(0,1)] + [(0,i) for i in range(N-2,0,-1)] if N > 2 else False # TODO: check that works as intended 
        try:
            out_arr = einsum(*einargs,optimize=path)
        except TypeError: # symbolic array
            from opt_einsum import contract as oe_contract
            out_arr = oe_contract(*einargs,optimize='greedy',backend='object') # only works with greedy optimization?
        except ValueError: # too many subscripts
            from opt_einsum import contract as oe_contract
            out_arr = oe_contract(*einargs,optimize='greedy')
    return out_arr

#@profile
def _assemble_label_dict(tensors):
    """ 
    Helper function for contract_many. Assigns integers to each bond of the network.
    TODO: Seems like the bond_labels(min(id])) strategy is a bit slow.
    Could we make it faster somehow?
    next_label is fast.
    We could assign negative labels to some of them?
    """
        
    next_label = count().__next__
    bond_labels = defaultdict(next_label)
        
    bonds = sum([T.bonds for T in tensors],[])
    bond_set = set(bonds)
    
    assert len(bond_set) == len(bonds), 'Repeated bond in network'
    
    is_bonded = [bond.dual in bond_set for bond in bonds]
    
    # Assign labels to each bond and free index
    label_dict = {b:(bond_labels[frozenset((b,b.dual))] if x else next_label())\
                   for x,b in zip(is_bonded,bonds)}
    # Hmm, 
        
    indices = sum([T.indices for T in tensors],())
    default_out_order = [idx for x,idx in zip(is_bonded,indices) if not x] # default
    
    return label_dict, default_out_order, next_label()
            

def _prepare_merge(merge,idx_order):
    """Helper function for contract_many. Rearranges idx_order. copies name from first argument."""
    for idx1,idx2 in merge:  # merge indices as requested. TODO: make this handle sets of more than 2.  
        i1 = idx_order.index(idx1)
        i2 = idx_order.index(idx2)
        if i1 > i2: i1,i2 = i2,i1 # swap
        
        idx_order.insert(i1 + 1,idx2) # move i2 over in idx_order
        del idx_order[i2 + 1] 
        
    merged_indices = idx_order.copy()
    newshape = [i.dim for i in idx_order] # goal: prepare a new shape for the array
    
    for idx1,idx2 in merge:
        i1 = merged_indices.index(idx1)
        i2 = merged_indices.index(idx2)
        
        merged_indices[i1] = idx1.name # copy name from i1, but not bond or dim
        del merged_indices[i2] # and remove i2 altogether.
        
        newshape[i1] = idx1.dim*idx2.dim  # dimension after merging
        del newshape[i2]
        
    return newshape, merged_indices

#%%
from functools import lru_cache

@lru_cache
def get_path(optimize,subscripts,shapes):
    """ TODO: Make it autocache reused contraction paths """
    raise Exception('TODO: Write this')
    N = len(shapes)
    if optimize=='ordered': 
        optimize = [(0,1),] + [(0,i) for i in range(N-2,0,-1)] if N > 2 else False
    