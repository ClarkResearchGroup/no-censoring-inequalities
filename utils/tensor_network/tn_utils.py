# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:12:55 2022

Some utilities

@author: Daniel
"""
import numpy as np

def twine(*args):
    """ Like zip but all one list instead of list of lists"""
    return [j for i in zip(*args) for j in i]


def randh(*shape):
    """ Random normal tensor, made hermitian over first two axes. """
    # TODO: Is that the right pair to choose?
    T = 0.5*(np.random.randn(*shape) + 1j*np.random.randn(*shape))
    
    return T + np.conj(np.swapaxes(T,0,1))

def randc(*shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)


def multicat(arrs,match_indices,stack_indices):
    """ 
    Like np.concatenate, but along several indices at once. Also transposes.
    both match_indices and stack_indices are lists of sets of indices to be grouped.
    e.g. match_indices = [(idx_of_arr1, idx_of_arr2,idx_of_arr3), (idx_of_arr1, idx_of_arr2...)]
    TODO: Add option to have unspecified indices automatically broadcast.
    
    
    """
    assert all([len(match_indices) + len(stack_indices) == len(a.shape) for a in arrs])    
    
    
    idx = np.concatenate((match_indices,stack_indices)).T
    arrs = [a.transpose(ii) for ii,a in zip(idx,arrs)]
    
    # Prep output
    nstack = len(stack_indices)
    shapes = np.array([a.shape for a in arrs])
    out_shape = np.concatenate((np.max(shapes[:,:-nstack],axis=0),np.sum(shapes[:,-nstack:],axis=0)))
    out_dtype = np.result_type(*arrs)
    out = np.zeros(out_shape, dtype=out_dtype)
    
    
    old_pos = np.zeros(nstack,dtype=int) # tracks how far into 
    for i, shape in enumerate(shapes):
        new_pos = old_pos + shape[-nstack:]
        idx = tuple(slice(a,b) for a,b in zip(old_pos,new_pos))
        old_pos  = new_pos
        
        out[(...,) + idx] = arrs[i] # add it in
    return out

'''
class IdDict(dict):
    def __missing__(self, key):
        return key
    
    def __or__(self,other):
        return IdDict(super().__or__(other))'''


"""
TODO: Brodcasting for multicat. 
x = np.zeros((2,10,5,6))
y = np.zeros((8,7,2,11,4))


# def multicat(*arrs,match_indices,stack_indices):
match_indices = ((0,2),(2,4))
stack_indices = ((3,0),(4,1))


arrs = x,y

# Next goal: Identify indices which are neither matched nor stacked.
# and put them first.
# hmm, do we need to add ones? Yes, we do.
# plan: iterate through and make a list of all unpaired indices for each arr.
idx = np.concatenate((match_indices,stack_indices)).T
xx = [[i  for i in range(len(a.shape)) if i not in ii] for ii,a in zip(idx,arrs)] # extras

# ok, now what? 
# goal: put the extras first
# and put in ones to match
n_total = np.sum([len(x) for x in xx]) + idx.shape[1] # total number of unpaired dimensions
new_dims = [n_total - len(a.shape) for a in arrs]

n = 0
start = 0
ii = list(idx[n])
a = arrs[n]

a = np.expand_dims(a,axis=tuple(range(-new_dims[n],0))) # add 1s at the end.
assert len(a.shape) == n_total
new_matches = []
# TODO: Write this brodcasting later, iff needed.

# ok, and... new matches will be...
# Well, the new match-sets will be 

# Ok, and now we simply want to add new matches...
# there will be 1 for each 
# new_matches = [tuple(tuple(range(-n,0))]
# Hmm... let's add all the new matches that start with x...
# better, let's make a blank template
# and fill it in
# so...


# new_matches += list(zip(xx[0],*[range(-n,0) for n in new_dims[1:])

"""