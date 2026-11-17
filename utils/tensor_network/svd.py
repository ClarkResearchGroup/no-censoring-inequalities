# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:54:31 2022

Related things I could want: Version that applies some function to the singular
values before returning?

Version specialized to the hermitian case

Version which takes in a tensor network, turns it into a scipy LinearOperator, 
and does scipy sparse svd.

Version which uses scipy.linalg.eigh to compute only a few singular values.

TODO: think about alternatives to the SVD_setting object.
@author: Daniel
"""

import numpy as np
from .tn import Tensor, make_bonds
from scipy.linalg import svd as scipy_svd
from scipy.linalg import eig as scipy_eig
from scipy.linalg import eigh as scipy_eigh
from time import time
from .operators import Operator
# from itertools import chain

#%%
class SVD_setting(object): # TODO: Could I get this by subclassing namedtuple?
    def __init__(self, max_bond=np.inf,abstol=0,reltol=0):
        self.max_bond=max_bond
        self.abstol=abstol
        self.reltol=reltol
        
    def is_nontrivial(self):
        return self.max_bond < np.inf or self.abstol > 0 or self.reltol > 0
    
    def merge(self,other,max_bond=None,abstol=None,reltol=None,mode='aggressive'):
        """ TODO: Maybe no need for the optional inputs?"""
        if mode == 'aggressive':
            if max_bond is None: max_bond = min(self.max_bond,other.max_bond)
            if abstol is None: abstol = max(self.abstol,other.abstol)
            if reltol is None: reltol = max(self.reltol,other.reltol)
        elif mode == 'conservative':
            if max_bond is None: max_bond = max(self.max_bond,other.max_bond)
            if abstol is None: abstol = min(self.abstol,other.abstol)
            if reltol is None: reltol = min(self.reltol,other.reltol)
        else:
            raise Exception()
        return SVD_setting(max_bond=max_bond,abstol=abstol,reltol=reltol)
    
    def override(self,max_bond=None,abstol=None,reltol=None):
        if max_bond is None: max_bond = self.max_bond
        if abstol is None: abstol = self.abstol
        if reltol is None: reltol = self.reltol
        return SVD_setting(max_bond=max_bond,abstol=abstol,reltol=reltol)
        # TODO: Could just return the tuple
        # TODO: This would be a good candidate for learning
        # to write numba classes
        
    def __iter__(self):
        return (self.max_bond, self.abstol, self.reltol).__iter__()
    
    def __repr__(self):
        return 'SVD_setting({},{},{})'.format(self.max_bond, self.abstol, self.reltol)

def parse_svd_setting(setting):
    """ TODO: Come back to this, figure out cleanest approach"""
    if isinstance(setting,SVD_setting):
        return setting
    elif isinstance(setting,dict):
        return SVD_setting(**setting)
    elif isinstance(setting,tuple):
        return SVD_setting(*setting)
    
#%%
def _svd_inner(A):
    assert np.prod(A.shape),"must have at least one element"
    try:
        u,s,v = scipy_svd(A,full_matrices=False,overwrite_a=True,check_finite=False) # Appears to be slightly faster.
    except:
        assert np.all(np.logical_not(np.isnan(A)))
        print('Warning: scipy svd failed on input of shape ' + str(A.shape) + '. Trying numpy...')
        u,s,v = np.linalg.svd(A,full_matrices=False) # Do the SVD
    
    return u,s,v

def svd(T,group1=None, group2=None,inner_names = ('svd_left','svd_right'),\
        setting=None, max_bond=None,abstol=None,reltol=None):
    """
    Tensor svd over two arbitrary subsets of dimensions.
    Groups are lists of indices or bonds. Samples usage: 
    U,S,V = svd(T, (T.i0,T.j0),(T.i1,T.j1))
    assert T == U @ S @ V
    
    U, V inherit outer bonds from T. 
    
    inner_names determines the index names for the new inner indices. Defaults
    to ('svd_left','svd_right')        
    """
    # Standardize arguments:
    if setting is None: setting = SVD_setting()
    max_bond,abstol,reltol = setting.override(max_bond,abstol,reltol)
    # Let other inputs override an explicit setting
    
    group1,group2 = T._split_indices(group1,group2)
    A,dims1,dims2 = T.matrix(group1,group2,return_dims=True)
        
    u,s,v = _svd_inner(A)
    
    # handle truncation:
    threshold = max(abstol,reltol*s[0]) # discard singular values smaller than this
    n = max(min(max_bond, np.count_nonzero(s > threshold)),1) # apply truncation criteria    
    if n < len(s):
        u,s,v = u[:,:n],s[:n],v[:n,:] # truncate
    
    inner_dim = len(s)
    u = u.reshape(dims1 + (inner_dim,))
    #print(u,group1,group2)
    u = Operator(u,outgroup=group1,ingroup=(inner_names[0],))
    
    s = Operator(np.diag(s),outgroup=(inner_names[0],),ingroup=(inner_names[1],))

    v = v.reshape((inner_dim,) + dims2)
    v = Operator(v,outgroup=(inner_names[1],),ingroup=group2)
    
    make_bonds((u.indices[-1],s.indices[0]),(s.indices[1],v.indices[0]))
    
    return u,s,v

def singvals(T,group1=None, group2=None, setting=None, max_bond=None,abstol=None,reltol=None):
    """
    Just gives singular values   
    """
    # Standardize arguments:    
    group1,group2 = T._split_indices(group1,group2)
    A = T.matrix(group1,group2)
    
    if setting is None: setting = SVD_setting()
    max_bond,abstol,reltol = setting.override(max_bond,abstol,reltol)
        
    _,s,_ = _svd_inner(A)
    
    # handle truncation:
    threshold = max(abstol,reltol*s[0]) # discard singular values smaller than this
    n = max(min(max_bond, np.count_nonzero(s > threshold)),1) # apply truncation criteria    
        
    return s[:n]
    

#%%
def polar_u(T,group1=None, group2=None):
    """
    Compute the U of the polar decomposition.
    U maps from group2 to group1...
    """
    group1,group2 = T._split_indices(group1,group2)
    A,dims1,dims2 = T.matrix(group1,group2,return_dims=True)
    # Ok, wait, so here,
    # A is a matrix which maps from which to which?
    
    u,s,v = _svd_inner(A)
        
    U = u @ v
    U = U.reshape(dims1 + dims2)
    U = Operator(U,outgroup=group1,ingroup=group2)
    return U

#%%
def pinv(T,group1=None,group2=None,abstol=1e-8,reltol=1e-8):
    u,s,v = svd(T,group1,group2,abstol=abstol,reltol=reltol)
    s.arr = np.diag(1/np.diag(s.arr))
    # v^* 1/s^* u^* u s v = v^* v = I over support
    return v.conj() @ s.conj() @ u.conj()

#%%
def nullspace(T,group1=None, group2=None,inner_name = 'null_inner',\
        setting=None, max_bond=None,abstol=1e-8,reltol=None):
    """
    Half-projectors into left and right nullspaces.
    """
    # Standardize arguments:
    if setting is None: setting = SVD_setting()
    max_bond,abstol,reltol = setting.override(max_bond,abstol,reltol)
    # Let other inputs override an explicit setting
    
    group1,group2 = T._split_indices(group1,group2)
    A,dims1,dims2 = T.matrix(group1,group2,return_dims=True)
        
    u,s,v = _svd_inner(A)
    
    # handle truncation:
    threshold = max(abstol,reltol*s[0]) # keep only singular values smaller than this
    n = max(min(max_bond, np.count_nonzero(s < threshold)),1) # apply truncation criteria    
    if n < len(s):
        u,v = u[:,-n:],v[-n:,:] # truncate
    
    inner_dim = u.shape[1]
    u = u.reshape(dims1 + (inner_dim,))
    u = Operator(u,outgroup=group1,ingroup=(inner_name,))
    
    v = v.reshape((inner_dim,) + dims2)
    v = Operator(v,outgroup=(inner_name,),ingroup=group2)
    
    u.indices[-1] - v.indices[0]    
    return u,v

#%%
def _diagonalize_inner(A,hermitian):
    if hermitian:
        e,u = scipy_eigh(A)
        v = u.conj().T # right?
    else:
        e, u = scipy_eig(A)
        v = np.linalg.inv(u)
    e = np.diag(e)
    assert np.allclose(u @ e @ v,A), "diagonalization failed"
    return u,e,v

def diagonalize(T,inner_names=('eig_left','eig_right'),hermitian=False):
    """Requires that T be an operator and self-bonded"""
    group2 = T.ingroup
    group1 = [i.bond.dual for i in T.ingroup]

    dims = tuple(i.dim for i in group1)

    A = T.matrix(group1,group2,return_dims=False)
        
    u,e,v = _diagonalize_inner(A,hermitian)

    inner_dim = np.prod(dims)
    u = u.reshape(dims + (inner_dim,))
    u = Operator(u,outgroup=group1,ingroup=(inner_names[0],))
    
    e = Operator(e,outgroup=(inner_names[0],),ingroup=(inner_names[1],))

    v = v.reshape((inner_dim,) + dims)
    v = Operator(v,outgroup=(inner_names[1],),ingroup=group2)
    
    make_bonds((u.indices[-1],e.indices[0]),(e.indices[1],v.indices[0]))
    
    return u,e,v

def eigvals(T,inner_names=('eig_left','eig_right'),hermitian=False):
    group2 = T.ingroup
    group1 = [i.bond.dual for i in T.ingroup]

    A = T.matrix(group1,group2,return_dims=False)
        
    if hermitian:
        e,u = scipy_eigh(A)
    else:
        e, u = scipy_eig(A)
        
    return e