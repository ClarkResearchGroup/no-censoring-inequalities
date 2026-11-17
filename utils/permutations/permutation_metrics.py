#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:45:12 2024

Goal: Assemble canonical copies of all the permutation metric tools in one place

@author: danielbelkin
"""

import numpy as np
from scipy.special import factorial
from itertools import permutations
from sympy.combinatorics import Permutation
from functools import lru_cache,cache


    
from ..tensor_network.tn import Tensor,groupcopy
from ..tensor_network.operators import Operator, make_operator, operator_contract
from ..tensor_network.svd import pinv,svd


from .murnak import character, partition_list

import sympy as sp




#%% Metric
@cache
def permutation_list(t):    
    return [Permutation(i) for i in permutations(range(t))]

@cache
def get_perm_idx_map(t):
    return {p:i for i,p in enumerate(permutation_list(t))}

@cache
def cycle_overlaps(t):
    St = permutation_list(t)
    
    tf = int(factorial(t))
    h = np.zeros((tf,tf),dtype=int)
    for i,x in enumerate(St):
        for j,y in enumerate(St):
            # h[i,j] = len((x*~y).cyclic_form)
            h[i,j] = t - len((x*~y).full_cyclic_form)
    return h


#@lru_cache
def get_metric_inner(q,t):
    """ q**-h, separated out for caching ... which seems to cause bugs"""
    h = cycle_overlaps(t)
    if isinstance(q,(int,np.integer)): q = float(q)
    return q**-h

def get_metric(q,t):
    """ Metric in permutation basis"""
    g = Operator(get_metric_inner(q,t)) #
    g.i0-g.i1
    return g
    

#%% Weingarten functions
def schur1(partition,dim):
    """ Sadowski eq. 5 """
    partition = partition + (0,)*(dim - len(partition))
    assert len(partition) >= dim
        
    out = 1
    for j in range(len(partition)):
        for i in range(j):
            out *= 1 + (partition[i] - partition[j])/(j-i)
    return out

def weingarten(sigma,dim):
    """ TODO: Add option to do the analytic continuation for non-integer dimensions?
    TODO: Make it work by only the cycle-type of sigma?
    """
    t = sigma.size
    I = Permutation(t-1)
    w = 0
    for partition in partition_list(t):
        if len(partition) <= dim:
            term = character(partition,sigma)*character(partition,I)**2/schur1(partition,dim)
            w += term
    w /= factorial(t)**2
    return w

weingarten_dict =  {(): (lambda d: 1), \
                    (1,): (lambda d: d), \
                    (2,): (lambda d: -d), \
                    (1,1): (lambda d: d**2), \
                    (3,): (lambda d: 2*d), \
                    (2,1): (lambda d: -d*2), \
                    (1,1,1): (lambda d: d*(d**2 - 2)), \
                    (4,): (lambda d: -5*d), \
                    (3,1): (lambda d: (2*d**2 - 3)), \
                    (2,2): (lambda d: d**2 + 6), \
                    (2,1,1): (lambda d: -d*(d**2 - 4)), \
                    (1,1,1,1): (lambda d: d**4 - 8*d**2 + 6), \
                    }

def symbolic_weingarten(sigma,dim):
    """
    Plan: extract cycle type, check the dict, apply the appropriate denominator
    """
    part = tuple(sorted((len(i) for i in sigma.full_cyclic_form),reverse=True))
    t = sum(part)
    numerator = weingarten_dict[part](dim)
    denominator = sp.Mul(*[(dim**2 - n**2) for n in range(t)])
    return numerator/denominator

@cache
def metric_pseudoinverse_inner(q,t,mode):
    """ Default is based on weingarten functions, 
    only for integer q.
    set mode='svd' to get a numerical pseudoinverse that works for noninteger q.
    Set mode='symbolic' to use the analytic continuation (t <= q) with symbolic q.
    """
    if mode == 'weingarten':
        q = int(round(q))
        perms = permutation_list(t)
        W = np.zeros((len(perms),)*2)
    
        for i,sigma in enumerate(perms):
            for j,tau in enumerate(perms):
                W[i,j] = weingarten(sigma*tau**-1,q)
        return W*q**t
    elif mode=='svd':
        return pinv(get_metric(q,t)).arr
    elif mode=='symbolic': # assuming q >= t
        assert t <= 4, "symbolic mode not supported for large t"
        perms = permutation_list(t)
        W = sp.zeros(len(perms),len(perms))
    
        for i,sigma in enumerate(perms):
            for j,tau in enumerate(perms):
                W[i,j] = symbolic_weingarten(sigma*tau**-1,q)
        return W*q**t
    else:
        raise Exception('Unrecognized mode')
        
        

def metric_pseudoinverse(q,t,mode='weingarten'):
    """ Default is based on weingarten functions, 
    only for integer q.
    set mode='svd' to get a numerical pseudoinverse that works for noninteger q.
    Set mode='symbolic' to use the analytic continuation (t <= q) with symbolic q.
    """
    M = metric_pseudoinverse_inner(q,t,mode)
    
    ginv = Operator(M)
    ginv.i0-ginv.i1
    return ginv
        
#A = metric_pseudoinverse(8,2,'weingarten')
#B = metric_pseudoinverse(16,2,'weingarten')


#%% Gate operators
def permutation_basis_half_projector(n,t):
    """ Up-projector """
    tf = int(factorial(t))
    x = np.zeros((tf,)*(n + 1),dtype=int)
    for i in range(tf):
        x[(i,)*(n+1)] = 1
        
    X = Tensor(x)
    make_operator(X,ingroup=X.i0)
    
    #assert len(X.outgroup) == n, "temp check"
    return X

#print('caching disabled')
@cache
def permutation_basis_halves(n,q,t,mode='svd'):
    """ Does the adjoint in the nontrivial metric 
    TODO: I think it would also be correct to conjugate by g1^-1
    and perhaps conceptually cleaner, since then both X and Y have a pseudoinverse 
    on their outgroup
    i.e. can be interpreted as living in the basis/cobasis representation
    
    """
    #print('reset_cache')
    g1 = get_metric(q,t)
    X = permutation_basis_half_projector(n,t)

    gninv = metric_pseudoinverse(q**n,t,mode=mode) # TODO: Play around with mode.
    gninv.outgroup[0] - X.ingroup[0]
    gninv.ingroup[0] - gninv.outgroup[0]
    Y = gninv @ X.conj()

    aa = []
    for i in Y.ingroup:
        aa.append(g1.copy(free='all'))
        aa[-1].outgroup[0] - i

    Y = operator_contract(Y,*aa)
    [i-j for i,j in zip(X.outgroup,Y.ingroup)] # self-bonded
    
    assert len(X.outgroup) == n, "temp check"
    return X,Y

def permutation_basis_projector(n,q,t,mode='svd'):
    B,A = permutation_basis_halves(n,q,t,mode=mode)
    return B @ A

def permutation_basis_halves_multiq(qvals,t,mode='svd'):
    """
    qvals is list by sites
    """
    n = len(qvals)
    qtotal = np.prod(qvals)

    X = permutation_basis_half_projector(n,t)
    
    gninv = metric_pseudoinverse(qtotal,t,mode=mode)
    gninv.outgroup[0] - X.ingroup[0]
    gninv.ingroup[0] - gninv.outgroup[0]
    Y = gninv @ X.conj()

    aa = []
    for i,q in zip(Y.ingroup,qvals):
        gi = get_metric(q,t).copy(free='all')
        aa.append(gi)
        aa[-1].outgroup[0] - i

    Y = operator_contract(Y,*aa)
    [i-j for i,j in zip(X.outgroup,Y.ingroup)] # self-bonded
    return X,Y


#%% Sparse gate operators
def sparse_half_projector(n,t):
    """ Up-projector """
    from sparse import DOK,COO
    tf = int(factorial(t))
    x = DOK((tf,)*(n + 1),dtype=bool)
    for i in range(tf):
        x[(i,)*(n+1)] = 1
        
    X = Tensor(COO(x))
    make_operator(X,ingroup=X.i0)
    return X


def sparse_halves(n,q,t,mode='svd'):
    """
    Does the adjoint in the nontrivial metric
    """
    from sparse import COO
    X = sparse_half_projector(n,t)

    gninv = metric_pseudoinverse(q**n,t,mode=mode) # TODO: Play around with mode.
    # gninv.arr = COO(gninv.arr) # make it sparse so that contractions work
    gninv.outgroup[0] - X.ingroup[0]
    
    Y = X.conj(free='all')
    gninv.ingroup[0] - Y.outgroup[0]
    
    u,s,B = svd(gninv,abstol=1e-8)
    A = u @ s
    A.arr = COO(A.arr)
    B.arr = COO(B.arr)

    g1 = get_metric(q,t)
    u,s,D = svd(g1,abstol=1e-8)
    C = u @ s
    C.arr = COO(C.arr)
    D.arr = COO(D.arr)
    
    Clist = []
    for i,j in zip(Y.ingroup,X.outgroup):
        c,d = groupcopy((C,D),snip=True)
        c.outgroup[0] - i
        d.ingroup[0] - j
        Clist.append((c,d))
    return X, (A,B), Y, Clist

#%%
def orthogonal_xform(metric,dual=False,check=False):
    """ Returns a tensor mapping from permutation basis to orthonormal basis """
    from ..tensor_network.svd import svd
    if not dual:
        u,s,v = svd(metric,reltol=1e-6)
        s.arr = np.diag(np.sqrt(np.diag(s.arr)))
        X = s @ v
        
        if check:
            assert X.conj() @ X == metric
    else:
        u,s,v = svd(metric,reltol=1e-6)
        s.arr = np.diag(1/np.sqrt(np.diag(s.arr)))
        X = v.conj() @ s.conj()

        if check:
            I = X.conj() @ metric @ X
            assert np.allclose(I.arr, np.eye(I.arr.shape[0]))
    return X

@cache
def half_cluster_projector(n,q,t):
    g1 = get_metric(q,t)
    gn = get_metric(q**n,t)
    
    A = orthogonal_xform(g1)
    B = orthogonal_xform(gn,dual=True)
    
    X = permutation_basis_half_projector(n,t)  
    
    aa = []
    for i in X.outgroup:
        aa.append(A.copy(free='all'))
        aa[-1].ingroup[0] - i
    B.outgroup[0] - X.ingroup[0]
    
    P = operator_contract(X,*aa,B)
    return P

def cluster_projector(n,q,t): # TODO: Is there a version of this based on weingarten functions we'd like to implement?
    B = half_cluster_projector(n,q,t)
    A = B @ B.conj()
    A.clear_bonds()
    return A


    
def half_cluster_projector_multiq(qvals,t):
    n = len(qvals)
    X = permutation_basis_half_projector(n,t)  

    gn = get_metric(np.prod(qvals),t)
    B = orthogonal_xform(gn,dual=True)
    B.outgroup[0] - X.ingroup[0]

    aa = []
    for i,q in zip(X.outgroup,qvals):
        gi = get_metric(q,t).copy(free='all')
        aa.append(orthogonal_xform(gi))
        aa[-1].ingroup[0] - i
    
    P = operator_contract(X,*aa,B)
    return P