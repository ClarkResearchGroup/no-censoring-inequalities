#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:13:56 2024

@author: danielbelkin
"""

from functools import cache

@cache # memoize
def mninner(nu,R):
    """ 
    see https://core.ac.uk/download/pdf/82701079.pdf 
    The computational complexity of rules for the character table of Sn
    Bernstein
    """
    R = list(R)
    if not nu: 
        chi = 1
    else:
        nu0 = nu[0]
        chi = 0
        sigma = 1
        if len(R) < nu0 - 1:
            print('weird case, need to verify correctness')
        for j in range(min(nu0-1,len(R))): # TODO: Is the "min" correct?
            if R[j] == 0: 
                sigma *= -1
        for i in range(len(R)-nu0):
            if R[i] != R[i + nu0-1]:
                sigma *= -1
            if (R[i],R[i + nu0]) == (1,0): # is rim hook
                R[i],R[i + nu0] = R[i + nu0],R[i]
                chi += sigma*mninner(nu[1:],tuple(R))
                R[i],R[i + nu0] = R[i + nu0],R[i] # unswap
    return chi

def partseq(tau):
    """ extracts essential part of a partition, returns list """
    out = []
    tau = tau+(0,)
    for i in range(len(tau)-2,-1,-1):
        for k in range(tau[i] - tau[i+1]):
            out.append(1)
        out.append(0)
    return tuple(out)

@cache
def murnak(tau,sigma,check=False):
    """ computes chi_tau(sigma). Inputs must be ordered tuples."""
    if check:
        assert sum(tau) == sum(sigma),f'Non-matching weights: {tau} and {sigma}'
        assert all([i >= j for i,j in zip(tau[:-1],tau[1:])]), f'Non-ordered partition: {tau}'
        assert all([i >= j for i,j in zip(sigma[:-1],sigma[1:])]), f'Non-ordered partition: {sigma}'

    R = partseq(tau)
    chi = mninner(sigma,R)
    return chi

def character(partition, permutation):
    """ Version for sympy permutations """
    return murnak(partition, tuple(sorted(len(i) for i in permutation.full_cyclic_form)[::-1]))

        
def test_murnak():
    assert partseq((5,4,2,1)) == (1,0,1,0,1,1,0,1,0)
    assert murnak((2,1),(2,1)) == 0 
    assert murnak((3,1),(1,1,1,1)) == 3
    assert murnak((2,1,1,1),(2,1,1,1)) == -2
    assert murnak((2,2,2,2),(8,)) == 0 # this one has some special properties
    


#%%
def partitions(n):
    """ jeromekelleher.net/generating-integer-partitions.html
    TODO: Replace with sympy.utilities.iterables.partitions
    
    """
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
        
def partition_list(n):
    """ Produces all and sorts them appropriately for murnak """
    return [tuple(i[::-1]) for i in partitions(n)]