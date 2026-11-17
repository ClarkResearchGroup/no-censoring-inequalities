# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:32:47 2022

Goal: Write an operator class.
An operator is a tensor,
with indices divided into to groups.
conjugation should swap the groups, I believe.
The key thing is we add a matmul operation
which works only with other operators?
and which contracts outgroup to ingroup?
Hmm, I think we can also matmul on a tensor.
It contracts only ingroup on to the tensor.

Also should write automorphism class
which requires that the groups line up with each other.
Do we also require that bonds match?
Do we insist on name/bond transfers, perhaps?

TODO: Add a function that contracts a sequence, then turns the result into an operator
according to indices inherited from first/last tensor. 

In fact, more generally, operator_contract should have any surviving ingroup indices auto-mapped to ingroup
and any surviving outgroup indices auto-mapped to outgroup.

TODO: Change how endomorphisms work.

TODO: Make it so that 2-legged operators have "in" and "out" shortcuts to ingroup[0] etc
@author: Daniel
"""
from scipy.linalg import fractional_matrix_power

from .tn import Tensor,Index,Bond,make_bonds
from .contract import contract
#from itertools import chain
#from ordered_set import OrderedSet

class Operator(Tensor):
    def __init__(self,input_array,indices=None,outgroup=None,ingroup=None):
        """ TODO: Add option to set outgroup or ingroup as tuples of ints?
        Oh, they can be set as letters or strings or w/e.
        in which case they default according to order
        but for this to work you need to specify both.
        """
        if isinstance(outgroup,Index): outgroup = (outgroup,)
        if isinstance(ingroup,Index): ingroup = (ingroup,)
        
        if indices is None and isinstance(input_array,Operator):
            raise Exception("Should use copy?")
            
        if indices is None and isinstance(input_array,Tensor):
            #indices = input_array.indices
            raise Exception('Should use make_operator probably')
        
        if indices is None and outgroup is not None and ingroup is not None:
            outgroup = tuple(outgroup)
            ingroup = tuple(ingroup)
            super().__init__(input_array,outgroup + ingroup)
            self.outgroup = self.indices[:len(outgroup)]
            self.ingroup = self.indices[len(outgroup):]
        else: # Tensor-style initialization
            super().__init__(input_array,indices)
            self.outgroup,self.ingroup = super()._split_indices(outgroup,ingroup)
        
        assert len(self.outgroup) + len(self.ingroup) == len(self.indices)
        
    def __str__(self):
        out = 'Operator[' # TODO: Make it give its real type name?
        for idx in self.outgroup:
            out += '(' + str(idx.name) + ',' + str(idx.dim) + '), '
        out += '<- '
        for idx in self.ingroup:
            out += '(' + str(idx.name) + ',' + str(idx.dim) + '), '
        out += ']'
        return out
    
    def __repr__(self):
        return self.__str__()
    
    def copy(self,**kwargs):
        out = super().copy(**kwargs)
        if isinstance(out,tuple): # return map was true, I guess
            O = out[0]
        else:
            O = out
        
        make_operator(O,\
                outgroup = tuple(O.indices[i.int] for i in self.outgroup),\
                ingroup = tuple(O.indices[i.int] for i in self.ingroup))
            
        return out
    
    def conj(self,**kwargs):
        out = super().conj(**kwargs)
        if isinstance(out,tuple): # return map was true, I guess
            O = out[0]
        else:
            O = out
        make_operator(O,\
                outgroup = tuple(O.indices[i.int] for i in self.ingroup),\
                ingroup = tuple(O.indices[i.int] for i in self.outgroup))
            
        return out
    
    def _split_indices(self,*groups):
        if groups == (None,None):
            return self.outgroup,self.ingroup # TODO: Consider ordering convention.
        else:
            return super()._split_indices(*groups)
        
    def __matmul__(self,other):
        """
        Permits operators on tensor-product subspaces and so on. Contracts
        only indicies that are already bonded.
        To act on a tensor, we currently require that *all* input indices  
        have a bond with that tensor. TODO: Consider relaxing this restriction.
        
        NOTE:
            Matmul and mul are not associative operations. Parenthesize carefully.
            Fails silently when repeated bonds show up?
        """
        if other is self: other = self.copy()
        if isinstance(other,Operator):
            out = contract(self,other,out_order=self.outgroup+other.ingroup)
            out.__class__ = Operator
            
            out.outgroup = tuple(dict.fromkeys(x for i in self.outgroup+other.outgroup if (x := out[i.bond]))) # should use ordered set here really...
            out.ingroup = tuple(dict.fromkeys(x for i in self.ingroup+other.ingroup if (x := out[i.bond])))
            
            assert len(out.outgroup) + len(out.ingroup) == len(out.indices), "possible repeated bond"
            
            return out
        elif isinstance(other,Tensor):
            # TODO: This syntax requires that every index of ingroup matches up to other. Hmm...
            return other*self.ingroup
        else:
            return NotImplemented
        
    def __rmatmul__(self,other):
        if isinstance(other,Operator):
            return Operator.__matmul__(other,self) # Needed by endomorphism, not sure if this is the cleanest way.
        if isinstance(other,Tensor):
            return other*self.outgroup
        else:
            return NotImplemented
        
    def __add__(self,other):
        if isinstance(other,Operator):
            assert {i.bond for i in self.ingroup} == {i.bond for i in other.ingroup}, "operator addition requires that ingroup/outgroup structures match"
            out = super().__add__(other)
            make_operator(out,ingroup=self.ingroup,outgroup=self.outgroup) # make operator in same format
            return out
        elif isinstance(other,Tensor):
            t_self = self.copy()
            t_self.to_tensor()
            return t_self + other # just use tensor addition
        elif other == 0: # shortcut for avoiding reduce calls
            return self.copy()
        
    def join(self,other):
        """ forms bonds assuming ingroup, outgroup orders line up"""
        make_bonds(*[(i,j) for i,j in zip(self.ingroup,other.outgroup)])
        # Tons of ways for this to go wrong. Let's not worry about it.
        
    def to_endomorphism(self):
        raise Exception('Deprecated')
        self.__class__ = Endomorphism
        
    def to_tensor(self):
        self.__class__ = Tensor
        
# TODO: Could overload copy and conj in a way that permits inclusion of operators in a groupconj?
# Hmm... yes, this is a good idea.
'''
class Endomorphism(Operator):
    """
    Like an operator, but is careful to not change bonds or names when applied.
    ordering within groups matters.
    TODO: Rethink weird matmul behavior. annoying option would be to ad a special symbol for 
    applying endomorphisms. another option would be to just mess with the bonds of the endomorphism itself.
    
    TODO: Add initializer that permits specification of only ingroup/only outgroup, has the other half default to
    
    TODO: Consider just forcing it to be self-bonded? If it's not self-bonded,
    then it doesn't count as an endomorphism. Feels reasonable.
    
    TODO: Add transpose option for endomorphisms? 
    well, how would it differ from conj?
    """
    
    def conj(self):
        out = super().conj()
        out.to_endomorphism()
        return out
    
    def copy(self):
        out = super().copy()
        out.to_endomorphism()
        return out
    
    def __pow__(self,c):
        M,dims1,dims2 = self.matrix(return_dims=True)
        M = fractional_matrix_power(M,c)
        M = M.reshape(dims1 + dims2)
        return Endomorphism(M,ingroup=self.ingroup,outgroup=self.outgroup)
        
    def __matmul__(self,other):
        if isinstance(other,Endomorphism):
            # If other is an endomorphism, make sure we return an endomorphism. 
            # And don't mess with bonds or anything.
            # TODO: Keep thinking about what is reasonable here...
            out = super().__matmul__(other)
            out.to_endomorphism()
            return out
            
        elif isinstance(other,(Operator,Tensor)):
            out = super().__matmul__(other)
            for i1,i2 in zip(self.outgroup,self.ingroup):
                if (original_index := other[i2.bond.dual]) is not None: # old index
                    new_index = out[i1.bond] 
                    new_index.name = original_index.name
                    new_index.bond = original_index.bond
            return out
        else:
            return NotImplemented
        
    def __rmatmul__(self,other):
        if isinstance(other,Tensor): # including operators... Wait, should not happen?
            out = super().__rmatmul__(other)
            for i2,i1 in zip(self.ingroup,self.outgroup):
                if (original_index := other[i2.bond.dual]) is not None:
                    new_index = out[i2.bond] 
                    new_index.name = original_index.name
                    new_index.bond = original_index.bond
            return out
        else:
            return NotImplemented
        
    def match_outgroup_to(self,*others):
        """
        Makes self.outgroup's names and bonds match the indices of others which are 
        bonded to self.ingroup.
        If there are repeated bonds, first is grabbed.
        """
        for i,j in zip(self.ingroup,self.outgroup):
            for other in others:
                if (k := other[i.bond.dual]):
                    j.name, j.bond = k.name, k.bond
                    break # breaks only inner loop
        
    def trace(self):
        self.save_bonds()
        make_bonds(*zip(self.ingroup,self.outgroup))
        out = self.contract().scalar()
        self.restore_bonds()
        return out
        
    def partial_trace(self,indices):
        """ indices are the indices to keep? Should be specifiable from either group.
        TODO: Switch it to indices to contract? eh.
        Assumes self-bondedness, hmm...
        """
        indices = self.guess_index_group(indices)
        out = self.contract(skip=indices) # Ok, so far so good.
        
        new_ingroup = out.guess_index_group(self.ingroup,allow_misses=True)
        new_outgroup = out.guess_index_group(self.outgroup,allow_misses=True)
        make_operator(out,ingroup=new_ingroup,outgroup=new_outgroup)
        out.to_endomorphism()
        return out
        # Plan: Turn this tensor into an operator.
''' 
    
#%%
def make_operator(T,outgroup=None,ingroup=None):
    """
    Mutates T, no copying.    
    """
    T.outgroup,T.ingroup = T._split_indices(outgroup,ingroup)
    T.__class__ = Operator
    assert len(T.outgroup) + len(T.ingroup) == len(T.indices)
        
        
#%%
'''def operator_contract(*operators,**kwargs):
    T = contract(*operators,**kwargs)
    outgroup=T.get_indices_inherited_from(*[i for o in operators for i in o.outgroup])
    ingroup=T.get_indices_inherited_from(*[i for o in operators for i in o.ingroup])
    make_operator(T,outgroup=outgroup,ingroup=ingroup)
    return T'''

def get_operator_product_bond_maps(*operators):
    """
    constructs a sequence of bond maps to deduplicate bonds 
    for an ordered operator product 
    TODO: Simplify logic of this section
    """
    unmatched = {} # dict{ bond:index_of_origin}
    pairs = []
    for O in operators:
        for idx in O.outgroup:
            if idx.bond.dual in unmatched:
                pairs.append([idx,unmatched.pop(idx.bond.dual)]) # return the pair of indices
                
        for idx in O.ingroup:
            if idx.bond in unmatched:
                raise Exception('Repeated bond in network')
            unmatched[idx.bond] = idx
    
    order_map = {id(O):i for i,O in enumerate(operators)}
    assert len(order_map) == len(operators), "no repeated operators!"

    bond_maps = [{} for _ in operators]
    for i,j in pairs:
        b = Bond(i.bond.dim)
        bond_maps[order_map[id(i.tensor)]][i.bond] = b
        bond_maps[order_map[id(j.tensor)]][j.bond] = b.dual
    return bond_maps
        

def arrange_bonds_ordered(*operators,cyclic=False):
    [o.save_bonds() for o in operators] # so we have two layers of messed-up bonds
    
    unmatched = {} # dict{ bond:index_of_origin}
    pairs = []
    for O in operators:
        for idx in O.outgroup:
            if idx.bond.dual in unmatched:
                pairs.append([idx,unmatched.pop(idx.bond.dual)]) # return the pair of indices
                
        for idx in O.ingroup:
            if idx.bond in unmatched:
                raise Exception('Repeated bond in network')
            unmatched[idx.bond] = idx
        
    if not cyclic:
        for idx in unmatched.values():
            idx.clear() # clean those bonds
            # TODO: figure out how to actually handle cyclicness.
    make_bonds(*pairs)
    
def operator_contract(*operators,mode='bond', out_order=[], skip=set(), extra_edges=(), \
             newnames=(), merge=(),optimize=False, warnbig=True,ordered=False):
    """ 
    Returns an operator, with ingroup coming from surviving ingroups and outgroup from
    surviving outgroups.
    
    If not ordered, can still bond two ingroup indices together if that's what the bonds say.

    If ordered, does acyclic
    Honestly, might be easier to just make a clean copy of everything.
    
    TODO: Add "trace" kwarg that lets you do ordered cyclic contractions.
    
    """
    from .contract import parse_args, arrange_bonds
    from .contract_internals import _contract_many
    
    operators, out_order, skip, extra_edges = \
        parse_args(operators, mode, out_order, skip, extra_edges, newnames, merge)
        
    # Rearrange bonds to match the input description
    need_to_restore = arrange_bonds(operators, mode, skip, extra_edges, out_order)
    if ordered: arrange_bonds_ordered(*operators)

    # Do the contraction
    out_arr,out_indices = _contract_many(operators,out_order=out_order,merge=merge,\
                                         optimize=optimize,warnbig=warnbig)

    
    # Construct the output tensor
    out = Tensor(out_arr, out_indices)
            
    for i,name in enumerate(newnames):
        out.indices[i].name = name
        
    outgroup = out.get_indices_inherited_from(*[i for o in operators for i in o.outgroup])
    ingroup = out.get_indices_inherited_from(*[i for o in operators for i in o.ingroup])
    make_operator(out,outgroup=outgroup,ingroup=ingroup)
        
    # Restore the bonds
    for i in range(ordered + need_to_restore): # do it either once or twice, depending
        bond_map = {j:i for T in operators for i,j in zip(T.old_bond_stack[-1],T.bonds)}
        out.map_bonds(bond_map)
        [T.restore_bonds() for T in operators]

    return out

#%% TODO: Move these to some sort of 'extras' file?
import numpy as np
def identity_operator(outgroup=None,ingroup=None,shape=None,dtype=float):
    """ Returns a operator whose entries are 1 only if the first half of indices 
    match the second half.
    Hmm, perhaps we should permit a shape to be specified?
    Or: Either indices or group1, group2 or both...
    Start with the restricted case. Indices must be all actual indices.
    
    Shape is the half-shape. At least one of shape, ingroup,outgroup must be specified
    TODO: Have it return an endomorphism by default?
    """
    if shape is None:
        if outgroup is None and ingroup is not None:
            outgroup = [(i.dual if isinstance(i,Bond) else i.bond.dual) for i in ingroup]
        elif ingroup is None and outgroup is not None:
            ingroup = [(i.dual if isinstance(i,Bond) else i.bond.dual)  for i in outgroup]
        shape = tuple(i.dim for i in outgroup)
        
    T = Operator(np.zeros(shape*2,dtype=dtype),ingroup=ingroup,outgroup=outgroup)
    
    for i in range(np.prod(shape)):
        idx = np.unravel_index(i,shape)
        T.arr[idx*2] = 1 # add ones where needed
        
    return T
        
def apply_to_all(O,indices,side=None,endo=True):
    """ Applies a single-index endomorphism-like operator O
    to all indices in a tuple (assuming they all share a tensor)
    either from left or right
    
    side is auto-inferred unless specified
    
    preserves bonds if endo is true
    """
    x = indices[0].tensor
    y = O
    O.save_bonds()
    
    if side is None:
        if isinstance(x,Operator): # infer side
            if indices[0] in x.ingroup:
                side = 'right'
            else:
                side = 'left'
        else: # it's a tensor, default to left
            side = 'left'
    else:
        assert side in ('left','right') # watch out for deprecated ingroup/outgroup keywords
        
        
    if side == 'left':
        for idx in indices:
            y.ingroup[0] - idx
            if endo: 
                y.outgroup[0] + idx
            else:
                y.ingroup[0].clear()
            x = y @ x # TODO: Attempt to make this in-place.
    elif side=='right':
        for idx in indices:
            y.outgroup[0] - idx
            if endo: 
                y.ingroup[0] + idx
            else:
                y.ingroup[0].clear()
            x = x @ y
    
    O.restore_bonds()
    return x
        
