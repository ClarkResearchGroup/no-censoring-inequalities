
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:05:32 2022

This version:
    More efficient index construction
    
TODO: Make imatmul call rmatmul of other by default. This way we can dispense
with the SingleSite stuff.

TODO: reconsider bond-making. i-j is easier to use than make_bonds, but weird to look at...    
@author: Daniel
"""

import numpy as np    
from collections import defaultdict
from itertools import chain
# from more_itertools import unzip
from numbers import Number

from numba import njit, types
from numba.typed import Dict


from .tn_utils import multicat
from .contract_internals import _index_index,_tensor_tensor, _tensor_tuple,_contract_self

#%% Classes
class Bond(object): # This is one class that is simple enough to compile...
    def __init__(self,dim,dual=None):
        self.dim = dim
        if dual is None:
            dual = Bond(dim=dim,dual=self)
        self.dual = dual
           
    def __repr__(self):
        return 'Bond(' + 'dim=' + str(self.dim) + ' @ ' + str(id(self))[-5:] + ')'
            
    def set_dim(self,dim): # TODO: change to check_dim.
        if self.dim is not None and self.dim != dim: # verify
            raise Exception('Dimensions of bonded indices must match')
        self.dim = dim
        self.dual.dim = dim
        # TODO: Could use property syntax here
    
class Index(object):
    """
    Each index points to a particular dimension of a particular tensor. Their 
    main role is to be unique, so that we don't need to worry about naming
    collisions. 
    A thought: Could add option for tags like "physical" or "mps internal"
    TODO: Add Index.join() syntax for easier version of make_bonds
    """
    def __init__(self,tensor,position,dim):
        """
        Should be set automatically by _set_indices.
        Possibly I should just switch to T['i'] syntax, but T.i sure is faster to type.
        Attributes:
            tensor
            name
            bond (bond object)
            int (position on the tensor)
            dim (dimension)
            
        You can change name and bond.
        Changing any other attributes will mess things up.
        
        TODO: Considr making this constructor lazier.
        dim could be lazy, and in principle name could be too.
        Yeah, might be smart to have names and bonds both work the same way.                                                            
        On the other hand, bonds only need to work like that to make 
        clear/restore operations painless...
        """
        # Index construction needs to be fast, so this is written a little weirdly.
        # Should be called only when a new tensor is created.
        self.tensor = tensor
        self.int = position
        # assert dim == tensor.arr.shape[position] # verify?
        self.dim = dim # orr just trust...

    
    @property
    def bond(self):
        return self.tensor.bonds[self.int]
    
    @bond.setter
    def bond(self,bond):
        bond.set_dim(self.dim)
        self.tensor.bonds[self.int] = bond
        self.tensor.bond_change_flag = True
        
    @property
    def name(self):
        return self.tensor.names[self.int]
    
    @name.setter
    def name(self,name):
        self.tensor.names[self.int] = name
        self.tensor.name_change_flag = True
            
    def clear(self):
        newbond = Bond(self.dim)
        self.tensor.bonds[self.int] = newbond
        self.tensor.bond_change_flag = True
        
    def join(self,other):
        """ Joins self to dual of other with a fresh bond """
        newbond = Bond(self.dim)
        other.bond = newbond.dual
        self.bond = newbond
        
    def __ior__(self,other): # not sure if this is a good idea or not.
        """ Joins self to dual of other. Doesn't mess with other.."""
        raise Exception('Use - instead')
        self.bond = other.bond.dual
        return self
    
    def __pow__(self,other):
        """ Joins self to dual of other. Doesn't mess with other. 
        TODO: Consider having this make a clean bond instead?
        Returns nothing."""
        raise Exception('Use - instead') 
        
    def __sub__(self,other):
        """ Joins self to dual of other. Doesn't mess with other. 
        TODO: Consider having this make a clean bond instead?
        Returns nothing."""
        if isinstance(other,Index):
            bond = other.bond.dual
        elif isinstance(other,Bond):
            bond = other.dual
        else:
            raise Exception('Bad input')
        self.bond = bond
    
    def __isub__(self,other):
        raise Exception('Deprecated. Use - instead')
        
    def __add__(self,other):
        """ Copies bond from other."""
        if isinstance(other,Index):
            bond = other.bond
        elif isinstance(other,Bond):
            bond = other
        else:
            raise Exception('Bad input')
        self.bond = bond
        
    def __mul__(self,other):
        """ To contract just one single bond, do X.i*Y.j"""
        out_arr,out_indices = _index_index(self,other)
        return Tensor(out_arr, out_indices) 
            
    def __imul__(self,other):
        raise Exception('Not allowed.')
                        
    def __str__(self):
        return 'Idx(' + str(self.name) + ')'
        # return self.__repr__()
        
    def __repr__(self):
        return 'Idx(' + str(self.name) + ', ' + str(self.dim) + ' @ ' + str(id(self))[-5:] + ')'
    
    def __getitem__(self,key):
        """ TODO: make this type-respecting?"""
        T = self.tensor
        j  = [slice(None)]*len(T.indices)
        j[self.int] = key
        arr = T.arr[tuple(j)]
        if len(arr.shape) == len(T.indices)-1:
            indices = T.indices[:self.int] + T.indices[self.int+1:]
        else:
            indices = T.indices[:self.int] + ('sliced_' + self.name,) + T.indices[self.int+1:]
        return Tensor(arr,indices)


#%%
#@profile
def _parse_index_info(x,d):
    """ TODO: Consider allowing (name,bond) pairs """
    if isinstance(x,Index):
        name = x.tensor.names[x.int]
        bond = x.tensor.bonds[x.int]
        assert bond.dim == d, 'Dimensions of bonded indices must match'
    elif isinstance(x,str):
        name = x
        bond = Bond(d)
    elif isinstance(x,Bond):
        name = ''
        bond = x
    elif isinstance(x,tuple):
        print('made it this far!')
        raise Exception
    else:
        raise Exception('Bad input')
    return name,bond

def unzip2(L):
    if L:
        return [list(i) for i in zip(*L)]
    else:
        return [],[]

#%%
class Tensor(object):
    """ Tensor object. 
    The important attribute is the set of indices. Access them by either T.p0
    or T['p0']. Also, T[B] (where B is a Bond) returns a list of indices
    of T which are connected to B. 
    Indices can be renamed using T.p0.name = 'p1'
    They can be joined using T.p0.bond = A.p1.bond or make_bonds((T.p0,A.p1))
    Contractions between tensors will contract over every pair of indices that
    shares a bond.
    
    TODO: autoset bond change flag when user accesses bond list directly?
    or deprecate that behavior.
    """
#%% Initialization
    def __init__(self, input_array, indices=None):
        """ If indices is a list of Index objects, bonds will be copied.
        Array must be... what?
        
        Attributes of input_array required:
            shape
            ndim
            copy
            conj
            np.moveaxis
            
        TODO for next version: Make Tensor.__init__ inflexible and fast,
        but add a make_tensor function that accepts a flexible input format.
            
        """
        if isinstance(input_array,Tensor) and indices is None: 
            indices = input_array.indices
            input_array = input_array.arr
            # Hmm, could have a more efficient copy constructor
            # which copies the bond lists...
            # Hmm...
        elif isinstance(input_array,Tensor):
            input_array = input_array.arr
            
        try:
            input_array = np.asarray(input_array) # make sure it's a numpy array
            # assert np.issubdtype(input_array.dtype, np.number)
        except:
            pass # TODO: Decide if this is a problem or not
            #print('Warning: Could not convert input to Tensor into numeric numpy array')
            #print(input_array)
            # raise Exception('Could not convert input to Tensor into numeric numpy array')
        
        assert all(input_array.shape),'Tensor must have at least one element'
        
        self.arr = input_array
        self._set_indices(input_array.shape,indices)
        self.old_bond_stack = []
        self.bond_change_flag = True # tracks if bonds have been changed since last index dict update
        self.name_change_flag = True # ditto
            
    #@profile
    def _set_indices(self,dims,indices):
        """ Utility to construct a list of Index objects, assign them"""
        if indices is None:
                indices = ['i' + str(i) for i in range(self.arr.ndim)] # default
        else:
            assert len(indices) == len(self.arr.shape), 'Must specify all tensor indices or none'
        
        self.names, self.bonds = unzip2([_parse_index_info(i,d) for i,d in zip(indices,dims)])
        self.indices = tuple(Index(self,i,d) for i,d in enumerate(dims))
        assert all([b.dim == d for b,d in zip(self.bonds,dims)]), \
            "Index or bond dimension does not match array dimension"
        
        
    def __str__(self):
        out = 'Tensor[' # TODO: Make it give its real type name?
        for idx in self.indices:
            out += '(' + str(idx.name) + ',' + str(idx.dim) + '), '
        out += ']'
        return out
    
    def __repr__(self):
        return self.__str__()

#%% Index-accessing    
    def __getitem__(self,key):
        if isinstance(key,Bond):
            return self.idx_by_bond.get(key) 
        elif isinstance(key,str):
            return self.idx_by_name[key] # idk, could come in handy to get indices from name-strings
        elif isinstance(key,dict):
            arr_idx = [slice(None) for _ in self.indices]
            for tidx,aidx in key.items():
                arr_idx[tidx.int] = aidx
            return self.arr[tuple(arr_idx)]
        else:
            raise Exception('Tensors only support [] for index name strings or Bond objects or dicts, not type ' + str(type(key)))
            # return self.arr.__getitem__(key)
            
    def __setitem__(self,key,value):
        """ Use sparingly. """
        assert isinstance(key,dict)
        arr_idx = [slice(None) for _ in self.indices]
        for tidx,aidx in key.items():
            arr_idx[tidx.int] = aidx
        self.arr[tuple(arr_idx)] = value
        
    def __getattr__(self,name):
        if name in ['name_change_flag','idx_by_name']: # avoid infinite recursions
            raise AttributeError('uninitialized tensor somehow?')
        try:
            return self.idx_by_name[name]
        except KeyError:
            raise AttributeError('Tensor ' + str(self) + 'has no index or attribute \'' + name + '\'.')
        
    @property
    def idx_by_name(self):
        """
        Maps from names to indices.
        Lazy property. Exists only once called.
        Currently unused. Not sure if I want to bother with this or not.
        Advantages: Makes it impossible to overwrite a tensor attribute we actually want.
        Disadvantages: Maybe slower? But not every tensor will ever require dict creation...
        """
        if self.name_change_flag:
            self._idx_by_name = dict(zip(self.names,self.indices)) # update as needed
            self.name_change_flag = False
        return self._idx_by_name
    
    @property
    def idx_by_bond(self):
        """
        Maps from bonds to indices.
        Lazy property. Exists only once called.
        """
        if self.bond_change_flag:
            self._idx_by_bond = dict(zip(self.bonds,self.indices))
            self.bond_change_flag = False
        return self._idx_by_bond
        
    def guess_index(self,x,allow_miss=False):
        """
        Tries to guess what you mean. Input can be index of self,
        index of other (with same bond, not dual), bond, or name.
        """
        try:                
            if isinstance(x,Index):
                if x.tensor is self:
                    return x
                else:
                    return self.idx_by_bond[x.bond] # hoping self has an index with the same bond
            elif isinstance(x,str):
                return self.idx_by_name[x]
            elif isinstance(x,Bond):
                return self.idx_by_bond[x] # return the index with this bond
        except Exception as e:
            if not allow_miss:
                raise Exception('guess_index failed to read your mind on input ' + str(x))
            # returns None otherwise
        
    def guess_index_group(self,idx_grp,*,allow_misses=False):
        """
        Permits flexible input. If idx_grp is specified in a uniform way
        (all indices of self, all indices of other...) then don't use this.
        """
        if isinstance(idx_grp,(Index,Bond)): 
            idx_grp = (idx_grp,)
            
        if not set(idx_grp).issubset(self.indices):
            if allow_misses:
                return tuple(j for i in idx_grp if (j := self.guess_index(i,allow_miss=True)))
            else:   
                return tuple(self.guess_index(i) for i in idx_grp) # flexible option
        else:
            return tuple(idx_grp)
        
    def get_all_indices_named(self,*names):
        names = set(names)
        return tuple(i for i in self.indices if i.name in names)
    
    def get_indices_inherited_from(self,*others):
        """ supported inputs: Tensor, Index, Bond, tuple of indices """
        bonds = []
        for o in others:
            if isinstance(o,Tensor):
                bonds.extend(o.bonds)
            elif isinstance(o,Index):
                bonds.append(o.bond)
            elif isinstance(o,Bond):
                bonds.append(o)
            elif isinstance(o,tuple):
                bonds.extend([oo.bond for oo in o])
            else:
                raise Exception('bad input')

        return tuple(j for bond in bonds if (j := self[bond]))
            
    def get_indices_bonded_to(self,*others):
        """ supported inputs: Tensor, Index, Bond, tuple of indices """
        bonds = []
        for o in others:
            if isinstance(o,Tensor):
                bonds.extend(o.bonds)
            elif isinstance(o,Index):
                bonds.append(o.bond)
            elif isinstance(o,Bond):
                bonds.append(o)
            elif isinstance(o,tuple):
                bonds.extend([oo.bond for oo in o])
            else:
                raise Exception('bad input')
        return tuple(j for bond in bonds if (j := self[bond.dual]))

#%% Copying        
    def copy(self,*,idx_map=dict(),free=[],bond_map={},copy_arr=False):
        """
        idx_map is a list of pairs (index of self, index of other)
        from which bonds are copied. (can also be dict.)
        Note that underlying array is not copied. 
        However, the underlying bond list and indices are.
        TODO: Add option to free only certain bonds? 
        TODO: Rename idx_map to bond map?
        Does not preserve subtype. Copyable subclasses will need to write their own.
        """
        out =  Tensor(self) # Gives copy with same bonds and names. Does not pres
        # out = type(self)(self)
        
        if free=='all':
            out.clear_bonds(save=False)
            #bond_map = {j:i for j,i in zip(self.bonds,out.bonds)}
        elif idx_map or free or bond_map:
            if idx_map:
                print('Didn\'t think this got used...')
            
            if isinstance(idx_map, dict): idx_map = idx_map.items()
            bond_map = bond_map.copy() # avoid mutating
            bond_map |= {old.bond:new.bond for old,new in idx_map}
            
            for idx in self.guess_index_group(free):
                bond_map[idx.bond] = Bond(idx.dim)
                
            out.map_bonds(bond_map)
            
        if copy_arr:
            out.arr = out.arr.copy()
        
        return out
    

    def conj(self,*,idx_map={},free=(),bond_map={},return_map=False):
        """
        bond_map is map from bond of original to bond of new.
        TODO: Figure out cleanest way to make this work with...
        idx_map currently takes old.bond.dual:new.bond
        Hmm...
        Can set free to 'all' to get clean bonds.
        
        TODO: 
            A common idx_map use case is: we have some tensor network. We groupconj it.
            Now we modify a tensor X in the first network, then copy that modification over
            into the mirror network.
            For this, we really want idx_map to permit us to specify new bonds, or something.
            Yeah, one option is to have groupconj return a bond map that we can apply.
            Option: Permit a user-specified bond map. Easy-peasy.
            Option: Have groupconj return a bond map that we can apply after.
            
            Ok, ok, so we have our X and Xt. 
            We set bond_map = {a.dual:b for a,b in zip(X.bonds,Xt.bonds)}
            Note that if self is guaranteed same array order as Xt, can do
            self.bonds = Xt.bonds
            and set bond_change_flag=True.
        """
        
        if idx_map:
            raise Exception('Didn\'t think idx_map option got used...')
            
            
        out =  self.copy()
        out.arr = out.arr.conj()
        out.bonds = [b.dual for b in self.bonds] # flip directions on all of them

        if free=='all':
            out.clear_bonds(save=False)
            bond_map = {j.dual:i for j,i in zip(self.bonds,out.bonds)}
        elif idx_map or free or bond_map:
            if isinstance(idx_map, dict): idx_map = idx_map.items()
            bond_map = bond_map.copy() # NOTE: Removing this line causes this function to modify its default bond map every time, very bad.
            bond_map |= {old.bond.dual:new.bond for old,new in idx_map}
            
            for idx in self.guess_index_group(free):
                bond_map[idx.bond.dual] = Bond(idx.dim) # make the dual of this index clean.
                
            out.map_bonds(bond_map)
        
        if return_map:
            
            return out,bond_map
        else:
            return out
    
    def deepcopy(self):
        """
        copies the underlying array. Is this deep enough?
        Does not preserv subclasses...
        """
        return Tensor(self.arr.copy(), self.indices) # Gives copy with same bonds
     
    def copy_arr_from(self,other): # keeps index order the same
        arrs,_ = match_bonds(self,other)
        self.arr = arrs[1]
        
#%% Bond manipulation
    def save_bonds(self):
        self.old_bond_stack.append(self.bonds.copy()) # save a copy, not the originals!
        
    def restore_bonds(self):
        self.bonds = self.old_bond_stack.pop()
        self.bond_change_flag = True
    
    def map_bonds(self,bond_map):
        if bond_map is not None: # Allow a no-mapping-needed shortcut, for compatibility with MPT.
            self.bonds = [bond_map.get(b,b) for b in self.bonds]
            self.bond_change_flag = True
            # a tensor will never have more than ~10 bonds, so this should be fine speed-wise.
            
    def clear_bonds(self,*,return_map=False,save=True):
        """ Returns a dict mapping new:old"""
        newbonds = [Bond(b.dim) for b in self.bonds]
        if return_map: bond_map = dict(zip(newbonds,self.bonds))
        if save: self.save_bonds()
            
        self.bonds = newbonds # the update
        self.bond_change_flag = True
        
        if return_map:
            return bond_map
        
    def copy_bonds(self,other,safe=True):
        """
        Copies over the bonds in order. Relying on index ordering feels
        philosophically out of place, is it useful enough to be worth it?
        """
        if safe:
            assert len(self.bonds) == len(other.bonds)
            assert all([a.dim ==b.dim for a,b in zip(self.bonds,other.bonds)])
        self.bonds = other.bonds
        self.bond_change_flag = True

#%% Index manipulation            
    def merge_indices(self,idx1,idx2, newname='merged_index'):
        """ 
        Return new tensor with idx1, idx2 merged. 
        Equivalent to contracting with some sort of identity-ish tensor, 
        but faster.
        TODO: Consider making in-place by calling self.__init__ at the end?
        """
        if isinstance(idx1,str): idx1 = self[idx1]
        if isinstance(idx2,str): idx2 = self[idx2]
        if newname is None: newname = idx1.name
        # if isinstance(idx1,Bond): idx1 = self[idx1][0] # not sure if we want this
         
        i1 = self.indices.index(idx1)
        i2 = self.indices.index(idx2)
        
        new_indices = list(self.indices)
        new_indices[i1] = newname 
        new_indices.pop(i2)
        
        new_arr = np.moveaxis(self.arr,i2,i1+1) # put the axes next to each oter
        new_arr = new_arr.reshape(new_arr.shape[:i1] + (-1,) + new_arr.shape[i1 + 2:]) # merge them
        return Tensor(new_arr, new_indices) # return view

    def split_index(self,idx,dims,newnames=None):
        """
        Returns a new tensor with idx decomposed into several indices.
        dims is list of dimensions of new indices.
        Equivalent to contracting with some sort of identity-ish tensor, 
        but faster.
        But in general might want to do a unitary as we split or something?
        """
        idx = self.guess_index(idx)
        if newnames is None: newnames = tuple('split' + str(i) for i in range(len(dims)))
        if np.prod(dims) == idx.dim:
            old_arr = self.arr
        else: # need to pad
            target_dim = np.prod(dims)
            assert target_dim > idx.dim, "split_index can\'t match the requested dimensions"
            # print('Warning: Padding with zeros in split_index')
            fill_shape = list(self.arr.shape)
            fill_shape[idx.int] = target_dim - fill_shape[idx.int] # amount we're missing
            old_arr = np.concatenate((self.arr,np.zeros(fill_shape)),axis=idx.int)
            
        
        dims = tuple(dims)
        newnames = tuple(newnames)
        newshape = old_arr.shape
        newshape = newshape[:idx.int] + dims + newshape[idx.int + 1:] # insert the new shape
        new_arr = old_arr.reshape(newshape)
        new_indices = self.indices[:idx.int] + newnames + self.indices[idx.int + 1:]
        
        return Tensor(new_arr,new_indices)
        
    def drop_singleton_index(self,idx):
        assert idx.dim == 1
        new_indices = tuple(i for i in self.indices if i is not idx)
        new_arr = self.arr.reshape(self.arr.shape[:idx.int] + self.arr.shape[idx.int+1:])
        return Tensor(new_arr,new_indices)
    
    def add_singleton_index(self,idx):
        new_indices = self.indices + (idx,)
        new_arr = self.arr.reshape(self.arr.shape+(1,))
        return Tensor(new_arr,new_indices)
    
#%% Arithmetic:
    def __eq__(self,other):
        arrs,_ = match_bonds(self,other)
        return np.allclose(*arrs)
        # TODO: Make this work with symbolic tensors?
    
    def __matmul__(self,other):
        return NotImplemented
        # Don't want to implement this.
    
    def __add__(self,other):
        """
        Elementwise addition, indices inherited from self. Preserves index order
        of first input too.
        """
        if isinstance(other,Tensor):
            assert set(other.bonds) == set(self.bonds)
            out = self.copy()
            out += other
            return out
        elif other == 0: # shortcut for avoiding reduce calls
            return self.copy()
        else:
            return NotImplemented
        
    def __radd__(self,other):
        if other == 0: # shortcut for avoiding reduce calls
            return self.copy()
        else:
            return NotImplemented
            
    def __sub__(self,other):
        return self + -1*other
    
    def __neg__(self):
        return -1*self
        
    def __iadd__(self,other):
        #assert isinstance(other,Tensor)
        # assert set(other.bonds) == set(self.bonds)
        arrs,_ = match_bonds(self,other) # Hmm...
        self.arr = np.add(*arrs)
        return self
    
    def __isub__(self,other):
        arrs,_ = match_bonds(self,other) # Hmm...
        self.arr = np.add(arrs[0],-arrs[1])
        return self
        
    #@profile
    def __mul__(self,other):
        """Scalar multiplication or contraction. Various formats."""
        from sympy import Expr
        if isinstance(other,(Number,Expr)): # Scalar multiplication
            out = self.copy()
            out.arr = out.arr*other
            return out
        elif isinstance(other,Tensor):
            out_arr,out_indices = _tensor_tensor(self,other)
            out = Tensor(out_arr, out_indices)
        elif isinstance(other,Index):
            self_idx = self[other.bond.dual]
            assert self_idx is not None, 'Attempt to contract with unbonded index'
            out_arr,out_indices = _index_index(self_idx,other)
            out = Tensor(out_arr,out_indices)
        elif isinstance(other,tuple):
            out_arr,out_indices = _tensor_tuple(self,other)
        else:
            print('Bad input to tensor.__mul__: ' + str(other))
            return NotImplemented
        
        return Tensor(out_arr, out_indices) 
    
    def __rmul__(self,c):
        return self*c
    
    def __truediv__(self,c):
        return self*(1/c)
    
    def __imul__(self,other):
        if isinstance(other,Number):
            try:
                self.arr *= other
            except: # UFuncTyperror
                self.arr = self.arr*other
            return self 
        else: # nothing fancy
            return self*other
    
    def __itruediv__(self,c):
        try:
            self.arr /= c
        except: # UFuncTyperror
            self.arr = self.arr/c
        return self
    
    def __imatmul__(self,other):
        """ Do other @ self instead of the default self @ other."""
        return other @ self 

    def contract(self,skip=set()):
        """ Contracts only self-bonds """
        if isinstance(skip,Index):
            skip = {skip,}
        else:
            skip = skip if isinstance(skip,set) else set(skip)
            
        out_arr,out_indices = _contract_self(self,skip)
        return Tensor(out_arr,out_indices)
    
    # Things useful for quantum in particular:
    def norm(self):
        """ 2-norm """
        return np.linalg.norm(self.arr)
    
    def scalar(self):
        """ Turns a 1-element tensor into a scalar."""
        assert max(self.arr.shape,default=0) <= 1, 'Can\'t turn multi-element tensor into scalar'
        return np.squeeze(self.arr)[()]
    
    def _split_indices(self,*groups):
        """ Makes sure group1 and group2 are a partition of the indices of T. 
        Inputs can be indices of self, indices of other that share a bond, the bonds
        themselves, or just a string.
        Returns two tuples.
        TODO: rewrite to work with arbitrary # of groups?
        In which case we lose the ability to specify only group1...
        Ah, but then you can do explicit Nones.
        Right, right, use None to fill in one of the groups.
        """
        if groups == (None,):
            return (self.indices,)
        elif groups == (None,None): # Special case is useful.
            n = len(self.indices)//2
            return self.indices[:n], self.indices[n:]
        
        groups = list(groups)
        fill_in = None
        for i,group in enumerate(groups):
            if group is None:
                assert fill_in is None, 'Can\'t leave more than one index group unspecified'
                fill_in = i
                groups[i] = () # default empty
            else:
                groups[i] = self.guess_index_group(group)
                
        full_set = {i for g in groups for i in g}
        
        if fill_in is not None:
            groups[fill_in] = tuple(x for x in self.indices if x not in full_set)
            full_set.update(groups[fill_in])
        
        assert full_set == set(self.indices) \
            and len(full_set) == len(self.indices) \
            and sum([len(i) for i in groups]) == len(self.indices), \
                '_split_indices has failed'
            # Hmm, huh?
        # having a case where we toss in an overcomplete set, and it doesn't complain.
        # right, duh.
        
        return groups
    
    
    def matrix(self,group1=None, group2=None, return_dims=False):
            """
            Turns tensor into a 2D numpy array. 
            group1, group2 are indices, bonds, names, or lists thereof.
            Resulting matrix maps from group2 to group1 using the
            usual convention.
            
            """
            # Standardize arguments:
            group1,group2 = self._split_indices(group1,group2)
                
            # Reshape into matrix:    
            newaxes = [x.int for x in group1] + [x.int for x in group2]
            A = np.transpose(self.arr,axes = newaxes) # reorder the axes
            
            dims1 = tuple(x.dim for x in group1)
            dims2 = tuple(x.dim for x in group2)
            
            A = np.reshape(A,[np.prod(dims1,dtype=int),np.prod(dims2,dtype=int)]) # unroll into a single matrix
            
            try:
                if np.may_share_memory(A,self.arr): # necessary when we overwrite A, which sometimes happens.
                    A = A.copy()
            except Exception: # happens with sparse arrays
                A = A.copy()
            
            if return_dims:
                return A, dims1, dims2
            else:
                return A
            
    def vector(self,order=None,return_dims=False):
        group1, = self._split_indices(order)
            
        # Reshape into matrix:    
        newaxes = [x.int for x in group1]
        A = np.transpose(self.arr,axes = newaxes) # reorder the axes
        
        dims1 = tuple(x.dim for x in group1)
        
        A = np.reshape(A,(np.prod(dims1,dtype=int),)) # unroll into a single matrix
        
        if np.may_share_memory(A,self.arr): # necessary when we overwrite A, which sometimes happens.
            A = A.copy()
        
        if return_dims:
            return A, dims1
        else:
            return A
        
    def array(self,groups,return_dims=None):
        """ Turns tensor into an array with specified index ordering/grouping.
        generalization of Tensor.matrix() and Tensor.vector().
        
        I think T.array(T.indices) == T.arr
        """
        # Standardize arguments:
        groups = self._split_indices(*groups)
            
        # Reshape into matrix:    
        newaxes = [x.int for group in groups for x in group]
        A = np.transpose(self.arr,axes = newaxes) # reorder the axes
        
        dims = [tuple(x.dim for x in group) for group in groups]
        newshape = tuple(np.prod(d,dtype=int) for d in dims)
        
        A = np.reshape(A,newshape) # unroll into a single matrix
        
        try:
            if np.may_share_memory(A,self.arr): # necessary when we overwrite A, which sometimes happens.
                A = A.copy()
        except Exception: # happens with sparse arrays
            A = A.copy()
        
        if return_dims:
            return A, dims
        else:
            return A

#%% Multi-tensor helper functions
def make_bonds(*edges):
    """ Input is of the form (index1,index2). TODO: Maybe remove use_existing option."""
    assert all([len(e) == 2 for e in edges]), "Can only join indices in pairs"
    for edge in edges:
        b = Bond(edge[0].dim)
        edge[0].bond = b
        edge[1].bond = b.dual
        
def bond_by_name(*tensors):
    """ Handles dimension, collision checking automatically"""
    names = defaultdict(lambda : [])
    for tensor in tensors:
        for idx in tensor.indices:
            names[idx.name].append(idx)
            
    edges = [e for e in names.values() if len(e) > 1]
    make_bonds(*edges)

def groupcopy(tensors,*,idx_map=[],free=[],leave=[],snip=False,\
              return_map=False,bond_map=None,copy_arrs=False):
    """ 
    Copies a group of tensors, creating
    new internal bonds. Also, bonds in free will not be matched up.
    free is a list of indices to have bonds cleared. They can be internal or external.
    Note that if an index in free has a repeated bond, that will mess things up.
    If snip=True, then all external indices will also have their bonds cleared. 
    
    TODO: Make the snip=True/free='all' syntax consistent
    
    Hmm, what about when we have repeated internal bonds?
    
    Want to be able to additionally ignore certain indices.
    'leave' kwarg, perhaps?
    Hmm... use case, e.g.:
        Have A @ b, where A is automorphism. Want to move
        the A-b index, while leaving A.outgroup unchanged.
        Can't use bond map to do it. 
        Hmm...
    
    
    """
    leave_bonds = [idx.bond for idx in leave]
    
    bonds = sum([T.bonds for T in tensors],[])
    bond_set = set(bonds)
    assert len(bond_set) == len(bonds), 'Repeated bond in network'
    
    if bond_map is None: bond_map = {}
    
    is_bonded = [bond.dual in bond_set for bond in bonds]
    
    for x,bond in zip(is_bonded,bonds):
            if (snip or x) and bond not in bond_map: # no need to repeat ourselves
                newbond = Bond(bond.dim)
                bond_map[bond] = newbond
                bond_map[bond.dual] = newbond.dual
        
    for idx in free:
        bond_map[idx.bond] = Bond(idx.dim)
        
        
    new_tensors = [T.copy(copy_arr=copy_arrs) for T in tensors]
    for T in new_tensors:
        T.map_bonds(bond_map)
        
    for idx,bond in zip(leave,leave_bonds):
        idx.bond = bond # put these ones back wherever they started.
    
    if return_map:
        return new_tensors, bond_map
    else:
        return new_tensors

def groupconj(tensors,*,idx_map=[],free=[],snip=False,return_map=False,bond_map={}):
    """ 
    Copies a group of tensors, creating
    new internal bonds. Also, bonds in free will not be matched up.
    idx_map is list of pairs of indices (old,new) to have bonds moved.
    TODO: Currently idx_map takes old.bond.dual:new.bond. We may want it to take old.bond.dual:new.bond.dual. Hmm...
    
    free is a list of indices to have bonds cleared. They can be internal or external.
    Note that if an index in free has a repeated bond, that will mess things up.
    If snip=True, then all external indices will also have their bonds cleared. 
    
    Can use bond_map to supply an explicit map, which allows you to match part or all of
    an earlier groupconj.
    
    Hmm, question: Should further indices be ignored if explicit map is provided?
    No, further internal indices should also be mapped
    """
    
    if bond_map is None: bond_map = {} # TODO: Clean this up too?
        
    bonds = sum([T.bonds for T in tensors],[])
    bond_set = set(bonds)
    assert len(bond_set) == len(bonds), 'Repeated bond in network'
    
    
    is_bonded = [bond.dual in bond_set for bond in bonds] 
    # TODO: Can make this faster with listcomps.
    bond_map = bond_map.copy()
    for x,bond in zip(is_bonded,bonds):
            if (snip or x) and bond.dual not in bond_map: # no need to repeat ourselves
                newbond = Bond(bond.dim)
                bond_map[bond] = newbond
                bond_map[bond.dual] = newbond.dual
    
    for idx in free:
        bond_map[idx.bond.dual] = Bond(idx.dim) # make the dual of this index clean.
        
    new_tensors = [T.conj() for T in tensors] # conjugates all the bonds by default
    for T in new_tensors: # and then move some internal bonds
        T.map_bonds(bond_map)
     
    if return_map:
        return new_tensors, bond_map
    else:
        return new_tensors
    
def find_free_bonds(*tensors):
    """ 
    Tells you which indices will survive contraction and which tensors they
    came from.
    TODO: Decide if this should return bonds or indices.
    TODO: Decide if this function is worth having.
    """
    bonds = sum([T.bonds for T in tensors],[])
    bond_set = set(bonds)
    assert len(bond_set) == len(bonds), 'Repeated bond in network'
    
    is_bonded = {bond:(bond.dual in bond_set) for bond in bonds}
    return [[b for b in T.bonds if not is_bonded[b]] for T in tensors]                                       
    
    

def map_external_bonds(tensors,bond_map):
    raise Exception('Not sure I need this... can finish writing later if useful.')
    bond_map = bond_map.copy()
    
    bonds = sum([T.bonds for T in tensors],[])
    bond_set = set(bonds)
    assert len(bond_set) == len(bonds), 'Repeated bond in network'
    
    is_bonded = [bond.dual in bond_set for bond in bonds]
    
    for x,bond in zip(is_bonded,bonds):
        if x:
            bond_map.pop(bond,None) # remove internal bonds from the dict.
    

def grouprestore(old,*,new=()):
    """
    Restores bonds using the most recent save of old. Also maps bonds for new accordingly.
    """
    if new:
        bond_map = dict(chain(*[zip(T.bonds,T.old_bond_stack[-1]) for T in old]))    
        [T.map_bonds(bond_map) for T in new]
    [T.restore_bonds() for T in old]


def match_names(*tensors):
    """ Given N tensors with the same set of (unique) names, returns N numpy arrays permuted
    to the same order. Also returns the names"""
    
    arrs = []
    name_pos = {idx.name:i for i,idx in enumerate(tensors[0].indices)}
    permutation = np.empty(len(tensors[0].indices),dtype=int) # used for numpy.transpose
    for i,T in enumerate(tensors): # could skip first one
        for idx in T.indices: 
            permutation[idx.int] = name_pos[idx.name]
            
        arrs.append(np.transpose(T.arr, permutation))
        
    return arrs, list(name_pos.keys())

#@profile
def match_bonds(*tensors):
    """ Given N tensors with the same set of (unique) bonds, returns N numpy arrays permuted
    to the same order. Also returns the bonds.
    Preserves order...
    TODO: Maybe remove bond-returning? if not used.
    also let's special-case the especially common n=2 version.
    """
    try:
        arrs = []
        T0 = tensors[0] # the template
        bond_pos = {bond:i for i,bond in enumerate(T0.bonds)}
        permutation = np.empty(len(bond_pos),dtype=int) # used for numpy.transpose
        arrs = [T0.arr]
        for T in tensors[1:]: # skip first one
            #for idx in T.indices: 
                #permutation[bond_pos[idx.bond]] = idx.int
            for i,bond in enumerate(T.bonds):
                permutation[bond_pos[bond]] = i
            arrs.append(np.transpose(T.arr, permutation))
            
        return arrs, list(bond_pos.keys())
    except KeyError:
        raise Exception('Bonds do not match')
    except ValueError:
        raise Exception('Cannot compare tensors with repeated bonds... or possibly bonds just dont match?')
        # TODO: Get these error messages right.
        
@njit
def compute_permutations_numba(bond_labels):
    """Compute axis permutations to align tensors with template (first tensor)."""
    n_tensors, n_bonds = bond_labels.shape
    permutations = np.zeros((n_tensors - 1, n_bonds), dtype=np.int64)
    template = bond_labels[0]
    
    # Build dictionary: {bond_id -> template_index}
    template_dict = Dict.empty(types.int64, types.int64)
    for idx, bond_id in enumerate(template):
        template_dict[bond_id] = idx
    
    # Compute permutations for subsequent tensors
    for i in range(n_tensors - 1):
        current_bonds = bond_labels[i + 1]
        perm = np.empty(n_bonds, dtype=np.int64)
        
        for j in range(n_bonds):
            bond_id = current_bonds[j]
            perm[template_dict[bond_id]] = j
            
        permutations[i] = perm
    
    return permutations

#@profile
def match_bonds_numba(*tensors):
    """
    Alternate version of match_bonds, empirically seems slower.
    """
    try:
        bond_labels = np.array([[id(b) for b in T.bonds] for T in tensors])
        permutations = compute_permutations_numba(bond_labels)
        arrs = [tensors[0].arr] + [np.transpose(T.arr, perm) 
                                   for T, perm in zip(tensors[1:], permutations)]
        return arrs, tensors[0].bonds.copy()
    except Exception as e:
        raise Exception(f'Error matching bonds: {e}')
     #%%
def tensor_sum(*tensors):
    """ add several tensors, faster than sum() """
    out = tensors[0].copy()
    bond_labels = np.array([[id(b) for b in T.bonds] for T in tensors])
    permutations = compute_permutations_numba(bond_labels)
    for T,perm in zip(tensors[1:],permutations):
        out.arr += np.transpose(T.arr, perm)
    return out
        
def direct_sum(tensors,match_indices,stack_indices):
    """
    Like block diagonal. Handy for MPS adition.
    
    match_indices, stack_indices are both lists of tuples indices
    with each tuple containing (exactly?) one index from each tensor
    match_indices = [(T0.i, T1.j), (T0.k, T1.i)...]
    and they are matched up in sets as specified.
    
    TODO: Specify only stack_indices. Handle matching automatically by bonds. 
    """
    out_arr = multicat([t.arr for t in tensors],\
                       [[i.int for i in j] for j in match_indices],\
                       [[i.int for i in j] for j in stack_indices])
        
    new_indices = tuple(i[np.argmax([j.dim for j in i])] for i in match_indices) \
        + tuple(i[0].name for i in stack_indices)
        # can't copy bonds for the stacked indices, since dimension is different.
        # Of course, if we take "bonds are vector spaces" seriously,
        # there'd be a natural direct-product space...
    
    return Tensor(out_arr,new_indices)


def tensor_stack(*tensors, newname = 'i'):
    """
    Automatically matches by bond
    and creates one new index.
    can we use multicat?

    """
    pass
    
    

