"""
Numbers quoted in the caption of Figure 6 of the paper: a 7-qubit circuit that violates censoring for
multiplicative error. At t=2 the multiplicative error is 2.91 with the red
gate and 1.96 without it (the red gate is deleted in the "censored" circuit).

Provenance: random_circuit_concentration/scripts/s01_31_multerr_censor_search.py
(last cell; the random-search cells and earlier candidate sequences were removed).
Commit in the old repo: unknown.

Deviations from the original: dead code and unused helpers removed, imports made
absolute from utils/, connectivity assert dropped. The circuit diagram
itself is a schematic and is not generated here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from numpy.fft import fftn
from itertools import product

from utils.tensor_network.tn import Bond
from utils.tensor_network.operators import Operator, operator_contract
from utils.permutations.permutation_metrics import (
    permutation_basis_projector, metric_pseudoinverse)


def get_haar_comparison_templates(N):
    """Boolean masks selecting the entries of the 2N-index array (even total
    parity) split by the parity of the first N indices."""
    cut = N
    m = 2*cut
    masks = np.zeros((2,)*(m + 1), dtype=bool)
    for ii in product(range(2), repeat=m-1):
        jj = ii + (np.sum(ii) % 2,)  # force even total parity
        hemiparity = np.sum(jj[:cut]) & 1
        masks[hemiparity, *jj] = True
    return masks


def gates_from_sequence(sequence, t=2, q=2):
    """One 2-qubit gate (permutation-basis projector) per pair in `sequence`."""
    N = np.max(np.asarray(sequence)) + 1
    bonds = np.array([Bond(t) for _ in range(N)])

    A = permutation_basis_projector(n=2, q=2, t=2)
    gates = []
    for pair in sequence:
        U = A.copy(free='all')
        for k in range(2):
            U.ingroup[k] - bonds[pair[k]]
            U.outgroup[k] - U.ingroup[k]
        gates.append(U)
    return bonds, gates


N = 7
q = 2
t = 2
sequence = [(0, 1), (2, 3), (4, 5), (6, 5), (3, 4), (1, 2), (0, 2), (1, 5)]
delete = 5  # index of the gate (1,2) in `sequence`: the red gate

W = metric_pseudoinverse(q**N, t, mode='svd')
h = np.diag(fftn(W.arr))  # Haar eigenvalues
masks = get_haar_comparison_templates(N)

g0 = metric_pseudoinverse(q, t, mode='svd')
f0 = Operator([[1, 1], [1, -1]])

bonds, gates = gates_from_sequence(sequence)

gg = []
ff = []
for b in bonds:
    g = g0.copy()
    g.outgroup[0] + b
    g.ingroup[0] - g.outgroup[0]
    gg.append(g)

    F = f0.copy()
    F.outgroup[0] + b
    F.ingroup[0] - F.outgroup[0]
    ff.append(F)

ffcopy = [f.copy() for f in ff]


def mult_err_of(gate_list):
    C = operator_contract(*ffcopy, *gate_list, *gg, *ff, ordered=True, optimize='greedy')
    ratios = np.array([C.arr[mask]/hh for mask, hh in zip(masks, h)])
    return np.max(np.abs(ratios - 1))


mult_err = mult_err_of(gates)
mult_err_censored = mult_err_of(gates[:delete] + gates[delete+1:])
print(f"multiplicative error with red gate: {mult_err:.3f}, without: {mult_err_censored:.3f}, "
      f"ratio {mult_err/mult_err_censored:.3f}")

