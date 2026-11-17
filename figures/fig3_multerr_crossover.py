"""
Figure 3: crossover of the multiplicative error for the 5-qubit circuit of
Fig. 1. Multiplicative error of the complete circuit versus the censored
circuit (gates 1 and 3 of each period deleted), as a function of the number
of periods.

Provenance: random_circuit_concentration/scripts/s02_03_multerr_crossover_plots.py
(last commit 6aafad2), first plotting cell only.

Deviations from the original: only the code for this figure is kept; the later
thick-line plotting cell, unused imports/helpers and dead code were removed;
the figure is saved to figures/output/multerr_crossover.png.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from itertools import product

import numpy as np
from numpy.fft import fftn
import matplotlib.pyplot as plt

from utils.tensor_network.tn import Bond
from utils.tensor_network.operators import Operator, operator_contract
from utils.permutations.permutation_metrics import (
    permutation_basis_halves_multiq, metric_pseudoinverse)


def get_haar_comparison_templates(N):
    cut = N
    m = 2*cut
    masks = np.zeros((2,)*(m + 1), dtype=bool)
    for ii in product(range(2), repeat=m-1):
        jj = ii + (np.sum(ii) % 2,)  # force even total parity
        hemiparity = np.sum(jj[:cut]) & 1
        masks[hemiparity, *jj] = True
    return masks


def gates_from_sequence(sequence, qvals, t=2):
    """Many-site gates, different local dimensions qvals."""
    N = np.max(np.asarray(sequence)) + 1
    bonds = np.array([Bond(t) for _ in range(N)])
    gates = []
    for group in sequence:
        X, Y = permutation_basis_halves_multiq([qvals[site] for site in group], t=t)
        U = X @ Y
        for k in range(len(group)):
            U.ingroup[k] - bonds[group[k]]
            U.outgroup[k] - U.ingroup[k]
        gates.append(U)
    return gates, bonds


q = 2
q2 = 2**3
t = 2
sequence = [(1, 2), (0, 1), (0, 2), (0, 1), (1, 2)]
qvals = (q, q, q2)
N = 3
delete = (1, 3)
reps = 4

gates, bonds = gates_from_sequence(sequence, qvals, t=t)

W = metric_pseudoinverse(np.prod(qvals), t, mode='svd')
h = np.diag(fftn(W.arr))  # haar eigenvalues
masks = get_haar_comparison_templates(N)

gg = []
ff = []
for b, qq in zip(bonds, qvals):
    g = metric_pseudoinverse(qq, t, mode='svd')
    g.outgroup[0] + b
    g.ingroup[0] - g.outgroup[0]
    gg.append(g)

    F = Operator([[1, 1], [1, -1]])
    F.outgroup[0] + b
    F.ingroup[0] - F.outgroup[0]
    ff.append(F)

post = operator_contract(*ff, ordered=True)


def multiplicative_errors(skip=()):
    errs = []
    state = operator_contract(*gg, *ff, ordered=True)
    for rep in range(reps):
        for i, gate in enumerate(gates):
            if i not in skip:
                state = gate @ state
            C = post @ state
            ratios = np.array([C.arr[mask]/hh for mask, hh in zip(masks, h)])
            errs.append(np.max(np.abs(ratios - 1)))
    return errs


mult_err = multiplicative_errors()
mult_err_censored = multiplicative_errors(skip=delete)

ngates = np.arange(len(mult_err))/len(gates)
plt.plot(ngates, mult_err, '.-')
plt.plot(ngates, mult_err_censored, '.-')
plt.yscale('log')
plt.legend(['Complete', 'Censored'])
plt.xlabel('Period')
plt.ylabel('Multiplicative error')
plt.grid()

out = Path(__file__).resolve().parent / 'output'
out.mkdir(exist_ok=True)
plt.savefig(out / 'multerr_crossover.png', dpi=150)
plt.show()
