"""
Figure 1(c): multiplicative error of the t=2 moment operator of a single period
of the Fig. 1(a) architecture, with and without the red gates, vs. number of qubits N.

The architecture is modelled as 3 sites: two qubits (sites 0, 1) and one site of
dimension q2 = 2**(N-2) standing in for the remaining N-2 qubits. One period is
the gate sequence [(1,2), (0,1), (0,2), (0,1), (1,2)]; "with deletion" removes the
red gates, i.e. those at indices 1 and 3. q2 is swept continuously, as in Fig. 1(b),
so N = 2 + log2(q2) is non-integer between the qubit counts.

Provenance: adapted from scripts/s02_03_multerr_crossover_plots.py in
random_circuit_concentration. No script for this panel survives; the original
s02_03 code computes the multiplicative error after each gate for fixed q2 = 8 over
4 periods. Deviations from it: only one period is applied, the error is evaluated
only at the end of the period, and q2 is swept over linspace(2, 65, ...) like the
Fig. 1(b) script (scripts/2024/s11_29_nocensor.py). The original plotting cells
(error vs. gate count) are removed.
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


def get_haar_comparison_templates(n_sites):
    """Boolean masks selecting the Fourier components of the moment operator that
    should equal the Haar value of each parity sector."""
    m = 2 * n_sites
    masks = np.zeros((2,) * (m + 1), dtype=bool)
    for ii in product(range(2), repeat=m - 1):
        jj = ii + (np.sum(ii) % 2,)  # force even total parity
        hemiparity = np.sum(jj[:n_sites]) & 1
        masks[hemiparity, *jj] = True
    return masks


def gates_from_sequence(sequence, qvals, t=2):
    """Haar-random gates (as t-th moment operators) on the given groups of sites."""
    n_sites = np.max(np.asarray(sequence)) + 1
    bonds = np.array([Bond(t) for _ in range(n_sites)])
    gates = []
    for group in sequence:
        X, Y = permutation_basis_halves_multiq([qvals[site] for site in group], t=t)
        U = X @ Y
        for k in range(len(group)):
            U.ingroup[k] - bonds[group[k]]
            U.outgroup[k] - U.ingroup[k]
        gates.append(U)
    return gates, bonds


def multiplicative_error(qvals, sequence, delete=(), t=2):
    """Multiplicative error of one period of `sequence` (minus gates in `delete`)."""
    gates, bonds = gates_from_sequence(sequence, qvals, t=t)

    W = metric_pseudoinverse(np.prod(qvals), t, mode='svd')
    h = np.diag(fftn(W.arr))  # Haar eigenvalues
    masks = get_haar_comparison_templates(len(qvals))

    gg, ff = [], []
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
    state = operator_contract(*gg, *ff, ordered=True)
    for i, gate in enumerate(gates):
        if i not in delete:
            state = gate @ state
    C = post @ state
    ratios = np.array([C.arr[mask] / hh for mask, hh in zip(masks, h)])
    return np.max(np.abs(ratios - 1))


#%%
q = 2
t = 2
sequence = [(1, 2), (0, 1), (0, 2), (0, 1), (1, 2)]
delete = (1, 3)  # the red gates

q2vals = np.linspace(2, 65, 500)
errs = np.array([[multiplicative_error((q, q, q2), sequence, delete=d, t=t)
                  for d in ((), delete)] for q2 in q2vals])
nvals = 2 + np.log(q2vals) / np.log(q)

for n in range(3, 9):
    a, b = (multiplicative_error((q, q, 2.0**(n - 2)), sequence, delete=d, t=t)
            for d in ((), delete))
    print(f'N={n}: without deletion {a:.4f}, with deletion {b:.4f}')

#%%
plt.plot(nvals, errs, linewidth=3)
plt.xlabel('Number of qubits')
plt.ylabel('Multiplicative Error')
plt.legend(['Without deletion', 'With deletion'])
plt.grid()

out = Path(__file__).resolve().parent / 'output' / 'fig1c_multerr_vs_qubits.pdf'
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches='tight')
plt.show()
