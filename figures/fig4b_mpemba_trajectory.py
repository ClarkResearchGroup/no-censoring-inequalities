"""
Figure 4 of arXiv:2502.15995: Mpemba-like trajectory example.

Panel (b): multiplicative error vs. number of gates for the 13-gate, N=8
circuit of panel (a), with and without the red gate (sequence index 6),
log y-axis. Panel (a) is a schematic and is not generated here.

Output: figures/output/mpemba_trajectory.png

Provenance: random_circuit_concentration/scripts/s02_06_neat_trajectory_search.py
(the cell with the 13-gate sequence and delete=6). Commit unknown.

Deviations from the original: pattern-based sequence, the random search over
circuits, and the later restyled plot cell were removed; axis label changed to
'Number of gates' and the grid removed to match the paper.
Note: the paper caption says the first five points are omitted, but the code
(kept as is) omits the first four (errors[4:]).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from numpy.fft import fftn
from itertools import product
import matplotlib.pyplot as plt

from utils.tensor_network.tn import Bond
from utils.tensor_network.operators import Operator, operator_contract
from utils.permutations.permutation_metrics import (
    permutation_basis_projector, metric_pseudoinverse)

OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

N = 8
q = 2
t = 2

# 13-gate sequence on 8 qubits; gate index 6 is the "red" gate
sequence = [(0, 1), (3, 2), (5, 4), (6, 7), (2, 5), (4, 3), (2, 0),
            (3, 5), (6, 5), (1, 2), (0, 4), (3, 7), (6, 3)]
delete = 6


def get_haar_comparison_templates(N):
    """Boolean masks selecting the even-parity entries, split by hemiparity."""
    cut = N
    m = 2 * cut
    masks = np.zeros((2,) * (m + 1), dtype=bool)
    for ii in product(range(2), repeat=m - 1):
        jj = ii + (np.sum(ii) % 2,)  # force even total parity
        hemiparity = np.sum(jj[:cut]) & 1
        masks[hemiparity, *jj] = True
    return masks


def gates_from_sequence(sequence):
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


def prep_states(bonds):
    pre = []
    post = []
    for b in bonds:
        g = metric_pseudoinverse(q, t, mode='svd')
        g.outgroup[0] + b
        g.ingroup[0] - g.outgroup[0]

        F = Operator([[1, 1], [1, -1]])
        F.outgroup[0] + b
        F.ingroup[0] - F.outgroup[0]

        pre.append(g @ F)
        post.append(F)
    return pre, post


W = metric_pseudoinverse(q**N, t, mode='svd')
h = np.diag(fftn(W.arr))  # Haar eigenvalues
masks = get_haar_comparison_templates(N)


def multerr_sequence(gates, bonds, skip=[]):
    """Multiplicative error after each gate, skipping gates whose index is in skip."""
    pre, post = prep_states(bonds)

    errors = []
    state = operator_contract(*pre, ordered=True, optimize='greedy')
    for i, gate in enumerate(gates):
        if i not in skip:
            state = gate @ state
        C = operator_contract(*post, state, ordered=True, optimize='greedy')
        ratios = np.array([C.arr[mask] / hh for mask, hh in zip(masks, h)])
        errors.append(np.max(np.abs(ratios - 1)))
    return errors


# Error trajectories
bonds, gates = gates_from_sequence(sequence)
errors = multerr_sequence(gates, bonds)
errors2 = multerr_sequence(gates, bonds, skip=[delete])

gvals = np.arange(4, len(errors)) + 1
fig_t, ax_t = plt.subplots()
ax_t.plot(gvals, errors[4:], '.-')
ax_t.plot(gvals, errors2[4:], '.-')
ax_t.set_yscale('log')
ax_t.set_ylabel('Multiplicative error')
ax_t.set_xlabel('Number of gates')
ax_t.legend(['With red gate', 'Without'])
fig_t.savefig(OUT_DIR / 'mpemba_trajectory.png', dpi=200, bbox_inches='tight')

plt.show()
