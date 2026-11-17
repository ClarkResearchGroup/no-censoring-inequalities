"""
Numbers quoted in the caption of Figure 5 of the paper: a 5-qubit
circuit (drawn in original_paper/plots/minimal_circuit.png) that violates the censoring inequality for spectral gaps. Deleting the
final (red) gate changes the largest non-unit eigenvalue of the second-moment
operator from 0.2512 (with the gate) to 0.2423 (without it).

Provenance (old repo random_circuit_concentration):
scripts/2023/s01_31_censorship_search.py (last cell). Commit: not recorded.

Deviations from the original: random-search cells and unused helpers removed.
The circuit diagram itself is a schematic and is not generated here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.tensor_network.tn import Bond
from utils.tensor_network.operators import operator_contract
from utils.tensor_network.svd import eigvals
from utils.permutations.permutation_metrics import half_cluster_projector


def gates_from_sequence(sequence, t=2):
    """One two-qubit gate (second-moment operator) per pair of qubits, applied in order."""
    N = np.max(np.asarray(sequence)) + 1
    bonds = np.array([Bond(t) for _ in range(N)])

    B = half_cluster_projector(n=2, q=2, t=2)
    A = B @ B.conj()
    gates = []
    for pair in sequence:
        U = A.copy(free='all')
        for k in range(2):
            U.ingroup[k] - bonds[pair[k]]
            U.outgroup[k] - U.ingroup[k]
        gates.append(U)
    return gates


def largest_nonunit_eigenvalue(gates, tol=1e-6):
    M = operator_contract(*gates, ordered=True, optimize=True)
    e = eigvals(M)
    return M, np.max(e.real[~np.isclose(e, 1)]), np.sum(e > 1 - tol)


sequence = [(0, 1), (3, 2), (3, 4), (2, 1), (0, 3), (0, 2)]
delete = 5  # the final gate (red in the figure)

# Eigenvalues with and without the final gate
gates = gates_from_sequence(sequence)
M, emax1, n1 = largest_nonunit_eigenvalue(gates)
gates = gates[:delete] + gates[delete + 1:]
M2, emax2, n2 = largest_nonunit_eigenvalue(gates)
assert n1 == n2, 'Connectivity changed'
print(f'with final gate: {emax1:.4f}, without: {emax2:.4f}, ratio {emax1/emax2:.4f}')

