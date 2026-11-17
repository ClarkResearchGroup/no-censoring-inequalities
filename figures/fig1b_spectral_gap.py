"""
Paper figure 1, panel (b): t=2 spectral gap vs. number of qubits, for the
three-site circuit with gates (1,2), (2,3), (1,3), with and without deletion.
Site 1 has a variable number of qubits (q3 = local dimension 2..65, plotted as
N = 2 + log2(q3)); sites 2 and 3 are single qubits.
Legend: 'Without deletion' / 'With deletion'.
Output: output/fig1b_spectral_gap.pdf (panel b only; the paper's
t2_choi_spectra_and_spectral_gap.pdf combines panels b and c).

Provenance: random_circuit_concentration/scripts/2024/s11_29_nocensor.py
(the get_ssv / svdvals cell; the earlier eigvals cell is dropped).

Deviation from the original code: the original plots 1 - sigma2 (largest
non-unit singular value). The paper plots 1 - sigma2**2 (the gap of the
palindromic period G1 G2 G3 G2 G1 defined in Sec. 4 of arXiv:2502.15995), so the
gap is computed as 1 - sigma2**2. The resulting curves agree with the closed
forms for lambda and lambda' in Lemma 1 of the paper.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svdvals

from utils.tensor_network.tn import Bond
from utils.tensor_network.operators import operator_contract
from utils.permutations.permutation_metrics import half_cluster_projector_multiq

t = 2
q = 2


def get_second_sv(bonds, *gates):
    """Largest singular value below 1 of the product of gates."""
    O = operator_contract(*gates, ordered=True)
    duals = [b.dual for b in bonds]
    O = O.matrix(group1=bonds, group2=duals)
    s = svdvals(O)
    return np.max(s[s <= 0.999]).real


q3vals = np.linspace(2, 65, 500)
gaps = []
for q3 in q3vals:
    qvals = (q3, q, q)
    bonds = [None] * 3

    def get_gate(i, j):
        half = half_cluster_projector_multiq((qvals[i], qvals[j]), t)
        for site, idx in zip((i, j), half.outgroup):
            if bonds[site] is None:
                bonds[site] = Bond(idx.dim)
            idx - bonds[site]
        return half @ half.conj()

    U12 = get_gate(0, 1)
    U23 = get_gate(1, 2)
    U13 = get_gate(0, 2)

    a = get_second_sv(bonds, U13, U23, U12)
    b = get_second_sv(bonds, U13, U12)
    gaps.append((1 - a**2, 1 - b**2))

gaps = np.array(gaps)
nvals = 2 + np.log(q3vals) / np.log(q)

fig, ax = plt.subplots()
ax.plot(nvals, gaps)
ax.set_xlabel('Number of qubits')
ax.set_xticks(np.round(nvals))
ax.set_xlim(np.min(nvals), np.max(nvals))
ax.set_ylabel('Spectral gap')
ax.legend(['Without deletion', 'With deletion'])
ax.grid()

for n in (3, 4, 8):
    i = np.argmin(np.abs(nvals - n))
    print(f'N={n}: without={gaps[i, 0]:.4f}, with={gaps[i, 1]:.4f}')

out = Path(__file__).resolve().parent / 'output'
out.mkdir(exist_ok=True)
fig.savefig(out / 'fig1b_spectral_gap.pdf', bbox_inches='tight')
plt.show()
