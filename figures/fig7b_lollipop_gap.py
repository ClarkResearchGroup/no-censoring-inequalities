"""
Figure 7(b): spectral gap scaling, N(1 - lambda) vs number of qubits N (log y),
for the linear graph and the lollipop graph (q=2, t=2).

Provenance: random_circuit_concentration/
scripts/2024/s04_27_lollipop_numerics.py (commit ef4dfd6 "Reorganizing" is the
latest touching it).

Deviations from the original: @file_cache removed (get_sev is recomputed each
run); only the plot used in the paper is kept; imports and paths updated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functools import cache
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import factorial

from utils.tensor_network.tn import Bond, Tensor
from utils.permutations.permutation_metrics import cluster_projector

NMAX = 20  # paper uses 20


def graphsample_LO(edges, q=2, t=2):
    n = np.max(edges) + 1
    U = cluster_projector(2, q, t)

    d = U.ingroup[0].dim

    sites = [Bond(d) for _ in range(n)]
    template = Tensor(np.zeros((d,) * n), sites)
    out = Tensor(np.zeros((d,) * n), sites)

    c = 1 / len(edges)  # normalizer

    def matvec(v, out=out):
        template.arr = c * np.reshape(v, (d,) * n)
        out.arr[:] = 0
        for edge in edges:
            [i - sites[j] for i, j in zip(U.ingroup, edge)]
            [i + sites[j] for i, j in zip(U.outgroup, edge)]
            out += U @ template
        return out.vector(sites)

    return LinearOperator(shape=(d ** n,) * 2, matvec=matvec, dtype=float)


@cache
def get_sev(edges, q=2, t=2):
    """Smallest of the top t!+1 eigenvalues of the averaged projector."""
    tf = int(factorial(t))
    LO = graphsample_LO(edges, q, t)
    e, v = eigsh(LO, k=tf + 1)
    return np.min(e)


def linear_graph(n):
    return tuple((i, i + 1) for i in range(n - 1))


def lollipop_graph(n):
    edges = []
    for i in range(n // 2):
        for j in range(i):
            edges.append((j, i))
    for i in range(n // 2 - 1, n - 1):
        edges.append((i, i + 1))
    return tuple(edges)


q = 2

nlin = np.arange(3, NMAX + 1)
elin = []
t0 = time()
for n in nlin:
    elin.append(get_sev(linear_graph(n), q, t=2))
    print('linear', n, time() - t0, flush=True)

nlol = np.arange(4, NMAX + 1)
elol = []
t0 = time()
for n in nlol:
    elol.append(get_sev(lollipop_graph(n), q, t=2))
    print('lollipop', n, time() - t0, flush=True)

elin = np.array(elin)
elol = np.array(elol)

fig, ax = plt.subplots()
ax.plot(nlin, nlin * (1 - elin), '.-')
ax.plot(nlol, nlol * (1 - elol), '.-')
ax.set_yscale('log')
ax.set_xlabel('Number of qubits')
ax.set_ylabel(r'$N (1 - \lambda)$')
ax.legend(['Linear graph', 'Lollipop graph'])
ax.set_xticks(nlin[1::2])

out_dir = Path(__file__).resolve().parent / 'output'
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / 'lollipop_gap.png', dpi=200, bbox_inches='tight')
plt.show()
