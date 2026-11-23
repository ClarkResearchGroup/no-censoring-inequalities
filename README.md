# no-censoring-inequalities

Code to reproduce the numerical results of [arXiv:2502.15995](https://arxiv.org/abs/2502.15995). Multiplicative error calculations use an algorithm similar to that described in [arXiv:2510.23726](https://arxiv.org/abs/2510.23726).

## Contents

| Script | Reproduces |
|---|---|
| `figures/fig1b_spectral_gap.py` | Fig. 1b: spectral gap vs. number of qubits, with and without the red gates |
| `figures/fig1c_multerr_vs_qubits.py` | Fig. 1c: multiplicative error of one period vs. number of qubits |
| `figures/fig3_multerr_crossover.py` | Fig. 3: crossover of multiplicative error over several periods |
| `figures/fig4b_mpemba_trajectory.py` | Fig. 4b: multiplicative error vs. number of gates for the circuit of Fig. 4a |
| `figures/fig5_minimal_circuit_eigenvalues.py` | Fig. 5 caption: largest non-unit eigenvalues 0.2512 and 0.2423 (printed) |
| `figures/fig6_superminimal_multerr.py` | Fig. 6 caption: multiplicative errors 2.91 and 1.96 (printed) |
| `figures/fig7b_lollipop_gap.py` | Fig. 7b: spectral gaps of the linear and lollipop graphs |
| `figures/lemma3.py` | Proof of Lemma 1: symbolic eigenvalues of a three-site brickwork with local dimensions (Q1, Q2, Q3) (printed) |

The circuit and graph diagrams (Figs. 1a, 2, 4a, 5, 6, 7a) are schematics and are not generated here. Plots use matplotlib's default style rather than the paper's.

`utils/` contains the supporting code: `tensor_network/` (a small tensor-network library) and `permutations/` (moment operators of Haar-random gates in the permutation basis).

## Running

Requires Python 3.11 or later, with numpy, scipy, matplotlib, numba, sympy and opt_einsum.

```sh
python figures/fig1b_spectral_gap.py
```

Scripts can be run from any directory. Plots are saved to `figures/output/` and also shown on screen. Each script runs in a few seconds, except `fig7b_lollipop_gap.py`, which takes about 8 minutes (set `NMAX` in the script to a smaller value for a quicker run).
