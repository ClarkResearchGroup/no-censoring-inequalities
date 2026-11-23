"""
Symbolic eigenvalues of the t=2 moment operator of a three-site brickwork
with local dimensions (Q1, Q2, Q3). The non-trivial eigenvalue printed is
(Q1^2-1) Q2^2 (Q3^2-1) / ((Q1^2 Q2^2-1)(Q2^2 Q3^2-1)), the spectrum quoted
from Theorem 37 of Allen et al. (2024) in the proof of Lemma 1 of the paper.

Author: James Allen.

Provenance: supplied directly by the author; not from random_circuit_concentration.

Deviations from the original: sp.eye(2) in place of sp.Matrix(np.eye(2)),
so the output has exact integer coefficients rather than 1.0 factors.
"""
import sympy as sp

q1, q2, q3 = sp.symbols('Q1,Q2,Q3') # Local Hilbert space indices of the three sites

i2 = sp.eye(2)
# The W_{12} matrix
on_diag = 1/((q1*q2)**2-1)
off_diag = -1/(q1*q2*((q1*q2)**2-1))
Wg12 = sp.Matrix([[on_diag, 0,0,off_diag],[0,0,0,0],[0,0,0,0],[off_diag,0,0,on_diag]])
Wg12 = sp.kronecker_product(Wg12, i2)

# The C_{23} matrix
Cyq1 = sp.Matrix([[q1**2, q1],[q1,q1**2]])
Cyq2 = sp.Matrix([[q2**2, q2],[q2,q2**2]])
Cyq3 = sp.Matrix([[q3**2, q3],[q3,q3**2]])
Cy23 = sp.kronecker_product(i2, sp.kronecker_product(Cyq2, Cyq3))

# The W_{23} matrix
on_diag = 1/((q3*q2)**2-1)
off_diag = -1/(q3*q2*((q3*q2)**2-1))
Wg23 = sp.Matrix([[on_diag, 0,0,off_diag],[0,0,0,0],[0,0,0,0],[off_diag,0,0,on_diag]])
Wg23 = sp.kronecker_product(i2, Wg23)

# The C_{12} matrix
Cy12 = sp.kronecker_product(sp.kronecker_product(Cyq1, Cyq2), i2)

Block2 = sp.simplify(Wg23 @ Cy23 @ Wg12 @ Cy12 @ Wg23 @ Cy23)
print(sp.factor(Block2.eigenvals()))
