import numpy as np
import pytest
from fermionic_operators import FermionicOperators  # adjust if class is in same file

@pytest.mark.parametrize("N", [1, 2, 3])
def test_operator_shapes(N):
    f = FermionicOperators(N)
    for i in range(N):
        c = f.get_annihilation(i)
        cd = f.get_creation(i)
        assert c.shape == (2**N, 2**N)
        assert cd.shape == (2**N, 2**N)

def test_hermitian_conjugation():
    N = 2
    f = FermionicOperators(N)
    for i in range(N):
        c = f.get_annihilation(i)
        cd = f.get_creation(i)
        assert np.allclose(cd, c.conj().T)

def test_anticommutation_relations():
    N = 3
    f = FermionicOperators(N)
    dim = 2**N
    I = np.eye(dim, dtype=complex)
    
    for i in range(N):
        for j in range(N):
            ci = f.get_annihilation(i)
            cj = f.get_annihilation(j)
            cdi = f.get_creation(i)
            cdj = f.get_creation(j)

            # {c_i, c_j} = 0
            assert np.allclose(ci @ cj + cj @ ci, np.zeros((dim, dim)))

            # {c_i^†, c_j^†} = 0
            assert np.allclose(cdi @ cdj + cdj @ cdi, np.zeros((dim, dim)))

            # {c_i, c_j^†} = δ_ij
            lhs = ci @ cdj + cdj @ ci
            rhs = I if i == j else np.zeros((dim, dim))
            assert np.allclose(lhs, rhs)

def test_number_operator():
    N = 2
    f = FermionicOperators(N)
    for i in range(N):
        c = f.get_annihilation(i)
        cd = f.get_creation(i)
        n = cd @ c  # number operator
        # n^2 = n
        assert np.allclose(n @ n, n)
        # Eigenvalues are 0 or 1
        vals = np.linalg.eigvals(n)
        assert np.allclose(sorted(set(np.round(vals.real, 8))), [0, 1])
