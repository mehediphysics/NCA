import unittest
import numpy as np
from fermionic_operators import FermionicOperators  # adjust if class is in the same file

class TestFermionicOperators(unittest.TestCase):
    def test_shapes(self):
        """Check that operators have the correct dimensions."""
        for N in [1, 2, 3, 4, 5]:
            f = FermionicOperators(N)
            for i in range(N):
                c = f.get_annihilation(i)
                cd = f.get_creation(i)
                self.assertEqual(c.shape, (2**N, 2**N))
                self.assertEqual(cd.shape, (2**N, 2**N))

    def test_hermitian_conjugation(self):
        """Verify that c_i^† = c_i† (Hermitian conjugate)."""
        for N in [1, 2, 3, 4, 5]:
            f = FermionicOperators(N)
            for i in range(N):
                c = f.get_annihilation(i)
                cd = f.get_creation(i)
                self.assertTrue(np.allclose(cd, c.conj().T))

    def test_anticommutation_relations(self):
        """Check {c_i, c_j^†} = δ_ij and {c_i, c_j} = 0."""
        for N in [1, 2, 3, 4, 5]:
            f = FermionicOperators(N)
            dim = 2**N
            I = np.eye(dim, dtype=complex)
            zero = np.zeros((dim, dim), dtype=complex)

            for i in range(N):
                for j in range(N):
                    ci, cj = f.get_annihilation(i), f.get_annihilation(j)
                    cdi, cdj = f.get_creation(i), f.get_creation(j)

                    # {c_i, c_j} = 0
                    self.assertTrue(np.allclose(ci @ cj + cj @ ci, zero))

                    # {c_i^†, c_j^†} = 0
                    self.assertTrue(np.allclose(cdi @ cdj + cdj @ cdi, zero))

                    # {c_i, c_j^†} = δ_ij
                    lhs = ci @ cdj + cdj @ ci
                    rhs = I if i == j else zero
                    self.assertTrue(np.allclose(lhs, rhs))

    def test_number_operator_properties(self):
        """Check n_i = c_i^† c_i has correct idempotent behavior."""
        for N in [1, 2, 3, 4, 5]:
            f = FermionicOperators(N)
            for i in range(N):
                c = f.get_annihilation(i)
                cd = f.get_creation(i)
                n = cd @ c
                # n^2 = n
                self.assertTrue(np.allclose(n @ n, n))
                # Eigenvalues should be 0 or 1
                vals = np.linalg.eigvals(n)
                rounded_vals = np.unique(np.round(vals.real, 10))
                self.assertTrue(np.all(v in [0, 1] for v in rounded_vals))

    def test_action_on_vac_state(self):
        """Check that any anihilation operator acting on the vacuum state gives zero."""
        for N in [1, 2, 3, 4, 5]:
            dim = 2**N
            f = FermionicOperators(N)
            zero = np.zeros((dim), dtype=complex)
            vac_state = np.zeros((dim), dtype=complex)
            vac_state[0] = 1
            for i in range(N):
                c = f.get_annihilation(i)
                self.assertTrue(np.allclose(c @ vac_state, zero))




if __name__ == "__main__":
    unittest.main()
