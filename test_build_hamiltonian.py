import unittest
import numpy as np
import tempfile
from build_hamiltonian import read_hamiltonian_and_params, build_hamiltonian

class TestBuildHamiltonian(unittest.TestCase):

    def setUp(self):
        # Create a small temporary text file with a test Hamiltonian
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.temp_file.write("""
# Number of single-particle states
N 2

# One-body terms: i j coeff
0 0 1.0
1 1 2.0
0 1 0.5
1 0 0.5

# Two-body terms: i j k l coeff
0 0 1 1 0.25

# Time parameters
t_max 10.0
dt 0.05

# Green's function indices (0-based)
i 0
j 0
""")
        self.temp_file.flush()

    def tearDown(self):
        self.temp_file.close()

    def test_hamiltonian_shape(self):
        N, one_body, two_body, t_max, dt, i_gf, j_gf = read_hamiltonian_and_params(self.temp_file.name)
        H = build_hamiltonian(N, one_body, two_body)
        self.assertEqual(H.shape, (2**N, 2**N))

    def test_hamiltonian_hermitian(self):
        N, one_body, two_body, t_max, dt, i_gf, j_gf = read_hamiltonian_and_params(self.temp_file.name)
        H = build_hamiltonian(N, one_body, two_body)
        self.assertTrue(np.allclose(H, H.conj().T), "Hamiltonian is not Hermitian")

    def test_known_entry(self):
        # Optional: check a known matrix element
        N, one_body, two_body, t_max, dt, i_gf, j_gf = read_hamiltonian_and_params(self.temp_file.name)
        H = build_hamiltonian(N, one_body, two_body)
        # For this small test case, the diagonal element corresponding to |00> should be 0
        # |00> is index 0 in the 2^2 basis
        self.assertAlmostEqual(H[0, 0].real, 0.0, places=12)

if __name__ == '__main__':
    unittest.main()

