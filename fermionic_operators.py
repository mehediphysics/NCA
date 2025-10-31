import numpy as np

class FermionicOperators:
    def __init__(self, N):
        """Initialize with N single-particle states."""
        self.N = N
        self.dim = 2**N
        # Pauli matrices and identity
        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.c_ops = [self._annihilation_op(i) for i in range(N)]
        self.cd_ops = [self._creation_op(i) for i in range(N)]

    def _kron_list(self, ops):
        """Kronecker product of a list of matrices."""
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _annihilation_op(self, i):
        """Jordan–Wigner annihilation operator for site i."""
        ops = [self.Z]*i + [(self.X + 1j*self.Y)/2] + [self.I]*(self.N - i - 1)
        return self._kron_list(ops)

    def _creation_op(self, i):
        """Jordan–Wigner creation operator for site i."""
        ops = [self.Z]*i + [(self.X - 1j*self.Y)/2] + [self.I]*(self.N - i - 1)
        return self._kron_list(ops)

    def get_creation(self, i):
        """Return c_i^dagger."""
        return self.cd_ops[i]

    def get_annihilation(self, i):
        """Return c_i."""
        return self.c_ops[i]

    def print_occupation(self, psi_n):
        """
        Print the occupation number representation of a basis state vector psi_n.
        Assumes psi_n has a single 1 corresponding to basis state index n.
        """
        n_index = np.argmax(np.abs(psi_n))  # find which basis vector is occupied
        occ = np.array(list(np.binary_repr(n_index, width=self.N)), dtype=int)
        print(f"Basis index {n_index} -> occupation: {occ}")
        return occ


# Example usage:
if __name__ == "__main__":
    N = 3
    fermions = FermionicOperators(N)
    c0 = fermions.get_annihilation(0)
    cd0 = fermions.get_creation(0)
    print("c_0 shape:", c0.shape)
    print("Anticommutator check:", np.allclose(c0 @ cd0 + cd0 @ c0, np.eye(2**N)))
